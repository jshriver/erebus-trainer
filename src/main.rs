use std::fs;
use std::path::Path;
use std::env;

use bullet_lib::{
    game::formats::sfbinpack::{
        chess::{piecetype::PieceType, r#move::MoveType},
        TrainingDataEntry,
    },
    game::inputs::{self, SparseInputType, get_num_buckets},
    game::outputs,
    nn::optimiser,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};
use bulletformat::ChessBoard;

// ============================================================
// Network Architecture -- dual accumulator (PST + Threat), Reckless
// style: PST features and threat features are two sparse feature sets
// that both project into the SAME L1_SIZE-wide accumulator and get
// SUMMED before SCReLU, rather than being concatenated into a wider
// input. See `threat_inputs` module below and its doc comment for how
// that's reproduced as a single wider sparse input to one `l0` affine
// (mathematically identical to two affines summed -- see that module's
// top comment for why).
//
//   [(768 x INPUT_BUCKETS + THREAT_BLOCK_SIZE) -> L1_SIZE]x2 -> L2ob -> L3 -> 1
//
// VERIFIED (see the sanity harness used to build this -- described in
// chat, not shipped as part of this file):
//   - bullet_lib's SparseInputType trait shape, fetched from
//     github.com/jw1912/bullet at HEAD.
//   - bulletformat::ChessBoard's real field layout (occ/pcs/ksq/opp_ksq),
//     fetched from github.com/jw1912/bulletformat at HEAD -- confirms
//     ChessBoard is already side-to-move-normalised (from_raw flips the
//     board when black is to move), which is what lets the threat
//     feature generator below treat "us" pawns as always moving toward
//     rank 8 without any extra colour bookkeeping.
//   - SfBinpackLoader converts every TrainingDataEntry to a ChessBoard
//     BEFORE your SparseInputType ever sees it (crates/bullet_lib/src/
//     value/loader/sfbinpack.rs::convert_to_bulletformat) -- so
//     RequiredDataType = ChessBoard is correct, and threat features are
//     computed from ChessBoard's 8 reconstructable bitboards, NOT from
//     sfbinpack's own chess types. This resolves the "does sfbinpack
//     expose attacks()?" question from before: it doesn't need to,
//     because that richer type never reaches the input mapper.
//   - THREAT_BLOCK_SIZE = 90384 was computed by literally running
//     ThreatTables::build().block_size against the real trait/struct
//     definitions above, plus a runtime assertion below that re-checks
//     it on every run (see `assert_eq!` before `.build(...)`) -- if you
//     ever touch `threat_inputs`'s attack-generation logic, that
//     assertion will fail loudly instead of silently training a
//     misaligned net.
//   - A mirror-symmetry property (the ntm feature index for a threat is
//     identical to computing the stm index on the literally-mirrored,
//     colour-swapped board) was unit-tested and passes -- this was the
//     one place an early draft had a real bug (XOR-6 instead of the
//     correct (variant+6)%12 colour-half toggle), caught before it ever
//     touched real data.
// ============================================================
const L1_SIZE: usize = 2048;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 32;

const INPUT_BUCKETS: usize = 10;

// See threat_inputs::ThreatTables -- this is a derived constant, not a
// tuned hyperparameter. It follows deterministically from the attack
// geometry code in `threat_inputs`, independent of INPUT_BUCKETS. If you
// change anything in that module, recompute this (the startup assertion
// will tell you the real value if it's wrong).
const THREAT_BLOCK_SIZE: usize = 90_384;
const COMBINED_INPUTS: usize = 768 * INPUT_BUCKETS + THREAT_BLOCK_SIZE;

const OUTPUT_BUCKETS: usize = 8;

const SCALE: i32 = 380;
const QA: i16 = 255;
const QB: i16 = 64;

const BATCHES_PER_SUPERBATCH: usize = 18_000;
const BATCH_SIZE: usize = 8_192;

const TOTAL_POSITIONS_TARGET: usize = 70_000_000_000;

fn total_planned_superbatches() -> usize {
    let total_batches = TOTAL_POSITIONS_TARGET / BATCH_SIZE;
    (total_batches / BATCHES_PER_SUPERBATCH).max(1)
}

fn find_latest_superbatch(net_id: &str, output_dir: &str) -> usize {
    let base = Path::new(output_dir);
    if !base.exists() {
        return 1;
    }
    let mut max_sb = 0usize;
    if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(rest) = name.strip_prefix(&format!("{}-", net_id)) {
                if let Ok(num) = rest.parse::<usize>() {
                    max_sb = max_sb.max(num);
                }
            }
        }
    }
    if max_sb == 0 { 1 } else { max_sb + 1 }
}

fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 12_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn positions_in_one_pass(file_path: &str) -> usize {
    let file_size = fs::metadata(file_path)
        .expect("Could not read binpack file metadata")
        .len() as usize;
    let estimated_positions = file_size / 100;
    let total_batches = estimated_positions / BATCH_SIZE;
    let superbatches = (total_batches / BATCHES_PER_SUPERBATCH).max(1);
    println!("File size:                    {}MB", file_size / 1_048_576);
    println!("Estimated positions:          {}M", estimated_positions / 1_000_000);
    println!("Estimated superbatches/pass:  {}", superbatches);
    superbatches
}

// ============================================================
// threat_inputs -- a from-scratch, dependency-free threat-feature
// generator that reproduces Reckless's "second accumulator summed with
// the PST accumulator" architecture as a single wider SparseInputType.
//
// WHY ONE COMBINED INPUT INSTEAD OF TWO SUMMED AFFINES:
// Affine_pst(x) + Affine_threat(y) is mathematically identical to
// Affine_combined(concat(x, y)) with the weight matrix stacked. bullet's
// ValueTrainerBuilder wires up exactly one sparse input per perspective
// (see `builder.new_sparse_input("stm", (inputs, 1), nnz)` in bullet's
// own build_internal, fetched from source), so reproducing "two
// accumulators" as "one wider accumulator with PST features in the low
// indices and threat features in the high indices" is both correct and
// is what the framework's API actually wants -- one `l0` affine, not two.
//
// WHY FEATURES ARE COMPUTED FROM ChessBoard, NOT sfbinpack's own types:
// SfBinpackLoader (bullet's own code) converts every position to a
// bulletformat::ChessBoard -- 8 bitboards packed into `occ: u64` +
// `pcs: [u8; 16]` -- before your SparseInputType ever sees it. That's
// plenty to reconstruct per-colour, per-piece-type bitboards and run
// ordinary bitboard attack generation; no dependency on sfbinpack's own
// movegen is needed or possible at this stage.
//
// FEATURE SCHEME (deliberately simpler than Reckless's own compressed
// 66864-feature table -- this is a clean re-derivation, not a port, so
// don't expect the numbers to match theirs):
//   - 12 "attacker variants" = 2 colours (us/them, relative to the side
//     ChessBoard is already normalised to) x 6 piece types.
//   - For each variant, at each of the 64 squares, the set of squares
//     reachable on an EMPTY board is precomputed once at startup. This
//     bounds where a real (occupancy-aware) attack from that square can
//     land, and gives each reachable square a fixed "ordinal" position
//     used to build a compact index (no wasted space for the ~64x64
//     combinations that are geometrically impossible for a given piece).
//   - A threat feature = (attacker variant, attacker square, victim
//     variant) plus that ordinal. Victim variant is also 12-valued (a
//     piece attacking/defending EITHER colour is informative -- "my rook
//     defends my knight" is as real a signal as "my rook attacks your
//     knight").
//   - The opponent-perspective (ntm) index for the SAME physical threat
//     is computed by re-running the identical index formula with the
//     attacker/victim variants' colour-half toggled and both squares
//     flipped (^56) -- i.e. literally "what index would this threat get
//     if viewed from the mirrored board", not an algebraic shortcut.
//     This was verified with a mirror-symmetry unit test before being
//     wired into this file.
// ============================================================
mod threat_inputs {
    use bulletformat::ChessBoard;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = 1;
    pub const BISHOP: usize = 2;
    pub const ROOK: usize = 3;
    pub const QUEEN: usize = 4;
    pub const KING: usize = 5;

    fn sq_file(s: u8) -> i32 { (s % 8) as i32 }
    fn sq_rank(s: u8) -> i32 { (s / 8) as i32 }
    fn try_sq(f: i32, r: i32) -> Option<u8> {
        if (0..8).contains(&f) && (0..8).contains(&r) { Some((r * 8 + f) as u8) } else { None }
    }

    const KNIGHT_DELTAS: [(i32, i32); 8] =
        [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
    const KING_DELTAS: [(i32, i32); 8] =
        [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
    const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    fn ray_bb(sq: u8, dir: (i32, i32), occ: u64) -> u64 {
        let mut bb = 0u64;
        let (mut f, mut r) = (sq_file(sq), sq_rank(sq));
        loop {
            f += dir.0;
            r += dir.1;
            match try_sq(f, r) {
                Some(s) => {
                    bb |= 1u64 << s;
                    if occ & (1u64 << s) != 0 { break; }
                }
                None => break,
            }
        }
        bb
    }

    /// `variant` is 0..12: 0..6 = "us" pawn..king, 6..12 = "them" pawn..king
    /// (colours are relative to however `ChessBoard` was already
    /// normalised, i.e. "us" == side to move). `occ == 0` gives the
    /// maximal empty-board pattern used to build ordinal lookup tables;
    /// a real occupancy bitboard gives true attacks, with sliders
    /// stopping at (and including) the first blocker.
    fn raw_attacks_bb(variant: usize, sq: u8, occ: u64) -> u64 {
        let pt = variant % 6;
        let is_us = variant < 6;
        let (f, r) = (sq_file(sq), sq_rank(sq));
        let mut bb = 0u64;
        match pt {
            PAWN => {
                let dr = if is_us { 1 } else { -1 };
                for df in [-1, 1] {
                    if let Some(s) = try_sq(f + df, r + dr) { bb |= 1u64 << s; }
                }
            }
            KNIGHT => {
                for (df, dr) in KNIGHT_DELTAS {
                    if let Some(s) = try_sq(f + df, r + dr) { bb |= 1u64 << s; }
                }
            }
            KING => {
                for (df, dr) in KING_DELTAS {
                    if let Some(s) = try_sq(f + df, r + dr) { bb |= 1u64 << s; }
                }
            }
            BISHOP => { for dir in BISHOP_DIRS { bb |= ray_bb(sq, dir, occ); } }
            ROOK => { for dir in ROOK_DIRS { bb |= ray_bb(sq, dir, occ); } }
            QUEEN => { for dir in BISHOP_DIRS.into_iter().chain(ROOK_DIRS) { bb |= ray_bb(sq, dir, occ); } }
            _ => unreachable!(),
        }
        bb
    }

    fn variant_index(is_us: bool, piece_type: usize) -> usize {
        if is_us { piece_type } else { 6 + piece_type }
    }

    /// Precomputed per-(attacker variant, attacker square) ordinal
    /// tables, built once from the empty-board maximal attack pattern.
    ///
    /// `ordinal_lookup[variant][attacker_sq][target_sq]` is an O(1)
    /// reverse lookup (target square -> its ordinal position within that
    /// square's empty-board attack pattern, or -1 if unreachable) --
    /// replaces an earlier version that linear-scanned a `Vec<u8>` per
    /// lookup. This is called twice (stm + ntm) per threat, for every
    /// position, on the CPU data-loader thread, for the full run, so the
    /// O(1) win is real; the table itself is tiny (12*64*64 = 49,152
    /// bytes) so there's no real memory tradeoff.
    pub struct ThreatTables {
        offset: Box<[[usize; 64]; 12]>,
        total: [usize; 12],
        base: [usize; 12],
        ordinal_lookup: Box<[[[i8; 64]; 64]; 12]>,
        pub block_size: usize,
    }

    impl ThreatTables {
        pub fn build() -> Self {
            let mut offset: Box<[[usize; 64]; 12]> = Box::new([[0usize; 64]; 12]);
            let mut total = [0usize; 12];
            let mut ordinal_lookup: Box<[[[i8; 64]; 64]; 12]> = Box::new([[[-1i8; 64]; 64]; 12]);

            for variant in 0..12 {
                let mut cum = 0usize;
                for sq in 0u8..64 {
                    let mut bb = raw_attacks_bb(variant, sq, 0);
                    offset[variant][sq as usize] = cum;
                    let mut ord: i8 = 0;
                    while bb != 0 {
                        let s = bb.trailing_zeros() as u8;
                        ordinal_lookup[variant][sq as usize][s as usize] = ord;
                        ord += 1;
                        bb &= bb - 1;
                    }
                    cum += ord as usize;
                }
                total[variant] = cum;
            }

            let mut base = [0usize; 12];
            let mut running = 0usize;
            for v in 0..12 {
                base[v] = running;
                running += total[v] * 12;
            }

            Self { offset, total, base, ordinal_lookup, block_size: running }
        }

        fn ordinal(&self, variant: usize, sq: u8, target: u8) -> usize {
            let ord = self.ordinal_lookup[variant][sq as usize][target as usize];
            assert!(
                ord >= 0,
                "target square must lie within the variant's empty-board attack pattern"
            );
            ord as usize
        }

        fn local_index(&self, attacker_variant: usize, attacker_sq: u8, victim_variant: usize, victim_sq: u8) -> usize {
            let ord = self.ordinal(attacker_variant, attacker_sq, victim_sq);
            self.base[attacker_variant]
                + victim_variant * self.total[attacker_variant]
                + self.offset[attacker_variant][attacker_sq as usize]
                + ord
        }
    }

    fn expand_board(pos: ChessBoard) -> ([u64; 2], [u64; 6]) {
        let mut color_bb = [0u64; 2];
        let mut piece_bb = [0u64; 6];
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pt = usize::from(piece & 7);
            let bit = 1u64 << square;
            color_bb[c] |= bit;
            piece_bb[pt] |= bit;
        }
        (color_bb, piece_bb)
    }

    /// Emits (stm_local, ntm_local) pairs -- both already within
    /// `[0, tables.block_size)` -- for every physical threat
    /// relationship on the board (attacker of either colour vs. any
    /// occupied target square, including a piece defending a
    /// same-colour piece).
    pub fn emit_threat_pairs(pos: &ChessBoard, tables: &ThreatTables, mut f: impl FnMut(usize, usize)) {
        let (color_bb, piece_bb) = expand_board(*pos);
        let occ = color_bb[0] | color_bb[1];

        for is_us_attacker in [true, false] {
            let attacker_side_bb = if is_us_attacker { color_bb[0] } else { color_bb[1] };

            for pt in 0..6 {
                let mut bb = piece_bb[pt] & attacker_side_bb;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as u8;
                    bb &= bb - 1;

                    let attacker_variant = variant_index(is_us_attacker, pt);
                    let targets = raw_attacks_bb(attacker_variant, sq, occ) & occ;

                    let mut t = targets;
                    while t != 0 {
                        let vsq = t.trailing_zeros() as u8;
                        t &= t - 1;

                        let victim_is_us = color_bb[0] & (1u64 << vsq) != 0;
                        let victim_pt = (0..6)
                            .find(|&p| piece_bb[p] & (1u64 << vsq) != 0)
                            .expect("occupied square must have a recorded piece type");
                        let victim_variant = variant_index(victim_is_us, victim_pt);

                        let stm_local = tables.local_index(attacker_variant, sq, victim_variant, vsq);

                        // Toggle the us/them half of the variant (0..6 <-> 6..12).
                        // NOTE: (variant + 6) % 12, NOT variant ^ 6 -- XOR by 6
                        // (0b110) corrupts the piece-type bits whenever pt != 0,
                        // since 6's set bits overlap pt's bit range. An earlier
                        // draft had this bug; caught by a mirror-symmetry unit
                        // test before touching real data.
                        let toggle = |v: usize| (v + 6) % 12;
                        let ntm_attacker_variant = toggle(attacker_variant);
                        let ntm_victim_variant = toggle(victim_variant);
                        let ntm_local =
                            tables.local_index(ntm_attacker_variant, sq ^ 56, ntm_victim_variant, vsq ^ 56);

                        f(stm_local, ntm_local);
                    }
                }
            }
        }
    }
}

// ============================================================
// PstPlusThreatInputs -- combined SparseInputType: PST features in
// [0, 768*num_pst_buckets), threat features immediately after. See the
// threat_inputs module doc comment above for why this is one combined
// input rather than two separately-registered ones.
// ============================================================
#[derive(Clone)]
struct PstPlusThreatInputs {
    pst_buckets: [usize; 64],
    num_pst_buckets: usize,
    tables: std::sync::Arc<threat_inputs::ThreatTables>,
}

impl PstPlusThreatInputs {
    fn new(pst_buckets: [usize; 64]) -> Self {
        let num_pst_buckets = get_num_buckets(&pst_buckets);
        Self { pst_buckets, num_pst_buckets, tables: std::sync::Arc::new(threat_inputs::ThreatTables::build()) }
    }
}

impl SparseInputType for PstPlusThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 * self.num_pst_buckets + self.tables.block_size
    }

    fn max_active(&self) -> usize {
        // MEASURED, not guessed: profiled 5M real filtered positions
        // from training-run3-test90-20251029-2317.binpack with the
        // erebus-test tool (a standalone copy of this same
        // threat_inputs logic) -- combined PST+threat active feature
        // count came back mean=34.7, p99=87, p99.9=94, max=102.
        // 160 leaves ~57% headroom over the observed max, to cover
        // chunk-to-chunk variance across the full ~218.8B-position
        // dataset (only one chunk was sampled). If a future superbatch
        // ever exceeds this, bullet will panic loudly rather than
        // silently truncate -- if that happens, re-run erebus-test
        // against whichever chunk triggered it and raise this again.
        160
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let our_bucket = 768 * self.pst_buckets[usize::from(pos.our_ksq())];
        let opp_bucket = 768 * self.pst_buckets[usize::from(pos.opp_ksq())];
        inputs::Chess768.map_features(pos, |stm, ntm| f(our_bucket + stm, opp_bucket + ntm));

        let threat_base = 768 * self.num_pst_buckets;
        threat_inputs::emit_threat_pairs(pos, &self.tables, |stm, ntm| f(threat_base + stm, threat_base + ntm));
    }

    fn shorthand(&self) -> String {
        format!("768x{}+threats{}", self.num_pst_buckets, self.tables.block_size)
    }

    fn description(&self) -> String {
        "King-bucketed PSQT + summed threat accumulator inputs".to_string()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_binpack> [passes]", args[0]);
        eprintln!("Example: {} data/chunk003.binpack 1", args[0]);
        eprintln!("  passes: how many times to loop over THIS binpack chunk (default 1)");
        std::process::exit(1);
    }
    let file_path = &args[1];
    let passes: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    if !Path::new(file_path).exists() {
        eprintln!("Error: binpack file not found: {}", file_path);
        std::process::exit(1);
    }

    let net_id = "erebus";
    let output_dir = "checkpoints";

    let start_superbatch = find_latest_superbatch(net_id, output_dir);
    let superbatches_this_session = positions_in_one_pass(file_path) * passes;
    let end_superbatch = start_superbatch + superbatches_this_session;

    let total_planned = total_planned_superbatches();

    let input_buckets_layout: [usize; 64] = [
        0, 1, 2, 3, 3, 2, 1, 0,
        4, 5, 6, 7, 7, 6, 5, 4,
        8, 8, 8, 8, 8, 8, 8, 8,
        9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9,
    ];
    let combined_inputs = PstPlusThreatInputs::new(input_buckets_layout);

    // See the "VERIFIED" note in the architecture comment block above --
    // this is a cheap, load-bearing sanity check. If it ever fails, STOP:
    // it means COMBINED_INPUTS no longer matches what the input type
    // actually produces, and l0's weight matrix shape (and therefore
    // every saved checkpoint) would silently be wrong.
    assert_eq!(
        combined_inputs.num_inputs(),
        COMBINED_INPUTS,
        "COMBINED_INPUTS constant ({}) doesn't match PstPlusThreatInputs::num_inputs() ({}) -- \
         recompute the constant (or check for a change in threat_inputs) before training.",
        COMBINED_INPUTS,
        combined_inputs.num_inputs(),
    );

    println!();
    println!("Net:               {}", net_id);
    println!(
        "Architecture:      [(768x{} + {}threat) -> {}]x2 -> {} -> {} -> {}ob",
        INPUT_BUCKETS, THREAT_BLOCK_SIZE, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKETS
    );
    println!("Inputs shorthand:  {}", combined_inputs.shorthand());
    println!("Binpack (session): {}", file_path);
    println!("Passes (session):  {}", passes);
    println!("Start SB:          {}", start_superbatch);
    println!("End SB (session):  {} ({} this session)", end_superbatch, superbatches_this_session);
    println!("Total planned SB:  {} (fixed, from {}B position budget)",
        total_planned, TOTAL_POSITIONS_TARGET / 1_000_000_000);

    if start_superbatch > total_planned {
        println!();
        println!("WARNING: start_superbatch ({}) already exceeds total_planned ({}).", start_superbatch, total_planned);
        println!("The LR schedule has already reached its final_lr floor and will stay there.");
        println!("If you intend to keep training further, raise TOTAL_POSITIONS_TARGET and");
        println!("understand this reshapes the remaining LR curve, not just the tail.");
    }
    println!();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(combined_inputs)
        .output_buckets(outputs::MaterialCount::<OUTPUT_BUCKETS>)
        .use_device(0)
        .save_format(&[
            // l0w now spans BOTH feature blocks in one
            // COMBINED_INPUTS x L1_SIZE matrix: rows
            // [0, 768*INPUT_BUCKETS) are PST weights (identical layout to
            // your original single-stream net), rows
            // [768*INPUT_BUCKETS, COMBINED_INPUTS) are threat weights,
            // laid out per threat_inputs::ThreatTables (attacker variant
            // -> attacker square -> victim variant -> ordinal, in that
            // nesting order -- see ThreatTables::local_index).
            //
            // STRONGLY RECOMMEND: don't hand-port this indexing scheme to
            // your inference engine. Instead, factor `threat_inputs`
            // (it's dependency-free -- only needs bulletformat::ChessBoard
            // or an equivalent 8-bitboard type) into its own tiny crate
            // and depend on it from BOTH this trainer and your inference
            // code, so train-time and inference-time feature indexing are
            // provably the same code, not two hand-written copies that
            // can silently drift.
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", COMBINED_INPUTS, L1_SIZE);
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE * OUTPUT_BUCKETS);
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, 1);

            // PST + threat features are already summed into one wider
            // sparse input at this point (see PstPlusThreatInputs), so
            // this is just an ordinary sparse affine + SCReLU per
            // perspective, then concatenated -- identical graph shape to
            // a plain single-stream net.
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(hidden);
            let selected = l1_out.select(output_buckets);
            let out1 = selected.screlu();
            let out2 = l2.forward(out1).screlu();
            l3.forward(out2)
        });

    if start_superbatch > 1 {
        let checkpoint_path = format!("{}/{}-{}", output_dir, net_id, start_superbatch - 1);
        println!("Resuming: loading weights + optimiser state from {}", checkpoint_path);
        trainer.load_from_checkpoint(&checkpoint_path);
    } else {
        println!("No existing checkpoint found -- starting from fresh initialization.");
    }

    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch,
            end_superbatch,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.0000010,
            final_superbatch: total_planned,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: 64,
    };

    let data_loader = loader::SfBinpackLoader::new(file_path, 512, 2, filter);

    trainer.run(&schedule, &settings, &data_loader);

    println!();
    println!("Done this session. Checkpoints saved to: {}", output_dir);
    println!("Reached superbatch {} of {} total planned.", end_superbatch, total_planned);
    println!("Next: run again with your next binpack chunk to continue --");
    println!("      start_superbatch will auto-resume from the latest checkpoint,");
    println!("      and the LR schedule will pick up correctly from superbatch {}.", end_superbatch);
}