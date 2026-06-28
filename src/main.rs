use std::fs;
use std::path::Path;
use std::env;

use bullet_lib::{
    game::formats::sfbinpack::{
        chess::{piecetype::PieceType, r#move::MoveType},
        TrainingDataEntry,
    },
    game::inputs,
    game::outputs,
    nn::optimiser,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

// ============================================================
// Network Architecture
// [768 x INPUT_BUCKETS -> L1_SIZE]x2 -> L2_SIZE -> L3_SIZE -> 1
// with OUTPUT_BUCKETS separate weight sets for the dense layers.
//
// Scaled up from the original 768/16/32 toward production scale.
// L1 2048 (up from 768): biggest lever for representational capacity;
//   SCReLU sparsity keeps inference cost sub-linear in width.
// L2 32 (up from 16): original 768->16 ratio was a steep bottleneck;
//   32 is in line with what current strong nets pair with a wide L1.
// L3 32: unchanged, already in the normal range.
//
// These constants must match your inference Parameters struct exactly.
// Do not change them after serious training begins.
// ============================================================
const L1_SIZE: usize = 2048;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 32;

// Input bucketing unchanged for now -- flagged separately as a future
// redesign (current layout collapses the whole own-half into one bucket,
// which is coarser than typical). Not touched in this pass since changing
// it requires a fresh run and should be tested independently.
const INPUT_BUCKETS: usize = 10;

// Output bucketing: 8 buckets by piece count, per the verified bullet
// MaterialCount<N> formula: divisor = ceil(32/N), bucket = (piece_count-2)/divisor.
const OUTPUT_BUCKETS: usize = 8;

// ============================================================
// Quantization -- unchanged, must match inference FT_QUANT / L1_QUANT.
// ============================================================
const SCALE: i32 = 380;
const QA: i16 = 255;
const QB: i16 = 64;

// ============================================================
// Training throughput constants
//
// BATCH_SIZE reduced from 16,384 -> 8,192 as a starting point for a
// Kaggle P100 (16GB) at L1=2048. This is a STARTING POINT, not a verified
// safe value -- run a short smoke test (a few dozen batches) before
// committing to a long session, and raise/lower based on actual memory
// headroom. Getting this wrong fails as a CUDA OOM crash mid-session,
// which is expensive to discover on a metered, session-limited resource.
// ============================================================
const BATCHES_PER_SUPERBATCH: usize = 1_000;
const BATCH_SIZE: usize = 8_192; // SMOKE-TEST THIS before a long run.

// ============================================================
// Fixed global training budget.
//
// THIS IS THE FIX for the cross-session LR schedule bug: TOTAL_POSITIONS_TARGET
// and the superbatch count derived from it are FIXED CONSTANTS, computed the
// same way every time this binary is invoked -- they do NOT depend on which
// binpack chunk this particular session happens to be loading, or how big
// that chunk is. This is what CosineDecayLR's `final_superbatch` anchors to,
// so the LR curve is consistent across the whole multi-month, many-session
// run instead of silently resetting its notion of "near the end" every time
// you restart on a new binpack file.
//
// 300B positions, as discussed. Update this constant only if your total
// planned data budget actually changes -- and if you do change it mid-run,
// understand that it reshapes the whole remaining LR curve, not just the
// tail end.
// ============================================================
const TOTAL_POSITIONS_TARGET: usize = 300_000_000_000;

fn total_planned_superbatches() -> usize {
    let total_batches = TOTAL_POSITIONS_TARGET / BATCH_SIZE;
    (total_batches / BATCHES_PER_SUPERBATCH).max(1)
}

// ============================================================
// Checkpoint resumption
// Scans output_dir for files named "<net_id>-<N>" and resumes
// from the next superbatch after the highest found.
// ============================================================
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

// ============================================================
// Position filter -- unchanged.
// ============================================================
fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 12_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

// ============================================================
// Estimate how many superbatches this SESSION's binpack covers.
// This is now used ONLY to compute end_superbatch for this invocation
// (i.e. where to stop and checkpoint) -- it is NOT used to anchor the
// LR schedule anymore. That's the whole point of the fix.
// ============================================================
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

    // Fixed across ALL invocations -- this is the anchor for the LR curve.
    let total_planned = total_planned_superbatches();

    println!();
    println!("Net:               {}", net_id);
    println!("Architecture:      [768x{}->{}]x2 -> {} -> {} -> {}ob",
        INPUT_BUCKETS, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKETS);
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
        .inputs(inputs::ChessBuckets::new([
            0, 1, 2, 3, 3, 2, 1, 0,
            4, 5, 6, 7, 7, 6, 5, 4,
            8, 8, 8, 8, 8, 8, 8, 8,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
        ]))
        .output_buckets(outputs::MaterialCount::<OUTPUT_BUCKETS>)
        .use_device(0)
        .save_format(&[
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
            let l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, L1_SIZE);
            // l1 fans out to L2_SIZE * OUTPUT_BUCKETS -- bucket-specialised right
            // after the FT, at the layer doing the heaviest compression (4096->32).
            // Confirmed via Colab smoke test (select() chained mid-pipeline, followed
            // by further forward()/screlu() calls, trained one superbatch cleanly with
            // finite loss) that this composes correctly -- not just a terminal-only op
            // as bullet's own example demonstrates it.
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE * OUTPUT_BUCKETS);
            // l2/l3 remain fully shared across all output buckets -- bullet has no
            // built-in mechanism for full per-bucket replication of these layers
            // (confirmed: new_affine_custom's extra param is bias_cols, not a bucket
            // count). Bucketing only at l1 concentrates the specialization where it
            // has the most leverage rather than spreading it thin.
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(hidden);             // shape: L2_SIZE * OUTPUT_BUCKETS
            let selected = l1_out.select(output_buckets); // shape: L2_SIZE, bucket-specific
            let out1 = selected.screlu();
            let out2 = l2.forward(out1).screlu();
            l3.forward(out2)
        });

    // ============================================================
    // Learning rate schedule -- CosineDecayLR, anchored to the FIXED
    // total_planned superbatch count, not to this session's local
    // superbatches_this_session. This is what makes the schedule behave
    // consistently across however many Kaggle sessions this run takes.
    //
    // initial_lr / final_lr are starting points, not verified-optimal for
    // this exact architecture -- bullet's own example progressions use
    // CosineDecayLR but with their own (smaller-net) initial_lr/final_lr
    // values, so treat these as reasonable defaults to monitor and adjust,
    // not as settled numbers.
    // ============================================================
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
        save_rate: 1, // every superbatch -- Kaggle sessions can die uncleanly;
                      // don't risk losing more than one superbatch of work.
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: 64, // re-check this against actual GPU utilization
                              // once BATCH_SIZE/L1_SIZE are smoke-tested --
                              // a wider net changes the compute-per-batch vs.
                              // data-loading balance.
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