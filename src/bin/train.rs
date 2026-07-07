//! `train` — a small, dependency-free CPU trainer for Mythos's NNUE.
//!
//! It reads a text dataset of `<FEN> | <score_cp> | <result>` samples, precomputes
//! each sample's active feature lists (for both perspectives) using the *exact*
//! feature convention from `mythos::nnue`, and fits the perspective net with
//! mini-batch stochastic gradient descent. Per-batch gradients are summed in
//! parallel across worker threads.
//!
//! Usage:
//!
//! ```text
//! train <data_file> <out_net_file> [--epochs E] [--lr LR] [--threads T] [--lambda L] [--batch B]
//! ```
//!
//! Defaults: epochs=30, lr=0.001, threads=available_parallelism (capped ~28),
//! lambda=0.7, batch=16384.
//!
//! The whole thing uses `f32` and a fixed-seed splitmix64 PRNG (no `rand` crate),
//! so a run is fully deterministic given the same data and flags.

use std::process;
use std::sync::Arc;
use std::thread;

use mythos::nnue::{active_features, Net, HIDDEN, NUM_FEATURES, SCALE};
use mythos::position::Position;

// ---------------------------------------------------------------------------
// A precomputed training sample.
// ---------------------------------------------------------------------------

/// One training position, reduced to what the hot loop needs: the active feature
/// indices for each perspective (as `u16` to save memory — every index is < 768)
/// and the blended training target `y`.
struct Sample {
    /// Active features from the side-to-move's perspective.
    feats_stm: Vec<u16>,
    /// Active features from the not-side-to-move's perspective.
    feats_nstm: Vec<u16>,
    /// The training target in [0, 1]: `lambda*sigmoid(cp/SCALE) + (1-lambda)*result`.
    y: f32,
}

// ---------------------------------------------------------------------------
// CLI configuration.
// ---------------------------------------------------------------------------

struct Config {
    data_file: String,
    out_file: String,
    epochs: usize,
    lr: f32,
    threads: usize,
    lambda: f32,
    batch: usize,
}

fn default_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 28)
}

fn parse_args() -> Config {
    let mut args = std::env::args().skip(1);

    let data_file = match args.next() {
        Some(a) => a,
        None => {
            eprintln!(
                "usage: train <data_file> <out_net_file> \
                 [--epochs E] [--lr LR] [--threads T] [--lambda L] [--batch B]"
            );
            process::exit(2);
        }
    };
    let out_file = match args.next() {
        Some(a) => a,
        None => {
            eprintln!("error: missing <out_net_file>");
            process::exit(2);
        }
    };

    let mut cfg = Config {
        data_file,
        out_file,
        epochs: 30,
        lr: 0.001,
        threads: default_threads(),
        lambda: 0.7,
        batch: 16384,
    };

    // Flags come in `--name value` pairs.
    while let Some(flag) = args.next() {
        let val = args.next();
        let need = |v: Option<String>| -> String {
            match v {
                Some(s) => s,
                None => {
                    eprintln!("error: flag {flag} needs a value");
                    process::exit(2);
                }
            }
        };
        match flag.as_str() {
            "--epochs" => cfg.epochs = parse_or_die(&need(val), "--epochs"),
            "--lr" => cfg.lr = parse_or_die(&need(val), "--lr"),
            "--threads" => cfg.threads = parse_or_die::<usize>(&need(val), "--threads").max(1),
            "--lambda" => cfg.lambda = parse_or_die(&need(val), "--lambda"),
            "--batch" => cfg.batch = parse_or_die::<usize>(&need(val), "--batch").max(1),
            other => {
                eprintln!("error: unknown flag '{other}'");
                process::exit(2);
            }
        }
    }
    cfg
}

fn parse_or_die<T: std::str::FromStr>(s: &str, name: &str) -> T {
    match s.parse::<T>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("error: could not parse value '{s}' for {name}");
            process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Math helpers.
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[inline]
fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64) — used for weight init and shuffling.
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f32` in `[0, 1)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Take the top 24 bits for a full-precision f32 mantissa.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// A uniform `f32` in `[-mag, mag)`.
    #[inline]
    fn next_signed(&mut self, mag: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * mag
    }

    /// A uniform `usize` in `[0, n)` (n > 0).
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

// ---------------------------------------------------------------------------
// Dataset loading.
// ---------------------------------------------------------------------------

/// Parse the dataset file into precomputed [`Sample`]s. Lines starting with `#`
/// and blank lines are skipped; malformed lines (bad FEN, bad numbers, wrong
/// field count) are skipped with a warning cap so a few bad rows don't abort.
fn load_dataset(path: &str, lambda: f32) -> std::io::Result<Vec<Sample>> {
    let text = std::fs::read_to_string(path)?;
    let mut samples: Vec<Sample> = Vec::new();

    let mut scratch: Vec<usize> = Vec::with_capacity(32);
    let mut skipped = 0usize;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // `<FEN> | <score_cp> | <result>`
        let parts: Vec<&str> = line.split('|').map(|p| p.trim()).collect();
        if parts.len() != 3 {
            skipped += 1;
            continue;
        }

        let pos = match Position::from_fen(parts[0]) {
            Ok(p) => p,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let score_cp: f32 = match parts[1].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let result: f32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        // Blended target: part game-theoretic result, part search-score sigmoid.
        // `result` is already side-to-move relative, in {0, 0.5, 1}.
        let y = lambda * sigmoid(score_cp / SCALE) + (1.0 - lambda) * result;

        // Precompute feature lists for both perspectives, converting usize -> u16.
        active_features(&pos, pos.side_to_move(), &mut scratch);
        let feats_stm = to_u16(&scratch);
        active_features(&pos, !pos.side_to_move(), &mut scratch);
        let feats_nstm = to_u16(&scratch);

        samples.push(Sample {
            feats_stm,
            feats_nstm,
            y,
        });
    }

    if skipped > 0 {
        eprintln!("note: skipped {skipped} malformed/blank data line(s)");
    }
    Ok(samples)
}

/// Convert a feature-index scratch buffer to a compact `Vec<u16>`, asserting the
/// invariant that every index fits the feature space (< 768).
fn to_u16(src: &[usize]) -> Vec<u16> {
    src.iter()
        .map(|&f| {
            debug_assert!(f < NUM_FEATURES, "feature index {f} out of range");
            f as u16
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gradient buffers.
// ---------------------------------------------------------------------------

/// Accumulated gradients for the whole net — same shape as [`Net`].
struct Grad {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

impl Grad {
    fn zeros() -> Grad {
        Grad {
            w1: vec![0.0; HIDDEN * NUM_FEATURES],
            b1: vec![0.0; HIDDEN],
            w2: vec![0.0; 2 * HIDDEN],
            b2: 0.0,
        }
    }

    /// Add another gradient into this one (used to reduce per-thread partials).
    fn add(&mut self, other: &Grad) {
        for (a, b) in self.w1.iter_mut().zip(other.w1.iter()) {
            *a += *b;
        }
        for (a, b) in self.b1.iter_mut().zip(other.b1.iter()) {
            *a += *b;
        }
        for (a, b) in self.w2.iter_mut().zip(other.w2.iter()) {
            *a += *b;
        }
        self.b2 += other.b2;
    }
}

// ---------------------------------------------------------------------------
// Per-sample forward + backward. Accumulates into `grad`, returns the sample loss.
//
// `scratch_stm` / `scratch_nstm` are reusable length-HIDDEN buffers so the hot
// loop makes no per-sample heap allocations.
// ---------------------------------------------------------------------------

fn accumulate_sample(
    net: &Net,
    sample: &Sample,
    grad: &mut Grad,
    acc_stm: &mut [f32],
    acc_nstm: &mut [f32],
) -> f32 {
    // --- Forward: two accumulators from the sparse feature lists. -----------
    acc_stm.copy_from_slice(&net.b1);
    for &f in &sample.feats_stm {
        let mut base = f as usize;
        for a in acc_stm.iter_mut() {
            *a += net.w1[base];
            base += NUM_FEATURES;
        }
    }
    acc_nstm.copy_from_slice(&net.b1);
    for &f in &sample.feats_nstm {
        let mut base = f as usize;
        for a in acc_nstm.iter_mut() {
            *a += net.w1[base];
            base += NUM_FEATURES;
        }
    }

    // combined = concat(CReLU(acc_stm), CReLU(acc_nstm)); output o.
    let mut o = net.b2;
    for j in 0..HIDDEN {
        o += net.w2[j] * crelu(acc_stm[j]);
        o += net.w2[HIDDEN + j] * crelu(acc_nstm[j]);
    }
    let p = sigmoid(o);
    let diff = p - sample.y;
    let loss = diff * diff;

    // --- Backward. ----------------------------------------------------------
    // dL/do = 2*(p - y) * p * (1 - p)   (MSE over the sigmoid).
    let dodo = 2.0 * diff * p * (1.0 - p);
    grad.b2 += dodo;

    // For each hidden neuron, per perspective: propagate through W2 and CReLU.
    // dW2[k] += dL/do * combined[k]; dAcc = dL/do * W2[k] gated by CReLU'.
    // The gradient of CReLU passes only where 0 < acc < 1.
    for j in 0..HIDDEN {
        // --- Side-to-move half. ---
        let a = acc_stm[j];
        grad.w2[j] += dodo * crelu(a);
        if a > 0.0 && a < 1.0 {
            let d_acc = dodo * net.w2[j];
            grad.b1[j] += d_acc;
            for &f in &sample.feats_stm {
                grad.w1[j * NUM_FEATURES + f as usize] += d_acc;
            }
        }

        // --- Not-side-to-move half. ---
        let a2 = acc_nstm[j];
        grad.w2[HIDDEN + j] += dodo * crelu(a2);
        if a2 > 0.0 && a2 < 1.0 {
            let d_acc = dodo * net.w2[HIDDEN + j];
            grad.b1[j] += d_acc;
            for &f in &sample.feats_nstm {
                grad.w1[j * NUM_FEATURES + f as usize] += d_acc;
            }
        }
    }

    loss
}

/// Compute the summed gradient (and summed loss) over `samples[indices]` using a
/// single thread. Returns `(grad, loss_sum)`.
fn batch_gradient_serial(net: &Net, samples: &[Sample], indices: &[usize]) -> (Grad, f32) {
    let mut grad = Grad::zeros();
    let mut acc_stm = vec![0.0f32; HIDDEN];
    let mut acc_nstm = vec![0.0f32; HIDDEN];
    let mut loss = 0.0f32;
    for &idx in indices {
        loss += accumulate_sample(net, &samples[idx], &mut grad, &mut acc_stm, &mut acc_nstm);
    }
    (grad, loss)
}

/// Compute the summed gradient (and summed loss) over `samples[indices]`, split
/// across `threads` worker threads. Each thread accumulates its own gradient
/// buffers; the partials are then reduced by summation. Returns `(grad, loss_sum)`.
fn batch_gradient_parallel(
    net: &Arc<Net>,
    samples: &Arc<Vec<Sample>>,
    indices: &[usize],
    threads: usize,
) -> (Grad, f32) {
    // Fall back to the serial path when parallelism won't help.
    if threads <= 1 || indices.len() <= 1 {
        return batch_gradient_serial(net, samples, indices);
    }

    let n = indices.len();
    let chunk = n.div_ceil(threads);

    thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            // Slice of this batch's indices for the worker (borrowed within scope).
            let sub = &indices[start..end];
            let net_ref: &Net = net;
            let samples_ref: &Vec<Sample> = samples;
            handles.push(scope.spawn(move || {
                let mut grad = Grad::zeros();
                let mut acc_stm = vec![0.0f32; HIDDEN];
                let mut acc_nstm = vec![0.0f32; HIDDEN];
                let mut loss = 0.0f32;
                for &idx in sub {
                    loss += accumulate_sample(
                        net_ref,
                        &samples_ref[idx],
                        &mut grad,
                        &mut acc_stm,
                        &mut acc_nstm,
                    );
                }
                (grad, loss)
            }));
        }

        // Reduce the per-thread partials by summation.
        let mut total = Grad::zeros();
        let mut loss = 0.0f32;
        for h in handles {
            let (g, l) = h.join().expect("worker thread panicked");
            total.add(&g);
            loss += l;
        }
        (total, loss)
    })
}

// ---------------------------------------------------------------------------
// Weight initialization.
// ---------------------------------------------------------------------------

/// Build a net with small random weights (`~±0.1`, biases 0) from a fixed seed.
fn init_net() -> Net {
    let mut net = Net::zeros();
    let mut rng = SplitMix64::new(0x5EED_5EED_5EED_5EED);

    // W1: fan_in = NUM_FEATURES; but the input is 1-hot sparse (32 active), so a
    // uniform ±0.1 keeps early accumulators in a sane range. W2 similarly ±0.1.
    let w1_mag = 0.1f32;
    let w2_mag = 0.1f32;
    for w in net.w1.iter_mut() {
        *w = rng.next_signed(w1_mag);
    }
    for w in net.w2.iter_mut() {
        *w = rng.next_signed(w2_mag);
    }
    // Biases start at zero (already zeroed by `Net::zeros`).
    net
}

// ---------------------------------------------------------------------------
// Training loop.
// ---------------------------------------------------------------------------

/// Apply an averaged-gradient SGD step: `w -= lr * (grad / batch_size)`.
fn sgd_step(net: &mut Net, grad: &Grad, lr: f32, batch_size: usize) {
    let scale = lr / (batch_size as f32);
    for (w, g) in net.w1.iter_mut().zip(grad.w1.iter()) {
        *w -= scale * *g;
    }
    for (b, g) in net.b1.iter_mut().zip(grad.b1.iter()) {
        *b -= scale * *g;
    }
    for (w, g) in net.w2.iter_mut().zip(grad.w2.iter()) {
        *w -= scale * *g;
    }
    net.b2 -= scale * grad.b2;
}

/// Fisher–Yates shuffle of `indices`, deterministic in `rng`.
fn shuffle(indices: &mut [usize], rng: &mut SplitMix64) {
    let n = indices.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        indices.swap(i, j);
    }
}

fn main() {
    let cfg = parse_args();

    eprintln!(
        "train: data={} out={} epochs={} lr={} threads={} lambda={} batch={}",
        cfg.data_file, cfg.out_file, cfg.epochs, cfg.lr, cfg.threads, cfg.lambda, cfg.batch
    );

    // --- Load and precompute the dataset. -----------------------------------
    let samples = match load_dataset(&cfg.data_file, cfg.lambda) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read data file '{}': {e}", cfg.data_file);
            process::exit(1);
        }
    };
    if samples.is_empty() {
        eprintln!("error: dataset '{}' has no usable samples", cfg.data_file);
        process::exit(1);
    }
    eprintln!("loaded {} samples", samples.len());

    // Share the (read-only during a step) samples with worker threads.
    let samples = Arc::new(samples);

    // --- Initialize weights. ------------------------------------------------
    let mut net = init_net();

    let n = samples.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut shuffle_rng = SplitMix64::new(0xD1CE_D1CE_D1CE_D1CE);

    // --- Epoch loop. --------------------------------------------------------
    for epoch in 1..=cfg.epochs {
        shuffle(&mut indices, &mut shuffle_rng);

        let mut epoch_loss = 0.0f32;
        let mut seen = 0usize;

        // Wrap the current net in an Arc for the parallel gradient; we take it
        // back out (unwrap) after each batch to mutate it for the SGD step.
        let mut start = 0usize;
        while start < n {
            let end = (start + cfg.batch).min(n);
            let batch = &indices[start..end];
            let batch_size = batch.len();

            let net_arc = Arc::new(net);
            let (grad, loss) =
                batch_gradient_parallel(&net_arc, &samples, batch, cfg.threads);
            // Reclaim ownership of the net (the only other Arc clones live inside
            // the scoped threads, which have all joined by now).
            net = Arc::try_unwrap(net_arc)
                .unwrap_or_else(|_| panic!("net Arc still shared after batch"));

            sgd_step(&mut net, &grad, cfg.lr, batch_size);

            epoch_loss += loss;
            seen += batch_size;
            start = end;
        }

        let avg_loss = epoch_loss / (seen as f32);
        eprintln!("epoch {epoch}/{}: avg_loss = {avg_loss:.6}", cfg.epochs);

        // Periodic checkpoint every 5 epochs (and always at the end below).
        if epoch % 5 == 0 {
            if let Err(e) = net.save(&cfg.out_file) {
                eprintln!("warning: checkpoint save failed: {e}");
            }
        }
    }

    // --- Final save. --------------------------------------------------------
    match net.save(&cfg.out_file) {
        Ok(()) => eprintln!("saved trained net to {}", cfg.out_file),
        Err(e) => {
            eprintln!("error: could not save net to '{}': {e}", cfg.out_file);
            process::exit(1);
        }
    }
}
