//! All-Rust NNUE trainer for Mythos, built on Burn (GPU via wgpu).
//!
//! This mirrors the PyTorch trainer but keeps the whole stack in Rust and — crucially
//! — reuses the *engine's own* feature function (`mythos::nnue::active_features`) and
//! `.nnue` writer (`mythos::nnue::Net::save`), so the trained net can never disagree
//! with the engine about what a feature index means or how weights are laid out.
//!
//! Architecture (identical to the engine): a king-bucketed HalfKA perspective net.
//! For a position we build two accumulators — one from the side-to-move's perspective,
//! one from the other's — each `acc[j] = b1[j] + Σ_{active f} W1[j][f]`. We CReLU both,
//! concatenate (stm half first), and a single linear layer reduces to one output `o`;
//! the engine reads `cp = o * SCALE`. Training target is the standard blend
//! `λ·sigmoid(cp/SCALE) + (1-λ)·result`, loss is MSE against `sigmoid(o)`.
//!
//! Usage:
//!   trainer-burn <out.nnue> [--data sf_big.txt] [--positions N] [--epochs E]
//!                [--batch B] [--lr LR] [--lambda L] [--shards K]

use std::time::Instant;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Int, Tensor};
use burn::tensor::backend::Backend;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use mythos::nnue::{active_features, Net, HIDDEN, NUM_FEATURES, SCALE};
use mythos::position::Position;

/// Max active features per perspective (a full board has ≤32 pieces). Feature lists
/// are padded to this with the sentinel `PAD` index, which is masked out of the sum.
const MAX_FEATS: usize = 32;
/// Padding feature index: an extra embedding row that the mask zeroes out.
const PAD: i32 = NUM_FEATURES as i32;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/// One training example, already reduced to feature indices so training never
/// re-parses a FEN. `stm`/`nstm` are padded to `MAX_FEATS`; `n_stm`/`n_nstm` are the
/// real counts (the rest are `PAD`). `target` is the blended win-probability label.
struct Sample {
    stm: [i32; MAX_FEATS],
    nstm: [i32; MAX_FEATS],
    target: f32,
}

/// Parse up to `limit` positions from the data shards `<base>.w0..w{shards-1}`, each
/// line `FEN | stm_cp | stm_result`, into feature-index samples via the engine's own
/// `active_features`. Lines that don't parse are skipped.
fn load_samples(base: &str, shards: usize, limit: usize, lambda: f32) -> Vec<Sample> {
    let mut out: Vec<Sample> = Vec::with_capacity(limit.min(1 << 20));
    let mut wf: Vec<usize> = Vec::with_capacity(MAX_FEATS);
    let mut bf: Vec<usize> = Vec::with_capacity(MAX_FEATS);

    'outer: for s in 0..shards {
        let path = format!("{base}.w{s}");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for line in text.lines() {
            let mut it = line.split('|');
            let fen = match it.next() {
                Some(f) => f.trim(),
                None => continue,
            };
            let cp: f32 = match it.next().and_then(|x| x.trim().parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let result: f32 = match it.next().and_then(|x| x.trim().parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let pos = match Position::from_fen(fen) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let stm_color = pos.side_to_move();
            let nstm_color = !stm_color;
            active_features(&pos, stm_color, &mut wf);
            active_features(&pos, nstm_color, &mut bf);
            if wf.len() > MAX_FEATS || bf.len() > MAX_FEATS {
                continue; // shouldn't happen (≤32 pieces), but stay safe
            }

            let mut stm = [PAD; MAX_FEATS];
            let mut nstm = [PAD; MAX_FEATS];
            for (i, &f) in wf.iter().enumerate() {
                stm[i] = f as i32;
            }
            for (i, &f) in bf.iter().enumerate() {
                nstm[i] = f as i32;
            }

            // target = λ·sigmoid(cp/SCALE) + (1-λ)·result, all stm-relative.
            let wp = 1.0 / (1.0 + (-cp / SCALE).exp());
            let target = lambda * wp + (1.0 - lambda) * result;

            out.push(Sample { stm, nstm, target });
            if out.len() >= limit {
                break 'outer;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The NNUE network as a Burn module. `ft` is the shared feature transformer
/// (`NUM_FEATURES + 1` rows including the padding row); `b1` its bias; `l2` the
/// `2*HIDDEN -> 1` output layer.
#[derive(Module, Debug)]
struct Nnue<B: Backend> {
    ft: Embedding<B>,
    b1: Param<Tensor<B, 1>>,
    l2: Linear<B>,
}

impl<B: Backend> Nnue<B> {
    fn new(device: &B::Device) -> Self {
        // Small uniform init (matching the PyTorch trainer): keeps the summed
        // accumulator in the CReLU linear region [0,1] at the start so gradients
        // flow — a large default init saturates CReLU and learning stalls.
        let init = Initializer::Uniform { min: -0.1, max: 0.1 };
        let ft = EmbeddingConfig::new(NUM_FEATURES + 1, HIDDEN)
            .with_initializer(init.clone())
            .init(device);
        let b1 = Param::from_tensor(Tensor::zeros([HIDDEN], device));
        let l2 = LinearConfig::new(2 * HIDDEN, 1)
            .with_initializer(init)
            .init(device);
        Self { ft, b1, l2 }
    }

    /// One accumulator half: sum the embedding rows of the (masked) active features,
    /// add the bias, and CReLU. `feats` is `[batch, MAX_FEATS]` indices, `mask` is
    /// `[batch, MAX_FEATS]` (1.0 for a real feature, 0.0 for padding).
    fn half(&self, feats: Tensor<B, 2, Int>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = feats.dims()[0];
        // [batch, MAX_FEATS, HIDDEN]
        let emb = self.ft.forward(feats);
        // zero the padding rows, then sum over the feature axis -> [batch, HIDDEN]
        let mask = mask.reshape([batch, MAX_FEATS, 1]);
        let acc = (emb * mask).sum_dim(1).reshape([batch, HIDDEN]);
        let acc = acc + self.b1.val().reshape([1, HIDDEN]);
        acc.clamp(0.0, 1.0)
    }

    /// Forward: raw output `o` (`[batch, 1]`), before the sigmoid used in the loss.
    fn forward(
        &self,
        stm: Tensor<B, 2, Int>,
        stm_mask: Tensor<B, 2>,
        nstm: Tensor<B, 2, Int>,
        nstm_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let a = self.half(stm, stm_mask);
        let b = self.half(nstm, nstm_mask);
        let h = Tensor::cat(vec![a, b], 1); // [batch, 2*HIDDEN], stm half first
        self.l2.forward(h)
    }
}

// ---------------------------------------------------------------------------
// Batch tensors
// ---------------------------------------------------------------------------

/// Build the GPU tensors for a batch of samples: feature indices, masks, targets.
fn batch_tensors<B: Backend>(
    samples: &[Sample],
    idx: &[usize],
    device: &B::Device,
) -> (
    Tensor<B, 2, Int>,
    Tensor<B, 2>,
    Tensor<B, 2, Int>,
    Tensor<B, 2>,
    Tensor<B, 2>,
) {
    let n = idx.len();
    let mut stm = vec![0i32; n * MAX_FEATS];
    let mut nstm = vec![0i32; n * MAX_FEATS];
    let mut stm_mask = vec![0f32; n * MAX_FEATS];
    let mut nstm_mask = vec![0f32; n * MAX_FEATS];
    let mut tgt = vec![0f32; n];
    for (row, &i) in idx.iter().enumerate() {
        let s = &samples[i];
        for c in 0..MAX_FEATS {
            let o = row * MAX_FEATS + c;
            stm[o] = s.stm[c];
            nstm[o] = s.nstm[c];
            stm_mask[o] = if s.stm[c] == PAD { 0.0 } else { 1.0 };
            nstm_mask[o] = if s.nstm[c] == PAD { 0.0 } else { 1.0 };
        }
        tgt[row] = s.target;
    }
    let stm = Tensor::<B, 1, Int>::from_data(stm.as_slice(), device).reshape([n, MAX_FEATS]);
    let nstm = Tensor::<B, 1, Int>::from_data(nstm.as_slice(), device).reshape([n, MAX_FEATS]);
    let stm_mask = Tensor::<B, 1>::from_floats(stm_mask.as_slice(), device).reshape([n, MAX_FEATS]);
    let nstm_mask =
        Tensor::<B, 1>::from_floats(nstm_mask.as_slice(), device).reshape([n, MAX_FEATS]);
    let tgt = Tensor::<B, 1>::from_floats(tgt.as_slice(), device).reshape([n, 1]);
    (stm, stm_mask, nstm, nstm_mask, tgt)
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Copy the trained weights into a `mythos::nnue::Net` and save it as a `.nnue`.
/// The inference backend (no autodiff) keeps this simple.
fn export<B: Backend>(model: &Nnue<B>, path: &str) {
    let mut net = Net::zeros();

    // Feature transformer: embedding row f (0..NUM_FEATURES) holds feature f's
    // contribution to all HIDDEN neurons. The engine's w1 is row-major
    // [HIDDEN][NUM_FEATURES], so w1[j*NUM_FEATURES + f] = emb[f][j].
    let ft = model.ft.weight.val(); // [NUM_FEATURES+1, HIDDEN]
    let ft = ft.into_data();
    let ftv: Vec<f32> = ft.convert::<f32>().into_vec().expect("ft to vec");
    for f in 0..NUM_FEATURES {
        for j in 0..HIDDEN {
            net.w1[j * NUM_FEATURES + f] = ftv[f * HIDDEN + j];
        }
    }

    // Layer-1 bias.
    let b1v: Vec<f32> = model.b1.val().into_data().convert::<f32>().into_vec().expect("b1");
    net.b1.copy_from_slice(&b1v[..HIDDEN]);

    // Layer 2: Linear weight is [1, 2*HIDDEN] (out, in); engine w2 is [2*HIDDEN].
    let w2v: Vec<f32> = model.l2.weight.val().into_data().convert::<f32>().into_vec().expect("w2");
    net.w2.copy_from_slice(&w2v[..2 * HIDDEN]);
    let b2v: Vec<f32> = model
        .l2
        .bias
        .as_ref()
        .expect("l2 bias")
        .val()
        .into_data()
        .convert::<f32>()
        .into_vec()
        .expect("b2");
    net.b2 = b2v[0];

    net.rebuild_w1t();
    net.save(path).expect("save nnue");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn arg_val(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "mythos_burn.nnue".to_string());
    let data = arg_val(&args, "--data").unwrap_or_else(|| "sf_big.txt".to_string());
    let shards: usize = arg_val(&args, "--shards").and_then(|v| v.parse().ok()).unwrap_or(24);
    let positions: usize =
        arg_val(&args, "--positions").and_then(|v| v.parse().ok()).unwrap_or(3_000_000);
    let epochs: usize = arg_val(&args, "--epochs").and_then(|v| v.parse().ok()).unwrap_or(30);
    let batch: usize = arg_val(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(8192);
    let lr: f64 = arg_val(&args, "--lr").and_then(|v| v.parse().ok()).unwrap_or(1e-3);
    let lambda: f32 = arg_val(&args, "--lambda").and_then(|v| v.parse().ok()).unwrap_or(0.7);
    // GPU throttle: after each batch, sleep `throttle * batch_time`. The GPU is busy
    // `1/(1+throttle)` of the wall clock, so 0.9 ≈ 53% and 0.7 ≈ 59% — keeps the card
    // (and the desktop) responsive. 0 = full speed.
    let throttle: f32 = arg_val(&args, "--throttle").and_then(|v| v.parse().ok()).unwrap_or(0.0);

    println!(
        "trainer-burn: out={out} data={data} shards={shards} positions={positions} epochs={epochs} batch={batch} lr={lr} lambda={lambda} throttle={throttle}"
    );

    let t0 = Instant::now();
    let samples = load_samples(&data, shards, positions, lambda);
    println!("loaded {} positions in {:.1}s", samples.len(), t0.elapsed().as_secs_f32());
    if samples.is_empty() {
        eprintln!("no data — nothing to train");
        return;
    }

    let mut model = Nnue::<MyBackend>::new(&device);
    let mut optim = AdamConfig::new().init();

    let mut order: Vec<usize> = (0..samples.len()).collect();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let te = Instant::now();
        let mut running = 0.0f64;
        let mut nb = 0usize;
        for chunk in order.chunks(batch) {
            let bt = Instant::now();
            let (stm, stm_m, nstm, nstm_m, tgt) =
                batch_tensors::<MyBackend>(&samples, chunk, &device);
            let o = model.forward(stm, stm_m, nstm, nstm_m);
            let pred = sigmoid(o);
            let diff = pred - tgt;
            let loss = (diff.clone() * diff).mean();

            let lv: f32 = loss.clone().into_scalar(); // forces GPU sync
            running += lv as f64;
            nb += 1;

            let grads = loss.backward();
            let gp = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, gp);

            // Throttle: idle the GPU for a fraction of the batch's compute time so
            // average utilization (and the desktop) stays responsive.
            if throttle > 0.0 {
                std::thread::sleep(bt.elapsed().mul_f32(throttle));
            }
        }
        println!(
            "epoch {:>3}/{}  loss {:.6}  ({:.1}s)",
            epoch + 1,
            epochs,
            running / nb as f64,
            te.elapsed().as_secs_f32()
        );
        // Checkpoint after every epoch: overwrite `out` with the latest net so a
        // reaped/killed run still leaves the most-trained net on disk (export is
        // cheap ~1s vs a ~465s epoch).
        export(&model, &out);
        println!("  checkpoint -> {out}");
    }

    export(&model, &out);
    println!("saved {out}  (total {:.1}s)", t0.elapsed().as_secs_f32());
}
