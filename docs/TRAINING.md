# Training Mythos's NNUE evaluation

Mythos can evaluate positions two ways: a **hand-crafted evaluation (HCE)** that is
always available, or an **optional NNUE neural network** loaded from a `mythos.nnue`
file. This document explains the full NNUE pipeline — how to generate data, train a
net on a GPU, and use it in the engine.

```
  self-play games            PyTorch (GPU)              the engine
 ┌───────────────┐   data   ┌───────────────┐  .nnue   ┌───────────────┐
 │   datagen     │ ───────▶ │  train_nnue.py│ ───────▶ │ mythos.nnue    │
 │ (Mythos plays) │  FEN|cp| │  768→256→1 net│  weights │ (auto-loaded) │
 └───────────────┘  result  └───────────────┘          └───────────────┘
```

---

## 1. Generate training data (`datagen`)

`datagen` makes Mythos play thousands of self-play games and records every quiet
position together with (a) the search score and (b) the eventual game result.

```sh
cargo run --release --bin datagen -- nnue_data.txt --games 25000 --depth 8 --threads 24
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `<output>` | file to write samples to | *(required)* |
| `--games N` | total self-play games | `1000` |
| `--depth D` | search depth per move (higher = better labels, slower) | `8` |
| `--nodes X` | limit by nodes instead of depth | — |
| `--threads T` | worker threads (each plays games in parallel) | up to 8 |
| `--seed S` | RNG seed for reproducible openings | `1` |

Each line is one sample:

```
<FEN> | <score_cp> | <result>
```

- `score_cp` — the search score in centipawns, from the **side to move's** view.
- `result` — the game outcome from the side to move's view: `1.0` win, `0.5` draw, `0.0` loss.

Only **quiet** positions (not in check, best move not a capture/promotion) are kept,
which gives cleaner training labels. A run of ~25,000 games yields ~1.6M samples.

> **Tip:** higher `--depth` (e.g. `9`–`10`) produces better labels — the net learns
> to predict what a *deeper* search thinks, which is how it can become a better
> *static* evaluator than the hand-crafted eval. It just takes longer to generate.

## 2. Train the network (`train_nnue.py`)

The trainer is written in **PyTorch** and runs on a **CUDA GPU** (training on the
GPU is dozens of times faster than CPU).

```sh
# one-time setup — install a CUDA build of PyTorch (example: CUDA 12.8):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# train, exporting Mythos's .nnue format:
python train_nnue.py nnue_data.txt mythos.nnue --epochs 50
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `<data>` | the datagen output file | *(required)* |
| `<out.nnue>` | where to write the trained net | *(required)* |
| `--epochs E` | training passes over the data | `40` |
| `--lr LR` | Adam learning rate | `1e-3` |
| `--batch B` | mini-batch size | `16384` |
| `--lambda L` | blend of eval-score vs game-result in the target (1.0 = pure score) | `0.7` |

The trainer prints the train/validation loss each epoch and saves the net every few
epochs. On an RTX 5080, 50 epochs over 1.6M samples takes well under a minute.

**Target:** each sample's training target is
`y = λ · sigmoid(score/400) + (1−λ) · result` — a blend of what the search thought
and what actually happened.

## 3. Use the net in the engine

Place the trained net named **`mythos.nnue`** next to the engine binary (e.g. in
`target/release/`). The engine loads it automatically on startup and prints
`info string NNUE evaluation loaded`. You can also control it over UCI:

```
setoption name EvalFile value path/to/mythos.nnue   # load a specific net
setoption name UseNNUE value true                  # force NNUE on
setoption name UseNNUE value false                 # force the hand-crafted eval
```

If no net is found, the engine simply uses the (strong) hand-crafted evaluation.

## Architecture & file format

Mythos's net is a small **perspective network**:

- **Input:** 768 features = `2 (friendly/enemy) × 6 (piece type) × 64 (square)`,
  computed once from each side's point of view (the board is vertically mirrored
  for the black perspective).
- **Accumulator:** `768 → 256` per perspective, maintained **incrementally** across
  moves (only the changed features are updated — the "efficiently updatable" in NNUE).
- **Output:** `concat(CReLU(stm), CReLU(nstm)) = 512 → 1`, scaled by `400` to
  centipawns. `CReLU(x) = clamp(x, 0, 1)`.

The `.nnue` file is little-endian: `u32 magic (0x4E4E5545)`, `u32 hidden (256)`,
`u32 num_features (768)`, then `w1[256×768]`, `b1[256]`, `w2[512]`, `b2` as `f32`.
The Python trainer and the Rust inference share the exact same feature convention
(`nnue::feature_index`) so an exported net loads directly.

## How to make the net *stronger*

Training a net that clearly beats the hand-crafted evaluation is an iterative
process, not a single run:

1. **More & deeper data** — more games and higher `--depth` give better labels.
2. **Bigger / faster inference** — SIMD + int8 quantization so a good net is fast
   enough that its extra eval quality outweighs the speed cost.
3. **Iterative bootstrapping** — train a net, use the stronger engine to generate
   *better* games, retrain, and repeat. Each round lifts the ceiling. This is how
   top engines were built — over years, on huge amounts of data.
