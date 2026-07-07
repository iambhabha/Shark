# Making Mythos stronger

Concrete, high-value ways to improve Mythos's playing strength, roughly ordered by
value-for-effort. Elo figures are **rough estimates** — always confirm with the
`selfplay` match runner (see [CONTRIBUTING.md](../CONTRIBUTING.md)); a change that
"should" help often doesn't.

> The single most important habit: **measure every change.** Snapshot a baseline
> binary, make one change, play a few hundred self-play games, keep it only if the
> Elo gain clearly beats the error bar.

## The one rule that governs everything: depth × eval-quality

Playing strength comes from **searching deeper** and from a **better evaluation**
at the leaves. Almost every idea below improves one of those two — either by
making the engine *faster* (so it reaches greater depth in the same time) or by
making its judgement *smarter*.

---

## Near-term, reliable gains

### 1. Make NNUE fast — SIMD + int8 quantization  *(big; medium-hard)*
Today the NNUE evaluation is correct but ~5× slower than the hand-crafted eval, so
at a fixed time it searches shallower and loses. The fix is the standard NNUE
speed toolkit:
- **Integer quantization** — store weights as `int8`/`int16` instead of `f32`.
- **SIMD** — process 16–32 accumulator values per instruction (`std::arch` AVX2 /
  NEON, or portable `std::simd`).

Together these can make NNUE inference ~10–20× faster, closing the depth gap so a
good net's superior judgement finally wins. **This is the highest-leverage NNUE
work.**

### 2. Faster / better move generation and make-move  *(medium; measurable)*
Any speedup is depth, and depth is Elo. Profile the hot path; reduce allocations;
consider staged move generation (generate captures first, quiets only if needed).

### 3. Search tuning & new heuristics  *(steady gains; easy-to-medium)*
- **Singular extensions** — extend the search when one move is clearly forced.
- **Continuation history / counter-moves** — richer move ordering.
- **History-based LMR** — reduce less when a move has a good history.
- **SPSA tuning** — auto-tune the pruning margins / reduction constants by playing
  games (this is a large fraction of how Stockfish gains Elo).

## Bigger projects

### 4. A stronger NNUE — iterative bootstrapping  *(large; the real path)*
A net trained only on the hand-crafted eval's scores can at best *imitate* it. To
exceed it:
1. Train a net (v1).
2. Use the stronger engine to generate **better** self-play games.
3. Retrain on the new data (v2), which is now better than the HCE.
4. Repeat. Each round lifts the ceiling.
Also try **deeper data labels** (higher `datagen --depth`) and richer input
features (king-bucketed "HalfKA" features are much stronger than plain 768).
See [TRAINING.md](TRAINING.md).

### 5. Lazy SMP — multithreading  *(large; big gain on multi-core)*
Run the search on many threads that share the transposition table. On an 8-core
machine this is worth a lot of Elo. Needs a lock-free/relaxed-atomic TT and care
around shared state.

### 6. Syzygy tablebases  *(medium; endgame perfection)*
Perfect play for ≤7-piece endgames via the `shakmaty-syzygy` crate. Probe them at
the root (to pick the move) and inside the search (to prune). Optional — the
engine is fully functional without them.

### 7. Profile-guided optimization (PGO)  *(easy; a few % speed)*
Build with `-Cprofile-generate`, run `bench`, rebuild with `-Cprofile-use`. Free
speed, hence free Elo. Worth wiring into the release build.

---

## What is *not* worth doing (yet)

- **Chasing Stockfish's exact strength.** That is the product of 15+ years, a
  global testing network, and billions of training positions. It is not a
  realistic target for a small project — aim for "genuinely strong", not "#1".
- **Micro-optimizing before measuring.** Profile first; measure the Elo after.

## A suggested order

1. Quantize + SIMD the NNUE (unlock the net you already trained).
2. Add singular extensions + continuation history to the search.
3. Iterative NNUE training with deeper data.
4. Lazy SMP.
5. Syzygy + PGO for polish.

Each step is a self-contained, measurable project. Pick one, snapshot a baseline,
and let the games decide.
