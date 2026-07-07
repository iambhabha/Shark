# Contributing to Mythos

Thanks for your interest in Mythos! 🦈 This is a learning-driven chess engine, and
contributions — from bug fixes to strength patches to docs — are very welcome.

## Getting set up

You only need a stable **Rust** toolchain (via [rustup](https://rustup.rs)).

```sh
git clone https://github.com/iambhabha/MythosChess
cd Mythos
cargo build --release      # compile
cargo test                 # run the unit tests (must be green)
cargo run   --release      # start the UCI engine
```

For the optional NNUE trainer you also need Python + PyTorch — see
[docs/TRAINING.md](docs/TRAINING.md).

## The two golden rules

Chess-engine development has two hard rules. Please respect them.

### 1. Correctness first — `perft` must always pass

Any change touching move generation, `make_move`/`undo_move`, or the board must
keep **`perft`** exact:

```sh
cargo test                          # fast perft + all unit tests
cargo test --release -- --ignored   # the deep perft positions
```

If a single perft count is off, the change has a bug. `perft` is the ground truth
for move-generation correctness — never weaken or delete a perft test to make it
pass.

### 2. Strength changes must be *measured*, not guessed

A change that "should" help often doesn't — or makes it weaker. **Never merge a
strength patch on intuition.** Measure it with the built-in match runner:

```sh
# build your version and a baseline, then:
cargo run --release --bin selfplay -- ./mythos_new.exe ./mythos_baseline.exe --games 200 --movetime 200
```

It prints something like `+... -... =...  Elo +X ± Y`. Keep the change only if it
is a clear, positive result (the Elo gain comfortably exceeds the error bar).
Because a single machine is slow, batch several well-understood improvements or
run enough games (hundreds+) so the signal beats the noise. This is exactly what
Stockfish's Fishtest does — just at a much smaller scale.

> Tip: snapshot a baseline binary (`cp target/release/mythos mythos_baseline`)
> **before** you start, so you always have something to measure against.

## Code style

- Match the surrounding code — naming, comment density, and idioms. Read a nearby
  file before adding to it.
- Prefer clear, idiomatic Rust. `#[inline]` tiny hot functions; avoid allocation
  on the search/movegen hot paths (use fixed buffers, not `Vec`).
- Document *why*, not just *what* — especially for chess-specific tricks. Every
  module has a `//!` header explaining its role; keep that up.
- Keep integer widths deliberate in the search — the pruning math is tuned.
- Run `cargo fmt` and `cargo clippy` before submitting.

## Submitting a change

1. Fork the repo and create a branch.
2. Make your change; keep commits focused with clear messages.
3. Ensure `cargo test` is green (and deep perft if you touched move gen).
4. For a strength change, include the `selfplay` result (games, TC, Elo ± error)
   in the pull-request description.
5. Open a pull request describing *what* and *why*.

## Where to help

- **Make it stronger** — see [docs/IMPROVING.md](docs/IMPROVING.md) for concrete,
  high-value ideas (faster NNUE, search tuning, multithreading, tablebases…).
- **NNUE** — generate more/better data, train stronger nets
  ([docs/TRAINING.md](docs/TRAINING.md)).
- **Portability & tooling** — CI, cross-platform builds, benchmarks.
- **Docs & tests** — always appreciated.

## License

By contributing you agree that your contribution is licensed under the project's
**GPL-3.0-or-later** license.
