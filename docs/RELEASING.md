# Cutting a Mythos release

Most people won't compile from source — they want a **ready-to-run binary** they
can download and drop into a chess GUI. Publishing a release is the single best
way to get users. This is the process.

## 1. Pick a version

Mythos uses simple semantic-ish versions: `MAJOR.MINOR` (e.g. `0.1`, `0.2`, `1.0`).
Update the version in `Cargo.toml`:

```toml
[package]
version = "0.2.0"
```

Rough guidance: bump **MINOR** for a normal strength/feature release; bump
**MAJOR** for a big milestone (e.g. the first release that beats the previous one
by a large margin, or the first with a working NNUE).

## 2. Sanity-check before building

```sh
cargo test                          # unit tests green
cargo test --release -- --ignored   # deep perft exact
cargo run   --release -- bench      # note the perft nps (a quick health check)
```

If you keep a "bench" number, record it in the release notes so regressions are
easy to spot.

## 3. Build optimized binaries

The default release profile is already optimized (`lto = true`,
`codegen-units = 1`). For the machine building it:

```sh
cargo build --release                     # target/release/mythos(.exe)
```

For a **portable** binary that runs on any modern CPU, build with a conservative
target; for a **fast** binary tuned to your CPU, use `target-cpu=native`:

```sh
# fastest on THIS machine (don't distribute this one):
RUSTFLAGS="-C target-cpu=native" cargo build --release

# portable x86-64 build to distribute:
cargo build --release        # (baseline x86-64; safe everywhere)
```

Build per platform you want to support (Windows `.exe`, Linux, macOS). The
`webserver` and `selfplay` binaries are optional extras.

## 4. (Optional) include a network

If you ship an NNUE net, add `mythos.nnue` next to the binary in the release
archive and mention it in the notes. Without a net, the engine uses its
hand-crafted evaluation automatically.

## 5. Create the GitHub release

Tag and publish with the GitHub CLI (`gh`). Attach the built binaries as assets:

```sh
git tag v0.2.0
git push origin v0.2.0

gh release create v0.2.0 \
  target/release/mythos.exe \
  --title "Mythos 0.2" \
  --notes "What changed, measured Elo vs the previous release, and how to run it."
```

Or create the release from the GitHub web UI and drag the binaries in.

### Good release notes include
- A one-line summary of what's new.
- **Measured** strength change vs the previous release (games, time control,
  Elo ± error from `selfplay`).
- How to run it (drop the binary into a UCI GUI, or `./mythos` then `uci`).
- Any known issues.

## 6. After releasing

- Update the README's badges / "latest release" if you add one.
- Announce it (Reddit r/chess & r/rust, chess-programming Discord) — this is what
  actually brings users.
- Keep a baseline binary of the release so the *next* version can be measured
  against it.
