#!/usr/bin/env python3
"""A tiny SPSA tuner for Mythos's UCI search parameters.

Each iteration perturbs every parameter by +/- c (a random sign per parameter),
plays a short self-play match between the "+" and "-" engines, and nudges each
parameter toward whichever side scored better. The running best is checkpointed
to spsa_theta.json every iteration, so a killed run resumes where it left off.

Validate the result afterwards with a proper match:
    selfplay mythos_tuned.exe mythos.exe --games 200 --movetime 100
(where the tuned build just hard-codes the final numbers, or is run with the
final --opt-a values).

Usage:  python spsa.py [iters] [games_per_iter] [movetime_ms]
"""
import json
import os
import random
import re
import subprocess
import sys

ENGINE = "./target/release/mythos.exe"
SELFPLAY = "./target/release/selfplay.exe"
CKPT = "spsa_theta.json"

# name: [default, min, max, c (perturbation), a (step size)]
PARAMS = {
    "RfpMargin": [90, 20, 300, 16, 9],
    "NullRBase": [3, 1, 6, 1, 1],
    "LmrDiv":    [225, 120, 380, 28, 14],
    "FutScale":  [100, 30, 260, 22, 11],
}


def clamp(name, v):
    _, lo, hi, _, _ = PARAMS[name]
    return int(max(lo, min(hi, round(v))))


def load_theta():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            d = json.load(f)
        return {n: clamp(n, d.get(n, PARAMS[n][0])) for n in PARAMS}
    return {n: PARAMS[n][0] for n in PARAMS}


def save_theta(theta, it):
    with open(CKPT, "w") as f:
        json.dump({**theta, "_iter": it}, f)


def play(pa, pb, games, mt):
    args = [SELFPLAY, ENGINE, ENGINE, "--games", str(games), "--movetime", str(mt)]
    for k, v in pa.items():
        args += ["--opt-a", f"{k}={v}"]
    for k, v in pb.items():
        args += ["--opt-b", f"{k}={v}"]
    # A generous timeout so a single hung/crashed match can never freeze the
    # whole tuner; on any failure we report a neutral 0.5 and move on. A game can
    # run to ~200 plies, so budget for that much thinking plus a fixed buffer.
    budget = 90 + games * mt / 1000.0 * 220
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=budget).stdout
    except Exception as e:
        print(f"  (match failed: {e}; treating as 0.5)", flush=True)
        return 0.5
    m = re.search(r"score ([\d.]+)%", out)
    return float(m.group(1)) / 100.0 if m else 0.5


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    mt = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    theta = load_theta()
    start = 0
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            start = json.load(f).get("_iter", 0)
    print(f"start theta={theta} (from iter {start})", flush=True)

    rng = random.Random(0xC0FFEE + start)
    for it in range(start, start + iters):
        delta = {n: rng.choice([-1, 1]) for n in theta}
        plus = {n: clamp(n, theta[n] + PARAMS[n][3] * delta[n]) for n in theta}
        minus = {n: clamp(n, theta[n] - PARAMS[n][3] * delta[n]) for n in theta}
        s = play(plus, minus, games, mt)  # A=plus vs B=minus
        for n in theta:
            theta[n] = clamp(n, theta[n] + PARAMS[n][4] * (2 * s - 1) * delta[n])
        save_theta(theta, it + 1)
        print(f"iter {it + 1} s={s:.3f} theta={theta}", flush=True)

    print("FINAL " + json.dumps(theta), flush=True)


if __name__ == "__main__":
    main()
