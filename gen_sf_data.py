#!/usr/bin/env python3
"""Generate Stockfish-labeled NNUE training data ("distill Stockfish").

Stockfish plays itself from short random openings. For every quiet position we
record the FEN, Stockfish's own evaluation (centipawns, side-to-move view), and
the eventual game result — exactly the `FEN | score_cp | result` format that
`train_nnue.py` reads. Training a net on these labels teaches it Stockfish's
judgement instead of Mythos's hand-crafted eval, which is the whole point.

Usage:
    python gen_sf_data.py out.txt --games 4000 --depth 8 --workers 6
"""
import argparse
import os
import random
import time
from multiprocessing import Process

import chess
import chess.engine


def play_worker(wid, sf_path, out_path, games, depth, min_open, max_open, maxmoves, seed):
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        eng.configure({"Threads": 1, "Hash": 32})
    except Exception:
        pass
    limit = chess.engine.Limit(depth=depth)
    written = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for g in range(games):
            board = chess.Board()
            # Random opening for diversity; retry if it ends the game early.
            k = rng.randint(min_open, max_open)
            ok = True
            for _ in range(k):
                moves = list(board.legal_moves)
                if not moves:
                    ok = False
                    break
                board.push(rng.choice(moves))
            if not ok or board.is_game_over():
                continue

            samples = []  # (fen, cp, stm_is_white)
            while not board.is_game_over(claim_draw=True) and board.fullmove_number < maxmoves:
                try:
                    info = eng.analyse(board, limit)
                except Exception:
                    break
                score = info["score"].pov(board.turn)
                pv = info.get("pv") or []
                best = pv[0] if pv else None
                # Record only quiet, non-mate positions whose best move is not a
                # capture/promotion — those are what the net will judge at leaves.
                if best is not None and not score.is_mate() and not board.is_check():
                    if not board.is_capture(best) and best.promotion is None:
                        cp = score.score()
                        if cp is not None and abs(cp) < 3000:
                            samples.append((board.fen(), cp, board.turn == chess.WHITE))
                if best is None:
                    break
                board.push(best)

            # Game result from White's perspective.
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                white_res = 0.5
            else:
                white_res = 1.0 if outcome.winner == chess.WHITE else 0.0

            for fen, cp, stm_white in samples:
                res = white_res if stm_white else (1.0 - white_res)
                f.write(f"{fen} | {cp} | {res}\n")
                written += 1

            if (g + 1) % 25 == 0:
                f.flush()
                print(f"[w{wid}] game {g+1}/{games}  {written:,} samples  "
                      f"{time.time()-t0:.0f}s", flush=True)
    eng.quit()
    print(f"[w{wid}] DONE {written:,} samples -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sf", default="bench_engines/stockfish/stockfish-windows-x86-64-avx2.exe")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-open", type=int, default=4)
    ap.add_argument("--max-open", type=int, default=10)
    ap.add_argument("--maxmoves", type=int, default=160)
    args = ap.parse_args()

    per = max(1, args.games // args.workers)
    shards = [f"{args.out}.w{i}" for i in range(args.workers)]
    procs = []
    for i in range(args.workers):
        p = Process(target=play_worker, args=(
            i, args.sf, shards[i], per, args.depth,
            args.min_open, args.max_open, args.maxmoves, 1234 + i * 99))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Merge shards.
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# FEN | stm_score_cp | stm_result  (Stockfish-labeled)\n")
        for s in shards:
            if os.path.exists(s):
                with open(s, "r", encoding="utf-8") as f:
                    for ln in f:
                        out.write(ln)
                        total += 1
                os.remove(s)
    print(f"MERGED {total:,} samples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
