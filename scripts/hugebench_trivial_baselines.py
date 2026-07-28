"""Trivial action baselines on HUGE-Bench, reported per task family.

    python scripts/hugebench_trivial_baselines.py --data_root $D --anno_root $A

Required alongside every AeroV3 number. On UAV-Flow-Sim a 1.5B policy reached
pos_err_m 0.2156 against a class-mean lookup table at 0.2398 -- a 10.1% lead
that looked respectable until the baseline was on the page next to it. The
metric here is defined exactly as in ``training/v3_train.py``: mean L2 over the
first three action channels, per step of a 20-step chunk, in metres.

Predictors, weakest to strongest:

    zeros           hold position
    global-mean     one mean chunk for the whole dataset
    class-mean      one mean chunk per instruction
    class-progress  one mean chunk per (instruction, progress decile) -- this
                    is the lookup table a benchmark has to beat to be measuring
                    a policy rather than a prior
    pose-knn        k-NN over (instruction, initial pose, current pose), the
                    same estimator as the partial-observability audit; an upper
                    bound on what any memoryless predictor can do

Images are never opened, so this runs in minutes on the state columns alone.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.hugebench_dataset import (ACTION_HORIZON, EXEC_STEPS, build_index,
                                    read_episode)

# Yaw is in radians and one step spans ~0.09 rad while position spans ~1 m.
# Without this the neighbour search is decided entirely by position.
POSE_W = np.array([1.0, 1.0, 1.0, 10.0])


def load(eps, stride, horizon, max_frames_per_ep):
    """Flatten episodes into (context, chunk) rows. Context is what a memoryless
    predictor is allowed to see: instruction, initial pose, current pose."""
    rows = []
    for e in eps:
        try:
            st, ac, _, _ = read_episode(e.path, with_images=False)
        except Exception:
            continue
        T = min(len(st), len(ac))
        starts = list(range(1, T - horizon, stride))
        if len(starts) > max_frames_per_ep:
            step = len(starts) / max_frames_per_ep
            starts = [starts[int(i * step)] for i in range(max_frames_per_ep)]
        for f in starts:
            rows.append((e.index, e.instruction, e.family, st[0], st[f],
                         f / max(T - 1, 1), ac[f:f + horizon]))
    return rows


def err(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean over steps of the L2 position error, metres. Same definition the
    training script evaluates, so the two tables can sit side by side."""
    return float(np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--anno_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--frames_per_episode", type=int, default=24)
    ap.add_argument("--stride", type=int, default=EXEC_STEPS)
    ap.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    eps = build_index(args.data_root, args.anno_root, args.split)
    rng = random.Random(0)
    rng.shuffle(eps)
    eps = eps[:args.episodes]
    n_val = max(1, int(len(eps) * args.val_frac))
    # Held out by episode. A frame-level split lets a k-NN retrieve the
    # neighbouring frame of the same trajectory and score near zero.
    va_eps, tr_eps = eps[:n_val], eps[n_val:]
    print(f"[baselines] {len(tr_eps)} fit / {len(va_eps)} eval episodes",
          flush=True)

    tr = load(tr_eps, args.stride, args.horizon, args.frames_per_episode)
    va = load(va_eps, args.stride, args.horizon, args.frames_per_episode)
    print(f"[baselines] {len(tr)} fit / {len(va)} eval rows", flush=True)
    if not va:
        raise SystemExit("no evaluation rows")

    gmean = np.mean([r[6] for r in tr], axis=0)

    by_instr = collections.defaultdict(list)
    by_bin = collections.defaultdict(list)
    for r in tr:
        by_instr[r[1]].append(r[6])
        by_bin[(r[1], min(9, int(r[5] * 10)))].append(r[6])
    cmean = {k: np.mean(v, axis=0) for k, v in by_instr.items()}
    bmean = {k: np.mean(v, axis=0) for k, v in by_bin.items()}

    # k-NN index, grouped by instruction so the neighbour search never crosses
    # tasks. Features are the same (initial pose, current pose) pair the audit
    # used, so the two numbers are directly comparable.
    idx = collections.defaultdict(list)
    for i, r in enumerate(tr):
        idx[r[1]].append(i)
    feats = {k: np.stack([np.concatenate([tr[i][3] * POSE_W, tr[i][4] * POSE_W])
                          for i in v]) for k, v in idx.items()}

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in va:
        _, ins, fam, s0, sf, prog, gt = r
        z = np.zeros_like(gt)
        res["zeros"][fam].append(err(z, gt))
        res["global-mean"][fam].append(err(gmean, gt))
        res["class-mean"][fam].append(err(cmean.get(ins, gmean), gt))
        res["class-progress"][fam].append(
            err(bmean.get((ins, min(9, int(prog * 10))), cmean.get(ins, gmean)), gt))
        pool = idx.get(ins)
        if pool:
            q = np.concatenate([s0 * POSE_W, sf * POSE_W])
            d = np.linalg.norm(feats[ins] - q, axis=1)
            nn = np.argsort(d)[:args.knn]
            res["pose-knn"][fam].append(
                err(np.mean([tr[pool[j]][6] for j in nn], axis=0), gt))
        else:
            res["pose-knn"][fam].append(err(cmean.get(ins, gmean), gt))

    fams = sorted({r[2] for r in va})
    order = ["zeros", "global-mean", "class-mean", "class-progress", "pose-knn"]
    w = max(len(f) for f in fams) + 2
    print(f"\npos_err_m, mean L2 over a {args.horizon}-step chunk, metres\n")
    print(f"{'predictor':16s}{'ALL':>9s}" + "".join(f"{f:>{w}s}" for f in fams))
    table = {}
    for k in order:
        allv = float(np.mean([v for f in fams for v in res[k][f]]))
        table[k] = {"ALL": allv,
                    **{f: float(np.mean(res[k][f])) for f in fams if res[k][f]}}
        print(f"{k:16s}{allv:9.4f}" +
              "".join(f"{table[k].get(f, float('nan')):>{w}.4f}" for f in fams))

    print("\nA policy must beat class-progress by a margin worth reporting. "
          "pose-knn is the memoryless ceiling: anything at or below it is "
          "evidence the policy is using something beyond the current pose.")
    if args.out:
        json.dump({"table": table, "config": vars(args)},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    print("TRIVIAL_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
