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
                                    read_episode, split_by_episode)

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
            rows.append((e.index, (e.env_id, e.instruction), e.family, st[0],
                         st[f], f / max(T - 1, 1), ac[f:f + horizon]))
    return rows


def err(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Three errors, because they disagree and the disagreement is the point.

    step      mean per-step L2 on the raw deltas. This is what an L1 chunk loss
              optimises.
    path      mean L2 between the two cumulative paths. Closest offline proxy
              for the official soft-DTW path metric.
    endpoint  L2 between the two 20-step displacements. This is the quantity
              the partial-observability audit measured, where a pose-kNN lands
              within 0.24-0.78 m of an 11.33 m signal.

    A predictor can be excellent on endpoint and useless on step if the
    per-step decomposition is high-frequency. Reporting only one hides that.
    """
    d = pred[..., :3] - gt[..., :3]
    cum = np.cumsum(pred[..., :3], axis=-2) - np.cumsum(gt[..., :3], axis=-2)
    return {"step": float(np.linalg.norm(d, axis=-1).mean()),
            "path": float(np.linalg.norm(cum, axis=-1).mean()),
            "endpoint": float(np.linalg.norm(cum[..., -1, :], axis=-1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--anno_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--min_group", type=int, default=12,
                    help="min episodes per (env_id, instruction) group")
    ap.add_argument("--frames_per_episode", type=int, default=24)
    ap.add_argument("--stride", type=int, default=EXEC_STEPS)
    ap.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--policy_split", action="store_true",
                    help="split like training/v3_train.py so the numbers can "
                         "sit in the same table as a policy's")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Sample whole (env_id, instruction) groups, not random episodes. There are
    # 1,102 distinct instructions across 5,175 episodes, so a random draw of a
    # few hundred leaves most instructions with no fit episodes at all and the
    # class-mean and k-NN predictors silently degrade to the global mean. That
    # measures the absence of data, not the absence of information -- and it is
    # exactly how a trivial baseline gets understated.
    allep = build_index(args.data_root, args.anno_root, args.split)
    groups = collections.defaultdict(list)
    for e in allep:
        groups[(e.env_id, e.instruction)].append(e)

    if args.policy_split:
        # Same split procedure the policy trains under: whole episodes, 3% out,
        # every remaining episode in the fit pool. The dense-group sampler below
        # exists to keep the fit pool non-empty on a small draw, but it does so
        # by restricting to the most repeated instructions, which is an easier
        # subset than the policy is evaluated on. Numbers from the two modes
        # must not be put in the same table.
        tr_eps, va_eps = split_by_episode(allep, args.val_frac, seed=0)
        print(f"[baselines] policy split: {len(tr_eps)} fit / {len(va_eps)} "
              f"eval episodes over all {len(groups)} groups", flush=True)
    else:
        dense = [v for v in groups.values() if len(v) >= args.min_group]
        dense.sort(key=len, reverse=True)
        rng = random.Random(0)
        rng.shuffle(dense)

        tr_eps, va_eps = [], []
        for g in dense:
            if len(tr_eps) + len(va_eps) >= args.episodes:
                break
            rng.shuffle(g)
            # Split inside the group. Holding out whole groups would leave the
            # eval instructions with no fit data and reproduce the artefact.
            k = max(1, int(len(g) * args.val_frac))
            va_eps += g[:k]
            tr_eps += g[k:]
        print(f"[baselines] {len(dense)} dense groups (>= {args.min_group} eps) "
              f"of {len(groups)} total; using {len(tr_eps)} fit / {len(va_eps)} "
              f"eval episodes", flush=True)
    if not tr_eps:
        raise SystemExit("no fit episodes -- lower --min_group")

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
    # Per-channel spread of the cumulative chunk, as a fraction of the ground
    # truth's own. This is the reference the G5 collapse gate needs: a healthy
    # conditional predictor always has *less* spread than the data (it drops
    # the part the observation does not determine), so an arbitrary floor
    # cannot tell "collapsed" from "correctly unsure". Running sums instead of
    # keeping every array -- this is 30k rows x 20 x 4.
    spread = collections.defaultdict(lambda: np.zeros((2, 3, 4)))  # n,sum,sumsq

    def add(name, fam, pred, gt):
        res[name][fam].append(err(pred, gt))
        for i, x in enumerate((np.cumsum(pred, axis=-2), np.cumsum(gt, axis=-2))):
            s = spread[name]
            s[i, 0] += x.shape[-2]
            s[i, 1] += x.sum(axis=-2)
            s[i, 2] += (x ** 2).sum(axis=-2)

    for r in va:
        _, ins, fam, s0, sf, prog, gt = r
        add("zeros", fam, np.zeros_like(gt), gt)
        add("global-mean", fam, gmean, gt)
        add("class-mean", fam, cmean.get(ins, gmean), gt)
        add("class-progress", fam,
            bmean.get((ins, min(9, int(prog * 10))), cmean.get(ins, gmean)), gt)
        pool = idx.get(ins)
        if pool:
            q = np.concatenate([s0 * POSE_W, sf * POSE_W])
            d = np.linalg.norm(feats[ins] - q, axis=1)
            nn = np.argsort(d)[:args.knn]
            add("pose-knn", fam, np.mean([tr[pool[j]][6] for j in nn], axis=0), gt)
        else:
            add("pose-knn", fam, cmean.get(ins, gmean), gt)

    hit = sum(1 for r in va if idx.get(r[1]))
    print(f"[baselines] {100 * hit / len(va):.1f}% of eval rows have a "
          f"non-empty same-group fit pool", flush=True)

    fams = sorted({r[2] for r in va})
    order = ["zeros", "global-mean", "class-mean", "class-progress", "pose-knn"]
    w = max(max(len(f) for f in fams) + 2, 9)
    table = {}
    for metric, blurb in (
            ("step", "mean per-step L2 on raw deltas (what an L1 chunk loss fits)"),
            ("path", "mean L2 between cumulative paths (proxy for soft-DTW)"),
            ("endpoint", f"L2 of the {args.horizon}-step displacement")):
        print(f"\n{metric}_err_m -- {blurb}\n")
        print(f"{'predictor':16s}{'ALL':>9s}" + "".join(f"{f:>{w}s}" for f in fams))
        for k in order:
            per = {f: float(np.mean([d[metric] for d in res[k][f]]))
                   for f in fams if res[k][f]}
            allv = float(np.mean([d[metric] for f in fams for d in res[k][f]]))
            table.setdefault(k, {})[metric] = {"ALL": allv, **per}
            print(f"{k:16s}{allv:9.4f}" +
                  "".join(f"{per.get(f, float('nan')):>{w}.4f}" for f in fams))

    print("\nspread ratio -- std(pred) / std(GT) per channel on the cumulative "
          "chunk\n")
    print(f"{'predictor':16s}" + "".join(f"{c:>10s}"
                                         for c in ("dx", "dy", "dz", "dyaw")))
    ratios = {}
    for k in order:
        s = spread[k]
        sd = [np.sqrt(np.maximum(s[i, 2] / s[i, 0] - (s[i, 1] / s[i, 0]) ** 2, 0))
              for i in (0, 1)]
        r = sd[0] / np.maximum(sd[1], 1e-9)
        ratios[k] = {c: float(r[j]) for j, c in enumerate(("dx", "dy", "dz", "dyaw"))}
        print(f"{k:16s}" + "".join(f"{v:>10.3f}" for v in r))
    print("\nRead the pose-knn row as the per-channel G5 threshold: it is what "
          "a memoryless predictor with a same-instruction fit pool retains. A "
          "channel far below it is collapsed; a channel near it is as sure as "
          "the observation allows.")

    print("\nA policy must beat class-progress by a margin worth reporting. "
          "pose-knn is the memoryless ceiling: beating it is evidence the "
          "policy uses something beyond the current pose.\n"
          "If pose-knn is strong on endpoint and weak on step, the per-step "
          "decomposition is mostly high-frequency and step_err_m is the wrong "
          "headline metric -- fitting it means fitting jitter.")
    if args.out:
        json.dump({"table": table, "spread_ratio": ratios, "config": vars(args)},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    print("TRIVIAL_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
