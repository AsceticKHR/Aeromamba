#!/usr/bin/env python
"""What do you get on Stage-3 without a model at all?

`pos_err_m` and `end_pos_err_m` are only meaningful relative to what a
constant predictor achieves. UAV-Flow chunks are short (median per-step
displacement 0.19 m), so a policy that ignores its inputs entirely and emits
the dataset-mean chunk already scores well on an L1 metric. Any claim that the
policy "learned" something has to clear these bars first.

Predictors, all evaluated on trajectories held out at the *trajectory* level
with the same seed/val_frac as training:

  zeros        emit no motion at all
  global-mean  emit the mean training chunk, ignoring every input
  class-mean   emit the mean training chunk of the sample's motion class,
               i.e. a lookup table keyed on the instruction alone; this is the
               bar that "the policy understands the instruction" must beat
  oracle-traj  emit the mean chunk of the *same* trajectory (unavailable at
               test time) — an upper bound on what a per-trajectory constant
               can do, and therefore a measure of how much within-trajectory
               phase information any single-frame policy is missing

Metrics replicate `aero_action_loss_v5` exactly: pos_err_m is the mean absolute
error over all K waypoints and all 3 position axes; end_pos_err_m is the same
over the final waypoint only.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import UAVFlowDataset  # noqa: E402


def report(name: str, err: np.ndarray) -> None:
    """err: (N, K, 4) absolute error in metres / radians."""
    pos = err[..., :3].mean()
    end = err[:, -1, :3].mean()
    yaw = err[..., 3].mean() * 180.0 / np.pi
    print(f"{name:<14}{pos:>12.4f}{end:>16.4f}{yaw:>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--chunk_offset", type=int, default=1)
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--val_frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # split="val" turns off flip augmentation and class oversampling, so
    # ds.index carries each chunk exactly once.
    ds = UAVFlowDataset(data_root=args.data_root, chunk_size=args.chunk_size,
                        chunk_offset=args.chunk_offset, pos_scale=args.pos_scale,
                        tokenizer=None, transform=None, split="val")
    n_traj = len(ds.trajectories)

    # Reproduce training's trajectory-level split bit for bit.
    traj_ids = list(range(n_traj))
    random.Random(args.seed).shuffle(traj_ids)
    n_val_traj = max(1, int(n_traj * args.val_frac))
    val_traj = set(traj_ids[:n_val_traj])
    print(f"{n_traj} trajectories -> {n_traj - n_val_traj} train / "
          f"{n_val_traj} val (seed {args.seed})")

    cls_of = getattr(ds, "traj_motion_class", None)
    if cls_of is not None and len(cls_of) != n_traj:
        raise SystemExit(f"traj_motion_class has {len(cls_of)} entries for "
                         f"{n_traj} trajectories; refusing to mis-key class-mean")
    def class_of(t: int) -> int:
        return int(cls_of[t]) if cls_of is not None else -1
    tr_sum, tr_n = np.zeros((args.chunk_size, 4)), 0
    cls_sum, cls_n = defaultdict(lambda: np.zeros((args.chunk_size, 4))), defaultdict(int)
    val_rows, val_cls, val_traj_of = [], [], []
    # De-duplicate val entries: oversampling repeats indices, and a repeated
    # chunk would silently reweight the estimate.
    seen = set()

    for i, (t, s) in enumerate(ds.index):
        chunk = ds._extract_chunk(ds.trajectories[t], s).numpy()
        c = class_of(t)
        if t in val_traj:
            if (t, s) in seen:
                continue
            seen.add((t, s))
            val_rows.append(chunk); val_cls.append(c); val_traj_of.append(t)
        else:
            tr_sum += chunk; tr_n += 1
            cls_sum[c] += chunk; cls_n[c] += 1

    V = np.stack(val_rows)                       # (N, K, 4)
    gmean = tr_sum / max(tr_n, 1)
    print(f"{tr_n} train chunks | {len(V)} val chunks (deduplicated)\n")

    print(f"{'predictor':<14}{'pos_err_m':>12}{'end_pos_err_m':>16}{'yaw_err_deg':>14}")
    print("-" * 56)
    report("zeros", np.abs(V))
    report("global-mean", np.abs(V - gmean[None]))

    cmean = {c: cls_sum[c] / cls_n[c] for c in cls_sum if cls_n[c] > 0}
    pred = np.stack([cmean.get(c, gmean) for c in val_cls])
    report("class-mean", np.abs(V - pred))

    by_traj = defaultdict(list)
    for row, t in zip(V, val_traj_of):
        by_traj[t].append(row)
    tmean = {t: np.mean(v, axis=0) for t, v in by_traj.items()}
    pred = np.stack([tmean[t] for t in val_traj_of])
    report("oracle-traj", np.abs(V - pred))
    print("-" * 56)
    print("oracle-traj cheats (it sees the held-out trajectory's own mean); the\n"
          "gap between it and global-mean is the headroom a single-frame policy\n"
          "can reach, and the gap below it is what needs temporal context.")


if __name__ == "__main__":
    main()
