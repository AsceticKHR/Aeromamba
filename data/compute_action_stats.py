"""
Compute per-(k, dim) action-chunk statistics for UAV-Flow Stage-3 training.

Iterates the same UAVFlowDataset index / chunk-extraction code used during
training (guaranteeing exact target consistency) but skips image loading,
so it runs in a few minutes over the full dataset.

Output JSON (consumed by training/stage3_action.py --action_stats):
    {
        "chunk_size": 8, "action_dim": 4, "pos_scale": 100.0,
        "num_samples": N,
        "mean": [[..4..] x K],   # per-(k, dim) mean
        "std":  [[..4..] x K],   # per-(k, dim) std
        "turn_fraction_10deg": ...,
    }

Usage:
    python data/compute_action_stats.py \
        --data_root /root/autodl-tmp/datasets/uav-flow \
        --chunk_size 8 --symmetrize_lateral \
        --output /root/autodl-tmp/datasets/uav-flow/action_stats_k8.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import UAVFlowDataset


def main():
    ap = argparse.ArgumentParser(description="UAV-Flow action chunk statistics")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: <data_root>/action_stats_k<K>.json)")
    ap.add_argument("--symmetrize_lateral", action="store_true",
                    help="Force zero mean for y / yaw dims (matches aug_flip training, "
                         "where left-right flipping symmetrises the target distribution).")
    ap.add_argument("--turn_thresh_deg", type=float, default=10.0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[stats] Loading dataset index from {args.data_root} ...")
    ds = UAVFlowDataset(
        data_root=args.data_root,
        tokenizer=None,      # not needed: we never call __getitem__
        transform=None,
        chunk_size=args.chunk_size,
        pos_scale=args.pos_scale,
        aug_flip=False,
        split="train",
    )
    n_total = len(ds.index)
    print(f"[stats] {len(ds.trajectories)} trajectories | {n_total} chunk samples "
          f"({time.time()-t0:.1f}s)")

    K, D = args.chunk_size, 4
    acc_sum = np.zeros((K, D), dtype=np.float64)
    acc_sq = np.zeros((K, D), dtype=np.float64)
    turn_thresh_rad = math.radians(args.turn_thresh_deg)
    n_turn = 0
    n = 0
    abs_max = np.zeros((K, D), dtype=np.float64)

    for traj_idx, step_idx in ds.index:
        chunk = ds._extract_chunk(ds.trajectories[traj_idx], step_idx).numpy().astype(np.float64)
        acc_sum += chunk
        acc_sq += chunk ** 2
        abs_max = np.maximum(abs_max, np.abs(chunk))
        # dyaw targets are cumulative from the anchor (matches dataset
        # oversampling criterion: max |cumulative yaw| over the window)
        if np.abs(chunk[:, 3]).max() >= turn_thresh_rad:
            n_turn += 1
        n += 1
        if n % 200000 == 0:
            print(f"[stats]   {n}/{n_total} ({time.time()-t0:.1f}s)")

    mean = acc_sum / n
    var = np.maximum(acc_sq / n - mean ** 2, 0.0)
    std = np.sqrt(var)

    if args.symmetrize_lateral:
        for d in (1, 3):  # y (right) and yaw
            mean[:, d] = 0.0
            std[:, d] = np.sqrt(acc_sq[:, d] / n)

    out = {
        "chunk_size": K,
        "action_dim": D,
        "pos_scale": args.pos_scale,
        "num_samples": n,
        "num_trajectories": len(ds.trajectories),
        "symmetrize_lateral": bool(args.symmetrize_lateral),
        "turn_thresh_deg": args.turn_thresh_deg,
        "turn_fraction": n_turn / max(n, 1),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "abs_max": abs_max.tolist(),
    }
    out_path = Path(args.output) if args.output else (
        Path(args.data_root) / f"action_stats_k{K}.json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n[stats] samples={n}  turn(>{args.turn_thresh_deg}deg)={n_turn} "
          f"({100.0*n_turn/max(n,1):.2f}%)")
    header = f"{'k':>2} | " + " | ".join(f"{name+'_mean':>10} {name+'_std':>10}"
                                         for name in ("x", "y", "z", "yaw"))
    print(header)
    for k in range(K):
        row = f"{k:>2} | " + " | ".join(
            f"{mean[k, d]:>10.4f} {std[k, d]:>10.4f}" for d in range(D)
        )
        print(row)
    print(f"\n[stats] Saved -> {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
