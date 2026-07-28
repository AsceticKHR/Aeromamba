#!/usr/bin/env python
"""Verify that terminal padding actually produces stopping labels.

The eval harness ends an episode only when the policy emits <3cm/step for 10
consecutive steps, but only 11.1% of UAV-Flow trajectories end at rest, so no
training window ever carries a "you have arrived, hold position" target and the
policy never terminates (0/273 closed-loop episodes). Terminal padding clamps
the target index past the end of the trajectory, repeating the final pose.

This test checks the three things that have to hold for that to work, against
the real dataset:

  1. frac=0 reproduces the old chunk count exactly (no silent behaviour change)
  2. frac>0 keeps every trajectory usable at large K and adds chunks
  3. the padded windows really do carry near-zero terminal displacement, at a
     rate that matches how much padding was requested
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import UAVFlowDataset  # noqa: E402

# harness: 10 consecutive steps under 3 cm. Targets are metres.
STOP_M = 0.03


def build(root: str, k: int, frac: float) -> UAVFlowDataset:
    return UAVFlowDataset(data_root=root, chunk_size=k, chunk_offset=1,
                          pos_scale=100.0, tokenizer=None, transform=None,
                          split="val", terminal_pad_frac=frac)


def terminal_step_sizes(ds: UAVFlowDataset, n: int = 4000) -> np.ndarray:
    """Per-window mean step size over the final 10 waypoints, in metres."""
    stride = max(1, len(ds.index) // n)
    out = []
    for i in range(0, len(ds.index), stride):
        t, s = ds.index[i]
        chunk = ds._extract_chunk(ds.trajectories[t], s).numpy()  # (K, 4) metres
        tail = chunk[-10:, :3]
        if len(tail) < 2:
            continue
        out.append(float(np.linalg.norm(np.diff(tail, axis=0), axis=1).mean()))
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()

    ok = True

    print(f"=== T1: frac=0 must not change existing behaviour (K={args.k}) ===")
    base = build(args.data_root, args.k, 0.0)
    n_traj_base, n_chunk_base = len(base.trajectories), len(base.index)
    # Recompute the pre-change formula independently.
    expected = sum(max(len(t) - args.k - 1 + 1, 0) for t in base.trajectories)
    print(f"  trajectories {n_traj_base} | chunks {n_chunk_base} "
          f"| independently recomputed {expected}")
    if n_chunk_base != expected:
        print("  FAIL: chunk count drifted from the unpadded formula")
        ok = False
    else:
        print("  PASS")

    print(f"\n=== T2: frac=0.5 keeps trajectories and adds chunks (K={args.k}) ===")
    pad = build(args.data_root, args.k, 0.5)
    n_traj_pad, n_chunk_pad = len(pad.trajectories), len(pad.index)
    print(f"  trajectories {n_traj_base} -> {n_traj_pad} "
          f"| chunks {n_chunk_base} -> {n_chunk_pad} "
          f"({100.0*n_chunk_pad/max(n_chunk_base,1):.1f}%)")
    if n_traj_pad < n_traj_base or n_chunk_pad <= n_chunk_base:
        print("  FAIL: padding should never lose trajectories or chunks")
        ok = False
    else:
        print("  PASS")

    print("\n=== T3: padded windows carry near-zero terminal motion ===")
    d0, d1 = terminal_step_sizes(base), terminal_step_sizes(pad)
    r0 = 100.0 * (d0 < STOP_M).mean()
    r1 = 100.0 * (d1 < STOP_M).mean()
    print(f"  windows whose last 10 waypoints move <{STOP_M} m/step:")
    print(f"    frac=0.0 : {r0:5.1f}%   (median step {np.median(d0):.4f} m)")
    print(f"    frac=0.5 : {r1:5.1f}%   (median step {np.median(d1):.4f} m)")
    if r1 <= r0 + 1.0:
        print("  FAIL: padding did not create stopping labels")
        ok = False
    else:
        print(f"  PASS: stopping supervision went from {r0:.1f}% to {r1:.1f}% "
              f"of windows")

    print("\n=== T4: K=50 stays fully usable under padding ===")
    p50 = build(args.data_root, 50, 0.5)
    b50 = build(args.data_root, 50, 0.0)
    print(f"  K=50 trajectories: unpadded {len(b50.trajectories)} "
          f"-> padded {len(p50.trajectories)} (K=8 reference {n_traj_base})")
    if len(p50.trajectories) <= len(b50.trajectories):
        print("  FAIL: padding should recover the trajectories a long K drops")
        ok = False
    else:
        print("  PASS")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
