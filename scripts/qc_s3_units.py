#!/usr/bin/env python3
"""Strict unit + magnitude QC for Stage-3 UAV-Flow data.

Fails loudly if sim/real position units are wrong (the failure mode that made
the first Falcon-sim run report pos_err_m≈21 against 100×-inflated labels).

Gates (all must PASS):
  U1  detected/forced pos_unit matches --expect_unit (if given)
  U2  median |chunk endpoint| in metres is in [0.15, 15]
  U3  action_stats (if provided) mean/std endpoint scale matches the dataset
      within 2× (guards against stale stats from a previous unit bug)
  U4  real vs sim (optional --compare_root): endpoint medians within 5× of each
      other after correct loading (both should be metre-scale)

Usage:
  python scripts/qc_s3_units.py \\
    --data_root /root/autodl-tmp/datasets/stage3_uavflow_sim \\
    --expect_unit cm --chunk_size 8 --chunk_offset 1 \\
    --action_stats /root/autodl-tmp/datasets/uav-flow-sim/action_stats_k8_off1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import UAVFlowDataset


def _probe_actions(ds: UAVFlowDataset, n: int = 256) -> dict:
    acts = []
    for i in range(min(n * 40, len(ds.index))):
        t, start = ds.index[i]
        traj = ds.trajectories[t]
        try:
            chunk = ds._extract_body_frame_chunk(traj, start)
        except Exception:
            chunk = ds._extract_chunk(traj, start)
        acts.append(np.asarray(chunk, dtype=np.float64))
        if len(acts) >= n:
            break
    if not acts:
        raise RuntimeError("no action chunks extracted")
    A = np.stack(acts)  # [N,K,4]
    end = np.linalg.norm(A[:, -1, :3], axis=-1)
    step0 = np.linalg.norm(A[:, 0, :3], axis=-1)
    return {
        "n": int(len(A)),
        "end_norm_med": float(np.median(end)),
        "end_norm_p90": float(np.percentile(end, 90)),
        "step0_med": float(np.median(step0)),
        "abs_med_xyz": np.median(np.abs(A[..., :3]), axis=(0, 1)).tolist(),
        "yaw_abs_med_rad": float(np.median(np.abs(A[..., 3]))),
    }


def _gate(name: str, ok: bool, detail: str, failures: list) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--expect_unit", choices=["m", "cm", "auto"], default="auto")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--chunk_offset", type=int, default=1)
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--pos_unit", default="auto", choices=["auto", "m", "cm"])
    ap.add_argument("--action_stats", default="")
    ap.add_argument("--compare_root", default="",
                    help="Optional second root (usually real) for cross-domain scale check")
    ap.add_argument("--n_probe", type=int, default=256)
    ap.add_argument("--end_med_min", type=float, default=0.15)
    ap.add_argument("--end_med_max", type=float, default=15.0)
    args = ap.parse_args()

    failures: list = []
    print("=== U0 LOAD ===", flush=True)
    ds = UAVFlowDataset(
        data_root=args.data_root, tokenizer=None, transform=None,
        chunk_size=args.chunk_size, chunk_offset=args.chunk_offset,
        pos_scale=args.pos_scale, pos_unit=args.pos_unit,
        aug_flip=False, split="train",
    )
    print(f"  trajectories={len(ds.trajectories)} chunks={len(ds.index)} "
          f"pos_unit={ds.pos_unit}", flush=True)

    print("=== U1 UNIT ===", flush=True)
    if args.expect_unit != "auto":
        _gate("U1_expect_unit", ds.pos_unit == args.expect_unit,
              f"got {ds.pos_unit}, expect {args.expect_unit}", failures)
    else:
        _gate("U1_expect_unit", ds.pos_unit in ("m", "cm"),
              f"resolved {ds.pos_unit}", failures)

    print("=== U2 ACTION MAGNITUDE (metres after /pos_scale) ===", flush=True)
    probe = _probe_actions(ds, args.n_probe)
    print(f"  probe={probe}", flush=True)
    _gate(
        "U2_end_norm_med",
        args.end_med_min <= probe["end_norm_med"] <= args.end_med_max,
        f"end_norm_med={probe['end_norm_med']:.4f} "
        f"in [{args.end_med_min},{args.end_med_max}]",
        failures,
    )
    _gate(
        "U2_not_100x",
        probe["end_norm_med"] < 50.0,
        f"end_norm_med={probe['end_norm_med']:.4f} (<50 guards residual 100× bug)",
        failures,
    )

    if args.action_stats:
        print("=== U3 ACTION_STATS CONSISTENCY ===", flush=True)
        stats = json.loads(Path(args.action_stats).read_text(encoding="utf-8"))
        mean = np.asarray(stats["mean"], dtype=np.float64)
        end_mean_norm = float(np.linalg.norm(mean[-1, :3]))
        ratio = probe["end_norm_med"] / max(end_mean_norm, 1e-6)
        print(f"  stats_end_mean_norm={end_mean_norm:.4f} "
              f"probe_end_med={probe['end_norm_med']:.4f} ratio={ratio:.3f}",
              flush=True)
        stored_unit = stats.get("pos_unit")
        if stored_unit:
            _gate("U3_stats_unit", stored_unit == ds.pos_unit,
                  f"stats.pos_unit={stored_unit} vs ds={ds.pos_unit}", failures)
        _gate("U3_stats_scale", 0.25 <= ratio <= 4.0,
              f"ratio={ratio:.3f} in [0.25,4]", failures)
        _gate("U3_stats_geometry",
              int(stats.get("chunk_size", -1)) == args.chunk_size
              and int(stats.get("chunk_offset", 0)) == args.chunk_offset,
              f"stats K={stats.get('chunk_size')} off={stats.get('chunk_offset')}",
              failures)
        _gate("U3_has_quantiles", "q01" in stats and "q99" in stats,
              "q01/q99 present" if ("q01" in stats and "q99" in stats) else "MISSING",
              failures)

    if args.compare_root:
        print("=== U4 CROSS-DOMAIN SCALE ===", flush=True)
        ds2 = UAVFlowDataset(
            data_root=args.compare_root, tokenizer=None, transform=None,
            chunk_size=args.chunk_size, chunk_offset=args.chunk_offset,
            pos_scale=args.pos_scale, pos_unit="auto",
            aug_flip=False, split="train",
        )
        probe2 = _probe_actions(ds2, min(args.n_probe, 128))
        r = probe["end_norm_med"] / max(probe2["end_norm_med"], 1e-6)
        print(f"  other_unit={ds2.pos_unit} other_end_med={probe2['end_norm_med']:.4f} "
              f"ratio_this/other={r:.3f}", flush=True)
        _gate("U4_cross_scale", 0.2 <= r <= 5.0,
              f"ratio={r:.3f} in [0.2,5] (both should be metre-scale)", failures)

    print("=== VERDICT ===", flush=True)
    if failures:
        print(f"FAIL: {failures}", flush=True)
        return 2
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
