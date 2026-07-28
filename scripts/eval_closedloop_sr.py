#!/usr/bin/env python
"""Reproducible closed-loop success + stopping metrics for UAV-Flow.

The official protocol scores Success Rate by *manual visual inspection* and
ships no code for it; `UAV-Flow-Eval/metric.py` only computes nDTW. That makes
published SR numbers impossible to reproduce and impossible to compare against
without re-running a human study. This script defines a geometric SR that is
fully determined by the logs, and reports it alongside the stopping statistics
that the nDTW score is blind to.

Conventions follow `UAV-Flow-Eval/metric.py` exactly:
  * model log  : list of steps, step['state'][0] = xyz in UE cm,
                 step['state'][1] = (roll, yaw, pitch) in degrees
  * ground truth: test_jsons/<episode>.json, 'reference_path_preprocessed' is a
                 list of 6D [x, y, z, roll, yaw, pitch] in cm/deg
  * Turn and Rotate are orientation-only classes (metric.py zeroes their
    positions), so their success is judged on yaw alone.

Reported metrics
  SR@Xm        final pose within X m (and yaw tolerance) of the GT endpoint
  OSR@Xm       the trajectory came within X m at *some* point (oracle)
  overshoot    final error minus closest-approach error; the distance the
               policy travelled after it had already arrived
  stop_rate    fraction of episodes that ended before the step budget, i.e.
               the policy actually came to rest instead of being cut off
  steps        executed steps per episode (budget is 100)

Usage:
  python scripts/eval_closedloop_sr.py \
      --results_dir "../UAV-Flow-Eval/results/aerov2_v4_diverse" \
      --gt_dir      "../UAV-Flow-Eval/test_jsons" \
      --classified  "../UAV-Flow-Eval/classified_instr.json"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ORIENTATION_ONLY = {"Turn", "Rotate"}
STEP_BUDGET = 100  # batch_run_act_all.py max_steps


def load_model_traj(path: Path) -> Optional[np.ndarray]:
    """-> (T, 4) array of [x_cm, y_cm, z_cm, yaw_deg], or None if unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = []
    for step in data:
        st = step.get("state")
        if not st or len(st) < 2:
            continue
        pos, rot = st[0], st[1]
        # metric.py reads rot as (roll, yaw, pitch); yaw is index 1.
        rows.append([float(pos[0]), float(pos[1]), float(pos[2]), float(rot[1])])
    return np.asarray(rows, dtype=np.float64) if rows else None


def load_gt(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ref = data.get("reference_path_preprocessed") or []
    if not ref:
        return None
    ref = np.asarray(ref, dtype=np.float64)
    return {"path": ref[:, :3], "yaw": ref[:, 4], "end": ref[-1, :3],
            "end_yaw": float(ref[-1, 4])}


def yaw_err_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def episode_metrics(model: np.ndarray, gt: Dict[str, np.ndarray],
                    orientation_only: bool) -> Dict[str, float]:
    # The harness anchors model poses to the episode's first frame, and
    # reference_path_preprocessed is anchored the same way, so both are already
    # in the same frame and no alignment is needed.
    m_xyz, m_yaw = model[:, :3], model[:, 3]
    final_yaw_e = yaw_err_deg(float(m_yaw[-1]), gt["end_yaw"])

    if orientation_only:
        yaw_track = np.array([yaw_err_deg(y, gt["end_yaw"]) for y in m_yaw])
        return {"final_pos_err_m": float("nan"),
                "min_pos_err_m": float("nan"),
                "overshoot_m": float("nan"),
                "final_yaw_err_deg": final_yaw_e,
                "min_yaw_err_deg": float(yaw_track.min()),
                "steps": float(len(model)),
                "stopped": float(len(model) < STEP_BUDGET),
                "path_len_m": float(np.linalg.norm(np.diff(m_xyz, axis=0),
                                                   axis=1).sum() / 100.0)}

    d = np.linalg.norm(m_xyz - gt["end"][None, :], axis=1) / 100.0
    final_e, min_e = float(d[-1]), float(d.min())
    return {"final_pos_err_m": final_e,
            "min_pos_err_m": min_e,
            # How far it kept going after its closest approach. This is the
            # quantity that "cannot stop" failures show up in and that nDTW,
            # being a shape metric, largely absorbs.
            "overshoot_m": final_e - min_e,
            "final_yaw_err_deg": final_yaw_e,
            "min_yaw_err_deg": float("nan"),
            "steps": float(len(model)),
            "stopped": float(len(model) < STEP_BUDGET),
            "path_len_m": float(np.linalg.norm(np.diff(m_xyz, axis=0),
                                               axis=1).sum() / 100.0)}


def summarise(rows: List[Dict[str, float]], pos_thresh: List[float],
              yaw_thresh: float) -> Dict[str, float]:
    if not rows:
        return {}
    out: Dict[str, float] = {"n": float(len(rows))}
    for t in pos_thresh:
        sr, osr, n = 0, 0, 0
        for r in rows:
            if math.isnan(r["final_pos_err_m"]):        # orientation-only class
                ok = r["final_yaw_err_deg"] <= yaw_thresh
                ok_o = r["min_yaw_err_deg"] <= yaw_thresh
            else:
                ok = (r["final_pos_err_m"] <= t
                      and r["final_yaw_err_deg"] <= yaw_thresh)
                ok_o = r["min_pos_err_m"] <= t
            sr += int(ok); osr += int(ok_o); n += 1
        out[f"SR@{t:g}m"] = 100.0 * sr / max(n, 1)
        out[f"OSR@{t:g}m"] = 100.0 * osr / max(n, 1)
    for k in ("final_pos_err_m", "min_pos_err_m", "overshoot_m",
              "final_yaw_err_deg", "steps", "stopped", "path_len_m"):
        vals = [r[k] for r in rows if not math.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["stop_rate"] = 100.0 * out["stopped"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--classified", required=True)
    ap.add_argument("--pos_thresh", type=float, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--yaw_thresh", type=float, default=30.0)
    ap.add_argument("--out", default=None, help="write per-episode rows as JSON")
    args = ap.parse_args()

    res_dir, gt_dir = Path(args.results_dir), Path(args.gt_dir)
    classes: Dict[str, List[str]] = json.loads(
        Path(args.classified).read_text(encoding="utf-8"))

    per_class: Dict[str, List[Dict[str, float]]] = {}
    all_rows: List[Dict[str, float]] = []
    detail: Dict[str, Dict[str, float]] = {}
    missing = 0

    for cls, files in classes.items():
        rows: List[Dict[str, float]] = []
        for fn in files:
            mp, gp = res_dir / fn, gt_dir / fn
            model, gt = load_model_traj(mp), load_gt(gp)
            if model is None or gt is None:
                missing += 1
                continue
            r = episode_metrics(model, gt, cls in ORIENTATION_ONLY)
            rows.append(r)
            all_rows.append(r)
            detail[fn] = {"class": cls, **r}
        per_class[cls] = rows

    hdr = (f"{'Class':<16}{'N':>4}" +
           "".join(f"{'SR@'+format(t,'g')+'m':>10}" for t in args.pos_thresh) +
           f"{'OSR@3m':>9}{'final_m':>9}{'min_m':>8}{'over_m':>8}"
           f"{'yaw°':>7}{'steps':>7}{'stop%':>7}")
    print(f"\nresults: {res_dir}")
    if missing:
        print(f"WARNING: {missing} episodes had no usable log and were skipped")
    print(hdr)
    print("-" * len(hdr))
    for cls, rows in per_class.items():
        if not rows:
            continue
        s = summarise(rows, args.pos_thresh, args.yaw_thresh)
        line = f"{cls:<16}{int(s['n']):>4}"
        line += "".join(f"{s[f'SR@{t:g}m']:>10.1f}" for t in args.pos_thresh)
        line += (f"{s.get('OSR@3m', float('nan')):>9.1f}"
                 f"{s['final_pos_err_m']:>9.2f}{s['min_pos_err_m']:>8.2f}"
                 f"{s['overshoot_m']:>8.2f}{s['final_yaw_err_deg']:>7.1f}"
                 f"{s['steps']:>7.1f}{s['stop_rate']:>7.1f}")
        print(line)

    print("-" * len(hdr))
    # Macro average over classes, matching how WorldVLN reports UAV-Flow SR
    # (verified: their per-class values divide by the official class counts and
    # their Average is the unweighted mean of the 10 classes).
    macro = {}
    for t in args.pos_thresh:
        vals = [summarise(r, args.pos_thresh, args.yaw_thresh)[f"SR@{t:g}m"]
                for r in per_class.values() if r]
        macro[t] = float(np.mean(vals))
    overall = summarise(all_rows, args.pos_thresh, args.yaw_thresh)
    line = f"{'MACRO avg':<16}{int(overall['n']):>4}"
    line += "".join(f"{macro[t]:>10.1f}" for t in args.pos_thresh)
    line += (f"{overall.get('OSR@3m', float('nan')):>9.1f}"
             f"{overall['final_pos_err_m']:>9.2f}{overall['min_pos_err_m']:>8.2f}"
             f"{overall['overshoot_m']:>8.2f}{overall['final_yaw_err_deg']:>7.1f}"
             f"{overall['steps']:>7.1f}{overall['stop_rate']:>7.1f}")
    print(line)
    print(f"\nyaw tolerance {args.yaw_thresh:g}deg; step budget {STEP_BUDGET}. "
          f"stop% is the share of episodes that ended on their own rather than "
          f"being cut off by the budget.")

    if args.out:
        Path(args.out).write_text(json.dumps(detail, indent=2), encoding="utf-8")
        print(f"per-episode rows -> {args.out}")


if __name__ == "__main__":
    main()
