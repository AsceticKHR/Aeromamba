#!/usr/bin/env python3
"""Summarize inference --diagnose JSONL (runaway / z-dive triage)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="infer_diagnose.jsonl")
    p.add_argument("--episode", type=int, default=None, help="Only one episode id")
    p.add_argument("--tail", type=int, default=0, help="Print last N raw rows")
    args = p.parse_args()

    rows = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.episode is not None and int(row.get("episode", -1)) != args.episode:
                continue
            rows.append(row)

    if not rows:
        print("No rows.")
        return

    by_ep: dict[int, list] = defaultdict(list)
    for r in rows:
        by_ep[int(r.get("episode", 0))].append(r)

    print(f"rows={len(rows)} episodes={len(by_ep)}")
    for ep, erows in sorted(by_ep.items()):
        erows = sorted(erows, key=lambda x: int(x.get("step", 0)))
        flags = Counter(flg for r in erows for flg in (r.get("flags") or []))
        vel = [float(r.get("vel_norm_m") or 0.0) for r in erows]
        pose = [float(r.get("pose_norm_m") or 0.0) for r in erows]
        z_off = [float(z) for r in erows for z in (r.get("z_offset_cm") or [0.0])]
        xy = [float(x) for r in erows for x in (r.get("xy_norm_cm") or [0.0])]
        ep_z = [float(z) for r in erows for z in (r.get("episode_z_cm") or [0.0])]
        instr = (erows[0].get("instr") or "")[:60]
        first_flag = next((r for r in erows if r.get("flags")), None)
        print("=" * 72)
        print(f"ep={ep} steps={len(erows)} mode={erows[0].get('exec_mode')} instr={instr!r}")
        print(
            f"  pose_norm_m: first={pose[0]:.3f} mid={pose[len(pose)//2]:.3f} "
            f"last={pose[-1]:.3f} max={max(pose):.3f}"
        )
        print(
            f"  vel_norm_m:  first={vel[0]:.3f} mid={vel[len(vel)//2]:.3f} "
            f"last={vel[-1]:.3f} max={max(vel):.3f}"
        )
        print(
            f"  xy_cm: med={sorted(xy)[len(xy)//2]:.1f} max={max(xy):.1f} | "
            f"z_off_cm: med={sorted(z_off)[len(z_off)//2]:.1f} min={min(z_off):.1f} | "
            f"ep_z: min={min(ep_z):.1f} last={ep_z[-1]:.1f}"
        )
        print(f"  flags: {dict(flags) if flags else '{}'}")
        if first_flag:
            print(
                f"  first_flag @step{first_flag.get('step')}: {first_flag.get('flags')} "
                f"proprio={first_flag.get('proprio_cm')} vel={first_flag.get('vel_m')}"
            )
        # Show growth every ~25% of episode
        n = max(1, len(erows) // 4)
        for i, name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            seg = erows[i * n : (i + 1) * n] or erows[-1:]
            seg_xy = [float(x) for r in seg for x in (r.get("xy_norm_cm") or [0.0])]
            seg_z = [float(z) for r in seg for z in (r.get("z_offset_cm") or [0.0])]
            seg_v = [float(r.get("vel_norm_m") or 0.0) for r in seg]
            print(
                f"  {name}: xy_mean={sum(seg_xy)/len(seg_xy):.1f} "
                f"z_mean={sum(seg_z)/len(seg_z):.1f} vel_mean={sum(seg_v)/len(seg_v):.3f}"
            )

    if args.tail > 0:
        print("\n--- tail ---")
        for r in rows[-args.tail :]:
            print(
                json.dumps(
                    {
                        "ep": r.get("episode"),
                        "step": r.get("step"),
                        "flags": r.get("flags"),
                        "proprio_cm": r.get("proprio_cm"),
                        "vel_norm_m": r.get("vel_norm_m"),
                        "xy_norm_cm": r.get("xy_norm_cm"),
                        "z_offset_cm": r.get("z_offset_cm"),
                        "body_inc0": (r.get("body_increment_cm") or [None])[0],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
