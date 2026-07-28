"""Build L1 (action imitation) episode-level index for AeroMamba v2.

Scans extracted UAV-Flow episodes (<root>/<episode_id>/{NNNNNN.jpg, log.json})
plus metadata/manifest.jsonl and produces:

  <root>/metadata/l1_episode_index.jsonl   one row per episode:
      id, n_frames, instruction, instruction_unified,
      path_len_m, net_dxyz_m, yaw_delta_deg, z_delta_m,
      motion_hint (regex from instruction), magnitude_m / magnitude_deg
      (numbers parsed from instruction text), direction_word

  <root>/metadata/l1_windows_T{T}_S{S}.jsonl   sliding windows for streaming
      TBPTT training: episode_id, start, length (only windows fully inside
      the episode; short episodes emit one truncated window).

Pure-CPU, streams episode-by-episode (safe for 2GB-RAM no-GPU mode).

Usage:
  python data/build_l1_index.py --root /root/autodl-tmp/datasets/uav-flow \
      --window 48 --stride 24
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DIRECTION_WORDS = (
    "left", "right", "clockwise", "counterclockwise", "counter-clockwise",
    "forward", "backward", "back", "up", "down", "ascend", "descend",
)

MOTION_PATTERNS = [
    ("rotate", re.compile(r"\b(rotate|turn|spin|yaw)\b", re.I)),
    ("orbit", re.compile(r"\b(orbit|circle|surround|around)\b", re.I)),
    ("vertical", re.compile(r"\b(ascend|descend|altitude|rise|lower|climb|up|down)\b", re.I)),
    ("land", re.compile(r"\b(land|landing|touch down)\b", re.I)),
    ("retreat", re.compile(r"\b(away|retreat|back off|move back|backward)\b", re.I)),
    ("approach", re.compile(r"\b(approach|toward|to the|go to|navigate|head)\b", re.I)),
    ("lateral", re.compile(r"\b(shift|sideways|to the left|to the right|strafe)\b", re.I)),
    ("pass", re.compile(r"\b(pass|past|through|make way)\b", re.I)),
]

NUM_METERS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:meters?|m\b)", re.I)
NUM_DEGREES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)", re.I)


def parse_instruction(instr: str) -> dict:
    out: dict = {"motion_hint": None, "direction_word": None,
                 "magnitude_m": None, "magnitude_deg": None}
    for name, pat in MOTION_PATTERNS:
        if pat.search(instr):
            out["motion_hint"] = name
            break
    low = instr.lower()
    for w in DIRECTION_WORDS:
        if w in low:
            out["direction_word"] = w
            break
    m = NUM_METERS.search(instr)
    if m:
        out["magnitude_m"] = float(m.group(1))
    d = NUM_DEGREES.search(instr)
    if d:
        out["magnitude_deg"] = float(d.group(1))
    return out


def episode_stats(log_path: Path) -> dict | None:
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = log.get("raw_logs") or []
    if len(rows) < 3:
        return None
    # raw_logs rows: [x, y, z, roll?, yaw_deg, pitch?, timestamp] (metric, metres)
    xyz = [(r[0], r[1], r[2]) for r in rows]
    yaw = [r[4] for r in rows]
    path_len = 0.0
    for i in range(1, len(xyz)):
        path_len += math.dist(xyz[i], xyz[i - 1])
    net = tuple(xyz[-1][k] - xyz[0][k] for k in range(3))
    dyaw = yaw[-1] - yaw[0]
    dyaw = (dyaw + 180.0) % 360.0 - 180.0
    return {
        "n_log_rows": len(rows),
        "path_len_m": round(path_len, 3),
        "net_dx_m": round(net[0], 3),
        "net_dy_m": round(net[1], 3),
        "net_dz_m": round(net[2], 3),
        "yaw_delta_deg": round(dyaw, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.root)
    meta_dir = root / "metadata"
    manifest = {}
    mf = meta_dir / "manifest.jsonl"
    if mf.exists():
        with mf.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    manifest[row["id"]] = row
                except Exception:
                    continue

    ep_out = meta_dir / "l1_episode_index.jsonl"
    win_out = meta_dir / f"l1_windows_T{args.window}_S{args.stride}.jsonl"
    n_ep = n_win = n_bad = 0

    with ep_out.open("w", encoding="utf-8") as fe, win_out.open("w", encoding="utf-8") as fw:
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name == "metadata":
                continue
            log_path = d / "log.json"
            if not log_path.exists():
                n_bad += 1
                continue
            n_frames = sum(1 for p in d.glob("*.jpg"))
            if n_frames < 3:
                n_bad += 1
                continue
            meta = manifest.get(d.name, {})
            instr = meta.get("instruction", "")
            stats = episode_stats(log_path)
            if stats is None:
                n_bad += 1
                continue
            row = {
                "id": d.name,
                "n_frames": n_frames,
                "instruction": instr,
                "instruction_unified": meta.get("instruction_unified", ""),
                **stats,
                **parse_instruction(instr),
            }
            fe.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ep += 1

            # sliding windows for streaming TBPTT
            if n_frames <= args.window:
                fw.write(json.dumps({"id": d.name, "start": 0, "length": n_frames}) + "\n")
                n_win += 1
            else:
                s = 0
                while s + args.window <= n_frames:
                    fw.write(json.dumps({"id": d.name, "start": s, "length": args.window}) + "\n")
                    n_win += 1
                    s += args.stride
            if n_ep % 2000 == 0:
                print(f"[l1] {n_ep} episodes indexed...", flush=True)

    print(f"[l1] DONE episodes={n_ep} windows={n_win} bad={n_bad}")
    print(f"[l1] wrote {ep_out}")
    print(f"[l1] wrote {win_out}")


if __name__ == "__main__":
    main()
