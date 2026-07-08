"""
Validate prepared UAV-Flow Stage-3 folders.

Checks:
  - each trajectory has log.json
  - image count matches log length
  - images can be opened
  - body-frame action chunks are finite and not all zero
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared UAV-Flow Stage-3 data.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def yaw_delta_deg(yaw: float, yaw0: float) -> float:
    return (yaw - yaw0 + 180.0) % 360.0 - 180.0


def body_frame_actions(raw_logs: List[List[float]], chunk_size: int) -> np.ndarray:
    if not raw_logs:
        return np.zeros((0, 4), dtype=np.float32)
    anchor = raw_logs[0]
    x0, y0, z0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
    yaw0 = float(anchor[4]) if len(anchor) > 4 else 0.0
    yaw0_rad = math.radians(yaw0)
    cos_yaw = math.cos(yaw0_rad)
    sin_yaw = math.sin(yaw0_rad)
    actions = []
    for row in raw_logs[:chunk_size]:
        yaw = float(row[4]) if len(row) > 4 else yaw0
        dx_world = float(row[0]) - x0
        dy_world = float(row[1]) - y0
        actions.append(
            [
                cos_yaw * dx_world + sin_yaw * dy_world,
                -sin_yaw * dx_world + cos_yaw * dy_world,
                float(row[2]) - z0,
                math.radians(yaw_delta_deg(yaw, yaw0)),
            ]
        )
    return np.asarray(actions, dtype=np.float32)


def validate_trajectory(log_path: Path, chunk_size: int) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    length = int(payload.get("length") or len(payload.get("preprocessed_logs") or []))
    images = sorted(log_path.parent.glob("*.jpg"))
    bad_images = 0
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            bad_images += 1
    raw_logs = payload.get("raw_logs") or []
    actions = body_frame_actions(raw_logs, chunk_size)
    finite = bool(actions.size > 0 and np.isfinite(actions).all())
    nonzero = bool(actions.size > 0 and np.abs(actions[:, :3]).sum() > 1e-6)
    return {
        "id": payload.get("id") or log_path.parent.name,
        "length": length,
        "num_images": len(images),
        "bad_images": bad_images,
        "has_instruction": bool(payload.get("instruction_unified") or payload.get("instruction")),
        "finite_actions": finite,
        "nonzero_actions": nonzero,
        "action_abs_mean_cm_rad": actions.mean(axis=0).tolist() if actions.size else [],
        "ok": length >= chunk_size
        and len(images) == length
        and bad_images == 0
        and finite
        and nonzero,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    log_paths = sorted(data_root.rglob("log.json"))
    if args.max_trajectories is not None:
        log_paths = log_paths[: args.max_trajectories]

    results = [validate_trajectory(path, args.chunk_size) for path in log_paths]
    report = {
        "data_root": str(data_root),
        "checked": len(results),
        "ok": sum(1 for item in results if item["ok"]),
        "bad": sum(1 for item in results if not item["ok"]),
        "missing_instruction": sum(1 for item in results if not item["has_instruction"]),
        "bad_images": sum(item["bad_images"] for item in results),
        "zero_action_trajectories": sum(1 for item in results if not item["nonzero_actions"]),
        "examples_bad": [item for item in results if not item["ok"]][:10],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
