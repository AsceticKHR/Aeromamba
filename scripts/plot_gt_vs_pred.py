"""Overlay GT vs model-predicted trajectories for AeroMamba closed-loop eval.

For each UAV-Flow-Eval task we have:
  - model log:  results/<run>/<base>.json  -> list of {"state": [xyz_cm, [_, yaw_deg], ...]}
  - GT task:    test_jsons/<base>.json      -> {"reference_path_preprocessed": [[x,y,z,?,yaw_deg], ...],
                                                "instruction", "initial_pos", "target_pos", ...}

Picks one representative completed task per motion category (from classified_instr.json)
and draws:
  - a top-down (Y-right vs X-forward) GT-vs-model overlay grid with target markers
  - a matching 3D overlay grid

Saved to <Aeromamba>/reports/plots/gt_vs_pred_<run>.png and gt_vs_pred_3d_<run>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_model(path: Path):
    log = json.loads(path.read_text(encoding="utf-8"))
    if not log:
        return None, None
    xyz = np.array([it["state"][0] for it in log], dtype=float)
    yaw = np.array([it["state"][1][1] for it in log], dtype=float)
    return xyz, yaw


def world_to_local(target, init_pos):
    """Same convention as batch_run_act_all.py: yaw0 = initial_pos[4]."""
    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx, dy, dz = target[0] - x0, target[1] - y0, target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)
    return np.array([x_rel, y_rel, dz], dtype=float)


def load_gt(path: Path):
    task = json.loads(path.read_text(encoding="utf-8"))
    ref = np.array(task["reference_path_preprocessed"], dtype=float)
    instr = task.get("instruction") or task.get("task") or ""
    target_local = None
    tp, ip = task.get("target_pos"), task.get("initial_pos")
    if tp and ip and len(ip) >= 5:
        target_local = world_to_local(tp, ip)
    return ref[:, :3], ref[:, 4], instr, target_local


def pick_tasks(root: Path, result_dir: Path, test_dir: Path, per_class: int):
    classes = json.loads((root / "classified_instr.json").read_text(encoding="utf-8"))
    picks = []  # (category, base)
    for cat, files in classes.items():
        cnt = 0
        for fname in files:
            base = fname[:-5] if fname.endswith(".json") else fname
            mp = result_dir / f"{base}.json"
            gp = test_dir / f"{base}.json"
            if mp.exists() and gp.exists():
                try:
                    m_xyz, _ = load_model(mp)
                except Exception:
                    continue
                if m_xyz is not None and len(m_xyz) >= 3:
                    picks.append((cat, base))
                    cnt += 1
                    if cnt >= per_class:
                        break
    return picks


def plot_2d(picks, result_dir: Path, test_dir: Path, run: str, out: Path):
    n = len(picks)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)

    for i, (cat, base) in enumerate(picks):
        ax = axes[i // ncol][i % ncol]
        m_xyz, _ = load_model(result_dir / f"{base}.json")
        g_xyz, _, instr, tgt = load_gt(test_dir / f"{base}.json")

        # top-down: Y (right) on x-axis, X (forward) on y-axis (matches eval plots)
        ax.plot(g_xyz[:, 1], g_xyz[:, 0], "-", color="tab:green", lw=2.2, label="GT", zorder=2)
        ax.plot(m_xyz[:, 1], m_xyz[:, 0], "-", color="tab:blue", lw=1.8, label="model", zorder=3)
        ax.scatter([g_xyz[0, 1]], [g_xyz[0, 0]], c="black", s=40, marker="o", zorder=4, label="start")
        ax.scatter([g_xyz[-1, 1]], [g_xyz[-1, 0]], c="tab:green", s=70, marker="*", zorder=4)
        ax.scatter([m_xyz[-1, 1]], [m_xyz[-1, 0]], c="tab:blue", s=70, marker="*", zorder=4)
        if tgt is not None:
            ax.scatter([tgt[1]], [tgt[0]], c="red", s=60, marker="X", zorder=5, label="target")

        endpt = float(np.linalg.norm(m_xyz[-1, :2] - g_xyz[-1, :2]))
        ax.set_title(f"[{cat}] {instr[:34]}", fontsize=8)
        ax.text(0.02, 0.02, f"end_xy Δ={endpt:.0f}cm", transform=ax.transAxes,
                fontsize=7, va="bottom", color="dimgray")
        ax.set_xlabel("Y right (cm)", fontsize=7)
        ax.set_ylabel("X fwd (cm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axis("equal")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"GT (green) vs model (blue), target (red X) — {run}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_3d(picks, result_dir: Path, test_dir: Path, run: str, out: Path):
    n = len(picks)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 4.6 * nrow))

    for i, (cat, base) in enumerate(picks):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        m_xyz, _ = load_model(result_dir / f"{base}.json")
        g_xyz, _, instr, tgt = load_gt(test_dir / f"{base}.json")

        # axes: X=Y-right, Y=X-forward, Z=up (consistent with 2D top-down view)
        ax.plot(g_xyz[:, 1], g_xyz[:, 0], g_xyz[:, 2], "-", color="tab:green", lw=2.2, label="GT")
        ax.plot(m_xyz[:, 1], m_xyz[:, 0], m_xyz[:, 2], "-", color="tab:blue", lw=1.8, label="model")
        ax.scatter([g_xyz[0, 1]], [g_xyz[0, 0]], [g_xyz[0, 2]], c="black", s=40, marker="o", label="start")
        ax.scatter([g_xyz[-1, 1]], [g_xyz[-1, 0]], [g_xyz[-1, 2]], c="tab:green", s=80, marker="*")
        ax.scatter([m_xyz[-1, 1]], [m_xyz[-1, 0]], [m_xyz[-1, 2]], c="tab:blue", s=80, marker="*")
        if tgt is not None:
            ax.scatter([tgt[1]], [tgt[0]], [tgt[2]], c="red", s=70, marker="X", label="target")

        ax.set_title(f"[{cat}] {instr[:36]}", fontsize=8)
        ax.set_xlabel("Y right (cm)", fontsize=7)
        ax.set_ylabel("X fwd (cm)", fontsize=7)
        ax.set_zlabel("Z up (cm)", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(f"3D: GT (green) vs model (blue), target (red X) — {run}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default=r"C:\Users\user\学习\UAV source code\UAV-Flow-Eval")
    ap.add_argument("--run", default="aeromamba_stage3_v3_binding_ep4_clean_20260718")
    ap.add_argument("--per_class", type=int, default=1, help="tasks per category")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    root = Path(args.eval_root)
    result_dir = root / "results" / args.run
    test_dir = root / "test_jsons"

    picks = pick_tasks(root, result_dir, test_dir, args.per_class)
    if not picks:
        raise SystemExit("no completed tasks found for plotting")
    print(f"[plot] {len(picks)} tasks: {[c for c, _ in picks]}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "reports" / "plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_2d(picks, result_dir, test_dir, args.run, out_dir / f"gt_vs_pred_{args.run}.png")
    plot_3d(picks, result_dir, test_dir, args.run, out_dir / f"gt_vs_pred_3d_{args.run}.png")


if __name__ == "__main__":
    main()
