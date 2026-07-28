"""Per-task GT vs model trajectory comparison, in the official UAV-Flow-Eval style.

Follows the plotting conventions of UAV-Flow-Eval/batch_run_act_all.py
(draw_2d/3d_trajectory_from_log): x-axis = Y (right), y-axis = X (forward),
yaw shown as quiver arrows, target as red marker, equal aspect.

For every completed task in a result dir, writes one PNG that overlays:
  - GT reference_path_preprocessed (green, dark-green yaw arrows)
  - model closed-loop trajectory   (blue, orange yaw arrows)
with start / endpoints / target annotated, as a 2D top-down + 3D pair.

Usage:
  python scripts/plot_gt_vs_pred_per_task.py \
      --run aeromamba_stage3_v3_binding_ep4_clean_20260718
Output: <Aeromamba>/reports/plots/per_task_<run>/<base>_gt_vs_pred.png
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
    if not isinstance(log, list) or len(log) == 0:
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


def yaw_arrows(xyz, yaw_deg, arrow_len, step):
    """Subsampled yaw quiver components (official style: dx=cos, dy=sin)."""
    idx = np.arange(0, len(xyz), step)
    yaw = np.deg2rad(yaw_deg[idx])
    return xyz[idx], np.cos(yaw) * arrow_len, np.sin(yaw) * arrow_len


def plot_one(base, m_xyz, m_yaw, g_xyz, g_yaw, instr, tgt, out_path: Path):
    fig = plt.figure(figsize=(13, 6))

    # arrow length scaled to combined spread, per official heuristic
    allxy = np.vstack([m_xyz[:, :2], g_xyz[:, :2]])
    arrow_len = max(10.0, float(np.sqrt(allxy[:, 0].var() + allxy[:, 1].var())) / 4.0)
    g_step = max(1, len(g_xyz) // 20)
    m_step = max(1, len(m_xyz) // 20)

    # ---- 2D top-down (official axis mapping: horizontal Y-right, vertical X-fwd)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(g_xyz[:, 1], g_xyz[:, 0], color="tab:green", lw=2.2, label="GT trajectory", zorder=2)
    gp, gdx, gdy = yaw_arrows(g_xyz, g_yaw, arrow_len, g_step)
    ax.quiver(gp[:, 1], gp[:, 0], gdy, gdx, angles="xy", scale_units="xy", scale=1,
              color="darkgreen", width=0.003, alpha=0.55, label="GT yaw")
    ax.plot(m_xyz[:, 1], m_xyz[:, 0], color="tab:blue", lw=1.8, label="model trajectory", zorder=3)
    mp, mdx, mdy = yaw_arrows(m_xyz, m_yaw, arrow_len, m_step)
    ax.quiver(mp[:, 1], mp[:, 0], mdy, mdx, angles="xy", scale_units="xy", scale=1,
              color="darkorange", width=0.003, alpha=0.7, label="model yaw")
    ax.scatter([g_xyz[0, 1]], [g_xyz[0, 0]], c="black", s=50, marker="o", zorder=5, label="start")
    ax.scatter([g_xyz[-1, 1]], [g_xyz[-1, 0]], c="tab:green", s=90, marker="*", zorder=5)
    ax.scatter([m_xyz[-1, 1]], [m_xyz[-1, 0]], c="tab:blue", s=90, marker="*", zorder=5)
    if tgt is not None:
        ax.scatter([tgt[1]], [tgt[0]], c="red", s=70, marker="X", zorder=6, label="target")
    end_xy = float(np.linalg.norm(m_xyz[-1, :2] - g_xyz[-1, :2]))
    ax.text(0.02, 0.02, f"end_xy Δ={end_xy:.0f}cm", transform=ax.transAxes,
            fontsize=8, va="bottom", color="dimgray")
    ax.set_xlabel("Y (right) [cm]")
    ax.set_ylabel("X (forward) [cm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc="best")

    # ---- 3D (official axis mapping: x=Y-right, y=X-fwd, z=Z-up)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.plot(g_xyz[:, 1], g_xyz[:, 0], g_xyz[:, 2], color="tab:green", lw=2.2, label="GT")
    ax3.plot(m_xyz[:, 1], m_xyz[:, 0], m_xyz[:, 2], color="tab:blue", lw=1.8, label="model")
    ax3.quiver(gp[:, 1], gp[:, 0], gp[:, 2], gdy, gdx, np.zeros_like(gdx),
               color="darkgreen", linewidth=0.5, alpha=0.5)
    ax3.quiver(mp[:, 1], mp[:, 0], mp[:, 2], mdy, mdx, np.zeros_like(mdx),
               color="darkorange", linewidth=0.5, alpha=0.6)
    ax3.scatter([g_xyz[0, 1]], [g_xyz[0, 0]], [g_xyz[0, 2]], c="black", s=50, marker="o")
    ax3.text(g_xyz[0, 1], g_xyz[0, 0], g_xyz[0, 2], "Start", color="black", fontsize=9)
    ax3.scatter([g_xyz[-1, 1]], [g_xyz[-1, 0]], [g_xyz[-1, 2]], c="tab:green", s=90, marker="*")
    ax3.scatter([m_xyz[-1, 1]], [m_xyz[-1, 0]], [m_xyz[-1, 2]], c="tab:blue", s=90, marker="*")
    if tgt is not None:
        ax3.scatter([tgt[1]], [tgt[0]], [tgt[2]], c="red", s=70, marker="X")
        ax3.text(tgt[1], tgt[0], tgt[2], "Target", color="red", fontsize=9)
    ax3.set_xlabel("Y (right) [cm]", fontsize=8)
    ax3.set_ylabel("X (forward) [cm]", fontsize=8)
    ax3.set_zlabel("Z (up) [cm]", fontsize=8)
    ax3.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"{instr}\n{base}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default=r"C:\Users\user\学习\UAV source code\UAV-Flow-Eval")
    ap.add_argument("--run", default="aeromamba_stage3_v3_binding_ep4_clean_20260718")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--limit", type=int, default=0, help="only plot first N tasks (0 = all)")
    args = ap.parse_args()

    root = Path(args.eval_root)
    result_dir = root / "results" / args.run
    test_dir = root / "test_jsons"
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "reports" / "plots" / f"per_task_{args.run}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = sorted(p.stem for p in result_dir.glob("*.json"))
    if args.limit:
        bases = bases[: args.limit]

    n_ok, n_skip = 0, 0
    for base in bases:
        gp = test_dir / f"{base}.json"
        if not gp.exists():
            n_skip += 1
            continue
        try:
            m_xyz, m_yaw = load_model(result_dir / f"{base}.json")
            if m_xyz is None or len(m_xyz) < 2:
                n_skip += 1
                continue
            g_xyz, g_yaw, instr, tgt = load_gt(gp)
            plot_one(base, m_xyz, m_yaw, g_xyz, g_yaw, instr, tgt,
                     out_dir / f"{base}_gt_vs_pred.png")
            n_ok += 1
        except Exception as e:  # keep going on malformed tasks
            print(f"[skip] {base}: {e}")
            n_skip += 1

    print(f"[done] wrote {n_ok} figures to {out_dir} (skipped {n_skip})")


if __name__ == "__main__":
    main()
