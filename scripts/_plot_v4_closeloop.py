"""Overlay GT (reference_path_preprocessed) vs model executed trajectory for the
v4 proprio-free-query closed-loop smoke (2 tasks). Top-down XY path + yaw-vs-progress."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
EVAL = r"C:\Users\user\学习\UAV source code\UAV-Flow-Eval"
RESULT = EVAL + r"\results\aerov2_v4_diverse"
GT = EVAL + r"\test_jsons"
OUT = r"C:\Users\user\学习\UAV source code\Aeromamba\reports\plots\s3_v4_diverse_gt_vs_pred.png"
# 6 target tasks across categories (Move/Approach/Land/Retreat/Surround/Turn)
TASKS = ["2025-03-30_11-49-28", "2025-05-12_21-45-40", "2025-05-07_12-01-53",
         "2025-05-06_22-25-29", "2025-03-30_12-44-03", "2025-03-30_12-52-50"]


def load_model(base):
    m = json.load(open(f"{RESULT}\\{base}.json", encoding="utf-8"))
    xy = np.array([[s["state"][0][0], s["state"][0][1]] for s in m])   # cm
    yaw = np.array([s["state"][1][1] for s in m])                      # deg
    return xy, yaw


def w2l(target, init):
    import math
    x0, y0, z0, yaw0 = init[:4]
    dx, dy, dz = target[0] - x0, target[1] - y0, target[2] - z0
    th = -math.radians(yaw0)
    return np.array([dx * math.cos(th) - dy * math.sin(th),
                     dx * math.sin(th) + dy * math.cos(th)])


def load_gt(base):
    g = json.load(open(f"{GT}\\{base}.json", encoding="utf-8"))
    p = np.array(g["reference_path_preprocessed"], dtype=float)
    tgt = None
    if g.get("target_pos") and g.get("initial_pos"):
        tgt = w2l(g["target_pos"], g["initial_pos"])
    return p[:, :2], g.get("instruction", ""), tgt


fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for ax, base in zip(axes.flat, TASKS):
    mxy, _ = load_model(base)
    gxy, instr, tgt = load_gt(base)
    ax.plot(gxy[:, 0], gxy[:, 1], "-o", color="#b2182b", ms=3, lw=2, label="GT path")
    ax.plot(mxy[:, 0], mxy[:, 1], "-", color="#2166ac", lw=1.6, label="Pred path")
    ax.scatter([0], [0], c="green", s=110, marker="*", zorder=5, label="start")
    ax.scatter([gxy[-1, 0]], [gxy[-1, 1]], c="red", s=60, marker="X", zorder=5, label="GT end")
    ax.scatter([mxy[-1, 0]], [mxy[-1, 1]], c="blue", s=60, marker="X", zorder=5, label="Pred end")
    if tgt is not None:
        ax.scatter([tgt[0]], [tgt[1]], c="orange", s=130, marker="P", zorder=6,
                   edgecolors="k", label="target")
    ax.set_xlabel("x fwd (cm)"); ax.set_ylabel("y lateral (cm)")
    ax.set_title(f"{instr[:44]}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_aspect("equal", "datalim")

fig.suptitle("AeroMamba S3 v4 closed-loop (diverse) — GT vs Pred vs target: forward-creep / overshoot diagnosis",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print("SAVED", OUT)
