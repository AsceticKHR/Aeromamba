"""Plot pred-vs-gt endpoint scatter + per-channel std for the v4 S3 policy
(best_grounded.pth, proprio-free-query) from the dumped 2000-sample endpoints."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("reports/s3_v4_endpoints.npz")
pred, gt = d["pred_end"], d["gt_end"]          # [N,4] dx,dy,dz,dyaw_rad
pred = pred.copy(); gt = gt.copy()
pred[:, 3] = np.rad2deg(pred[:, 3]); gt[:, 3] = np.rad2deg(gt[:, 3])
names = ["dx (forward, m)", "dy (lateral, m)", "dz (vertical, m)", "dyaw (deg)"]
N = pred.shape[0]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 4 scatter panels (pred vs gt) with y=x
for j, ax in enumerate(axes.flat[:4]):
    x, y = gt[:, j], pred[:, j]
    ax.scatter(x, y, s=6, alpha=0.25, color="#2166ac")
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y=x (ideal)")
    # metrics
    mae = np.mean(np.abs(y - x))
    r = np.corrcoef(x, y)[0, 1] if x.std() > 1e-6 and y.std() > 1e-6 else float("nan")
    ratio = y.std() / (x.std() + 1e-9)
    ax.set_title(f"{names[j]}\nMAE={mae:.3f}  r={r:.2f}  std_ratio={ratio:.2f}",
                 fontsize=11)
    ax.set_xlabel("GT endpoint"); ax.set_ylabel("Pred endpoint")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# panel 5: per-channel std bars (pred vs gt) -> shows dz collapse
ax = axes.flat[4]
ch = ["dx", "dy", "dz", "dyaw"]
ps = [pred[:, j].std() for j in range(4)]
gs = [gt[:, j].std() for j in range(4)]
xpos = np.arange(4)
ax.bar(xpos - 0.2, gs, 0.4, label="GT std", color="#b2182b")
ax.bar(xpos + 0.2, ps, 0.4, label="Pred std", color="#2166ac")
ax.set_xticks(xpos); ax.set_xticklabels(ch)
ax.set_title("Per-channel endpoint std (pred vs GT)\ndz pred-std << GT-std = vertical under-prediction")
ax.set_ylabel("std"); ax.legend(); ax.grid(alpha=0.3, axis="y")

# panel 6: xy trajectory endpoints (pred vs gt) top-down
ax = axes.flat[5]
ax.scatter(gt[:, 1], gt[:, 0], s=6, alpha=0.3, color="#b2182b", label="GT")
ax.scatter(pred[:, 1], pred[:, 0], s=6, alpha=0.3, color="#2166ac", label="Pred")
ax.set_xlabel("dy (lateral, m)"); ax.set_ylabel("dx (forward, m)")
ax.set_title("Endpoint top-down (forward vs lateral)")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal", "box")

fig.suptitle(
    f"AeroMamba S3 v4 (best_grounded, proprio-free-query) — pred vs GT endpoints, N={N}\n"
    f"pos strong (dx/dy/yaw track GT), dz collapsed toward 0 (vertical weak)",
    fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "reports/plots/s3_v4_pred_vs_gt_endpoints.png"
fig.savefig(out, dpi=130)
print("SAVED", out)
