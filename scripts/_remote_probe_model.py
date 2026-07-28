"""Model-level probe: is the S3 action head conditioning on instruction/vision,
or shortcutting through the proprio velocity channel?

Feeds crafted state8 (proprio) and varied instructions directly to
forward_action, comparing the RAW z-space action chunk.
"""
import sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path("/root/AeroMamba") if Path("/root/AeroMamba").exists() else Path("/root/autodl-tmp/AeroMamba")
# fall back: find repo containing model/aerov2.py
for c in [ROOT, Path("/root/autodl-tmp/aeromamba"), Path.cwd()]:
    if (c / "model" / "aerov2.py").exists():
        ROOT = c
        break
sys.path.insert(0, str(ROOT))

from model.aerov2 import AeroV2
from training.v2_stage3_action import load_s2, load_action_stats
from scripts.eval_s3_systematic import load_s3

S2_DIR = "checkpoints/v2_stage2_full_cradio"
S3 = "checkpoints/v2_stage3_cradio/best.pth"
STATS = "/root/autodl-tmp/datasets/uav-flow/action_stats_k8.json"
IMG = "/root/autodl-tmp/datasets/stage3_uavflow/2025-04-02_15-17-46/000000.jpg"

dev = torch.device("cuda")
m = AeroV2(backbone_id="tiiuae/Falcon-H1-1.5B-Deep-Instruct", vision_type="cradio_v3_b").to(dev)
load_s2(m, Path(ROOT / S2_DIR) if (ROOT / S2_DIR).exists() else Path(S2_DIR), "best")
m.enable_action_head(chunk_size=8, proprio_dim=8)
load_action_stats(m, STATS, 8, 100.0)
load_s3(m, Path(ROOT / S3) if (ROOT / S3).exists() else Path(S3))
m.eval()

img = Image.open(IMG).convert("RGB")
pv = m.vision_encoder.transform(img).unsqueeze(0).to(dev)


def tok(instr):
    t = m.tokenizer(instr, max_length=64, padding="max_length", truncation=True,
                    return_tensors="pt")
    return t["input_ids"].to(dev)


@torch.no_grad()
def run(instr, state8):
    ids = tok(instr)
    p = torch.tensor([state8], dtype=torch.float32, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m.forward_action(pv, ids, p)
    z = out["action"][0].float().cpu()          # [K,4] z-space
    return z


def summ(z):
    # endpoint z-values per channel + overall std
    e = z[-1].tolist()
    return f"end_z=({e[0]:+.3f},{e[1]:+.3f},{e[2]:+.3f},{e[3]:+.3f}) std={z.std():.4f}"


Z8 = [0.0] * 8
print("=== A) vary INSTRUCTION, proprio=ZERO ===")
for instr in ["Fly straight forward.", "Turn left.", "Turn right.",
              "Climb up to a higher altitude.", "Descend and fly downward.",
              "Stop and hold position."]:
    print(f"  [{instr:38s}] {summ(run(instr, Z8))}")

print("=== B) fixed instr='Turn left.', vary VELOCITY channel (state8[4:8]) ===")
base = "Turn left."
for name, v in [("vel=0", [0, 0, 0, 0]),
                ("yaw_rate+0.5", [0, 0, 0, 0.5]),
                ("yaw_rate-0.5", [0, 0, 0, -0.5]),
                ("vx+0.5", [0.5, 0, 0, 0]),
                ("vz+0.5", [0, 0, 0.5, 0])]:
    s = [0, 0, 0, 0] + v
    print(f"  [{name:14s}] {summ(run(base, s))}")

print("=== C) fixed instr, vary POSE channel (state8[0:4]) ===")
for name, p in [("pose=0", [0, 0, 0, 0]),
                ("x+5", [5, 0, 0, 0]),
                ("yaw+1.57", [0, 0, 0, 1.57])]:
    s = p + [0, 0, 0, 0]
    print(f"  [{name:14s}] {summ(run(base, s))}")
