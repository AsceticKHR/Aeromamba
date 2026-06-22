"""
Stage 3: Action Head training on UAV trajectory data.

Trains only the UAVActionHead and ProprioEncoder.
The vision encoder, MLPProjector, and Mamba backbone (with Mamba-LoRA loaded)
are all frozen during Stage 3.

Loss: aero_action_loss (Smooth-L1 + smoothness regularisation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.uav_mamba_vla import AeroMambaVLA
from training.trainer    import BaseTrainer


class Stage3Trainer(BaseTrainer):
    """Stage 3: Action Head + ProprioEncoder fine-tuning."""

    def configure_model(self, model: AeroMambaVLA) -> None:
        # Freeze everything except ActionHead and ProprioEncoder
        model.configure_stage3()

    def load_pretrained(self, model: AeroMambaVLA) -> None:
        # Apply LoRA structure first (before loading checkpoint) so model state_dict matches Stage 2 checkpoint
        lora_r     = getattr(self.args, "lora_r",     16)
        lora_alpha = getattr(self.args, "lora_alpha", 32)
        model.apply_lora(r=lora_r, lora_alpha=lora_alpha)

        ckpt_path = getattr(self.args, "stage2_ckpt", None)
        if ckpt_path and Path(ckpt_path).exists():
            print(f"[Stage3] Loading Stage-2 checkpoint from: {ckpt_path}")
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model_state", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"         Loaded checkpoint state_dict.")
            print(f"         Missing keys (expected action_head/proprio): {len(missing)}")
            print(f"         Unexpected keys (should be empty): {len(unexpected)}")
        else:
            print("[Stage3] No stage2_ckpt provided — training policy layers from scratch")

    def compute_loss(self, model, batch, device):
        pixels    = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        proprio   = batch["proprio"].to(device, non_blocking=True)
        gt_action = batch["gt_action"].to(device, non_blocking=True)

        pred = model(
            pixel_values=pixels,
            input_ids=input_ids,
            proprio=proprio,
            gt_action=gt_action,
            return_loss=True,
            lambda_smooth=getattr(self.args, "lambda_smooth", 0.1),
        )
        return pred["loss"], pred.get("loss_detail", {})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="AeroMamba Stage 3: Action Head Training")
    p.add_argument("--dummy",          action="store_true")
    p.add_argument("--data_root",      default="")
    p.add_argument("--mamba_type",     default="mamba-130m")
    p.add_argument("--vision_type",    default="siglip_l_384")
    p.add_argument("--chunk_size",     type=int,   default=5)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch",          type=int,   default=4)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--lora_r",         type=int,   default=16, help="LoRA rank (must match Stage 2)")
    p.add_argument("--lora_alpha",     type=int,   default=32, help="LoRA alpha (must match Stage 2)")
    p.add_argument("--lambda_smooth",  type=float, default=0.1)
    p.add_argument("--val_frac",       type=float, default=0.1)
    p.add_argument("--save_dir",       default="./checkpoints/stage3")
    p.add_argument("--stage2_ckpt",    default=None, help="Stage-2 SFT model checkpoint")
    p.add_argument("--resume",         default=None, help="Resume from mid-stage checkpoint")
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--max_text_len",   type=int,   default=64)
    p.add_argument("--dummy_size",     type=int,   default=100)
    p.add_argument("--log_every",      type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    Stage3Trainer(args).run()
