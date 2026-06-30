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


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


class Stage3Trainer(BaseTrainer):
    """Stage 3: Action Head + ProprioEncoder fine-tuning."""

    def configure_model(self, model: AeroMambaVLA) -> None:
        # Freeze the large frozen stack; train the policy modules plus optional
        # resampler / Mamba-LoRA adapters for UAV-Flow dynamics adaptation.
        model.configure_stage3(train_lora=getattr(self.args, "stage3_train_lora", False))

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
            model_state = model.state_dict()
            filtered = {}
            skipped = []
            for key, value in state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered[key] = value
                elif key in model_state:
                    skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            if skipped:
                missing, unexpected = model.load_state_dict(filtered, strict=False)
                print(f"         Skipped {len(skipped)} tensors with incompatible shapes.")
                for key, old_shape, new_shape in skipped[:12]:
                    print(f"           - {key}: ckpt{old_shape} -> model{new_shape}")
            else:
                missing, unexpected = model.load_state_dict(filtered, strict=False)
            print("         Loaded compatible checkpoint tensors.")
            print(f"         Missing keys (expected action_head/proprio): {len(missing)}")
            print(f"         Unexpected keys (should be empty): {len(unexpected)}")
        else:
            print("[Stage3] No stage2_ckpt provided — training policy layers from scratch")

    def compute_loss(self, model, batch, device):
        pixels    = _to_device(batch["pixel_values"], device)
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
    p.add_argument(
        "--arch_preset",
        default="none",
        choices=["none", "uav_lite_compatible", "uav_lite_siglip"],
        help=(
            "Architecture preset. 'uav_lite_compatible' keeps the existing "
            "DinoSigLIP Stage-2 projector but adds Perceiver resampling and "
            "a dynamics head. 'uav_lite_siglip' switches to a SigLIP-only "
            "lightweight vision stack for a new Stage1/2/3 run."
        ),
    )
    p.add_argument("--dummy",          action="store_true")
    p.add_argument("--data_root",      default="")
    p.add_argument("--hf_dataset",     default="", help="HuggingFace dataset name, e.g. wangxiangyu0814/UAV-Flow")
    p.add_argument("--hf_split",       default="train")
    p.add_argument("--hf_data_files",  default=None, help="Optional local parquet glob for load_dataset('parquet').")
    p.add_argument("--hf_cache_dir",   default=None)
    p.add_argument("--instruction",    default="Navigate the UAV along the planned trajectory.")
    p.add_argument("--mamba_type",     default="mamba-130m")
    p.add_argument("--vision_type",    default="siglip_l_384")
    p.add_argument("--token_resampler", default="none", choices=["none", "perceiver"])
    p.add_argument("--num_visual_queries", type=int, default=32)
    p.add_argument("--resampler_layers", type=int, default=2)
    p.add_argument("--resampler_heads", type=int, default=8)
    p.add_argument("--action_head_type", default="mlp", choices=["mlp", "dynamics"])
    p.add_argument("--action_bound",   type=float, default=1.0)
    p.add_argument("--stage3_train_lora", action="store_true", help="Keep Mamba LoRA adapters trainable in Stage 3.")
    p.add_argument("--chunk_size",     type=int,   default=5)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch",          type=int,   default=4)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--lora_r",         type=int,   default=16, help="LoRA rank (must match Stage 2)")
    p.add_argument("--lora_alpha",     type=int,   default=32, help="LoRA alpha (must match Stage 2)")
    p.add_argument("--lambda_smooth",  type=float, default=0.1)
    p.add_argument("--pos_scale",      type=float, default=100.0)
    p.add_argument("--aug_flip",       action="store_true")
    p.add_argument("--val_frac",       type=float, default=0.1)
    p.add_argument("--save_dir",       default="./checkpoints/stage3")
    p.add_argument("--stage2_ckpt",    default=None, help="Stage-2 SFT model checkpoint")
    p.add_argument("--resume",         default=None, help="Resume from mid-stage checkpoint")
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--max_text_len",   type=int,   default=64)
    p.add_argument("--dummy_size",     type=int,   default=100)
    p.add_argument("--log_every",      type=int,   default=5)
    p.add_argument("--max_steps",      type=int,   default=None, help="Optional max train batches per epoch.")
    p.add_argument("--max_val_steps",  type=int,   default=100, help="Optional max validation batches.")
    p.add_argument("--no_amp",         action="store_true", help="Disable CUDA autocast/GradScaler for numerical stability.")
    p.add_argument("--save_every_steps", type=int, default=None, help="Save latest.pth every N training steps.")
    args = p.parse_args()

    if args.arch_preset == "uav_lite_compatible":
        args.vision_type = "dinosiglip_so_384"
        args.token_resampler = "perceiver"
        args.num_visual_queries = 32
        args.resampler_layers = 2
        args.resampler_heads = 8
        args.action_head_type = "dynamics"
        args.stage3_train_lora = True
        args.no_amp = True
        if args.lr == 5e-4:
            args.lr = 5e-5
    elif args.arch_preset == "uav_lite_siglip":
        args.vision_type = "siglip2_base_384"
        if args.mamba_type == "mamba-130m":
            args.mamba_type = "mamba-2-370m"
        args.token_resampler = "perceiver"
        args.num_visual_queries = 32
        args.resampler_layers = 2
        args.resampler_heads = 8
        args.action_head_type = "dynamics"
        args.stage3_train_lora = True
        args.no_amp = True
        if args.lr == 5e-4:
            args.lr = 5e-5

    return args


if __name__ == "__main__":
    args = get_args()
    Stage3Trainer(args).run()
