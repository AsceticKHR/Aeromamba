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
from training.arch_presets import add_arch_preset_arg, apply_arch_preset


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


class Stage3Trainer(BaseTrainer):
    """Stage 3: Action Head + ProprioEncoder fine-tuning."""

    def configure_model(self, model: AeroMambaVLA) -> None:
        # Attach the training-only instruction binding head BEFORE freezing
        # decisions so configure_stage3 can mark it trainable (AeroStream A4).
        if getattr(self.args, "lambda_binding", 0.0) > 0.0:
            model.enable_binding_head()
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

        self._load_action_stats(model)

    def _load_action_stats(self, model: AeroMambaVLA) -> None:
        """Load per-(k, dim) z-score stats into the action head buffers."""
        stats_path = getattr(self.args, "action_stats", None)
        if not stats_path:
            print("[Stage3] WARNING: no --action_stats given — loss runs in raw "
                  "physical space (forward-motion dominated). Strongly consider "
                  "running data/compute_action_stats.py first.")
            return
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"--action_stats file not found: {stats_path}")
        import json
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        if int(stats["chunk_size"]) != int(getattr(self.args, "chunk_size", 5)):
            raise ValueError(
                f"action_stats chunk_size={stats['chunk_size']} does not match "
                f"--chunk_size={getattr(self.args, 'chunk_size', 5)}"
            )
        if float(stats.get("pos_scale", 100.0)) != float(getattr(self.args, "pos_scale", 100.0)):
            raise ValueError(
                f"action_stats pos_scale={stats.get('pos_scale')} does not match "
                f"--pos_scale={getattr(self.args, 'pos_scale', 100.0)}"
            )
        model.action_head.set_normalization(stats["mean"], stats["std"])
        std = torch.as_tensor(stats["std"])
        print(f"[Stage3] Loaded action stats from {stats_path} "
              f"(samples={stats.get('num_samples', '?')}, "
              f"turn_fraction={stats.get('turn_fraction', 0):.3f})")
        print(f"         std range: min={std.min():.5f} max={std.max():.5f} "
              f"(floored at 1e-3 in head)")

    def _channel_weights(self, device) -> torch.Tensor | None:
        wz = float(getattr(self.args, "channel_weight_z", 1.0))
        wyaw = float(getattr(self.args, "channel_weight_yaw", 1.0))
        if wz == 1.0 and wyaw == 1.0:
            return None
        return torch.tensor([1.0, 1.0, wz, wyaw], device=device)

    def compute_loss(self, model, batch, device):
        pixels    = _to_device(batch["pixel_values"], device)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        proprio   = batch["proprio"].to(device, non_blocking=True)
        state     = batch.get("state8", proprio).to(device, non_blocking=True)
        delta_state = batch.get("delta_state8")
        if delta_state is not None:
            delta_state = delta_state.to(device, non_blocking=True)
        gt_action = batch["gt_action"].to(device, non_blocking=True)

        binding_labels = batch.get("binding_labels")
        sample_weights = None
        if binding_labels is not None:
            binding_labels = {
                k: v.to(device, non_blocking=True) for k, v in binding_labels.items()
            }
            if int(getattr(self.args, "magnitude_sample_weight", 0)):
                # 1 + log1p(endpoint displacement in metres): big-motion
                # samples get proportionally more main-term gradient.
                sample_weights = 1.0 + torch.log1p(
                    binding_labels["endpoint_norm_m"].float().clamp_min(0.0)
                )

        pred = model(
            pixel_values=pixels,
            input_ids=input_ids,
            proprio=proprio,
            state=state,
            delta_state=delta_state,
            gt_action=gt_action,
            return_loss=True,
            lambda_smooth=getattr(self.args, "lambda_smooth", 0.0),
            lambda_endpoint=getattr(self.args, "lambda_endpoint", 0.25),
            lambda_direction=getattr(self.args, "lambda_direction", 0.5),
            lambda_acc=getattr(self.args, "lambda_acc", 0.0),
            binding_labels=binding_labels,
            lambda_binding=getattr(self.args, "lambda_binding", 0.0),
            channel_weights=self._channel_weights(device),
            sample_weights=sample_weights,
        )
        return pred["loss"], pred.get("loss_detail", {})


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="AeroMamba Stage 3: Action Head Training")
    add_arch_preset_arg(p)
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
    p.add_argument("--proprio_dim",    type=int,   default=8)
    p.add_argument("--chunk_size",     type=int,   default=5)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch",          type=int,   default=4)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--lora_r",         type=int,   default=16, help="LoRA rank (must match Stage 2)")
    p.add_argument("--lora_alpha",     type=int,   default=32, help="LoRA alpha (must match Stage 2)")
    p.add_argument("--lambda_smooth",  type=float, default=0.0)
    p.add_argument("--lambda_endpoint", type=float, default=0.25)
    p.add_argument("--lambda_direction", type=float, default=0.5)
    p.add_argument("--lambda_acc",     type=float, default=0.0)
    p.add_argument("--pos_scale",      type=float, default=100.0)
    p.add_argument("--action_stats",   default=None,
                   help="JSON from data/compute_action_stats.py — per-(k,dim) "
                        "z-score stats for the loss / action head buffers.")
    p.add_argument("--oversample_turn_factor", type=int, default=1,
                   help="Replicate turn-heavy chunks N x in the training index.")
    p.add_argument("--oversample_turn_deg", type=float, default=10.0)
    p.add_argument("--oversample_class_factor", type=int, default=1,
                   help="Replicate all windows of Move/Shift/Ascend/Descend/"
                        "Surround/Rotate trajectories N x (max with turn "
                        "oversampling, not multiplied).")
    p.add_argument("--lambda_binding", type=float, default=0.0,
                   help="Weight of the instruction-binding CE loss (4 tasks "
                        "averaged); 0 disables the binding head entirely.")
    p.add_argument("--channel_weight_z", type=float, default=1.0,
                   help="z-space loss weight for the dz channel (main+endpoint).")
    p.add_argument("--channel_weight_yaw", type=float, default=1.0,
                   help="z-space loss weight for the dyaw channel (main+endpoint).")
    p.add_argument("--magnitude_sample_weight", type=int, default=0, choices=[0, 1],
                   help="Weight main loss per sample by 1+log1p(endpoint "
                        "displacement m) to fight small-motion collapse.")
    p.add_argument("--aug_flip",       action="store_true")
    p.add_argument("--aug_vision",     action="store_true",
                   help="Appearance augmentation (color jitter/grayscale/blur) "
                        "to bridge the real-photo -> Unreal domain gap.")
    p.add_argument("--val_frac",       type=float, default=0.1)
    p.add_argument("--save_dir",       default="./checkpoints/stage3")
    p.add_argument("--stage2_ckpt",    default=None, help="Stage-2 SFT model checkpoint")
    p.add_argument("--resume",         default=None, help="Resume from mid-stage checkpoint")
    p.add_argument("--resume_model_only", action="store_true",
                   help="Load model weights only (skip optimizer); use when trainable params change (e.g. Stage 3b).")
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--max_text_len",   type=int,   default=64)
    p.add_argument("--dummy_size",     type=int,   default=100)
    p.add_argument("--log_every",      type=int,   default=5)
    p.add_argument("--max_steps",      type=int,   default=None, help="Optional max train batches per epoch.")
    p.add_argument("--split_seed",     type=int,   default=42)
    p.add_argument("--max_val_steps",  type=int,   default=100, help="Optional max validation batches.")
    p.add_argument("--no_amp",         action="store_true", help="Disable CUDA autocast/GradScaler for numerical stability.")
    p.add_argument("--save_every_steps", type=int, default=None, help="Save latest.pth every N training steps.")
    args = p.parse_args()
    return apply_arch_preset(args, "stage3")


if __name__ == "__main__":
    args = get_args()
    Stage3Trainer(args).run()
