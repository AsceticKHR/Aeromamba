"""
Stage 1: MLP Projector alignment via Causal Language Modeling (CLM) loss.

Trains only the MLPProjector to align visual patch features with Mamba's
word embedding space. Both the vision encoder and Mamba backbone are frozen
throughout Stage 1.

Follows Cobra and LLaVA pre-training alignment methodology:
  Inputs:  [vis_tokens (N_vis) | text_tokens (L)]
  Outputs: Autoregressive Next-Token Prediction over the text tokens only.
  Loss:    Cross-Entropy Loss (masked for prompt and padding).
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.uav_mamba_vla import AeroMambaVLA
from training.trainer    import BaseTrainer
from training.arch_presets import add_arch_preset_arg, apply_arch_preset


class Stage1Trainer(BaseTrainer):
    """Stage 1: Autoregressive Causal LM projector alignment."""

    def configure_model(self, model: AeroMambaVLA) -> None:
        # Freeze everything except projector
        model.configure_stage1()

    def get_collate_fn(self):
        """Use LLaVA-specific collate for Stage 1 data."""
        from data.llava_dataset import llava_collate_fn
        return llava_collate_fn

    def get_dataset(self, model: AeroMambaVLA):
        """Build LLaVADataset for alignment training."""
        args = self.args
        from data.llava_dataset import LLaVADataset
        
        ds = LLaVADataset(
            data_root=args.data_root,
            tokenizer=model.tokenizer,
            transform=model.vision_encoder.transform,
            max_text_len=getattr(args, "max_text_len", 128),
            json_name=getattr(args, "json_name", "llava_subset.json"),
        )
        
        val_frac = getattr(args, "val_frac", 0.1)
        n_val    = max(1, int(len(ds) * val_frac))
        n_train  = len(ds) - n_val
        split_gen = torch.Generator().manual_seed(getattr(args, "split_seed", 42))
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=split_gen)
        print(f"[Trainer] Dataset: {n_train} train  |  {n_val} val")
        return train_ds, val_ds

    def compute_loss(self, model, batch, device):
        pixels    = batch["pixel_values"]
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        if isinstance(pixels, dict):
            pixels = {k: v.to(device, non_blocking=True) for k, v in pixels.items()}
        else:
            pixels = pixels.to(device, non_blocking=True)

        # ── 1. Vision path: vision → projector ────────────────────────────────
        with torch.no_grad():
            vis_patches = model._encode_vision(pixels)     # [B, N_vis, D_v]
        vis_tokens  = model.projector(vis_patches)          # [B, N_vis, D_m] (requires grad)
        vis_tokens  = model.token_resampler(vis_tokens)

        # ── 2. Text path: Mamba embed ─────────────────────────────────────────
        with torch.no_grad():
            text_embs = model._embed_text(input_ids)       # [B, L, D_m] (frozen embeddings)

        # ── 3. Concatenate: [vision | text] ───────────────────────────────────
        zero_state = torch.zeros(
            input_ids.size(0),
            model.proprio_encoder.proprio_dim,
            device=device,
            dtype=text_embs.dtype,
        )
        state_tokens = model.proprio_encoder.forward_pair(zero_state, zero_state)

        # AeroMamba-Opt order: [state | delta_state | vision | text]
        inputs_embeds = torch.cat([state_tokens, vis_tokens, text_embs], dim=1)

        # ── 4. Mamba backbone pass ────────────────────────────────────────────
        # Activations inside Mamba blocks are stored for backpropagation to projector
        hidden = model._run_mamba(inputs_embeds)           # [B, N_vis + L, D_m]

        # ── 5. Projection to vocabulary ───────────────────────────────────────
        logits = model.mamba.lm_head(hidden)               # [B, N_vis + L, Vocab]

        # Debug NaN detection
        has_nan = False
        for name, val in [("vis_patches", vis_patches), ("vis_tokens", vis_tokens), ("text_embs", text_embs), ("hidden", hidden), ("logits", logits)]:
            if torch.isnan(val).any():
                print(f"[DEBUG NaN] {name} contains NaNs! Mean={val.mean().item()} Std={val.std().item()}")
                has_nan = True
        if has_nan:
            print(f"[DEBUG NaN] input_ids: {input_ids.cpu().tolist()}")
            print(f"[DEBUG NaN] labels: {labels.cpu().tolist()}")

        # ── 6. Shift logits and labels for next-token prediction ─────────────
        # Logits at index i predict token at index i+1
        # Target labels are corresponding to the text segment which starts at index N_vis
        prefix_len = state_tokens.size(1) + vis_tokens.size(1)
        shift_logits = logits[:, prefix_len - 1 : -1, :].contiguous()  # [B, L, Vocab]
        shift_labels = labels.contiguous()                        # [B, L]

        # CrossEntropyLoss (will ignore -100 labels)
        if (shift_labels != -100).any():
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        else:
            loss = shift_logits.sum() * 0.0

        # Calculate Perplexity (PPL)
        with torch.no_grad():
            try:
                ppl = torch.exp(loss)
            except OverflowError:
                ppl = torch.tensor(float("inf"))

        return loss, {"clm_loss": loss.item(), "ppl": ppl.item()}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="AeroMamba Stage 1: CLM Projector Alignment")
    add_arch_preset_arg(p)
    p.add_argument("--dummy",         action="store_true",     help="Keep for compatibility, not active")
    p.add_argument("--data_root",     default="./data/llava_subset", help="Path to LLaVA subset folder")
    p.add_argument("--json_name",     default="llava_subset.json")
    p.add_argument("--mamba_type",    default="mamba-2-370m")
    p.add_argument("--vision_type",   default="siglip2_base_384")
    p.add_argument("--token_resampler", default="perceiver", choices=["none", "perceiver"])
    p.add_argument("--num_visual_queries", type=int, default=64)
    p.add_argument("--resampler_layers", type=int, default=2)
    p.add_argument("--resampler_heads", type=int, default=8)
    p.add_argument("--proprio_dim",   type=int,   default=8)
    p.add_argument("--chunk_size",    type=int,   default=5)
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--save_dir",      default="./checkpoints/stage1")
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--max_text_len",  type=int,   default=128)
    p.add_argument("--log_every",     type=int,   default=5)
    p.add_argument("--max_steps",     type=int,   default=None)
    p.add_argument("--split_seed",    type=int,   default=42)
    p.add_argument("--max_val_steps", type=int,   default=100)
    p.add_argument("--no_amp",        action="store_true")
    p.add_argument("--save_every_steps", type=int, default=None)
    args = p.parse_args()
    return apply_arch_preset(args, "stage1")


if __name__ == "__main__":
    args = get_args()
    Stage1Trainer(args).run()
