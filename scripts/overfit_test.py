"""
AeroMamba Overfit Test (Stage 3)

Verifies that the ActionHead + ProprioEncoder can overfit a tiny fixed dataset
of 32 samples. A consistently decreasing loss (reaching < 0.05 in 30 epochs)
confirms correct gradient flow and sufficient model capacity.

Usage:
    python scripts/overfit_test.py [--mamba_type mamba-130m] [--epochs 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import DummyUAVDataset, aero_collate_fn
from model.action_head import aero_action_loss
from model.uav_mamba_vla import AeroMambaVLA


def run_overfit_test(
    mamba_type:  str   = "mamba-130m",
    vision_type: str   = "siglip_so_384",
    n_samples:   int   = 32,
    batch_size:  int   = 16,
    epochs:      int   = 30,
    lr:          float = 1e-3,
) -> float:
    """
    Overfit Stage-3 (ActionHead + ProprioEncoder) on a tiny fixed dataset.

    Returns the final average training loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 60}")
    print(f"  AeroMamba Overfit Test")
    print(f"  Device      : {device}")
    print(f"  Mamba       : {mamba_type}")
    print(f"  Vision      : {vision_type}  (single encoder — faster)")
    print(f"  Samples     : {n_samples}  (fixed, non-shuffled for true overfit)")
    print(f"  Batch size  : {batch_size}")
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}")
    print(f"{'═' * 60}\n")

    # ── Fixed tiny dataset (dual_vision=False → single SigLIP, faster) ────────
    ds = DummyUAVDataset(
        size=n_samples,
        chunk_size=5,
        dual_vision=False,   # single encoder to avoid downloading DINOv2
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,   # no shuffle — exact same samples every epoch
        collate_fn=aero_collate_fn,
        num_workers=0,
    )

    # ── Model: Stage-3 configuration (only ActionHead + ProprioEncoder train) ──
    model = AeroMambaVLA(
        mamba_type=mamba_type,
        vision_type=vision_type,
        chunk_size=5,
    ).to(device)
    model.configure_stage3()
    model.print_trainable_params()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,   # no regularisation — we WANT to overfit
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'main':>10}  {'smooth':>10}")
    print(f"{'─' * 60}")

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_main   = 0.0
        epoch_smooth = 0.0
        n_batches    = 0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            pixels    = batch["pixel_values"].to(device)
            ids       = batch["input_ids"].to(device)
            proprio   = batch["proprio"].to(device)
            gt_action = batch["gt_action"].to(device)

            out  = model(pixels, ids, proprio, gt_action, return_loss=True)
            loss = out["loss"]
            detail = out.get("loss_detail", {})

            loss.backward()
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_main   += detail.get("main",   0.0)
            epoch_smooth += detail.get("smooth", 0.0)
            n_batches    += 1

        avg_loss   = epoch_loss   / n_batches
        avg_main   = epoch_main   / n_batches
        avg_smooth = epoch_smooth / n_batches
        history.append(avg_loss)

        # Print every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {avg_loss:>12.5f}  {avg_main:>10.5f}  {avg_smooth:>10.5f}")

    # ── Verdict ────────────────────────────────────────────────────────────────
    final_loss = history[-1]
    first_loss = history[0]
    reduction  = (first_loss - final_loss) / first_loss * 100.0

    print(f"\n{'─' * 60}")
    print(f"  Initial loss : {first_loss:.5f}")
    print(f"  Final loss   : {final_loss:.5f}")
    print(f"  Reduction    : {reduction:.1f}%")
    print(f"{'─' * 60}")

    if final_loss < 0.01:
        verdict = "✅ EXCELLENT — Model overfit fully (loss < 0.01)"
    elif final_loss < 0.05:
        verdict = "✅ GOOD — Model is converging correctly (loss < 0.05)"
    elif reduction > 50:
        verdict = "⚠️  PARTIAL — Loss is dropping but slowly; try more epochs or higher lr"
    else:
        verdict = "❌ POOR — Loss not decreasing; check gradient flow or model configuration"

    print(f"\n  {verdict}")
    print(f"{'═' * 60}\n")

    return final_loss


def parse_args():
    p = argparse.ArgumentParser(description="AeroMamba Stage-3 Overfit Test")
    p.add_argument("--mamba_type",  default="mamba-130m",
                   help="Mamba variant (default: mamba-130m for speed)")
    p.add_argument("--vision_type", default="siglip_so_384",
                   help="Vision encoder (default: siglip_so_384)")
    p.add_argument("--n_samples",   type=int,   default=32)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_overfit_test(
        mamba_type=args.mamba_type,
        vision_type=args.vision_type,
        n_samples=args.n_samples,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )
