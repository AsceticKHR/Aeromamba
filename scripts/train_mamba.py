"""
Quick-start training launcher for AeroMamba-VLA.

Wraps Stage-3 action training for convenience; can chain all 3 stages.

Usage:
    python scripts/train_mamba.py                     # stage3 dummy data
    python scripts/train_mamba.py --stage all         # stage1→2→3
    python scripts/train_mamba.py --data_root /path   # real UAV-Flow data
    python scripts/train_mamba.py --stage 3 --epochs 30 --batch 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_args():
    p = argparse.ArgumentParser(description="AeroMamba-VLA Training Launcher")

    # Stage selection
    p.add_argument(
        "--stage", default="3",
        choices=["1", "2", "3", "all"],
        help="Which training stages to run (1/2/3/all)"
    )

    # Shared model & data params
    p.add_argument("--mamba_type",  default="mamba-370m")
    p.add_argument("--vision_type", default="siglip_l_384")
    p.add_argument("--data_root",   default="",
                   help="UAV-Flow data root. Leave empty to use dummy dataset.")
    p.add_argument("--chunk_size",  type=int, default=5)
    p.add_argument("--batch",       type=int, default=8)
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--max_text_len",type=int, default=64)

    # Per-stage epoch overrides
    p.add_argument("--epochs_s1",   type=int, default=3,  help="Stage 1 epochs")
    p.add_argument("--epochs_s2",   type=int, default=5,  help="Stage 2 epochs")
    p.add_argument("--epochs",      type=int, default=20, help="Stage 3 epochs")

    # LR
    p.add_argument("--lr_s1",       type=float, default=1e-3)
    p.add_argument("--lr_s2",       type=float, default=2e-4)
    p.add_argument("--lr",          type=float, default=5e-4, help="Stage 3 LR")

    # LoRA
    p.add_argument("--lora_r",      type=int, default=16)
    p.add_argument("--lora_alpha",  type=int, default=32)

    # Checkpoint paths
    p.add_argument("--save_dir",    default="./checkpoints")
    p.add_argument("--resume",      default=None)

    # Misc
    p.add_argument("--dummy",       action="store_true",
                   help="Use synthetic dummy data for all stages")
    p.add_argument("--dummy_size",  type=int, default=2000)
    p.add_argument("--log_every",   type=int, default=20)

    return p.parse_args()


def run_stage(script: str, extra_args: list[str]):
    """Launch a training stage as a subprocess."""
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'━'*64}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'━'*64}")
    result = subprocess.run(cmd, check=True)
    return result


def build_common_args(args) -> list[str]:
    """Arguments shared across all stages."""
    common = [
        "--mamba_type",  args.mamba_type,
        "--vision_type", args.vision_type,
        "--chunk_size",  str(args.chunk_size),
        "--batch",       str(args.batch),
        "--workers",     str(args.workers),
        "--max_text_len",str(args.max_text_len),
        "--dummy_size",  str(args.dummy_size),
        "--log_every",   str(args.log_every),
    ]
    if args.data_root:
        common += ["--data_root", args.data_root]
    if args.dummy:
        common += ["--dummy"]
    return common


def main():
    args    = get_args()
    training_dir = ROOT / "training"

    stages_to_run = {
        "1": [1], "2": [2], "3": [3], "all": [1, 2, 3]
    }[args.stage]

    common = build_common_args(args)

    s1_ckpt = Path(args.save_dir) / "stage1" / "best.pth"
    s2_ckpt = Path(args.save_dir) / "stage2" / "best.pth"

    print(f"\n{'='*64}")
    print(f"  AeroMamba-VLA Training Launcher")
    print(f"  Stages     : {stages_to_run}")
    print(f"  Mamba      : {args.mamba_type}")
    print(f"  Vision     : {args.vision_type}")
    print(f"  Data       : {args.data_root or '(dummy)'}")
    print(f"{'='*64}")

    if 1 in stages_to_run:
        run_stage(
            str(training_dir / "stage1_align.py"),
            common + [
                "--epochs",   str(args.epochs_s1),
                "--lr",       str(args.lr_s1),
                "--save_dir", str(Path(args.save_dir) / "stage1"),
            ]
        )

    if 2 in stages_to_run:
        s2_extra = [
            "--epochs",     str(args.epochs_s2),
            "--lr",         str(args.lr_s2),
            "--lora_r",     str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--save_dir",   str(Path(args.save_dir) / "stage2"),
        ]
        if s1_ckpt.exists():
            s2_extra += ["--stage1_ckpt", str(s1_ckpt)]
        run_stage(str(training_dir / "stage2_vlm.py"), common + s2_extra)

    if 3 in stages_to_run:
        s3_extra = [
            "--epochs",   str(args.epochs),
            "--lr",       str(args.lr),
            "--save_dir", str(Path(args.save_dir) / "stage3"),
        ]
        if s2_ckpt.exists():
            s3_extra += ["--stage2_ckpt", str(s2_ckpt)]
        if args.resume:
            s3_extra += ["--resume", args.resume]
        run_stage(str(training_dir / "stage3_action.py"), common + s3_extra)

    print(f"\n{'='*64}")
    print(f"  All stages complete!")
    print(f"  Final checkpoint: {Path(args.save_dir)}/stage3/best.pth")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
