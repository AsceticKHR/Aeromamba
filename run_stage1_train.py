"""
AeroMamba Stage 1 Training Launcher
====================================
- Uses small LLaVA subset (./data/llava_subset)
- Enforces CUDA GPU usage; aborts if no GPU available
- Catches OOM and other critical errors: stops immediately and cleans GPU/disk cache
- Optimised for Windows + aeromamba conda env (num_workers=0)
- Uses lightweight mamba-130m + siglip_l_384 (single encoder) to minimise VRAM

Usage:
    conda activate aeromamba
    python run_stage1_train.py
    python run_stage1_train.py --batch 4 --epochs 1   # minimal test
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import traceback
from pathlib import Path

# ── Force UTF-8 output (avoids CP1252 errors on Windows) ─────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# GPU guard
# ─────────────────────────────────────────────────────────────────────────────

def enforce_gpu() -> None:
    """Exit early if CUDA GPU is not available."""
    import torch
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU detected. Aborting training.")
        print("  Make sure you are using the aeromamba conda environment")
        print("  and that the GPU driver / CUDA toolkit is correctly installed.")
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA built-with : {torch.version.cuda}")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"[GPU] CUDA version: {torch.version.cuda}")


# ─────────────────────────────────────────────────────────────────────────────
# Cache cleanup
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_cache(save_dir: str | None = None) -> None:
    """
    Release GPU memory and optionally remove incomplete checkpoint dir.
    Called on OOM or any other fatal training error.
    """
    import torch

    # 1. Empty GPU cache
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[Cleanup] GPU cache cleared.")
    except Exception as e:
        print(f"[Cleanup] GPU cache clear failed (non-fatal): {e}")

    # 2. Python garbage collection
    gc.collect()
    print("[Cleanup] Python garbage collection done.")

    # 3. Remove HuggingFace/transformers download cache lock files
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lock_files = list(hf_cache.rglob("*.lock"))
    for lf in lock_files:
        try:
            lf.unlink()
        except OSError:
            pass
    if lock_files:
        print(f"[Cleanup] Removed {len(lock_files)} HuggingFace lock file(s).")

    # 4. Optionally remove incomplete checkpoint output directory
    #    Skip the interactive prompt when running in a non-interactive context
    #    (e.g. background task / pipe) to avoid EOFError
    if save_dir:
        ckpt_path = Path(save_dir)
        if ckpt_path.exists():
            interactive = sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
            if interactive:
                answer = input(
                    f"\n[Cleanup] Remove incomplete checkpoint dir '{ckpt_path}'? [y/N] "
                ).strip().lower()
                if answer == "y":
                    shutil.rmtree(ckpt_path, ignore_errors=True)
                    print(f"[Cleanup] Removed: {ckpt_path}")
                else:
                    print(f"[Cleanup] Kept: {ckpt_path}")
            else:
                # Non-interactive — keep the checkpoint dir, just report
                print(f"[Cleanup] Non-interactive mode: keeping checkpoint dir '{ckpt_path}'")

    print("[Cleanup] Done. Training terminated safely.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="AeroMamba Stage 1 — LLaVA subset training launcher"
    )
    # Dataset
    p.add_argument("--data_root",    default="./data/llava_subset",
                   help="Path to LLaVA dataset folder")
    p.add_argument("--json_name",    default="llava_subset.json",
                   help="Name of LLaVA JSON file inside data_root (e.g. llava_subset.json)")
    p.add_argument("--max_text_len", type=int, default=32,
                   help="Max tokenised text length (shorter = less VRAM)")
    p.add_argument("--val_frac",     type=float, default=0.1)

    # Model — use lightweight combo by default to reduce VRAM
    p.add_argument("--use_token_pooling", action="store_true",
                   help="Enable vision token pooling (e.g. 576 -> 64) to speed up training")
    p.add_argument("--pool_size",    type=int, default=8,
                   help="Target spatial dimension size for pooling (default: 8, i.e. 64 tokens)")

    # Model — use lightweight combo by default to reduce VRAM
    p.add_argument("--mamba_type",   default="mamba-130m",
                   choices=["mamba-130m", "mamba-370m", "mamba-790m",
                            "mamba-1.4b", "mamba-2.8b"],
                   help="Mamba backbone variant (130m recommended for first run)")
    p.add_argument("--vision_type",  default="dinosiglip_so_384",
                   help="Vision encoder (dinosiglip_so_384 = dual dino+siglip encoder)")
    p.add_argument("--chunk_size",   type=int, default=5)

    # Training hyper-params
    p.add_argument("--epochs",  type=int,   default=3)
    p.add_argument("--batch",   type=int,   default=2,
                   help="Batch size. Reduce to 1 if you get OOM.")
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--workers", type=int,   default=0,
                   help="DataLoader workers. Keep 0 on Windows to avoid mp issues.")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Max steps to run per training epoch (None for all)")
    p.add_argument("--max_val_steps", type=int, default=100,
                   help="Max steps to run per validation epoch")

    # Output
    p.add_argument("--save_dir", default="./checkpoints/stage1")
    p.add_argument("--resume",   default=None,
                   help="Path to a checkpoint to resume from")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── 0. GPU guard ──────────────────────────────────────────────────────────
    enforce_gpu()

    # ── 0b. Enable memory-efficient CUDA allocator ────────────────────────────
    # This significantly reduces fragmentation and OOM risk
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:128"
    )
    import torch
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"[GPU] VRAM: {free_bytes/1e9:.2f} GB free / {total_bytes/1e9:.2f} GB total")
    if free_bytes < 1.5e9:
        print("[WARNING] Less than 1.5 GB VRAM free — training may OOM.")
        print("  Consider closing other GPU-intensive applications (e.g. LM Studio).")

    # ── 1. Validate data root ─────────────────────────────────────────────────
    data_root = Path(args.data_root)
    json_path = data_root / args.json_name
    if not json_path.exists():
        print(f"[FATAL] LLaVA JSON not found: {json_path}")
        print("  Make sure the data_root and json_name are correct.")
        sys.exit(1)
    print(f"[Data] Using dataset: {json_path}")

    # ── 2. Launch training with OOM / error guard ─────────────────────────────
    from training.stage1_align import Stage1Trainer

    trainer = Stage1Trainer(args)
    try:
        trainer.run()
        print("\n[Stage 1] Training finished successfully!")

    except torch.cuda.OutOfMemoryError as oom:
        print("\n" + "=" * 70)
        print("[OOM ERROR] CUDA out of memory! Stopping training immediately.")
        print(f"  Error: {oom}")
        print(f"  Suggestion: reduce --batch (current: {args.batch})")
        print("=" * 70)
        cleanup_cache(save_dir=args.save_dir)
        sys.exit(2)

    except KeyboardInterrupt:
        print("\n[Interrupted] Training cancelled by user.")
        cleanup_cache(save_dir=None)
        sys.exit(0)

    except Exception as exc:
        print("\n" + "=" * 70)
        print(f"[FATAL ERROR] Training crashed: {type(exc).__name__}: {exc}")
        print("=" * 70)
        traceback.print_exc()
        cleanup_cache(save_dir=args.save_dir)
        sys.exit(3)


if __name__ == "__main__":
    import torch  # import here so enforce_gpu() can use it
    main()
