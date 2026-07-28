"""
Extract a slim Stage-2 checkpoint (trainable / adapter weights only).

Base SigLIP2 + Mamba2 weights are expected to load from HuggingFace cache on
the eval machine; this keeps the transferable blob ~180MB instead of ~3.3GB.

Prefixes kept (PEFT LoRA + Stage1/2 modules):
  - any key containing 'lora_'
  - projector.
  - token_resampler.
  - proprio_encoder.   (Stage2 trains with zero-state pair, but weights exist)

Usage (remote, 2GB-safe via mmap):
  python scripts/make_stage2_slim.py \\
      --src /root/autodl-tmp/Aeromamba/checkpoints/.../stage2_v2/best.pth \\
      --dst /root/autodl-tmp/stage2_v2_best_slim.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


KEEP_PREFIXES = (
    "projector.",
    "token_resampler.",
    "proprio_encoder.",
)


def keep_key(k: str) -> bool:
    if "lora_" in k:
        return True
    return any(k.startswith(p) or f".{p}" in k or k.replace("base_model.model.", "").startswith(p)
               for p in KEEP_PREFIXES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    print(f"[slim] loading (mmap) {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    if not isinstance(state, dict):
        raise SystemExit("checkpoint has no model_state dict")

    slim = {k: v.detach().cpu().clone() for k, v in state.items() if keep_key(k)}
    n_bytes = sum(v.numel() * v.element_size() for v in slim.values())
    print(f"[slim] kept {len(slim)} / {len(state)} tensors ({n_bytes/1e6:.1f} MB raw)")

    out = {
        "model_state": slim,
        "slim": True,
        "stage": 2,
        "src": str(args.src),
    }
    # preserve handy metadata if present
    for meta in ("epoch", "global_step", "best_val_loss", "args"):
        if meta in ckpt:
            out[meta] = ckpt[meta]

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"[slim] wrote {args.dst} ({Path(args.dst).stat().st_size/1e6:.1f} MB file)")


if __name__ == "__main__":
    main()
