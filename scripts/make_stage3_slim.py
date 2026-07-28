"""
Extract a slim Stage-3 checkpoint (trainable / adapter weights only).

Keeps LoRA + projector + resampler + proprio + action_head (+ binding_head
if present). Frozen SigLIP2 / Mamba2 base weights are reloaded from HF on the
eval machine so the blob stays ~180-250MB instead of ~3.3GB.

Usage (remote, mmap-safe):
  python scripts/make_stage3_slim.py \\
      --src .../stage3_v3_binding_.../best.pth \\
      --dst .../stage3_v3/best_slim.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


KEEP_PREFIXES = (
    "projector.",
    "token_resampler.",
    "proprio_encoder.",
    "action_head.",
    "binding_head.",
)


def keep_key(k: str) -> bool:
    if "lora_" in k:
        return True
    k2 = k.replace("base_model.model.", "")
    return any(
        k.startswith(p) or k2.startswith(p) or f".{p}" in k
        for p in KEEP_PREFIXES
    )


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
    prefixes = {}
    for k in slim:
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print(f"[slim] key groups: {prefixes}")

    out = {
        "model_state": slim,
        "slim": True,
        "stage": 3,
        "src": str(args.src),
    }
    for meta in ("epoch", "global_step", "best_val", "best_val_loss", "args"):
        if meta in ckpt:
            out[meta] = ckpt[meta]

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"[slim] wrote {args.dst} ({Path(args.dst).stat().st_size/1e6:.1f} MB file)")


if __name__ == "__main__":
    main()
