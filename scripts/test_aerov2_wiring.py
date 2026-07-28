"""Offline structural test for AeroV2 S1 wiring (no data / weights needed).

AEROMAMBA_OFFLINE=1 python scripts/test_aerov2_wiring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.aerov2 import AeroV2  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    model = AeroV2(dtype=torch.float32)
    model.configure_stage1()
    model.print_param_census()

    B, L = 2, 16
    img = model.vision_encoder.img_size
    pv = torch.randn(B, 3, img, img)
    ids = torch.randint(5, 1000, (B, L))
    labels = ids.clone()
    labels[:, :4] = -100

    out = model.forward_clm(pv, ids, labels)
    loss = out["loss"]
    assert torch.isfinite(loss), "non-finite CLM loss"
    print(f"[wiring] loss={float(loss):.4f} n_vis={out['n_vis_tokens']}")

    loss.backward()
    grads = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    leaks = [n for n, p in model.named_parameters()
             if not p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert grads and not leaks, f"grad wiring broken: grads={len(grads)} leaks={leaks[:3]}"
    print(f"[wiring] projector grads OK ({len(grads)} tensors), no frozen leaks")

    # one optimizer step must reduce single-batch loss eventually — quick check
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    first = float(loss)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        l = model.forward_clm(pv, ids, labels)["loss"]
        l.backward()
        opt.step()
    print(f"[wiring] 20-step loss {first:.4f} -> {float(l):.4f}")
    assert float(l) < first, "loss not decreasing on fixed batch"
    print("WIRING_OK")


if __name__ == "__main__":
    main()
