"""
AeroMamba Module Unit Tests

Tests each sub-module independently for correct output shapes and gradient flow.
No real data or GPU required (runs on CPU).

Usage:
    python scripts/unit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.action_head import UAVActionHead, aero_action_loss
from model.projector import MLPProjector
from model.proprio_encoder import ProprioEncoder

# ── Test constants ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B      = 4      # batch size
D_v    = 2176   # vision hidden dim (DinoSigLIP fused)
D_m    = 1024   # Mamba hidden dim (mamba-370m)
K      = 5      # action chunk size
N_vis  = 729    # number of vision patches


def _section(name: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: MLPProjector
# ─────────────────────────────────────────────────────────────────────────────

def test_projector() -> None:
    _section("Test 1 — MLPProjector")
    proj = MLPProjector(
        vision_hidden_size=D_v,
        mamba_hidden_size=D_m,
    ).to(DEVICE)

    x = torch.randn(B, N_vis, D_v, device=DEVICE)
    y = proj(x)

    assert y.shape == (B, N_vis, D_m), (
        f"Shape mismatch: expected ({B}, {N_vis}, {D_m}), got {tuple(y.shape)}"
    )

    # Verify gradient flow
    y.sum().backward()
    grad_norms = [p.grad.norm().item() for p in proj.parameters() if p.grad is not None]
    assert all(g > 0 for g in grad_norms), "Some gradients are zero — check connectivity"

    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(y.shape)}")
    print(f"  Grad norms: {[f'{g:.4f}' for g in grad_norms]}")
    print(f"  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: ProprioEncoder
# ─────────────────────────────────────────────────────────────────────────────

def test_proprio_encoder() -> None:
    _section("Test 2 — ProprioEncoder")
    enc = ProprioEncoder(
        proprio_dim=4,
        mamba_hidden_size=D_m,
    ).to(DEVICE)

    x = torch.randn(B, 4, device=DEVICE)
    y = enc(x)

    assert y.shape == (B, 1, D_m), (
        f"Shape mismatch: expected ({B}, 1, {D_m}), got {tuple(y.shape)}"
    )

    # Check for NaN
    assert not torch.isnan(y).any(), "Output contains NaN"

    y.sum().backward()
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(y.shape)}")
    print(f"  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: UAVActionHead + aero_action_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_action_head() -> None:
    _section("Test 3 — UAVActionHead + aero_action_loss")
    head = UAVActionHead(
        mamba_hidden_size=D_m,
        chunk_size=K,
    ).to(DEVICE)

    global_token = torch.randn(B, D_m, device=DEVICE)
    out = head(global_token)

    assert "action" in out, "Missing 'action' key in output dict"
    assert out["action"].shape == (B, K, 4), (
        f"Shape mismatch: expected ({B}, {K}, 4), got {tuple(out['action'].shape)}"
    )

    # Loss computation
    gt_action = torch.randn(B, K, 4, device=DEVICE)
    loss, detail = aero_action_loss(out, gt_action, lambda_smooth=0.1)

    assert not torch.isnan(loss), f"Loss is NaN: {loss}"
    assert loss.item() >= 0.0, f"Loss is negative: {loss.item()}"

    loss.backward()

    print(f"  Input (global_token): {tuple(global_token.shape)}")
    print(f"  Output action chunk : {tuple(out['action'].shape)}")
    print(f"  Loss value          : {loss.item():.4f}")
    print(f"  Loss detail         : main={detail['main']:.4f}  smooth={detail['smooth']:.4f}")
    print(f"  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: TemporalEnsemble
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_ensemble() -> None:
    _section("Test 4 — TemporalEnsemble (inference utility)")
    from model.action_head import TemporalEnsemble

    ens = TemporalEnsemble(window=5, decay=0.7)

    results = []
    for step in range(8):
        chunk = torch.randn(K, 4)   # simulated model output per step
        action = ens.update(chunk)  # should return [4]
        results.append(action)
        assert action.shape == (4,), f"Step {step}: expected [4], got {tuple(action.shape)}"

    print(f"  Ran 8 control steps, all outputs shape [4]")
    print(f"  Sample output: {results[-1].tolist()[:2]} ...")
    print(f"  ✓ PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═' * 50}")
    print(f"  AeroMamba Unit Tests")
    print(f"  Device: {DEVICE}")
    print(f"{'═' * 50}")

    passed = 0
    failed = 0

    for test_fn in [test_projector, test_proprio_encoder, test_action_head, test_temporal_ensemble]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'═' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✅ All unit tests passed!")
    else:
        print("  ❌ Some tests failed — check output above.")
    print(f"{'═' * 50}\n")
