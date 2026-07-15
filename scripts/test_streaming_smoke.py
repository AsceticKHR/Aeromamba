"""
AeroStream streaming smoke test (change plan §B1) + Round-A unit smoke.

Pure engineering verification, no training:

  1. Binding labels: infer_motion_class / binding_labels_from_action sanity
     (signs, bins, flip consistency).
  2. Binding head + weighted loss: forward/backward on random tensors, loss
     finite, zero-init head produces uniform logits.
  3. Streaming numerical consistency: on the real (or offline-random) model,
     feed [state|delta|vision|text] then two incremental [state|delta|vision]
     frames through stream_step, and compare the hidden-state readout against
     the equivalent full-sequence forward. Tolerance 1e-3 (fp32).

Run on any machine with the aeromamba env (GPU optional):
  python scripts/test_streaming_smoke.py                 # full (downloads weights)
  AEROMAMBA_OFFLINE=1 python scripts/test_streaming_smoke.py --offline_ok
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_binding_labels() -> None:
    from data.dataset import (
        BINDING_IGNORE_INDEX,
        binding_labels_from_action,
        infer_motion_class,
    )
    from model.binding_head import MOTION_CLASSES

    assert MOTION_CLASSES[infer_motion_class("Orbit around the tower")] == "surround"
    assert MOTION_CLASSES[infer_motion_class("Please land near the car")] == "land"
    assert MOTION_CLASSES[infer_motion_class("Ascend to roof height")] == "ascend"
    assert MOTION_CLASSES[infer_motion_class("Rotate 90 degrees clockwise")] == "rotate"
    assert MOTION_CLASSES[infer_motion_class("Turn to face the statue")] == "turn"
    assert MOTION_CLASSES[infer_motion_class("Pass through the arch")] == "pass"
    assert MOTION_CLASSES[infer_motion_class("Move toward the red building")] == "approach"
    assert MOTION_CLASSES[infer_motion_class("Fly forward")] == "move"
    assert infer_motion_class("beep boop nonsense") == BINDING_IGNORE_INDEX

    # gt_action rows are cumulative offsets; endpoint defines the labels.
    K = 8
    gt = torch.zeros(K, 4)
    gt[-1] = torch.tensor([1.0, 0.0, 0.30, math.radians(15.0)])
    lab = binding_labels_from_action(gt, motion_class=0)
    assert lab["yaw_sign"].item() == 1, "positive yaw endpoint -> class 1"
    assert lab["dz_sign"].item() == 1, "positive dz endpoint -> class 1 (up)"
    # bins: [0,0.1)=0 [0.1,0.2)=1 [0.2,0.5)=2 [0.5,1)=3 [1,2)=4 ...
    assert lab["magnitude_bin"].item() == 4, "|1.04m| falls in [1,2) -> bin 4"

    # Flip consistency: labels computed from the flipped action flip signs.
    from data.dataset import UAVFlowDataset
    flipped = UAVFlowDataset._flip_action(gt)
    lab_f = binding_labels_from_action(flipped, motion_class=0)
    assert lab_f["yaw_sign"].item() == 2, "flip negates yaw -> class 2"
    assert lab_f["dz_sign"].item() == lab["dz_sign"].item(), "flip keeps dz"
    assert lab_f["magnitude_bin"].item() == lab["magnitude_bin"].item()

    small = torch.zeros(K, 4)
    small[-1] = torch.tensor([0.0, 0.0, 0.01, math.radians(0.5)])
    lab_s = binding_labels_from_action(small, motion_class=BINDING_IGNORE_INDEX)
    assert lab_s["yaw_sign"].item() == 0 and lab_s["dz_sign"].item() == 0
    assert lab_s["motion_class"].item() == BINDING_IGNORE_INDEX
    print("[smoke] binding labels ... OK")


def test_binding_head_and_loss() -> None:
    from model.binding_head import InstructionBindingHead, binding_loss
    from model.action_head import UAVActionHead, aero_action_loss

    torch.manual_seed(0)
    B, D, K = 6, 1024, 8
    head = InstructionBindingHead(hidden_size=D)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params < 1_000_000, f"binding head too large: {n_params}"

    h = torch.randn(B, D, requires_grad=True)
    logits = head(h)
    # Zero-init outputs -> all logits identically zero before training.
    for name, l in logits.items():
        assert torch.allclose(l, torch.zeros_like(l)), f"{name} not zero-init"

    labels = {
        "motion_class": torch.tensor([0, 1, -100, 3, 9, 2]),
        "yaw_sign": torch.tensor([0, 1, 2, -100, 1, 0]),
        "dz_sign": torch.tensor([0, 0, 1, 2, -100, 0]),
        "magnitude_bin": torch.tensor([0, 3, 5, 8, 2, -100]),
    }
    loss, detail = binding_loss(logits, labels)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()

    # Weighted action loss: finite + channel weights bias the gradient.
    action_head = UAVActionHead(mamba_hidden_size=D, chunk_size=K)
    pred = {"action": torch.randn(B, K, 4, requires_grad=True)}
    gt = torch.randn(B, K, 4)
    cw = torch.tensor([1.0, 1.0, 2.5, 2.5])
    sw = 1.0 + torch.log1p(torch.rand(B))
    total, det = aero_action_loss(
        pred, gt, head=action_head, lambda_acc=0.25,
        channel_weights=cw, sample_weights=sw,
    )
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(pred["action"].grad).all()
    # Backward-compat: no weights path unchanged shape/finite
    pred2 = {"action": torch.randn(B, K, 4)}
    total2, _ = aero_action_loss(pred2, gt, head=action_head)
    assert torch.isfinite(total2)
    print(f"[smoke] binding head ({n_params/1e3:.0f}K params) + weighted loss ... OK")


def test_streaming_consistency(args) -> None:
    from model.uav_mamba_vla import AeroMambaVLA

    offline = os.environ.get("AEROMAMBA_OFFLINE", "0") == "1"
    if offline and not args.offline_ok:
        raise SystemExit("set --offline_ok to run the streaming test offline")
    if offline:
        # Offline random init cannot build mamba2-370m; fall back to mamba-130m.
        mamba_type, vision_type = "mamba-130m", "siglip_b_224"
    else:
        mamba_type, vision_type = args.mamba_type, args.vision_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroMambaVLA(
        mamba_type=mamba_type,
        vision_type=vision_type,
        chunk_size=8,
        token_resampler="perceiver",
        num_visual_queries=args.num_visual_queries,
    ).to(device).eval()

    torch.manual_seed(42)
    L = 16
    input_ids = torch.randint(10, 1000, (1, L), device=device)

    def rand_pixels():
        hw = 224 if vision_type.endswith("224") else 384
        return torch.randn(1, 3, hw, hw, device=device)

    frames = [rand_pixels() for _ in range(3)]
    states = [torch.randn(1, 8, device=device) for _ in range(3)]
    deltas = [torch.randn(1, 8, device=device) for _ in range(3)]

    with torch.inference_mode():
        # ── streaming: frame0 (+text) then frame1, frame2 incrementally ────
        model.stream_reset()
        model.stream_step(frames[0], states[0], deltas[0], input_ids=input_ids)
        model.stream_step(frames[1], states[1], deltas[1])
        out_stream = model.stream_step(frames[2], states[2], deltas[2])
        act_stream = out_stream["action"]

        # ── reference: identical token sequence in ONE full forward ────────
        def tokens_for(i, with_text):
            s, d = model._prepare_state_inputs(
                1, device, next(model.action_head.parameters()).dtype,
                state=states[i], delta_state=deltas[i],
            )
            st = model.proprio_encoder.forward_pair(s, d)
            vis = model.token_resampler(model.projector(model._encode_vision(frames[i])))
            parts = [st, vis]
            if with_text:
                parts.append(model._embed_text(input_ids))
            return torch.cat(parts, dim=1)

        full = torch.cat(
            [tokens_for(0, True), tokens_for(1, False), tokens_for(2, False)], dim=1
        )
        hidden = model._run_mamba(full)
        global_token = hidden[:, -1, :]
        pred_ref = model.action_head(
            global_token,
            proprio=model._prepare_state_inputs(
                1, device, global_token.dtype, state=states[2], delta_state=deltas[2]
            )[0],
        )
        act_ref = model.action_head.denormalize(pred_ref["action"]) \
            if hasattr(model.action_head, "denormalize") else pred_ref["action"]

    diff = (act_stream.float() - act_ref.float()).abs().max().item()
    print(f"[smoke] streaming vs full-sequence max |Δaction| = {diff:.2e}")
    assert diff < args.tol, f"streaming mismatch {diff} >= tol {args.tol}"
    # State persistence sanity: cache must survive across calls
    assert model._stream_cache is not None and model._stream_pos == full.size(1)
    model.stream_reset()
    assert model._stream_cache is None and model._stream_pos == 0
    print("[smoke] streaming consistency ... OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba_type", default="mamba-2-370m")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--num_visual_queries", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--offline_ok", action="store_true")
    ap.add_argument("--skip_stream", action="store_true",
                    help="only run the fast unit checks (no model download)")
    args = ap.parse_args()

    test_binding_labels()
    test_binding_head_and_loss()
    if not args.skip_stream:
        test_streaming_consistency(args)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
