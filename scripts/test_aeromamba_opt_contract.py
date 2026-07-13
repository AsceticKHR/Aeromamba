from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import DummyUAVDataset, UAVFlowDataset, parse_state8
from model.action_head import aero_action_loss
from model.proprio_encoder import ProprioEncoder


def test_state8_and_flip_contract() -> None:
    state = [[1.0, 2.0, 3.0], [0.0, 90.0, 0.0], [0.1, 0.2, 0.3, 0.4]]
    state8 = torch.tensor(parse_state8(state))
    assert state8.shape == (8,)
    assert torch.isclose(state8[3], torch.tensor(torch.pi / 2), atol=1e-5)

    flipped = UAVFlowDataset._flip_state8(state8)
    assert flipped[0] == state8[0]
    assert flipped[1] == -state8[1]
    assert flipped[3] == -state8[3]
    assert flipped[5] == -state8[5]
    assert flipped[7] == -state8[7]


def test_proprio_encoder_pair_contract() -> None:
    encoder = ProprioEncoder(proprio_dim=8, mamba_hidden_size=16, num_freqs=2, dropout=0.0)
    state = torch.zeros(3, 8)
    delta_state = torch.ones(3, 8)
    tokens = encoder.forward_pair(state, delta_state)
    assert tokens.shape == (3, 2, 16)
    assert torch.isfinite(tokens).all()


def test_dummy_dataset_exposes_opt_state() -> None:
    sample = DummyUAVDataset(size=1, chunk_size=8, dual_vision=False)[0]
    assert sample["state8"].shape == (8,)
    assert sample["delta_state8"].shape == (8,)
    assert sample["gt_action"].shape == (8, 4)


def test_action_loss_s3a_s3b_contract() -> None:
    pred = {"action": torch.zeros(2, 8, 4)}
    gt = torch.randn(2, 8, 4) * 0.1
    loss_a, detail_a = aero_action_loss(
        pred,
        gt,
        lambda_smooth=0.0,
        lambda_endpoint=2.0,
        lambda_direction=0.0,
        lambda_acc=0.0,
    )
    loss_b, detail_b = aero_action_loss(
        pred,
        gt,
        lambda_smooth=0.0,
        lambda_endpoint=2.0,
        lambda_direction=0.0,
        lambda_acc=0.5,
    )
    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
    assert "acc" in detail_b
    assert "pos_err_m" in detail_a and "yaw_err_deg" in detail_a


def test_action_zscore_normalization_contract() -> None:
    from model.action_head import UAVActionHead

    head = UAVActionHead(mamba_hidden_size=32, chunk_size=8, action_dim=4)
    # Default buffers are identity
    assert not head.has_normalization()
    gt = torch.randn(2, 8, 4) * 0.1
    assert torch.allclose(head.normalize(gt), gt)

    mean = torch.randn(8, 4) * 0.05
    std = torch.rand(8, 4) * 0.1 + 0.01
    head.set_normalization(mean, std)
    assert head.has_normalization()
    # Roundtrip: denormalize(normalize(x)) == x
    assert torch.allclose(head.denormalize(head.normalize(gt)), gt, atol=1e-5)

    # Loss in z-space is finite and reports physical diagnostics
    pred = {"action": torch.zeros(2, 8, 4)}
    loss, detail = aero_action_loss(pred, gt, head=head)
    assert torch.isfinite(loss)
    assert "pos_err_m" in detail and "end_pos_err_m" in detail

    # Buffers persist through state_dict (checkpoint round-trip)
    head2 = UAVActionHead(mamba_hidden_size=32, chunk_size=8, action_dim=4)
    head2.load_state_dict(head.state_dict())
    assert torch.allclose(head2.action_mean, head.action_mean)
    assert torch.allclose(head2.action_std, head.action_std)

    # Zero-init output layer ⇒ z-space output 0 ⇒ physical output = dataset mean
    tok = torch.randn(2, 32)
    out = head(tok)["action"]
    phys = head.denormalize(out)
    assert torch.allclose(phys, mean.unsqueeze(0).expand(2, -1, -1), atol=1e-5)


def test_training_sources_use_opt_order() -> None:
    for rel_path in ("training/stage1_align.py", "training/stage2_vlm.py"):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "state_tokens = model.proprio_encoder.forward_pair" in text
        assert "torch.cat([state_tokens, vis_tokens, text_embs], dim=1)" in text
        assert "prefix_len = state_tokens.size(1) + vis_tokens.size(1)" in text

    model_text = (ROOT / "model/uav_mamba_vla.py").read_text(encoding="utf-8")
    assert "torch.cat([state_tokens, vis_tokens, text_embs], dim=1)" in model_text
    assert "last non-padding language token" in model_text
    assert "action_context_fuser" not in model_text


def test_legacy_dataset_rejects_missing_instruction() -> None:
    dataset_text = (ROOT / "data/dataset.py").read_text(encoding="utf-8")
    assert "refusing fixed fallback" in dataset_text


if __name__ == "__main__":
    test_state8_and_flip_contract()
    test_proprio_encoder_pair_contract()
    test_dummy_dataset_exposes_opt_state()
    test_action_loss_s3a_s3b_contract()
    test_action_zscore_normalization_contract()
    test_training_sources_use_opt_order()
    test_legacy_dataset_rejects_missing_instruction()
    print("AeroMamba-Opt contract tests passed.")
