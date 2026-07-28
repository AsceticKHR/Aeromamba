"""Offline contract test for the v5 action readout + distributional head.

Runs on CPU with random weights (no checkpoints, no dataset) so the shape /
gradient / decoding contract can be checked before spending remote GPU time.

  python scripts/test_v5_head_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.action_head import (  # noqa: E402
    ActionReadout, UAVActionHeadV5, aero_action_loss_v5, hl_gauss_targets,
)

B, K, D, N_VIS, A = 3, 8, 128, 49, 4
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


N_TXT = 16


def test_readout_is_spatial():
    """The readout must depend on the patch content, per query, with NO path
    from the text keys to the output that skips the attention."""
    print("\n=== T1 readout shape + spatial dependence ===")
    ro = ActionReadout(D, K, n_heads=4, n_layers=2)
    kv = torch.randn(B, N_VIS + N_TXT, D)
    out = ro(kv)
    check("output shape [B,K,D]", tuple(out.shape) == (B, K, D), str(tuple(out.shape)))

    # Perturbing ONE patch must move the output: this is the property the
    # terminal-hidden readout lacked (vis_sens 0.07). The perturbation has to be
    # a random direction, not a constant offset â€” ln_kv subtracts the per-token
    # mean, so a uniform shift lies exactly in its null space and would make
    # this test vacuously fail.
    kv_v = kv.clone()
    kv_v[:, N_VIS // 2] += torch.randn(D) * 3.0
    kv_t = kv.clone()
    kv_t[:, N_VIS + 1] += torch.randn(D) * 3.0
    delta_vis = (ro(kv_v) - out).abs().mean().item()
    delta_txt = (ro(kv_t) - out).abs().mean().item()
    check("single-patch perturbation moves output", delta_vis > 1e-4,
          f"delta={delta_vis:.5f}")
    check("text key also moves output", delta_txt > 1e-6, f"delta={delta_txt:.6f}")

    # The scale of the text keys must NOT be able to swamp vision: with both in
    # one LayerNormed KV the softmax arbitrates, so blowing up the text states
    # by 1000x (the real |ctx| 2955 vs |q| 0.82 situation) must leave the patch
    # sensitivity intact. This is the regression test for the v5-a2 failure.
    kv_big = kv.clone()
    kv_big[:, N_VIS:] *= 1000.0
    out_big = ro(kv_big)
    kv_big_v = kv_big.clone()
    kv_big_v[:, N_VIS // 2] += torch.randn(D) * 3.0
    delta_vis_big = (ro(kv_big_v) - out_big).abs().mean().item()
    check("1000x text scale does not drown vision",
          delta_vis_big > 0.2 * delta_vis,
          f"delta={delta_vis_big:.5f} vs {delta_vis:.5f} at unit text scale")

    # Padding must be excluded, or most of the text budget goes to pad tokens.
    mask = torch.zeros(B, N_VIS + N_TXT, dtype=torch.bool)
    mask[:, N_VIS + 8:] = True
    kv_pad = kv.clone()
    kv_pad[:, N_VIS + 8:] = torch.randn(B, N_TXT - 8, D) * 10.0
    check("masked keys are ignored",
          torch.allclose(ro(kv, mask), ro(kv_pad, mask), atol=1e-5))

    # Queries must not degenerate to identical tokens.
    spread = out.std(dim=1).mean().item()
    check("queries are differentiated", spread > 1e-3, f"per-query std={spread:.5f}")


def test_hl_gauss_targets():
    print("\n=== T2 HL-Gauss soft labels ===")
    n_bins, rng = 64, 1.5
    edges = torch.linspace(-rng, rng, n_bins + 1)
    sigma = 0.75 * (2 * rng / n_bins)
    y = torch.tensor([-1.0, 0.0, 0.37, 1.2])
    p = hl_gauss_targets(y, edges, sigma)
    check("normalised", torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5))
    centers = (edges[:-1] + edges[1:]) * 0.5
    recon = p @ centers
    err = (recon - y).abs().max().item()
    check("expectation recovers target", err < 2 * (2 * rng / n_bins),
          f"max err={err:.5f} (bin width={2*rng/n_bins:.5f})")
    # Out-of-range targets must clamp to the edge, keeping a usable gradient.
    # (Soft labels straddle a couple of bins by design, so the mass sits in the
    # top few bins rather than entirely in the last one.)
    p_far = hl_gauss_targets(torch.tensor([9.0]), edges, sigma)
    check("out-of-range target is finite", bool(torch.isfinite(p_far).all()))
    check("out-of-range target still normalised",
          abs(p_far.sum().item() - 1.0) < 1e-4, f"sum={p_far.sum().item():.5f}")
    check("out-of-range mass sits at the top edge",
          p_far[0, -3:].sum().item() > 0.95,
          f"top-3 bins={p_far[0, -3:].sum().item():.4f}, "
          f"bottom half={p_far[0, :n_bins // 2].sum().item():.2e}")


def test_head(n_bins: int):
    tag = f"hlgauss(n_bins={n_bins})" if n_bins else "l1"
    print(f"\n=== T3 head forward/backward â€” {tag} ===")
    head = UAVActionHeadV5(mamba_hidden_size=D, chunk_size=K, action_dim=A,
                           n_bins=n_bins, bin_range=1.5, n_layers=2, n_heads=4)
    # Physical scale roughly matching UAV-Flow (dx dominant, dz tiny).
    q99 = torch.tensor([[0.9, 0.55, 0.12, 0.25]]).repeat(K, 1) * torch.linspace(0.2, 1.6, K)[:, None]
    head.set_quantile_normalization(-q99, q99)

    kv = torch.randn(B, N_VIS + N_TXT, D)
    pred = head(kv)
    check("action shape [B,K,4]", tuple(pred["action"].shape) == (B, K, A))

    # Safe start: the initial physical action must be negligible for a UAV
    # (millimetre scale). L1 gets this from an exactly zeroed output layer;
    # HL-Gauss gets it from bin symmetry, so a small-scale init is enough.
    init_phys = head.denormalize(pred["action"]).abs().max().item()
    tol = 1e-5 if n_bins == 0 else 5e-3
    check("safe near-zero initial action", init_phys < tol,
          f"max|a|={init_phys:.2e} m (tol {tol:g})")

    if n_bins:
        check("logits shape", tuple(pred["logits"].shape) == (B, K, A, n_bins))

    # Gradient contract, measured on the pristine head (before any weight
    # tampering below), including targets far outside the bin span.
    gt = torch.randn(B, K, A) * 0.2
    gt[0] = 50.0                       # far out of range: must still teach
    loss, detail = aero_action_loss_v5(head(kv), gt, head)
    check("loss finite", bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
    check("detail finite", all(v == v for v in detail.values()), str(
        {k: round(v, 4) for k, v in detail.items()}))
    loss.backward()
    gnorm = sum(float(p.grad.norm()) ** 2 for p in head.parameters()
                if p.grad is not None) ** 0.5
    check("gradient reaches the head", gnorm > 0, f"|g|={gnorm:.4e}")
    q_grad = head.readout.queries.grad
    q_ok = q_grad is not None and float(q_grad.norm()) > 0
    if n_bins:
        # HL-Gauss uses a small-scale output init precisely so the readout is
        # live from step 1.
        check("gradient reaches the action queries at step 1", q_ok)
    else:
        # L1 keeps the zero-init output layer for a safe initial action, which
        # gates the upstream gradient for exactly one step. Assert it heals.
        check("L1 readout is gated at step 1 (expected)", not q_ok)
        with torch.no_grad():
            head.step_mlp[-1].weight.normal_(0, 1e-2)
        head.zero_grad()
        aero_action_loss_v5(head(kv), gt, head)[0].backward()
        healed = float(head.readout.queries.grad.norm()) > 0
        check("L1 readout gradient heals after one step", healed)

    if n_bins:
        # Bounded by construction: the expectation cannot leave the bin span,
        # however extreme the logits get.
        with torch.no_grad():
            head.step_mlp[-1].weight.normal_(0, 5.0)
            head.step_mlp[-1].bias.normal_(0, 5.0)
        a = head(kv)["action"]
        check("expectation stays inside the bin span",
              bool((a.abs() <= head.bin_range + 1e-5).all()),
              f"max|a_norm|={a.abs().max().item():.4f}")


def test_distributional_beats_l1_on_collapse():
    """The core claim of C3, as a falsifiable micro-experiment.

    The target distribution mirrors dz: mostly ~0 with a rare +-large excursion,
    and nothing in the input predicts which. L1's optimum is then the strict
    conditional median (0), so the head collapses to a constant and the rare
    excursions vanish â€” the measured dz std ratio of 0.124. A distributional
    head retains the outer modes.

    Note the mixture must have a mode AT the median. For a symmetric two-point
    target the L1 objective is exactly flat between the modes, so "L1 collapses"
    is not even well posed there.
    """
    print("\n=== T4 rare-excursion target: L1 collapses, HL-Gauss keeps the modes ===")
    torch.manual_seed(0)
    n_bins, rng = 64, 1.5
    edges = torch.linspace(-rng, rng, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    sigma = 0.75 * (2 * rng / n_bins)
    u = torch.rand(4096)
    y = torch.zeros(4096)
    y[u < 0.15] = -0.8
    y[u > 0.85] = 0.8

    # L1 point estimate: a single free scalar (the best input-independent value).
    v = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        (v - y).abs().mean().backward()
        opt.step()
    check("L1 optimum sits at the median (collapsed)", abs(v.item()) < 0.05,
          f"v={v.item():.4f}")

    # Distributional: free logits over bins, trained with HL-Gauss CE.
    logits = torch.zeros(n_bins, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.2)
    tgt = hl_gauss_targets(y, edges, sigma).mean(0)
    for _ in range(600):
        opt.zero_grad()
        (-(tgt * logits.log_softmax(-1)).sum()).backward()
        opt.step()
    p = logits.softmax(-1).detach()
    mass_pos = p[centers > 0.4].sum().item()
    mass_neg = p[centers < -0.4].sum().item()
    check("HL-Gauss keeps the rare excursions", mass_pos > 0.1 and mass_neg > 0.1,
          f"p(+)={mass_pos:.3f} p(-)={mass_neg:.3f} (true 0.15/0.15)")
    # The whole point: a point estimate has zero spread, the distribution does not.
    spread = float(((p * (centers - (p * centers).sum()) ** 2).sum()) ** 0.5)
    check("HL-Gauss retains spread that L1 destroys", spread > 0.3,
          f"std={spread:.3f} vs L1 std=0.0")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_readout_is_spatial()
    test_hl_gauss_targets()
    test_head(n_bins=0)
    test_head(n_bins=64)
    test_distributional_beats_l1_on_collapse()
    print(f"\n=== V5 CONTRACT: {'ALL PASS' if not FAILURES else 'FAIL ' + str(FAILURES)} ===")
    sys.exit(1 if FAILURES else 0)

