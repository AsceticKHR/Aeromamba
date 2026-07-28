"""AeroV3 contract test — CPU, no weights, seconds.

Run before every training launch:

    AEROMAMBA_OFFLINE=1 python scripts/test_aerov3_wiring.py

Each check maps to a claim that the architecture makes. The two that are easy
to break silently are the detached recurrence (if a gradient crosses a step
boundary we are doing truncated BPTT, not filtering, and the RoboMME argument
no longer applies) and task-frame cache equivalence (if caching is not an exact
identity, the per-episode-once latency claim is measuring a different model
than the one that was trained).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("AEROMAMBA_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.aerov3 import AeroV3, AeroV3Config

B, W, L = 2, 4, 7
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILED.append(name)


def make_batch(model: AeroV3, cfg: AeroV3Config, all_ignore: bool = False):
    sz = model.vision.img_size if hasattr(model.vision, "img_size") else 384
    g = torch.Generator().manual_seed(0)
    stage = torch.full((B, W), -1) if all_ignore else \
        torch.randint(0, 3, (B, W), generator=g)
    return {
        "pixel_values": torch.randn(B, W, 3, sz, sz, generator=g),
        "first_pixel_values": torch.randn(B, 3, sz, sz, generator=g),
        "vision_update": torch.tensor([[1.0, 0.0, 1.0, 0.0]] * B),
        "pose": torch.randn(B, W, 4, generator=g) * 10,
        "pose0": torch.randn(B, 4, generator=g) * 10,
        "action": torch.randn(B, W, cfg.horizon, cfg.action_dim, generator=g),
        "action_mask": torch.ones(B, W, cfg.horizon),
        "stage": stage,
        "progress": torch.rand(B, W, generator=g),
        "input_ids": torch.randint(0, 1000, (B, L), generator=g),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
    }


def main() -> int:
    cfg = AeroV3Config(train_lora=False)
    torch.manual_seed(0)
    model = AeroV3(cfg).eval()
    print(f"[AeroV3] {model.param_report()}\n", flush=True)

    batch = make_batch(model, cfg)

    # ── shapes ───────────────────────────────────────────────────────────────
    print("shapes")
    out = model(batch)
    check("action (B,W,H,4)", tuple(out["action"].shape) == (B, W, cfg.horizon, 4),
          str(tuple(out["action"].shape)))
    check("z_ep (B,d_task)", tuple(out["z_ep"].shape) == (B, cfg.d_task))
    check("phase (B,W,d_phase)", tuple(out["phase"].shape) == (B, W, cfg.d_phase))
    check("loss finite", torch.isfinite(out["loss"]).item(),
          f"loss={out['loss'].item():.4f}")

    # ── read-out KV order: [z_ep | patch tokens] ─────────────────────────────
    print("\ntoken order")
    seen = {}

    def grab(_mod, args):
        seen["cond"], seen["kv"] = args[0], args[1]

    h = model.readout.register_forward_pre_hook(grab)
    model(batch)
    h.remove()
    kv = seen["kv"]
    n_patch = model.vision.num_patches
    check("kv length == 1 + n_patches", kv.shape[1] == 1 + n_patch,
          f"{kv.shape[1]} vs {1 + n_patch}")
    z_tok = model.task_to_dec(out["z_ep"])
    check("kv[:,0] is the task frame",
          torch.allclose(kv[:, 0], z_tok, atol=1e-5))

    # ── gradient flow ────────────────────────────────────────────────────────
    print("\ngradient flow")
    model.zero_grad(set_to_none=True)
    model(batch)["loss"].backward()

    def gnorm(mod):
        return sum(float(p.grad.abs().sum()) for p in mod.parameters()
                   if p.grad is not None)

    check("frozen vision grad == 0", gnorm(model.vision) == 0.0)
    check("frozen LM grad == 0", gnorm(model.lm) == 0.0)
    for name, mod in (("readout", model.readout), ("phase_cell", model.phase_cell),
                      ("phase_in", model.phase_in), ("stage_head", model.stage_head),
                      ("task_head", model.task_head),
                      ("vis_to_dec", model.vis_to_dec)):
        check(f"{name} grad > 0", gnorm(mod) > 0.0)
    check("phase_init trains", model.phase_init.grad is not None
          and float(model.phase_init.grad.abs().sum()) > 0.0)

    # ── detached recurrence ──────────────────────────────────────────────────
    print("\ndetached recurrence")
    z = model.encode_task(batch["first_pixel_values"], batch["input_ids"],
                          batch["attention_mask"], batch["pose0"]).detach()
    vis = model.encode_vision(batch["pixel_values"][:, 0])
    phi0 = torch.zeros(B, cfg.d_phase, requires_grad=True)
    r = model.step(z, batch["pose"][:, 0], batch["pose0"], vis, phi0)
    r["phase"].sum().backward()
    leaked = phi0.grad is None or float(phi0.grad.abs().sum()) == 0.0
    check("no gradient crosses the step boundary", leaked,
          "" if phi0.grad is None else f"leak={float(phi0.grad.abs().sum()):.2e}")

    # ── task-frame cache equivalence ─────────────────────────────────────────
    print("\ntask-frame cache")
    with torch.no_grad():
        a = model.encode_task(batch["first_pixel_values"], batch["input_ids"],
                              batch["attention_mask"], batch["pose0"])
        b = model.encode_task(batch["first_pixel_values"], batch["input_ids"],
                              batch["attention_mask"], batch["pose0"])
    check("encode_task is bitwise deterministic", torch.equal(a, b))
    with torch.no_grad():
        s1 = model.step(a, batch["pose"][:, 0], batch["pose0"], vis, None)
        s2 = model.step(b, batch["pose"][:, 0], batch["pose0"], vis, None)
    check("cached z_ep gives bitwise identical action",
          torch.equal(s1["action"], s2["action"]))

    # ── vision hold ──────────────────────────────────────────────────────────
    print("\nvision rate")
    held = dict(batch)
    held["pixel_values"] = batch["pixel_values"].clone()
    held["pixel_values"][:, 1] = 999.0     # step 1 has vision_update = 0
    with torch.no_grad():
        o1, o2 = model(batch), model(held)
    check("masked frames do not reach the policy",
          torch.allclose(o1["action"], o2["action"], atol=1e-5))
    held2 = dict(batch)
    held2["pixel_values"] = batch["pixel_values"].clone()
    held2["pixel_values"][:, 0] = 999.0    # step 0 is always fresh
    with torch.no_grad():
        o3 = model(held2)
    check("unmasked frames do reach the policy",
          not torch.allclose(o1["action"], o3["action"], atol=1e-5))

    # ── stage masking ────────────────────────────────────────────────────────
    print("\nstage masking")
    model.zero_grad(set_to_none=True)
    ign = make_batch(model, cfg, all_ignore=True)
    o = model(ign)
    o["loss"].backward()
    check("stage loss is exactly 0 when all labels are -1",
          float(o["loss_stage"]) == 0.0, f"{float(o['loss_stage']):.3e}")
    check("stage head gets no gradient", gnorm(model.stage_head) == 0.0)

    # ── the three ablation arms actually differ ──────────────────────────────
    # The L2 result is a comparison between these three, so a silent leak --
    # "none" still routing the phase, or "oracle" quietly ignoring the label --
    # would not fail any other check and would invalidate the whole table.
    print("\nablation arms")
    for mode, phase_matters, stage_matters in (("none", False, False),
                                               ("filter", True, False),
                                               ("oracle", False, True)):
        torch.manual_seed(0)
        m = AeroV3(AeroV3Config(train_lora=False, phase_mode=mode)).eval()
        b = make_batch(m, m.cfg)
        with torch.no_grad():
            z = m.encode_task(b["first_pixel_values"], b["input_ids"],
                              b["attention_mask"], b["pose0"])
            vis = m.encode_vision(b["pixel_values"][:, 0])
            args = (z, b["pose"][:, 0], b["pose0"], vis)
            phi = torch.randn(B, m.cfg.d_phase)
            a0 = m.step(*args, torch.zeros(B, m.cfg.d_phase),
                        stage=b["stage"][:, 0])["action"]
            a_phi = m.step(*args, phi, stage=b["stage"][:, 0])["action"]
            a_st = m.step(*args, torch.zeros(B, m.cfg.d_phase),
                          stage=(b["stage"][:, 0] + 1))["action"]
        d_phi = float((a_phi - a0).abs().max())
        d_st = float((a_st - a0).abs().max())
        check(f"{mode}: action {'reads' if phase_matters else 'ignores'} "
              f"the estimated phase", (d_phi > 1e-6) == phase_matters,
              f"delta={d_phi:.2e}")
        check(f"{mode}: action {'reads' if stage_matters else 'ignores'} "
              f"the stage label", (d_st > 1e-6) == stage_matters,
              f"delta={d_st:.2e}")

    # ── numerics ─────────────────────────────────────────────────────────────
    print("\nnumerics")
    model.zero_grad(set_to_none=True)
    o = model(batch)
    o["loss"].backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("all gradients finite", not bad, ",".join(bad[:3]))

    print()
    if FAILED:
        print(f"AEROV3_WIRING: FAIL ({len(FAILED)}): {', '.join(FAILED)}")
        return 1
    print("AEROV3_WIRING: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
