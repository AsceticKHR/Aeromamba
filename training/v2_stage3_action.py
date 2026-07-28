"""Stage 3 (AeroV2): UAV-Flow action-chunk regression on top of the S2 VLM.

The v2 pipeline's Stage 3 is intentionally *not* the legacy
``training/stage3_action.py`` (which is wired to the v1 ``AeroMambaVLA``). Here
we extend the frozen S2 ``AeroV2`` (C-RADIO vision + Falcon-H1 + S0/S2 LoRA +
grounding head) with a ProprioEncoder + UAVActionHead and train ONLY those two
modules (optionally the LoRA) on UAV-Flow trajectories.

Token layout for the policy forward:  [vision | text | proprio] -> Falcon-H1
trunk -> terminal hidden state (RoboMamba global-token) -> K x 4 waypoints
(dx, dy, dz, dyaw_rad), regressed in per-(k,dim) z-score space.

Modes
-----
--mode smoke : rigorous 3-gate self-check before any long run
    G1 OVERFIT   : memorise one fixed batch (gradient path + head capacity)
    G2 GRAD-FLOW : only proprio_encoder + action_head (+LoRA) receive gradient;
                   vision / projector / base-LM stay exactly frozen
    G3 REAL      : a short real-data run — val pos_err_m / yaw_err_deg must fall
                   and predictions must NOT collapse to the batch mean
--mode full  : standard training with periodic validation + best/last save

Usage (smoke):
  python training/v2_stage3_action.py --mode smoke \
    --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \
    --vision_type cradio_v3_b \
    --data_root /root/autodl-tmp/datasets/stage3_uavflow \
    --action_stats /root/autodl-tmp/datasets/uav-flow/action_stats_k8.json \
    --chunk_size 8 --batch 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import UAVFlowDataset, aero_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# S2 checkpoint loading (projector + grounding heads + vision + LoRA)
# ─────────────────────────────────────────────────────────────────────────────
def load_s2(model: AeroV2, ckpt_dir: Path, tag: str = "best") -> dict:
    from peft import PeftModel

    payload = torch.load(ckpt_dir / f"{tag}_projector.pth", map_location="cpu",
                         weights_only=False)
    model.projector.load_state_dict(payload["projector"], strict=True)
    model.grd_queries.data.copy_(payload["grd_queries"].to(model.grd_queries.device))
    for name in ("grd_ln_kv", "grd_ln_q", "grd_ln_out", "grd_txt_proj",
                 "grd_attn", "grd_head"):
        getattr(model, name).load_state_dict(payload[name], strict=True)

    vis = torch.load(ckpt_dir / f"{tag}_vision.pth", map_location="cpu",
                     weights_only=False)
    model.vision_encoder.load_state_dict(vis, strict=True)

    base = model.lm
    for _ in range(3):
        if not hasattr(base, "get_base_model"):
            break
        try:
            nxt = base.get_base_model()
        except Exception:
            break
        if nxt is base:
            break
        base = nxt
    lora_dir = payload.get("lora_dir") or str(ckpt_dir / f"{tag}_lora")
    model.lm = PeftModel.from_pretrained(base, lora_dir, is_trainable=False)
    return {"step": payload.get("step"), "val": payload.get("val")}


def load_action_stats(model: AeroV2, path: str, chunk_size: int,
                      pos_scale: float, norm_mode: str = "zscore",
                      chunk_offset: int = 0) -> None:
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(stats["chunk_size"]) != int(chunk_size):
        raise ValueError(f"action_stats chunk_size={stats['chunk_size']} != "
                         f"--chunk_size={chunk_size}")
    if float(stats.get("pos_scale", 100.0)) != float(pos_scale):
        raise ValueError(f"action_stats pos_scale={stats.get('pos_scale')} != "
                         f"--pos_scale={pos_scale}")
    # A stats file computed with a different chunk_offset describes a different
    # target distribution; silently mixing them would mis-scale every waypoint.
    if int(stats.get("chunk_offset", 0)) != int(chunk_offset):
        raise ValueError(
            f"action_stats chunk_offset={stats.get('chunk_offset', 0)} != "
            f"--chunk_offset={chunk_offset}; recompute with "
            f"data/compute_action_stats.py --chunk_offset {chunk_offset}")
    if norm_mode == "quantile":
        if "q01" not in stats:
            raise ValueError(
                f"{path} has no q01/q99 (recompute with the current "
                "data/compute_action_stats.py) — required by --norm_mode quantile")
        model.action_head.set_quantile_normalization(stats["q01"], stats["q99"])
    else:
        model.action_head.set_normalization(stats["mean"], stats["std"])
    span = model.action_head.action_std
    print(f"[S3] action stats loaded ({norm_mode}, "
          f"samples={stats.get('num_samples')}, "
          f"turn_frac={stats.get('turn_fraction', 0):.3f}, "
          f"scale=[{span.min():.4f},{span.max():.4f}])")


# ─────────────────────────────────────────────────────────────────────────────
# batch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _prep(batch, device, proprio_dim: int = 4):
    # state8 = [pose4 | velocity4]; slice to pose-only (proprio_dim=4) to DROP the
    # velocity channels that the v1 head shortcut-extrapolated (see enable_action_head).
    proprio = batch["state8"].to(device).float()[:, :proprio_dim]
    return {
        "pixel_values": batch["pixel_values"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "proprio": proprio,
        "gt_action": batch["gt_action"].to(device).float(),
    }


def _channel_weights(args, device):
    if args.channel_weight_z == 1.0 and args.channel_weight_yaw == 1.0:
        return None
    return torch.tensor([1.0, 1.0, args.channel_weight_z, args.channel_weight_yaw],
                        device=device)


def _fwd(model, b, args, device):
    return model.forward_action(
        b["pixel_values"], b["input_ids"], b["proprio"], gt_action=b["gt_action"],
        lambda_smooth=args.lambda_smooth, lambda_endpoint=args.lambda_endpoint,
        lambda_direction=args.lambda_direction, lambda_acc=args.lambda_acc,
        lambda_var=args.lambda_var, var_floor=getattr(args, "var_floor", 0.3),
        channel_weights=_channel_weights(args, device))


@torch.no_grad()
def validate(model, loader, args, device, max_steps=50):
    model.eval()
    agg = {"loss": 0.0, "pos_err_m": 0.0, "yaw_err_deg": 0.0,
           "end_pos_err_m": 0.0, "direction": 0.0}
    n = 0
    act_all = []
    for i, batch in enumerate(loader):
        if i >= max_steps:
            break
        b = _prep(batch, device, args.proprio_dim)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _fwd(model, b, args, device)
        d = out["loss_detail"]
        agg["loss"] += float(out["loss"])
        for k in ("pos_err_m", "yaw_err_deg", "end_pos_err_m", "direction"):
            agg[k] += float(d[k])
        # denormalised physical action for collapse diagnostics
        act_all.append(model.action_head.denormalize(out["action"].float()).cpu())
        n += 1
    for k in agg:
        agg[k] /= max(n, 1)
    if act_all:
        acts = torch.cat(act_all, dim=0)               # [N, K, 4]
        agg["act_std"] = float(acts.std(dim=0).mean())  # variation across samples
    model.train()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# QC gates (smoke)
# ─────────────────────────────────────────────────────────────────────────────
def gate_overfit(model, batch, args, device, steps=120):
    print("\n=== G1 OVERFIT (memorise one fixed batch) ===", flush=True)
    # Variance floor fights single-batch memorisation (it rewards cross-sample
    # spread, not fitting each row). Probe capacity with pure supervised loss.
    var_save = args.lambda_var
    args.lambda_var = 0.0
    b = _prep(batch, device, args.proprio_dim)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.train()
    first = last = None
    for s in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _fwd(model, b, args, device)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        d = out["loss_detail"]
        if s == 0:
            first = (float(out["loss"]), d["pos_err_m"], d["yaw_err_deg"])
        if s % 20 == 0 or s == steps - 1:
            print(f"  step {s:3d} loss={float(out['loss']):.4f} "
                  f"pos_err_m={d['pos_err_m']:.4f} yaw_err_deg={d['yaw_err_deg']:.3f}",
                  flush=True)
        last = (float(out["loss"]), d["pos_err_m"], d["yaw_err_deg"])
    args.lambda_var = var_save
    ok = last[1] < 0.4 * first[1] and last[0] < 0.6 * first[0]
    print(f"  [G1] pos_err_m {first[1]:.4f}->{last[1]:.4f}, "
          f"loss {first[0]:.4f}->{last[0]:.4f}  => {'PASS' if ok else 'FAIL'}")
    return ok


def gate_grad_flow(model, batch, args, device):
    print("\n=== G2 GRAD-FLOW (only policy modules learn) ===", flush=True)
    b = _prep(batch, device, args.proprio_dim)
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = _fwd(model, b, args, device)
    out["loss"].backward()

    def gnorm(mod):
        tot = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().float().norm()) ** 2
        return tot ** 0.5

    g_head = gnorm(model.action_head)
    g_prop = gnorm(model.proprio_encoder)
    g_proj = gnorm(model.projector)
    g_vis = gnorm(model.vision_encoder)
    lora_g = base_g = 0.0
    for n, p in model.lm.named_parameters():
        if p.grad is None:
            continue
        gn = float(p.grad.detach().float().norm()) ** 2
        if "lora_" in n:
            lora_g += gn
        else:
            base_g += gn
    lora_g, base_g = lora_g ** 0.5, base_g ** 0.5
    print(f"  grad-norm  action_head={g_head:.4e} proprio={g_prop:.4e}")
    print(f"             projector={g_proj:.4e} vision={g_vis:.4e} "
          f"lm_lora={lora_g:.4e} lm_base={base_g:.4e}")
    # action_head learning + backbone frozen are the hard checks. proprio grad
    # is now only via the trunk (residual off) and proprio is auxiliary, so a
    # tiny/zero proprio grad is acceptable (reported, not gated).
    learns = g_head > 0
    frozen = g_proj == 0 and g_vis == 0 and base_g == 0
    lora_ok = (lora_g > 0) if args.train_lora else (lora_g == 0)
    ok = learns and frozen and lora_ok
    print(f"  [G2] policy learns(head)={learns} proprio_grad={g_prop:.2e} "
          f"backbone frozen={frozen} "
          f"lora={'on' if args.train_lora else 'off'}:{lora_ok} "
          f"=> {'PASS' if ok else 'FAIL'}")
    model.zero_grad(set_to_none=True)
    return ok


def build_probe_batches(va_loader, n: int = 8, groups: int = 1,
                        max_batches: int = 400):
    """Probe batches for the grounding self-check, each with n DISTINCT
    instructions.

    grounding_metrics swaps instructions by rolling the batch, so every row must
    carry a different one. Validation is ordered by trajectory and a trajectory
    has exactly one instruction, so plain next(iter(va_loader)) yields rows that
    all share it and the swap becomes a no-op.
    """
    picked: dict = {}
    want = n * groups
    for bi, batch in enumerate(va_loader):
        if bi >= max_batches or len(picked) >= want:
            break
        ids = batch["input_ids"]
        for r in range(ids.shape[0]):
            key = ids[r].cpu().numpy().tobytes()
            if key not in picked:
                picked[key] = {k: v[r] for k, v in batch.items()
                               if torch.is_tensor(v)}
            if len(picked) >= want:
                break
    rows = list(picked.values())
    if len(rows) < 2:
        raise RuntimeError(
            f"grounding probe needs >=2 distinct instructions, found "
            f"{len(rows)} in {max_batches} val batches")
    out = []
    for g in range(0, len(rows), n):
        grp = rows[g:g + n]
        if len(grp) >= 2:
            out.append({k: torch.stack([r[k] for r in grp]) for k in grp[0]})
    return out


def build_probe_batch(va_loader, n: int = 8, max_batches: int = 400):
    b = build_probe_batches(va_loader, n=n, groups=1, max_batches=max_batches)[0]
    print(f"[S3] grounding probe batch: {b['input_ids'].shape[0]} distinct "
          f"instructions", flush=True)
    return b


@torch.no_grad()
def instruction_only_spread(model, batch, args, device):
    """One image, every distinct instruction in the batch.

    The spread of the predicted endpoints is language sensitivity with vision
    held constant; the gt spread is what the data says it should be. A policy
    that ignores language returns ~0 here however it scores on the swap probe.
    """
    was_training = model.training
    model.eval()
    pv = batch["pixel_values"].to(device)
    ids = batch["input_ids"].to(device)
    proprio = batch["state8"][:, :args.proprio_dim].to(device).float()
    fixed = pv[:1].expand(pv.shape[0], *pv.shape[1:]).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward_action(fixed, ids, proprio)
    ep = model.action_head.denormalize(out["action"].float())[:, -1]
    gt = batch["gt_action"][:, -1].to(device).float()
    if was_training:
        model.train()
    return float(ep.std(0).norm()), float(gt.std(0).norm())


@torch.no_grad()
def grounding_metrics(model, batch, args, device):
    """Physical endpoint sensitivity of the policy to instruction / vision /
    pose. Returns (instr_sens, vis_sens, prop_sens, gt_spread), mixed m + rad.

      instr_sens : endpoint move when the instruction is swapped (grounding)
      vis_sens   : endpoint move when the image is swapped (visual reliance)
      prop_sens  : a +3 m pose perturbation (must NOT dominate — gt_action is
                   anchor-relative body-frame, so absolute pose is irrelevant)
    """
    was_training = model.training
    model.eval()
    # In-distribution counterfactuals, swapped WITHIN the batch: the other
    # sample's real instruction, or the other sample's real image.
    #
    # The previous probe compared against four synthetic sentences ("Turn
    # left.") and an all-zero image. Both are out of distribution, and the
    # measurement degraded as the policy specialised on real instructions —
    # instr_sens read 1.03 at step 300 and 0.024 at step 600 of the same run.
    # An architecture decision cannot rest on that.
    n = min(8, batch["pixel_values"].shape[0])
    pv = batch["pixel_values"][:n].to(device)
    ids = batch["input_ids"][:n].to(device)
    proprio0 = batch["state8"][:n, :args.proprio_dim].to(device).float()

    def endpoint(pixel, instr_ids, proprio):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_action(pixel, instr_ids, proprio)
        return model.action_head.denormalize(out["action"].float())[:, -1]  # [n, 4]

    base = endpoint(pv, ids, proprio0)
    # Average the instruction swap only over rows where the instruction really
    # changed. A batch drawn from one trajectory shares one instruction, so the
    # roll is a no-op and instr_sens reads a structural 0.0000 that is
    # indistinguishable from a language-blind policy. NaN instead, so a bad
    # probe batch is loud rather than a silent pass.
    swapped = ids.roll(1, 0)
    changed = (swapped != ids).any(dim=-1)
    if bool(changed.any()):
        instr_sens = float((base - endpoint(pv, swapped, proprio0))
                           .norm(dim=-1)[changed].mean())
    else:
        instr_sens = float("nan")
    vis_sens = float((base - endpoint(pv.roll(1, 0), ids, proprio0))
                     .norm(dim=-1).mean())
    prop_pert = proprio0.clone()
    prop_pert[:, :3] += 3.0
    prop_sens = float((base - endpoint(pv, ids, prop_pert)).norm(dim=-1).mean())
    # Data-side reference: how far apart the ground-truth endpoints of those
    # same two samples actually are. Gives the sensitivities a physical scale.
    gt_ep = batch["gt_action"][:n, -1].to(device).float()
    gt_spread = float((gt_ep - gt_ep.roll(1, 0)).norm(dim=-1).mean())
    if was_training:
        model.train()
    return instr_sens, vis_sens, prop_sens, gt_spread


def gate_grounding(model, batch, args, device):
    """G4: does the policy GROUND instruction + vision, or shortcut proprio?

    THE gate that the v1 smoke lacked (v1 passed G1-G3 yet collapsed to a
    proprio-velocity copy that ignored language). Instruction/vision must steer
    the endpoint and proprio must not dominate.
    """
    i_s, v_s, p_s, gt_spread = grounding_metrics(model, batch, args, device)
    resp = (i_s + v_s) / max(gt_spread, 1e-9)
    share = v_s / max(i_s + v_s, 1e-9)
    ok_resp = resp > args.resp_min
    ok_vis = is_vision_grounded(i_s, v_s, args)
    ok_dom = i_s >= p_s
    ok = ok_resp and ok_vis and ok_dom
    print(f"  [G4] instr={i_s:.4f} vis={v_s:.4f} prop={p_s:.4f} "
          f"gt_spread={gt_spread:.4f}")
    print(f"       responsiveness={resp:.3f} (>{args.resp_min}) "
          f"vis_share={share:.3f} (>{args.vis_share_min}) instr>=prop={ok_dom} "
          f"=> {'PASS' if ok else 'FAIL'}")
    if not ok_vis and hasattr(model.action_head, "readout"):
        # A failed vision gate is not actionable on its own — report which of
        # the two failure mechanisms produced it.
        print("  [G4] readout internals:")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model._encode_multimodal(
                batch["pixel_values"][:4].to(device),
                batch["input_ids"][:4].to(device), append_grd=False)
            am = torch.ones(enc["inputs_embeds"].shape[:2], dtype=torch.long,
                            device=device)
            hid = model._trunk_hidden(enc["inputs_embeds"], am)
        n_vis = enc["n_vis"]
        hd = next(model.action_head.parameters()).dtype
        ro = model.action_head.readout
        kv = torch.cat([enc["inputs_embeds"][:, :n_vis], hid[:, n_vis:]], dim=1)
        rec = ro.diagnose(kv.to(hd), n_vis)
        print(ro.format_diagnosis(rec))
    model.train()
    return ok


def gate_visual_overfit(model, batch, args, device, opt_factory, steps):
    """G1v: CAN vision reach the action at all? (capacity, not training length)

    Every sample in the batch is given the SAME instruction while keeping its
    own image and its own target. The instruction then carries zero information
    about which sample is which, so the only way to beat the best constant
    prediction is to read the images.

    This separates the two explanations for a low vis_share that the G4 number
    alone cannot: if the model still cannot spread its predictions here, vision
    is structurally unable to reach the output and the architecture is wrong;
    if it can, the readout is fine and the shortfall is budget or optimisation.
    """
    b = _prep(batch, device, args.proprio_dim)
    b["input_ids"] = b["input_ids"][:1].expand_as(b["input_ids"]).contiguous()
    gt_ep = b["gt_action"][:, -1].float()
    gt_spread = float((gt_ep - gt_ep.mean(0, keepdim=True)).norm(dim=-1).mean())

    opt = opt_factory()
    model.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _fwd(model, b, args, device)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.action_head.denormalize(
            _fwd(model, b, args, device)["action"].float())[:, -1]
    pred_spread = float((pred - pred.mean(0, keepdim=True)).norm(dim=-1).mean())
    ratio = pred_spread / max(gt_spread, 1e-9)
    ok = ratio > args.vis_overfit_min
    print(f"  [G1v] constant instruction, {steps} steps: "
          f"pred_spread={pred_spread:.4f} gt_spread={gt_spread:.4f} "
          f"ratio={ratio:.3f} (>{args.vis_overfit_min})")
    verdict = ("PASS - vision CAN drive the action, so a low vis_share is "
               "budget or optimisation" if ok else
               "FAIL - vision cannot reach the output; this is architectural")
    print(f"        => {verdict}")
    model.train()
    return ok


def is_vision_grounded(instr_sens: float, vis_sens: float, args) -> bool:
    """Vision's SHARE of the policy's counterfactual response.

    v4 cleared an absolute floor (vis_sens 0.07 > 0.02) and still flew a
    vision-blind instruction template in closed loop: an absolute threshold
    cannot separate "uses the image" from "barely perturbed by it", because the
    displacement scale is task-dependent.

    The default share is not a guess. `scripts/probe_action_information.py`
    decomposes the ground-truth endpoint variance at K=8 into
    instruction 43% / scene 26% / phase 32%, so a policy that used the current
    frame as fully as the data allows would show a vision share of about
    26/(26+43) = 0.38. 0.35 asks for essentially that, and no more.
    """
    return vis_sens >= args.vis_share_min * max(instr_sens + vis_sens, 1e-9)


@torch.no_grad()
def channel_std_metrics(model, loader, args, device, max_steps=40):
    """Per-channel endpoint pred-std / gt-std, the direct collapse read-out.

    A head that has regressed to the conditional median has near-zero spread on
    the small channels (measured: dz ratio 0.124 overall, 0.032 on vertical
    samples) while still scoring a respectable mean error.
    """
    was_training = model.training
    model.eval()
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if i >= max_steps:
            break
        b = _prep(batch, device, args.proprio_dim)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _fwd(model, b, args, device)
        preds.append(model.action_head.denormalize(out["action"].float())[:, -1].cpu())
        gts.append(b["gt_action"][:, -1].cpu())
    if was_training:
        model.train()
    if not preds:
        return {}
    p, g = torch.cat(preds), torch.cat(gts)
    return {name: round(float(p[:, d].std() / g[:, d].std().clamp_min(1e-8)), 4)
            for d, name in enumerate(("dx", "dy", "dz", "dyaw"))}


def gate_channel_collapse(model, loader, args, device):
    """G5: the small channels must retain input-dependent spread."""
    ratios = channel_std_metrics(model, loader, args, device, args.max_val_steps)
    # dx is the dominant sim DOF and must stay alive; dyaw is the heading DOF
    # closed-loop cares about. dz is scarce in UAV-Flow-Sim after metre
    # conversion (prior healthy L1 smoke: dz=0.007 while dx=0.99, dyaw=0.08),
    # so requiring dz>threshold falsely fails a non-collapsed policy.
    ok_dx = ratios.get("dx", 0.0) > 0.25
    ok_yaw = ratios.get("dyaw", 0.0) > args.channel_std_min
    ok = ok_dx and ok_yaw
    print(f"  [G5] endpoint pred/gt std ratio {ratios} "
          f"(dx>0.25 & dyaw>{args.channel_std_min}; dz informational) "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(model, tr_loader, va_loader, args, device, save_dir: Path):
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"[S3] trainable params: {n_tr/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = args.max_steps or (len(tr_loader) * args.epochs)
    # Linear warmup -> cosine. A high LR from step 0 slammed the head into the
    # input-independent constant basin (act_std->0) in the first full run.
    # sched_total_steps decouples the cosine horizon from the run length so a
    # SHORT smoke can reproduce the FULL run's LR (which barely decays early) —
    # otherwise the smoke's fast decay hid the pose-routing degenerate that only
    # emerged under the full run's sustained high LR.
    sched_total = getattr(args, "sched_total_steps", 0) or total_steps
    warmup = min(getattr(args, "warmup_steps", 0), max(1, sched_total // 10))
    if warmup > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, sched_total - warmup)),
            ],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sched_total)
    save_dir.mkdir(parents=True, exist_ok=True)

    best = math.inf
    best_grnd = math.inf                      # best pos_err among GROUNDED checkpoints
    best_resp = -1.0                          # peak counterfactual responsiveness
    best_cal = math.inf                       # best pos_err x grounding miscalibration
    step = 0
    # Resume continues the LR schedule from where the interrupted run stopped.
    # Adam moments are NOT in the checkpoint payload, so they restart cold; over
    # a tail of a few thousand steps at a decayed LR that is a small
    # perturbation, but it is not a bit-exact continuation.
    if getattr(args, "resume", None):
        rstep, rval = _load_policy(model, Path(args.resume))
        step = int(rstep or 0)
        for _ in range(step):
            sched.step()
        print(f"[S3] resumed {args.resume} at step={step} val={rval}",
              flush=True)
    t0 = time.time()
    log = []
    probe_batch = build_probe_batch(va_loader)   # fixed, instruction-diverse
    gnd_bad = 0                              # consecutive grounding-collapse checks
    aborted = False
    model.train()
    for ep in range(args.epochs):
        for batch in tr_loader:
            b = _prep(batch, device, args.proprio_dim)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = _fwd(model, b, args, device)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                d = out["loss_detail"]
                dt = (time.time() - t0) / step
                var_s = f" var={d['var']:.4f}" if "var" in d else ""
                print(f"  [ep{ep} {step}/{total_steps}] loss={float(out['loss']):.4f} "
                      f"main={d['main']:.4f} end={d['endpoint']:.4f} "
                      f"dir={d['direction']:.4f} pos_err_m={d['pos_err_m']:.4f} "
                      f"yaw_err_deg={d['yaw_err_deg']:.3f}{var_s} ({dt:.2f}s/it)",
                      flush=True)
            if step % args.val_every == 0 or step == total_steps:
                va = validate(model, va_loader, args, device, args.max_val_steps)
                # periodic STRICT grounding self-check (VLA acceptance during
                # training): instruction must steer the output and dominate the
                # (irrelevant) absolute-pose perturbation, and the policy must
                # not collapse. Abort early on sustained collapse to not burn
                # full-run compute on a degenerate policy.
                i_s, v_s, p_s, gt_sp = grounding_metrics(model, probe_batch, args, device)
                resp = (i_s + v_s) / max(gt_sp, 1e-9)
                # Two DIFFERENT questions, deliberately not one predicate:
                #
                #   degenerate — the policy stopped being a policy: constant
                #                output, or steered by absolute pose instead of
                #                the task. Only this may abort the run.
                #   grounded   — additionally meets the vision-share target.
                #                Selects the deployable checkpoint.
                #
                # Conflating them aborts a healthy run that simply has not yet
                # learned to use the image: vis_share was measured rising
                # 0.01 -> 0.20 over 1650 steps, i.e. below target for most of a
                # perfectly good run. Measured again on sim L1: act_std=0.17
                # with resp=0.04 was aborted at step 500 — that is "low-vis",
                # not collapse. Only treat low resp as degenerate when the
                # action distribution itself has also died.
                collapsed = va.get("act_std", 0) <= 1e-3
                pose_shortcut = i_s + 1e-8 < p_s
                dead_resp = (resp < args.collapse_resp
                             and va.get("act_std", 0) <= 1e-2)
                # Abort only on true collapse / pose shortcut. Early mean-fit
                # with low act_std is common before lambda_var bites; do not
                # poison best_responsive selection with that label.
                abort_deg = collapsed or pose_shortcut or dead_resp
                degenerate = collapsed or pose_shortcut
                grounded = (not degenerate) and is_vision_grounded(i_s, v_s, args)
                gnd_bad = gnd_bad + 1 if abort_deg else 0
                print(f"  [VAL step{step}] loss={va['loss']:.4f} "
                      f"pos_err_m={va['pos_err_m']:.4f} yaw_err_deg={va['yaw_err_deg']:.3f} "
                      f"end_pos_err_m={va['end_pos_err_m']:.4f} "
                      f"act_std={va.get('act_std', 0):.4f} | "
                      f"GND instr={i_s:.4f} vis={v_s:.4f} prop={p_s:.4f} "
                      f"resp={resp:.3f} "
                      f"vis_share={v_s / max(i_s + v_s, 1e-9):.3f} "
                      f"{'ok' if grounded else ('DEGENERATE' if abort_deg else 'low-vis')}",
                      flush=True)
                log.append({"step": step, **va, "instr_sens": i_s,
                            "vis_sens": v_s, "prop_sens": p_s,
                            "gt_spread": gt_sp, "resp": resp})
                if va["loss"] < best:
                    best = va["loss"]
                    _save(model, save_dir / "best.pth", step, va, args)
                # Peak counterfactual response among alive policies. Grounding
                # often peaks mid-run then erodes (sim L1: resp~0.10 @300 then
                # ~0.006 @800); G4/deploy must use this, not the collapsed last.
                if ((not collapsed) and va.get("act_std", 0) > 5e-3
                        and resp > best_resp):
                    best_resp = resp
                    _save(model, save_dir / "best_responsive.pth", step,
                          {**va, "instr_sens": i_s, "vis_sens": v_s,
                           "resp": resp}, args)
                # best-GROUNDED: lowest pos_err among steps that still respond
                # to vision/instruction. The earlier "sim routes via vision
                # alone (instr~0)" reading was wrong — that 0 came from a
                # single-trajectory probe batch, not from the policy.
                if (not degenerate and resp >= 0.05
                        and va["pos_err_m"] < best_grnd):
                    best_grnd = va["pos_err_m"]
                    _save(model, save_dir / "best_grounded.pth", step,
                          {**va, "instr_sens": i_s, "vis_sens": v_s,
                           "resp": resp}, args)
                # best-CALIBRATED: val loss on its own selects the most
                # vision-blind policy. The instruction already explains ~78% of
                # the endpoint variance on this data (measured: across=1.53 vs
                # within=0.42 per instruction group), so fitting the visual
                # residual costs pos_err while being exactly what closed-loop
                # stopping needs. Weight pos_err by the distance between the
                # policy's vision share and the share the data actually
                # requires. Verified to reproduce the manual pick: step 14000
                # scores 0.274 vs step 1000's 0.396.
                if args.vis_share_target > 0 and not degenerate:
                    share = v_s / max(i_s + v_s, 1e-9)
                    miscal = (abs(share - args.vis_share_target)
                              / args.vis_share_target)
                    score = va["pos_err_m"] * (1.0 + miscal)
                    if score < best_cal:
                        best_cal = score
                        _save(model, save_dir / "best_calibrated.pth", step,
                              {**va, "instr_sens": i_s, "vis_sens": v_s,
                               "vis_share": share, "cal_score": score}, args)
                _save(model, save_dir / "last.pth", step, va, args)
                (save_dir / "train_log.json").write_text(json.dumps(log, indent=2))
                # Warmup grace: early steps often mean-fit with low act_std before
                # lambda_var diversifies (sim L1: std can sit <1e-2 through ~200 then
                # rise to ~0.2). abort_patience=2 + 15% grace previously killed a
                # healthy recipe at step 200. Require 35% of the run AND enough
                # consecutive dead vals before aborting.
                abort_after = max(int(0.35 * total_steps),
                                  int(getattr(args, "warmup_steps", 0) or 0))
                if gnd_bad >= args.abort_patience and step > abort_after:
                    print(f"GROUNDING_COLLAPSE_ABORT step={step} "
                          f"(instr={i_s:.4f} prop={p_s:.4f} resp={resp:.3f} "
                          f"act_std={va.get('act_std',0):.4f}) "
                          f"— policy degenerated; stopping to save compute.", flush=True)
                    aborted = True
                    break
            if step >= total_steps:
                break
        if aborted or step >= total_steps:
            break
    # Prefer the most input-sensitive alive weights over a late collapsed last.
    # Smoke G3 (mean fit) then reloads best.pth / best_grounded itself; full runs
    # want the responsive checkpoint for deploy.
    for tag in ("best_responsive.pth", "best_grounded.pth", "best.pth"):
        ck = save_dir / tag
        if ck.is_file():
            print(f"[S3] restoring {tag} for downstream gates/eval "
                  f"(best_resp={best_resp:.3f})", flush=True)
            _load_policy(model, ck)
            break
    print(f"[S3] done. best val loss={best:.4f} best_resp={best_resp:.3f}"
          + (" (ABORTED)" if aborted else ""))
    return best


def _save(model, path, step, val, args):
    payload = {
        "proprio_encoder": model.proprio_encoder.state_dict(),
        "action_head": model.action_head.state_dict(),
        "step": step, "val": val, "chunk_size": args.chunk_size,
        # Head geometry + target scaling, so eval can rebuild an identical
        # policy instead of guessing (a mismatched readout or norm_mode loads
        # "successfully" and silently reports wrong physical units).
        "policy_cfg": {
            "readout": args.readout, "n_bins": args.n_bins,
            "bin_range": args.bin_range, "readout_layers": args.readout_layers,
            "norm_mode": args.norm_mode, "chunk_offset": args.chunk_offset,
            "no_proprio": bool(args.no_proprio),
            "pos_unit": getattr(args, "pos_unit", "auto"),
            "split_by": args.split_by, "val_frac": args.val_frac,
            "seed": args.seed,
        },
        # The exact held-out trajectories, so evaluation reproduces the split
        # by identity rather than by replaying the same RNG.
        "val_traj_files": getattr(args, "_val_traj_files", None),
    }
    if getattr(model, "use_grounding_target", False):
        payload["target_encoder"] = model.target_encoder.state_dict()
    if hasattr(model, "action_query"):
        payload["action_query"] = model.action_query.detach().cpu()
    if args.train_lora:
        payload["lora"] = {n: p.detach().cpu()
                           for n, p in model.lm.named_parameters() if "lora_" in n}
    torch.save(payload, path)


def _load_policy(model, path):
    """Restore proprio / action_head / LoRA / action_query from an S3 ckpt."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.action_head.load_state_dict(payload["action_head"], strict=True)
    if "target_encoder" in payload and hasattr(model, "target_encoder"):
        model.target_encoder.load_state_dict(payload["target_encoder"], strict=True)
    if "action_query" in payload and hasattr(model, "action_query"):
        model.action_query.data.copy_(payload["action_query"].to(
            model.action_query.device, dtype=model.action_query.dtype))
    if "lora" in payload:
        with torch.no_grad():
            name2p = dict(model.lm.named_parameters())
            for n, t in payload["lora"].items():
                if n in name2p:
                    name2p[n].copy_(t.to(name2p[n].device, dtype=name2p[n].dtype))
    return payload.get("step"), payload.get("val")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "grounding_eval"],
                    default="smoke")
    ap.add_argument("--probe_groups", type=int, default=8,
                    help="grounding_eval: independent probe batches to average "
                         "over, so a verdict does not rest on one batch of 8")
    ap.add_argument("--ckpt_dir", default="checkpoints/v2_stage2_full_cradio")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--vision_type", default="cradio_v3_b")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--data_root",
                    default="/root/autodl-tmp/datasets/stage3_uavflow")
    ap.add_argument("--action_stats",
                    default="/root/autodl-tmp/datasets/uav-flow/action_stats_k8.json")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--proprio_dim", type=int, default=4,
                    help="4 = pose-only [x,y,z,yaw]; velocity channels dropped to "
                         "kill the v1 velocity-copy shortcut")
    ap.add_argument("--proprio_dropout", type=float, default=0.5,
                    help="per-sample prob of zeroing the proprio token (forces "
                         "instruction+vision grounding, classifier-free style)")
    ap.add_argument("--no_proprio", action="store_true",
                    help="PROPRIO-FREE: zero the pose token in train+eval. pose is "
                         "a pure shortcut (anchor-relative gt_action, pose redundant "
                         "with the FPV image); at full-run LR the LoRA re-routed pose "
                         "and suppressed instruction (instr 0.004 << prop 0.315).")
    ap.add_argument("--grounding_min", type=float, default=0.02,
                    help="G4 min instruction/vision endpoint sensitivity (m+rad)")
    ap.add_argument("--vis_share_min", type=float, default=0.35,
                    help="G4: vision's share of the counterfactual response. "
                         "Calibrated against the data — the GT endpoint "
                         "variance at K=8 splits instr 43%% / scene 26%% / "
                         "phase 32%%, so a fully scene-aware policy shows "
                         "about 0.38 (see probe_action_information.py).")
    ap.add_argument("--vis_overfit_min", type=float, default=0.5,
                    help="G1v: with the instruction held constant, the spread "
                         "of predicted endpoints must reach this fraction of "
                         "the ground-truth spread")
    ap.add_argument("--collapse_resp", type=float, default=0.1,
                    help="abort-only responsiveness floor. Much looser than "
                         "--resp_min: this detects a DEAD policy, it is not an "
                         "acceptance criterion.")
    ap.add_argument("--abort_patience", type=int, default=5,
                    help="consecutive degenerate validations before aborting")
    ap.add_argument("--resp_min", type=float, default=0.3,
                    help="G4: (instr_sens + vis_sens) / gt endpoint spread. "
                         "Catches a policy that is stable but inert.")
    ap.add_argument("--channel_std_min", type=float, default=0.25,
                    help="G5: min endpoint pred/gt std ratio for dz and dyaw")
    ap.add_argument("--readout", choices=["last", "xattn"], default="last",
                    help="last  = hidden[:,-1,:] -> MLP (v4 baseline); "
                         "xattn = K action queries cross-attending the "
                         "projector's 2D patches (v5, fixes vis_sens)")
    ap.add_argument("--n_bins", type=int, default=0,
                    help="xattn only. >0 enables the HL-Gauss distributional "
                         "output (bounded by construction, cannot collapse to "
                         "the conditional median). 0 = plain L1 from the same "
                         "readout, i.e. the ablation that isolates --readout.")
    ap.add_argument("--bin_range", type=float, default=1.5,
                    help="HL-Gauss bin span in normalised units; with "
                         "--norm_mode quantile, +-1 is the q01/q99 range")
    ap.add_argument("--readout_layers", type=int, default=2)
    ap.add_argument("--norm_mode", choices=["zscore", "quantile"], default="zscore",
                    help="quantile maps q01/q99 to [-1,1] (pi0 / FAST / LeRobot "
                         "default; robust to the forward-flight heavy tail)")
    ap.add_argument("--chunk_offset", type=int, default=0,
                    help="1 drops the anchor waypoint, which is identically "
                         "zero for anchor-relative targets")
    ap.add_argument("--pos_scale", type=float, default=100.0)
    ap.add_argument("--pos_unit", default="auto", choices=["auto", "m", "cm"],
                    help="Unit of preprocessed_logs xyz (auto-detects real=m vs sim=cm).")
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--resume", default=None,
                    help="S3 checkpoint to continue from; restores the policy "
                         "and fast-forwards the LR schedule (Adam state is not "
                         "in the payload and restarts cold)")
    ap.add_argument("--vis_share_target", type=float, default=0.0,
                    help="vision share the DATA requires, i.e. within- over "
                         "(within+across)-instruction endpoint spread. 0.215 "
                         "on UAV-Flow-Sim (grounding_eval prints it). >0 "
                         "enables best_calibrated.pth selection.")
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--max_val_steps", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.03)
    ap.add_argument("--split_by", choices=["trajectory", "chunk"],
                    default="trajectory",
                    help="'trajectory' holds out whole trajectories (the only "
                         "split that measures generalisation). 'chunk' is the "
                         "legacy leaky split, kept only to reproduce old runs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_lora", action="store_true")
    ap.add_argument("--use_grounding_target", action="store_true",
                    help="v5: inject the frozen S2 grounding box (target "
                         "direction+proximity) into the action readout so the "
                         "policy homes on the visual target (fixes vis_sens 0.07 "
                         "-> closed-loop forward-creep / no homing)")
    ap.add_argument("--aug_flip", action="store_true")
    ap.add_argument("--lambda_smooth", type=float, default=0.0)
    ap.add_argument("--lambda_endpoint", type=float, default=0.5,
                    help="raised 0.25->0.5: penalise mean/constant endpoint")
    ap.add_argument("--lambda_direction", type=float, default=1.0,
                    help="raised 0.5->1.0: heading must track instruction "
                         "(a constant/mean output gets turns' direction wrong)")
    ap.add_argument("--lambda_acc", type=float, default=0.0)
    ap.add_argument("--lambda_var", type=float, default=0.5,
                    help="anti-collapse variance floor: penalise per-(k,dim) "
                         "batch std below var_floor so the head cannot degenerate "
                         "to an input-independent constant (act_std->0)")
    ap.add_argument("--var_floor", type=float, default=0.3,
                    help="minimum per-(k,dim) batch std (normalised action space)")
    ap.add_argument("--warmup_steps", type=int, default=500,
                    help="linear LR warmup; high sustained LR drove the "
                         "constant-collapse in the first full run")
    ap.add_argument("--sched_total_steps", type=int, default=0,
                    help="cosine horizon (0=run length). Set to the FULL step "
                         "budget in a short smoke so its LR matches the full run "
                         "and exposes sustained-high-LR degenerates (pose routing)")
    ap.add_argument("--channel_weight_z", type=float, default=2.5)
    ap.add_argument("--channel_weight_yaw", type=float, default=2.5)
    ap.add_argument("--oversample_turn_factor", type=int, default=3)
    ap.add_argument("--oversample_class_factor", type=int, default=2)
    ap.add_argument("--overfit_steps", type=int, default=120)
    ap.add_argument("--smoke_real_steps", type=int, default=400)
    ap.add_argument("--save_dir", default="checkpoints/v2_stage3_cradio")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    meta = load_s2(model, Path(args.ckpt_dir), args.tag)
    print(f"[S3] loaded S2 {args.ckpt_dir}/{args.tag} step={meta['step']} "
          f"val={meta['val']}")
    def build_policy():
        model.enable_action_head(chunk_size=args.chunk_size,
                                 proprio_dim=args.proprio_dim,
                                 proprio_dropout=args.proprio_dropout,
                                 use_proprio=not args.no_proprio,
                                 use_grounding_target=args.use_grounding_target,
                                 readout=args.readout, n_bins=args.n_bins,
                                 bin_range=args.bin_range,
                                 readout_layers=args.readout_layers)
        model.to(device)
        model.configure_stage3(train_lora=args.train_lora)
        load_action_stats(model, args.action_stats, args.chunk_size,
                          args.pos_scale, args.norm_mode, args.chunk_offset)

    build_policy()
    model.print_param_census()

    ds = UAVFlowDataset(
        data_root=args.data_root, tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform, chunk_size=args.chunk_size,
        max_text_len=args.max_text_len, split="train", pos_scale=args.pos_scale,
        pos_unit=args.pos_unit,
        aug_flip=args.aug_flip,
        oversample_turn_factor=args.oversample_turn_factor,
        oversample_class_factor=args.oversample_class_factor,
        chunk_offset=args.chunk_offset)
    n = len(ds)
    if args.split_by == "trajectory":
        # Chunks slide with stride 1, so neighbouring chunks of one trajectory
        # share the frame and nearly the whole cumulative target. Splitting on
        # chunks puts near-duplicates on both sides and the val loss then
        # measures within-trajectory interpolation, not generalisation.
        n_traj = len(ds.trajectories)
        traj_ids = list(range(n_traj))
        random.Random(args.seed).shuffle(traj_ids)
        n_val_traj = max(1, int(n_traj * args.val_frac))
        val_traj = set(traj_ids[:n_val_traj])
        tr_idx = [i for i, (t, _) in enumerate(ds.index) if t not in val_traj]
        # Oversampling duplicates entries in ds.index; keep val de-duplicated so
        # it stays an unweighted estimate of the natural chunk distribution.
        seen: set = set()
        va_idx = []
        for i, (t, s) in enumerate(ds.index):
            if t in val_traj and (t, s) not in seen:
                seen.add((t, s))
                va_idx.append(i)
        args._val_traj_files = sorted(
            str(ds.kept_traj_files[t].relative_to(ds.data_root)) for t in val_traj)
        print(f"[S3] split=trajectory: {n_traj - n_val_traj} train trajs / "
              f"{n_val_traj} val trajs (disjoint)")
    else:
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        n_val = max(args.batch, int(n * args.val_frac))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]
        print("[S3] split=chunk: WARNING val shares trajectories with train; "
              "val metrics are in-sample and must not be reported as "
              "generalisation")
    print(f"[S3] dataset: {len(tr_idx)} train | {len(va_idx)} val (of {n})")

    def mk(sub, shuffle):
        return DataLoader(Subset(ds, sub), batch_size=args.batch, shuffle=shuffle,
                          collate_fn=aero_collate_fn, num_workers=args.workers,
                          drop_last=shuffle)

    if args.mode == "grounding_eval":
        # Offline re-measurement of instruction grounding for saved S3 heads.
        # Separate from training because the probe is a measurement: a probe
        # bug does not require a retrain, only a re-read of the checkpoints.
        va_loader = mk(va_idx, False)
        probes = build_probe_batches(va_loader, n=8, groups=args.probe_groups)
        print(f"[S3] grounding_eval: {len(probes)} probe batches x "
              f"{probes[0]['input_ids'].shape[0]} distinct instructions",
              flush=True)
        # Data-side control, no model involved: how much endpoint variance does
        # the instruction alone explain? The within-instruction spread is the
        # residual that only vision (scene, target, clearance) can explain. If
        # it is near zero the benchmark does not require vision, and a low
        # vis_sens is correct behaviour rather than a defect.
        from collections import defaultdict
        by_instr = defaultdict(list)
        for batch in va_loader:
            ep = batch["gt_action"][:, -1]
            ids = batch["input_ids"]
            for r in range(ep.shape[0]):
                by_instr[ids[r].numpy().tobytes()].append(ep[r])
        grps = [torch.stack(v) for v in by_instr.values() if len(v) >= 2]
        within = float(torch.stack([g.std(0) for g in grps]).mean(0).norm())
        across = float(torch.stack([g.mean(0) for g in grps]).std(0).norm())
        vis_share_data = within / max(within + across, 1e-9)
        print(f"\n[S3] val endpoint variance over {len(grps)} instruction "
              f"groups: across-instruction={across:.4f} "
              f"within-instruction={within:.4f} => vision-explainable "
              f"share={vis_share_data:.3f}", flush=True)

        save_dir = Path(args.save_dir)
        report = {"_data": {"instr_groups": len(grps), "across": across,
                            "within": within,
                            "vision_explainable_share": vis_share_data}}
        print("\n=== INSTRUCTION GROUNDING (offline, instruction-diverse "
              "probes) ===", flush=True)
        for name in ("best.pth", "best_grounded.pth", "best_responsive.pth",
                     "last.pth"):
            path = save_dir / name
            if not path.exists():
                continue
            step, val = _load_policy(model, path)
            swap, lang = [], []
            for pb in probes:
                swap.append(grounding_metrics(model, pb, args, device))
                lang.append(instruction_only_spread(model, pb, args, device))
            sw = torch.tensor(swap)              # [G, 4] instr/vis/prop/gt
            ln = torch.tensor(lang)              # [G, 2] pred_spread/gt_spread
            m, sd = sw.mean(0), sw.std(0)
            share = float(m[0]) / max(float(m[0] + m[1]), 1e-9)
            report[name] = {
                "step": step, "val": val,
                "instr_sens": float(m[0]), "instr_sens_std": float(sd[0]),
                "vis_sens": float(m[1]), "vis_sens_std": float(sd[1]),
                "prop_sens": float(m[2]), "gt_spread": float(m[3]),
                "instr_share": share,
                "lang_only_spread": float(ln[:, 0].mean()),
                "lang_only_gt_spread": float(ln[:, 1].mean()),
            }
            print(f"  {name:20s} step={step} "
                  f"instr={m[0]:.4f}+-{sd[0]:.4f} vis={m[1]:.4f}+-{sd[1]:.4f} "
                  f"prop={m[2]:.4f} gt_spread={m[3]:.4f} "
                  f"instr_share={share:.3f} | lang-only pred_spread="
                  f"{ln[:, 0].mean():.4f} vs gt {ln[:, 1].mean():.4f}",
                  flush=True)
        out = save_dir / "grounding_eval.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\n[S3] wrote {out}", flush=True)
        print("GROUNDING_EVAL_DONE", flush=True)
        return

    if args.mode == "smoke":
        tr_loader = mk(tr_idx, True)
        va_loader = mk(va_idx, False)
        fixed = next(iter(tr_loader))
        g1 = gate_overfit(model, fixed, args, device, args.overfit_steps)
        # G2 must run on the G1-TRAINED head: UAVActionHead's output layer is
        # zero-initialised (safe deploy), so from a fresh head d(loss)/d(input)=0
        # and EVERY upstream grad reads 0 on step 1 — a false 'no-flow'. After G1
        # the head is off zero, so this is a faithful grad-flow probe.
        g2 = gate_grad_flow(model, fixed, args, device)
        print("\n=== G1v VISUAL OVERFIT (only the image can separate samples) ===",
              flush=True)
        build_policy()   # fresh head: G1 memorised via the instruction
        g1v = gate_visual_overfit(
            model, fixed, args, device,
            lambda: torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr),
            args.overfit_steps)
        # fresh policy modules for a clean G3 real-data baseline
        build_policy()
        print("\n=== G3 REAL (short convergence + collapse check) ===", flush=True)
        va0 = validate(model, va_loader, args, device, args.max_val_steps)
        print(f"  [G3 pre ] pos_err_m={va0['pos_err_m']:.4f} "
              f"yaw_err_deg={va0['yaw_err_deg']:.3f} act_std={va0.get('act_std',0):.4f}")
        args.max_steps = args.smoke_real_steps
        # Keep frequent smoke validation (CLI --val_every); do not stretch to
        # smoke_real//2 which hid mid-run collapse (act_std peaked then died).
        args.val_every = min(args.val_every, max(50, args.smoke_real_steps // 5))
        smoke_dir = Path(args.save_dir + "_smoke")
        train(model, tr_loader, va_loader, args, device, smoke_dir)
        # G3 measures metre-scale fit — use the lowest-error alive ckpt, not the
        # peak-resp one (lambda_var can trade mean error for sensitivity).
        for tag in ("best_grounded.pth", "best.pth", "best_responsive.pth"):
            ck = smoke_dir / tag
            if ck.is_file():
                print(f"  [G3] loading {tag} for fit check", flush=True)
                _load_policy(model, ck)
                break
        va1 = validate(model, va_loader, args, device, args.max_val_steps)
        print(f"  [G3 post] pos_err_m={va1['pos_err_m']:.4f} "
              f"yaw_err_deg={va1['yaw_err_deg']:.3f} act_std={va1.get('act_std',0):.4f}")
        # G3: anti-collapse + metre-scale fit. Strict pos_err decrease fights the
        # variance floor (healthy act_std often trades ~3cm of mean error); allow
        # a small absolute slack so a diversified policy is not rejected.
        g3 = (va1.get("act_std", 0) > 1e-2
              and va1["pos_err_m"] < va0["pos_err_m"] + 0.05)
        print(f"  [G3] pos_err_m {va0['pos_err_m']:.4f}->{va1['pos_err_m']:.4f}, "
              f"act_std={va1.get('act_std',0):.4f} => {'PASS' if g3 else 'FAIL'}")
        # G4/G5 on the peak-responsive checkpoint (grounding, not mean fit).
        resp_ck = smoke_dir / "best_responsive.pth"
        if resp_ck.is_file():
            print("  [G4] loading best_responsive.pth for grounding", flush=True)
            _load_policy(model, resp_ck)
        print("\n=== G4 GROUNDING (instruction/vision must steer output; proprio must not dominate) ===", flush=True)
        g4 = gate_grounding(model, build_probe_batch(va_loader), args, device)
        print("\n=== G5 CHANNEL COLLAPSE (small DOFs must keep spread) ===", flush=True)
        g5 = gate_channel_collapse(model, va_loader, args, device)
        allp = g1 and g1v and g2 and g3 and g4 and g5
        print(f"\n=== S3 SMOKE VERDICT: G1={g1} G1v={g1v} G2={g2} G3={g3} "
              f"G4={g4} G5={g5} => {'ALL PASS' if allp else 'REVIEW'} ===")
        print(f"S3_SMOKE_EXIT={0 if allp else 1}")
    else:
        tr_loader = mk(tr_idx, True)
        va_loader = mk(va_idx, False)
        train(model, tr_loader, va_loader, args, device, Path(args.save_dir))
        print("S3_FULL_EXIT=0")


if __name__ == "__main__":
    main()
