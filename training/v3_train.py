"""AeroV3 training on HUGE-Bench.

    # gates first -- no full run starts before every gate passes
    python training/v3_train.py --mode smoke --data_root $HUGE --anno_root $ANNO

    # full. sched_total_steps must equal max_steps: on the v2 line a shorter
    # schedule annealed the LR to zero early and produced a plateau that looked
    # like convergence for two days.
    python training/v3_train.py --mode full --max_steps 20000 \
        --sched_total_steps 20000 --tag c_phase_supervised

Ablation switches, matching the L2 table in the validation guide:

    A  --no_phase                     stateless, parameter-matched control
    B  --loss_stage 0 --loss_prog 0   recurrent but action-loss only (RoboMME)
    C  (defaults)                     recurrent + stage supervision
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.hugebench_dataset import (ACTION_HORIZON, EXEC_STEPS, ActionStats,
                                    HugeBenchWindows, build_index, collate,
                                    compute_action_stats, split_by_episode)
from model.aerov3 import AeroV3, AeroV3Config


# ── args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--data_root", required=True)
    p.add_argument("--anno_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--families", default="", help="comma list; empty = all")
    p.add_argument("--action_stats", default="")

    p.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    p.add_argument("--vision_type", default="cradio_v3_b")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--no_phase", action="store_true")
    p.add_argument("--train_lora", action="store_true")

    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=EXEC_STEPS)
    p.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    p.add_argument("--label_mode", choices=["cumulative", "delta"],
                   default="cumulative")
    p.add_argument("--windows_per_episode", type=int, default=4)

    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--sched_total_steps", type=int, default=0)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=30)
    p.add_argument("--val_frac", type=float, default=0.03)

    p.add_argument("--loss_action", type=float, default=1.0)
    p.add_argument("--loss_stage", type=float, default=0.5)
    p.add_argument("--loss_prog", type=float, default=0.1)

    # Gate budgets. 60 steps was not an overfit test, it was a warm-up: the
    # action loss was still falling steeply when it was cut off.
    p.add_argument("--g1_steps", type=int, default=400)
    p.add_argument("--g1_batches", type=int, default=3,
                   help="batches held fixed for the overfit gates")
    p.add_argument("--g3_steps", type=int, default=3000)

    p.add_argument("--out", default="checkpoints")
    p.add_argument("--tag", default="v3")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ── setup ────────────────────────────────────────────────────────────────────

def build(args, device):
    cfg = AeroV3Config(
        backbone_id=args.backbone, vision_type=args.vision_type,
        img_size=args.img_size, horizon=args.horizon,
        use_phase=not args.no_phase, train_lora=args.train_lora,
        loss={"action": args.loss_action, "stage": args.loss_stage,
              "progress": args.loss_prog})
    model = AeroV3(cfg).to(device)
    print(f"[v3] {model.param_report()}", flush=True)
    return model, cfg


def loaders(args, model):
    fams = [f for f in args.families.split(",") if f] or None
    eps = build_index(args.data_root, args.anno_root, args.split, families=fams)
    if not eps:
        raise SystemExit("[v3] no episodes indexed -- check data_root/anno_root")
    tr, va = split_by_episode(eps, args.val_frac, args.seed)
    byfam = defaultdict(int)
    for e in eps:
        byfam[e.family] += 1
    print(f"[v3] episodes {len(eps)}  train {len(tr)}  val {len(va)}  "
          f"families {dict(byfam)}", flush=True)

    if args.action_stats and Path(args.action_stats).exists():
        stats = ActionStats.load(args.action_stats)
    else:
        stats = compute_action_stats(tr, horizon=args.horizon,
                                     mode=args.label_mode, stride=args.stride)
        if args.action_stats:
            stats.save(args.action_stats)
    print(f"[v3] action q01={np.round(stats.q01, 4).tolist()} "
          f"q99={np.round(stats.q99, 4).tolist()}", flush=True)

    tf = model.vision.transform
    kw = dict(stats=stats, transform=tf, window=args.window, stride=args.stride,
              horizon=args.horizon, seed=args.seed, label_mode=args.label_mode)
    tr_ds = HugeBenchWindows(tr, windows_per_episode=args.windows_per_episode,
                             infinite=True, **kw)
    va_ds = HugeBenchWindows(va, windows_per_episode=2, infinite=True,
                             vision_rates=(1,), **kw)

    # partial over the tokenizer, not a closure over the model: a lambda here
    # would ship the whole 0.7B policy to every dataloader worker.
    fn = partial(collate, tokenizer=model.tokenizer)

    def mk(ds, workers):
        return DataLoader(ds, batch_size=args.batch, num_workers=workers,
                          collate_fn=fn, pin_memory=True,
                          persistent_workers=workers > 0)

    return mk(tr_ds, args.workers), mk(va_ds, min(2, args.workers)), stats, tr, va


def lr_at(step, args):
    total = args.sched_total_steps or args.max_steps
    if step < args.warmup:
        return args.lr * step / max(args.warmup, 1)
    t = min(1.0, (step - args.warmup) / max(total - args.warmup, 1))
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))


def to_dev(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in b.items()}


# ── evaluation ───────────────────────────────────────────────────────────────

def _cum_and_step(x: torch.Tensor, mode: str):
    """(B,W,H,4) in the trained representation -> (cumulative, per-step)."""
    if mode == "cumulative":
        return x, x - F.pad(x[..., :-1, :], (0, 0, 1, 0))
    return x.cumsum(-2), x


@torch.no_grad()
def evaluate(model, loader, stats, device, n_batches, amp, mode="cumulative"):
    """Reports the same three errors as scripts/hugebench_trivial_baselines.py,
    so a policy number can be put straight next to the baseline table. They
    disagree by an order of magnitude and the disagreement is informative:
    step is what an L1 chunk loss fits, path is the offline proxy for the
    official soft-DTW, endpoint is what the observability audit measured."""
    model.eval()
    span = torch.as_tensor(np.maximum(stats.q99 - stats.q01, 1e-6), device=device)
    lo = torch.as_tensor(stats.q01, device=device)
    agg, fam = defaultdict(list), defaultdict(lambda: defaultdict(list))
    it = iter(loader)
    for _ in range(n_batches):
        try:
            b = to_dev(next(it), device)
        except StopIteration:
            break
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            o = model(b)

        def denorm(a):
            return (a.float() + 1.0) * 0.5 * span + lo

        pc, ps = _cum_and_step(denorm(o["action"]), mode)
        gc, gs = _cum_and_step(denorm(b["action"]), mode)
        m = b["action_mask"].bool()
        n = m.sum((1, 2)).clamp(min=1)

        step = (((ps - gs)[..., :3].norm(dim=-1)) * m).sum((1, 2)) / n
        path = (((pc - gc)[..., :3].norm(dim=-1)) * m).sum((1, 2)) / n
        endp = (pc - gc)[:, :, -1, :3].norm(dim=-1).mean(1)

        agg["step_err_m"].append(float(step.mean()))
        agg["path_err_m"].append(float(path.mean()))
        agg["endpoint_err_m"].append(float(endp.mean()))
        agg["loss"].append(float(o["loss"]))
        agg["stage_acc"].append(float(o["stage_acc"]))
        # Per-channel spread as a fraction of the data's own, on the
        # cumulative chunk so it is directly comparable to the pose-kNN row of
        # scripts/hugebench_trivial_baselines.py. Measuring it on the per-step
        # decomposition instead compares against a different quantity and the
        # two disagree by an order of magnitude.
        sp, gsp = pc.std((0, 1, 2)), gc.std((0, 1, 2))
        for i, c in enumerate(("dx", "dy", "dz", "dyaw")):
            agg[f"spread_ratio_{c}"].append(float(sp[i] / gsp[i].clamp(min=1e-9)))
        for j, f in enumerate(b["family"]):
            fam[f]["path_err_m"].append(float(path[j]))
    model.train()
    out = {k: round(float(np.mean(v)), 5) for k, v in agg.items() if v}
    out["by_family"] = {f: round(float(np.mean(d["path_err_m"])), 4)
                        for f, d in fam.items()}
    return out


# ── smoke gates ──────────────────────────────────────────────────────────────

# Best per-channel spread any trivial predictor retains on the cumulative
# chunk, over 5,425 held-out windows (scripts/hugebench_trivial_baselines.py).
# Channel-wise max of pose-kNN (0.816/0.900/0.453/0.623) and class-progress
# (0.348/0.649/0.714/0.480). G5 wants at least half of it.
#
# The point of keying the gate to a measurement: a well-behaved conditional
# predictor *always* has less spread than the data, because it drops the part
# the observation does not determine, so a single floor cannot separate
# "collapsed" from "correctly unsure". The two references also disagree per
# channel in a way that is itself informative -- dz is the one channel where
# class-progress beats pose-kNN, 0.714 against 0.453, and where class-mean
# without progress collapses to 0.088. Altitude is set by how far along the
# flight is, not by the instruction, which is precisely the quantity the phase
# state exists to estimate.
#
# Both references are optimistic and neither is a conditional mean: kNN over a
# handful of neighbours keeps sampling noise, and class-progress reads the
# true progress off the frame index, which no policy has at test time. Halving
# them is the concession to that.
#
# Over-dispersion is a failure too -- exceeding the data's own spread means
# variance is being injected, which is how the v2 line's lambda_var produced a
# policy that moved a lot and tracked nothing.
TRIVIAL_SPREAD = {"dx": 0.816, "dy": 0.900, "dz": 0.714, "dyaw": 0.623}

def _mean_loss(batches) -> float:
    """Masked L1 of the best constant-per-horizon-index predictor on the
    overfit set. This is the reference G1 has to beat.

    The loss at random init is not a usable reference: it depends on the label
    representation, so the same 0.25x threshold means different things for
    deltas and for cumulative displacement. Beating the set's own mean ramp by
    4x means the same thing in both.
    """
    a = torch.cat([b["action"] for b in batches])
    m = torch.cat([b["action_mask"] for b in batches]).unsqueeze(-1)
    mu = (a * m).sum((0, 1), keepdim=True) / m.sum((0, 1), keepdim=True).clamp(min=1)
    return float(((a - mu).abs() * m).sum() / (m.sum() * a.shape[-1]).clamp(min=1))


def gate_overfit(model, batches, device, steps, lr, amp, label,
                 const_instr=False, target=0.25):
    """G1 / G1v. G1v pins every instruction to one string: if the loss still
    falls, vision and pose alone can drive the action, which separates a wrong
    architecture from an undertrained one.

    Full-batch gradient over several batches, not one. On a single batch of 8
    the same code gave 0.03x and 0.24x on consecutive runs, and G1v swung
    0.11x / 0.57x -- a gate that decides whether the architecture is sound
    cannot be a coin flip on which windows the loader happened to yield.
    """
    if const_instr:
        pinned = []
        for b in batches:
            b = dict(b)
            n = b["input_ids"].shape[0]
            b["input_ids"] = batches[0]["input_ids"][:1].repeat(n, 1)
            b["attention_mask"] = batches[0]["attention_mask"][:1].repeat(n, 1)
            pinned.append(b)
        batches = pinned
    ref = _mean_loss(batches)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    # Cosine to zero. A constant lr leaves AdamW taking a fixed-size step
    # forever, which puts a noise floor under the loss and makes an overfit
    # test unable to overfit.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    first = last = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for b in batches:
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                o = model(b)
            (o["loss"] / len(batches)).backward()
            tot += float(o["loss_action"]) / len(batches)
        opt.step()
        sch.step()
        last = tot
        if i == 0:
            first = last
        elif (i + 1) % 100 == 0:
            print(f"    {label} step {i + 1}: {last:.4f} "
                  f"({last / max(ref, 1e-9):.2f}x mean)", flush=True)
    r = last / max(ref, 1e-9)
    ok = r < target
    print(f"  [{label}] action loss {first:.4f} -> {last:.4f}; "
          f"mean-ramp predictor {ref:.4f} -> {r:.2f}x, "
          f"need <{target:.2f}  {'PASS' if ok else 'FAIL'}", flush=True)
    return r


def run_smoke(args, model, tr_loader, va_loader, stats, device, amp):
    print("\n[v3] ===== smoke gates =====", flush=True)
    res = {}
    it = iter(tr_loader)
    batches = [to_dev(next(it), device) for _ in range(args.g1_batches)]
    batch = batches[0]

    import copy
    base = copy.deepcopy(model.state_dict())

    r1 = gate_overfit(model, batches, device, args.g1_steps, 1e-3, amp,
                      "G1 overfit")
    res["G1"] = bool(r1 < 0.25)
    model.load_state_dict(base)

    # G1v is strictly harder than G1: an input has been removed, so it cannot
    # share G1's threshold. What it has to show is that most of what the model
    # learns is reachable without the instruction, so it is scored against G1
    # itself -- it must close at least half the gap G1 closes. Self-calibrating,
    # and it tightens automatically if G1 gets stronger.
    rv = gate_overfit(model, batches, device, args.g1_steps, 1e-3, amp,
                      "G1v const-instruction", const_instr=True,
                      target=1.0 - 0.5 * (1.0 - r1))
    res["G1v"] = bool((1.0 - rv) > 0.5 * (1.0 - r1))
    print(f"       G1v closes {100 * (1 - rv) / max(1 - r1, 1e-9):.0f}% of the "
          f"gap G1 closes; need >50%", flush=True)
    model.load_state_dict(base)

    # G2: the frozen parts must be exactly zero, not small. A non-zero here
    # invalidates every parameter-count and latency claim we make.
    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
        model(batch)["loss"].backward()
    gv = sum(float(p.grad.abs().sum()) for p in model.vision.parameters()
             if p.grad is not None)
    gl = sum(float(p.grad.abs().sum()) for n, p in model.lm.named_parameters()
             if p.grad is not None and "lora" not in n)
    res["G2"] = (gv == 0.0 and gl == 0.0)
    print(f"  [G2 grad-flow] vision={gv:.3e} base-LM={gl:.3e}  "
          f"{'PASS' if res['G2'] else 'FAIL'}", flush=True)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(base)

    # G3/G4/G5: a short real-data run, then read the gates off validation.
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.wd)
    n = args.g3_steps
    sched = argparse.Namespace(lr=args.lr, warmup=min(100, n // 6),
                               sched_total_steps=n, max_steps=n)
    v0 = evaluate(model, va_loader, stats, device, 8, amp, args.label_mode)
    for step in range(n):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, sched)
        b = to_dev(next(it), device)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            o = model(b)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"    G3 step {step + 1}/{n} act {float(o['loss_action']):.4f} "
                  f"stage_acc {float(o['stage_acc']):.3f}", flush=True)
        # Spread as a curve, not a verdict. A channel that is climbing at the
        # end of the budget is undertrained; one that is flat near zero is
        # collapsed, and only the second is an architecture problem.
        if (step + 1) % max(n // 4, 1) == 0:
            vm = evaluate(model, va_loader, stats, device, 6, amp, args.label_mode)
            print(f"      spread@{step + 1} " + " ".join(
                f"{c}={vm[f'spread_ratio_{c}']:.3f}" for c in TRIVIAL_SPREAD)
                + f"  path {vm['path_err_m']:.2f}", flush=True)
    v1 = evaluate(model, va_loader, stats, device, 12, amp, args.label_mode)

    res["G3"] = bool(v1["path_err_m"] < v0["path_err_m"]
                     and np.isfinite(v1["path_err_m"]))
    print(f"  [G3 converges] path_err_m {v0['path_err_m']:.3f} -> "
          f"{v1['path_err_m']:.3f}  (step {v0['step_err_m']:.3f} -> "
          f"{v1['step_err_m']:.3f}, endpoint {v0['endpoint_err_m']:.2f} -> "
          f"{v1['endpoint_err_m']:.2f})  {'PASS' if res['G3'] else 'FAIL'}",
          flush=True)

    # G4 is new for v3. The phase state is the contribution; if its head cannot
    # learn, there is nothing downstream to ablate.
    res["G4"] = v1["stage_acc"] > v0["stage_acc"] + 0.05
    print(f"  [G4 stage head] acc {v0['stage_acc']:.3f} -> {v1['stage_acc']:.3f}  "
          f"{'PASS' if res['G4'] else 'FAIL'}", flush=True)

    sp = {c: v1[f"spread_ratio_{c}"] for c in TRIVIAL_SPREAD}
    res["G5"] = all(0.5 * TRIVIAL_SPREAD[c] < v < 1.25 for c, v in sp.items())
    print("  [G5 spread / GT spread] " + "  ".join(
        f"{c}={v:.3f}(>{0.5 * TRIVIAL_SPREAD[c]:.2f})" for c, v in sp.items())
        + f"  {'PASS' if res['G5'] else 'FAIL'}", flush=True)

    ok = all(res.values())
    print(f"\n[v3] S3 SMOKE VERDICT: {'ALL PASS' if ok else 'FAIL'}  {res}",
          flush=True)
    print("V3_SMOKE_DONE", flush=True)
    return ok


# ── full run ─────────────────────────────────────────────────────────────────

def run_full(args, model, tr_loader, va_loader, stats, device, amp):
    save = Path(args.out) / f"v3_{args.tag}"
    save.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.wd)
    best = float("inf")
    it = iter(tr_loader)
    t0 = time.time()
    hist = []

    for step in range(1, args.max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        b = to_dev(next(it), device)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            o = model(b)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"[v3] step {step}/{args.max_steps} loss {float(o['loss']):.4f} "
                  f"act {float(o['loss_action']):.4f} "
                  f"stage {float(o['loss_stage']):.4f} "
                  f"acc {float(o['stage_acc']):.3f} "
                  f"lr {lr_at(step, args):.2e} "
                  f"{(time.time() - t0) / step:.2f}s/it", flush=True)

        if step % args.val_every == 0:
            v = evaluate(model, va_loader, stats, device, args.val_batches, amp,
                         args.label_mode)
            v["step"] = step
            hist.append(v)
            print(f"[v3] VAL step {step} {json.dumps(v)}", flush=True)
            json.dump(hist, open(save / "val_history.json", "w"), indent=1)
            # Selected on path error, not step: step is dominated by jitter
            # the data does not determine, and the official metric scores the
            # path. Selecting on the wrong one is how the v2 line ended up
            # picking its most vision-blind checkpoints.
            if v["path_err_m"] < best:
                best = v["path_err_m"]
                torch.save({"model": {k: p for k, p in model.state_dict().items()
                                      if p.dtype.is_floating_point},
                            "cfg": vars(args), "val": v, "step": step},
                           save / "best.pth")
    print("V3_FULL_DONE", flush=True)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.bf16 and device == "cuda"
    if args.mode == "full" and not args.sched_total_steps:
        raise SystemExit("[v3] --sched_total_steps must be set for a full run "
                         "and should equal --max_steps")

    model, _ = build(args, device)
    tr, va, stats, _, _ = loaders(args, model)
    if args.mode == "smoke":
        raise SystemExit(0 if run_smoke(args, model, tr, va, stats, device, amp) else 1)
    run_full(args, model, tr, va, stats, device, amp)


if __name__ == "__main__":
    main()
