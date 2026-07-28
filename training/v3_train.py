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
    p.add_argument("--g1_steps", type=int, default=300)
    p.add_argument("--g3_steps", type=int, default=1500)

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
        stats = compute_action_stats(tr)
        if args.action_stats:
            stats.save(args.action_stats)
    print(f"[v3] action q01={np.round(stats.q01, 4).tolist()} "
          f"q99={np.round(stats.q99, 4).tolist()}", flush=True)

    tf = model.vision.transform
    kw = dict(stats=stats, transform=tf, window=args.window, stride=args.stride,
              horizon=args.horizon, seed=args.seed)
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

@torch.no_grad()
def evaluate(model, loader, stats, device, n_batches, amp):
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
        denorm = lambda a: (a.float() + 1.0) * 0.5 * span + lo
        pe = (denorm(o["action"])[..., :3] - denorm(b["action"])[..., :3]) \
            .norm(dim=-1)                                    # (B,W,H) metres
        m = b["action_mask"].bool()
        per_sample = (pe * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)

        agg["pos_err_m"].append(float(per_sample.mean()))
        agg["loss"].append(float(o["loss"]))
        agg["stage_acc"].append(float(o["stage_acc"]))
        # Per-channel spread as a fraction of the data's own spread. An
        # absolute floor is not interpretable here: one step of yaw spans
        # ~0.09 rad while one step of dx spans ~1 m, so the same number means
        # "healthy" for one channel and "collapsed" for the next.
        sp, gsp = denorm(o["action"]).std((0, 1, 2)), denorm(b["action"]).std((0, 1, 2))
        for i, c in enumerate(("dx", "dy", "dz", "dyaw")):
            agg[f"spread_{c}"].append(float(sp[i]))
            agg[f"spread_ratio_{c}"].append(float(sp[i] / gsp[i].clamp(min=1e-9)))
        for j, f in enumerate(b["family"]):
            fam[f]["pos_err_m"].append(float(per_sample[j]))
    model.train()
    out = {k: float(np.mean(v)) for k, v in agg.items() if v}
    out["by_family"] = {f: float(np.mean(d["pos_err_m"])) for f, d in fam.items()}
    return out


# ── smoke gates ──────────────────────────────────────────────────────────────

def gate_overfit(model, batch, device, steps, lr, amp, label, const_instr=False,
                 target=0.25):
    """G1 / G1v. G1v pins every instruction to one string: if the loss still
    falls, vision and pose alone can drive the action, which separates a wrong
    architecture from an undertrained one."""
    if const_instr:
        batch = dict(batch)
        n = batch["input_ids"].shape[0]
        batch["input_ids"] = batch["input_ids"][:1].repeat(n, 1)
        batch["attention_mask"] = batch["attention_mask"][:1].repeat(n, 1)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    first = last = None
    for i in range(steps):
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            o = model(batch)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        opt.step()
        last = float(o["loss_action"])
        if i == 0:
            first = last
        elif (i + 1) % 100 == 0:
            print(f"    {label} step {i + 1}: {last:.4f}", flush=True)
    ok = last < target * first
    print(f"  [{label}] action loss {first:.4f} -> {last:.4f} "
          f"({last / first:.2f}x, need <{target})  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_smoke(args, model, tr_loader, va_loader, stats, device, amp):
    print("\n[v3] ===== smoke gates =====", flush=True)
    res = {}
    it = iter(tr_loader)
    batch = to_dev(next(it), device)

    import copy
    base = copy.deepcopy(model.state_dict())

    res["G1"] = gate_overfit(model, batch, device, args.g1_steps, 1e-3, amp,
                             "G1 overfit")
    model.load_state_dict(base)
    res["G1v"] = gate_overfit(model, batch, device, args.g1_steps, 1e-3, amp,
                              "G1v const-instruction", const_instr=True)
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
    v0 = evaluate(model, va_loader, stats, device, 8, amp)
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
    v1 = evaluate(model, va_loader, stats, device, 12, amp)

    res["G3"] = v1["pos_err_m"] < v0["pos_err_m"] and np.isfinite(v1["pos_err_m"])
    print(f"  [G3 converges] pos_err_m {v0['pos_err_m']:.3f} -> "
          f"{v1['pos_err_m']:.3f}  {'PASS' if res['G3'] else 'FAIL'}", flush=True)

    # G4 is new for v3. The phase state is the contribution; if its head cannot
    # learn, there is nothing downstream to ablate.
    res["G4"] = v1["stage_acc"] > v0["stage_acc"] + 0.05
    print(f"  [G4 stage head] acc {v0['stage_acc']:.3f} -> {v1['stage_acc']:.3f}  "
          f"{'PASS' if res['G4'] else 'FAIL'}", flush=True)

    sp = {c: v1[f"spread_ratio_{c}"] for c in ("dx", "dy", "dz", "dyaw")}
    res["G5"] = all(v > 0.10 for v in sp.values())
    print(f"  [G5 spread / GT spread] "
          f"{({k: round(v, 3) for k, v in sp.items()})}  need >0.10  "
          f"{'PASS' if res['G5'] else 'FAIL'}", flush=True)

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
            v = evaluate(model, va_loader, stats, device, args.val_batches, amp)
            v["step"] = step
            hist.append(v)
            print(f"[v3] VAL step {step} {json.dumps(v)}", flush=True)
            json.dump(hist, open(save / "val_history.json", "w"), indent=1)
            if v["pos_err_m"] < best:
                best = v["pos_err_m"]
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
