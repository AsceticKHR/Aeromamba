"""AeroMamba v2 — S1 projector alignment (Falcon-H1 backbone, no resampler).

Trains ONLY the MLP projector with a caption CLM loss over [vision | text],
LLaVA-Pretrain style. `--smoke` runs the
wiring gates (init-loss / grad-flow / single-batch overfit) and exits; a full
run refuses to start unless the smoke gates pass first (--skip_smoke to
override, e.g. when resuming).

Usage:
  # gates only (~3 min)
  python training/v2_stage1_align.py --data_root data/llava_pretrain --smoke

  # pilot: 2000 steps
  python training/v2_stage1_align.py --data_root data/llava_pretrain \
      --max_steps 2000 --out checkpoints/v2_stage1_pilot

  # full
  python training/v2_stage1_align.py --data_root data/llava_pretrain \
      --epochs 1 --out checkpoints/v2_stage1
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
from torch.utils.data import DataLoader, Subset, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.llava_dataset import LLaVADataset, llava_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402


def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    model.configure_stage1()
    model.print_param_census()
    ds = LLaVADataset(
        data_root=args.data_root,
        tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform,
        max_text_len=args.max_text_len,
        json_name=args.json_name,
    )
    return model, ds, device


def fwd(model, batch, device):
    pv = batch["pixel_values"].to(device)
    ids = batch["input_ids"].to(device)
    lbl = batch["labels"].to(device)
    return model.forward_clm(pv, ids, lbl)


# ─────────────────────────────────────────────────────────── smoke gates
def run_smoke(model, ds, device, args) -> int:
    """S1 wiring gates. Returns number of hard failures."""
    fails = 0
    g = random.Random(0)
    batch = llava_collate_fn([ds[g.randrange(len(ds))] for _ in range(4)])

    # G1: init loss ~ ln(vocab) for a random projector (upper bound); must be finite
    model.eval()
    with torch.no_grad():
        out = fwd(model, batch, device)
    init_loss = float(out["loss"])
    ln_v = math.log(model.tokenizer.vocab_size)
    ok = math.isfinite(init_loss) and init_loss < ln_v * 1.5
    print(f"[G1 init-sanity] loss={init_loss:.3f} ln(V)={ln_v:.2f} "
          f"n_vis={out['n_vis_tokens']} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # G2: grads reach projector only
    model.train()
    model.zero_grad(set_to_none=True)
    fwd(model, batch, device)["loss"].backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    leaked = [n for n, p in model.named_parameters()
              if not p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    ok = not missing and not leaked
    print(f"[G2 grad-flow] missing={len(missing)} leaked={len(leaked)} "
          f"-> {'PASS' if ok else 'FAIL'} {missing[:3] + leaked[:3]}")
    fails += 0 if ok else 1
    model.zero_grad(set_to_none=True)

    # G3: single-batch overfit (projector only; expect large but real drop)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    first = last = None
    for step in range(args.smoke_steps):
        opt.zero_grad(set_to_none=True)
        loss = fwd(model, batch, device)["loss"]
        loss.backward()
        opt.step()
        last = float(loss)
        if first is None:
            first = last
        if last < 0.35 * first:
            break
    ok = last < 0.5 * first
    print(f"[G3 overfit-batch] {first:.3f} -> {last:.3f} in {step + 1} steps "
          f"-> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    print(f"[smoke] hard failures = {fails}")
    return fails


# ───────────────────────────────────────────────────────────── training
def train(model, ds, device, args) -> None:
    n_val = max(64, int(len(ds) * 0.02))
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    if args.max_steps > 0:
        need = min(len(train_ds), args.max_steps * args.batch)
        train_ds = Subset(train_ds, list(range(need)))
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    collate_fn=llava_collate_fn, num_workers=args.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    collate_fn=llava_collate_fn, num_workers=2)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total_steps, 2), pct_start=0.03)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    t0 = time.time()
    log_hist = []

    def validate() -> float:
        model.eval()
        tot = cnt = 0.0
        with torch.no_grad():
            for b in vl:
                tot += float(fwd(model, b, device)["loss"])
                cnt += 1
        model.train()
        return tot / max(cnt, 1)

    model.train()
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in tl:
            opt.zero_grad(set_to_none=True)
            loss = fwd(model, batch, device)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"[S1] step {step}/{total_steps} loss={float(loss):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {rate:.1f} smp/s", flush=True)
                log_hist.append({"step": step, "loss": float(loss)})
            if step % args.val_every == 0 or step == total_steps:
                v = validate()
                print(f"[S1] step {step} VAL={v:.4f} (best {best_val:.4f})", flush=True)
                log_hist.append({"step": step, "val": v})
                if v < best_val:
                    best_val = v
                    torch.save({"projector": model.projector.state_dict(),
                                "step": step, "val": v,
                                "config": {"backbone": args.backbone,
                                           "vision_type": args.vision_type}},
                               out_dir / "best_projector.pth")
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break
    torch.save({"projector": model.projector.state_dict(), "step": step,
                "config": {"backbone": args.backbone, "vision_type": args.vision_type}},
               out_dir / "last_projector.pth")
    (out_dir / "train_log.json").write_text(json.dumps(log_hist), encoding="utf-8")
    print(f"[S1] done: {step} steps, best_val={best_val:.4f} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--json_name", default="llava_subset.json")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--out", default="checkpoints/v2_stage1")
    ap.add_argument("--smoke", action="store_true", help="run wiring gates and exit")
    ap.add_argument("--smoke_steps", type=int, default=150)
    ap.add_argument("--skip_smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, ds, device = build(args)

    if args.smoke:
        sys.exit(run_smoke(model, ds, device, args))
    if not args.skip_smoke:
        if run_smoke(model, ds, device, args) > 0:
            sys.exit("smoke gates failed — refusing to start full training")
        # re-init projector so gate overfitting doesn't leak into the run
        for m in model.projector.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        model.projector.to(model.dtype)
    train(model, ds, device, args)


if __name__ == "__main__":
    main()
