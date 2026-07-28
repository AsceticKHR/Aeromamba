"""AeroMamba v2 — S0 aerial-domain CPT (LoRA + projector + vision tail).

Loads the S1-aligned projector, then continues CLM training on L0 mixed
aerial VQA / grounding / motion data (design doc §4).

Usage:
  # smoke / wiring
  python training/v2_stage0_cpt.py --smoke

  # pilot
  python training/v2_stage0_cpt.py --max_steps 2000 --out checkpoints/v2_stage0_pilot

  # full 1 epoch
  python training/v2_stage0_cpt.py --epochs 1 --out checkpoints/v2_stage0_full
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

from data.l0_dataset import L0CPTDataset  # noqa: E402
from data.llava_dataset import llava_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402


def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    if args.s1_ckpt:
        model.load_projector(args.s1_ckpt)
    model.configure_stage0(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        vision_unfreeze_last_n=args.vision_tail,
        grad_ckpt=not args.smoke,
    )
    model.print_param_census()

    report = Path(args.jsonl).parent / "build_report.json"
    image_roots = None
    if report.exists():
        image_roots = json.loads(report.read_text(encoding="utf-8")).get("image_roots")

    ds = L0CPTDataset(
        jsonl_path=args.jsonl,
        tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform,
        image_roots=image_roots,
        max_text_len=args.max_text_len,
    )
    return model, ds, device


def fwd(model, batch, device):
    return model.forward_clm(
        batch["pixel_values"].to(device),
        batch["input_ids"].to(device),
        batch["labels"].to(device),
    )


def run_smoke(model, ds, device, args) -> int:
    fails = 0
    g = random.Random(0)
    batch = llava_collate_fn([ds[g.randrange(len(ds))] for _ in range(2)])

    model.eval()
    with torch.no_grad():
        out = fwd(model, batch, device)
    init_loss = float(out["loss"])
    ok = math.isfinite(init_loss) and init_loss < 20.0
    print(f"[G1 init] loss={init_loss:.3f} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    model.train()
    model.zero_grad(set_to_none=True)
    fwd(model, batch, device)["loss"].backward()
    # Only require grads on modules that participate: projector, LoRA, and
    # vision params that are not pure buffers. Skip unused LayerNorm edge cases
    # by requiring that >=95% of trainable params received a grad.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    covered = 1.0 - len(missing) / max(len(trainable), 1)
    ok = covered >= 0.95
    print(f"[G2 grad-flow] missing={len(missing)}/{len(trainable)} "
          f"covered={covered:.3f} -> {'PASS' if ok else 'FAIL'} {missing[:5]}")
    fails += 0 if ok else 1

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    first = last = None
    for step in range(args.smoke_steps):
        opt.zero_grad(set_to_none=True)
        loss = fwd(model, batch, device)["loss"]
        loss.backward()
        opt.step()
        last = float(loss)
        if first is None:
            first = last
        if last < 0.5 * first:
            break
    ok = last < 0.7 * first
    print(f"[G3 overfit] {first:.3f} -> {last:.3f} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    print(f"[smoke] hard failures = {fails}")
    return fails


def save_ckpt(model, out_dir: Path, tag: str, step: int, val: float | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "projector": model.projector.state_dict(),
        "step": step,
        "val": val,
        "config": {"stage": "s0"},
    }
    if hasattr(model.lm, "save_pretrained"):
        adapter_dir = out_dir / f"{tag}_lora"
        model.lm.save_pretrained(adapter_dir)
        payload["lora_dir"] = str(adapter_dir)
    torch.save(model.vision_encoder.state_dict(), out_dir / f"{tag}_vision.pth")
    torch.save(payload, out_dir / f"{tag}_projector.pth")
    print(f"[S0] saved {tag} -> {out_dir}")


def train(model, ds, device, args):
    n_val = max(128, int(len(ds) * 0.01))
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
                            weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total_steps, 2), pct_start=0.03)

    out_dir = Path(args.out)
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
                if cnt >= args.val_batches:
                    break
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
                print(f"[S0] step {step}/{total_steps} loss={float(loss):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {rate:.1f} smp/s",
                      flush=True)
                log_hist.append({"step": step, "loss": float(loss)})
            if step % args.val_every == 0 or step == total_steps:
                v = validate()
                print(f"[S0] step {step} VAL={v:.4f} (best {best_val:.4f})",
                      flush=True)
                log_hist.append({"step": step, "val": v})
                if v < best_val:
                    best_val = v
                    save_ckpt(model, out_dir, "best", step, v)
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break

    save_ckpt(model, out_dir, "last", step, best_val)
    (out_dir / "train_log.json").write_text(json.dumps(log_hist), encoding="utf-8")
    print(f"[S0] done: {step} steps, best_val={best_val:.4f} -> {out_dir}")
    print("S0_EXIT=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl")
    ap.add_argument("--s1_ckpt",
                    default="checkpoints/v2_stage1_full/best_projector.pth")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/v2_stage0_full")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--vision_tail", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=40)
    ap.add_argument("--skip_smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, ds, device = build(args)
    if args.smoke:
        sys.exit(run_smoke(model, ds, device, args))
    if not args.skip_smoke:
        if run_smoke(model, ds, device, args) > 0:
            sys.exit("S0 smoke failed — refusing full train")
    train(model, ds, device, args)


if __name__ == "__main__":
    main()
