"""AeroMamba v2 — S2 instruction SFT + aerial grounding emphasis.

Continues from S0 CPT. Addresses S0 eval failure (blank/shuffle ablation):
language shortcuts on uav_motion — by up-weighting vision-dependent sources
(airspatial grounding, aerial_spatial, hrvqa) and down-weighting short
motion-phrase copies.

Usage:
  python training/v2_stage2_sft.py --smoke
  python training/v2_stage2_sft.py --epochs 1 --out checkpoints/v2_stage2_full
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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.l0_dataset import L0CPTDataset  # noqa: E402
from data.llava_dataset import llava_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402

# Higher = more vision-dependent / grounding. uav_motion answers are often
# short phrases that S0 memorized without looking (blank ablation failed).
SOURCE_WEIGHT = {
    "airspatial": 4.0,
    "aerial_spatial": 2.5,
    "hrvqa": 2.0,
    "cognitive": 1.5,
    "general": 1.0,
    "uav_motion": 0.35,
}


def load_s0_into(model: AeroV2, s0_dir: Path) -> None:
    from peft import PeftModel

    proj = torch.load(s0_dir / "best_projector.pth", map_location="cpu",
                      weights_only=False)
    model.projector.load_state_dict(proj["projector"], strict=True)
    vis = torch.load(s0_dir / "best_vision.pth", map_location="cpu",
                     weights_only=False)
    model.vision_encoder.load_state_dict(vis, strict=True)
    model.lm = PeftModel.from_pretrained(model.lm, str(s0_dir / "best_lora"),
                                         is_trainable=True)
    print(f"[S2] loaded S0 from {s0_dir} (step={proj.get('step')} "
          f"val={proj.get('val')})")


def configure_s2(model: AeroV2, vision_tail: int = 4, grad_ckpt: bool = True):
    """Train LoRA (already present) + projector + vision tail."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = True
    # enable LoRA params
    for n, p in model.lm.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    model._unfreeze_vision_tail(vision_tail)
    if grad_ckpt:
        model._enable_grad_ckpt()
    model.print_param_census()


def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    load_s0_into(model, Path(args.s0_dir))
    configure_s2(model, vision_tail=args.vision_tail, grad_ckpt=not args.smoke)

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
    # prefer aerial samples for smoke
    aerial = [i for i, r in enumerate(ds.rows)
              if r.get("source") in ("airspatial", "aerial_spatial", "hrvqa")]
    pick = aerial if aerial else list(range(len(ds)))
    batch = llava_collate_fn([ds[g.choice(pick)] for _ in range(2)])

    model.eval()
    with torch.no_grad():
        loss0 = float(fwd(model, batch, device)["loss"])
    ok = math.isfinite(loss0) and loss0 < 10
    print(f"[G1 init] loss={loss0:.3f} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    model.train()
    model.zero_grad(set_to_none=True)
    fwd(model, batch, device)["loss"].backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    covered = 1.0 - len(missing) / max(len(trainable), 1)
    ok = covered >= 0.95
    print(f"[G2 grad] missing={len(missing)}/{len(trainable)} "
          f"covered={covered:.3f} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # blank must be worse than real after a few steps (soft vision use check)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    for _ in range(min(20, args.smoke_steps)):
        opt.zero_grad(set_to_none=True)
        fwd(model, batch, device)["loss"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        real = float(fwd(model, batch, device)["loss"])
        blank_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        blank_b["pixel_values"] = torch.zeros_like(batch["pixel_values"])
        blank = float(fwd(model, blank_b, device)["loss"])
    delta = blank - real
    ok = delta > -0.05  # soft: at least not strongly preferring blank
    print(f"[G3 blank-vs-real] real={real:.3f} blank={blank:.3f} "
          f"Δ={delta:+.3f} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    print(f"[smoke] hard failures = {fails}")
    return fails


def save_ckpt(model, out_dir: Path, tag: str, step: int, val: float | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "projector": model.projector.state_dict(),
        "step": step,
        "val": val,
        "config": {"stage": "s2"},
    }
    if hasattr(model.lm, "save_pretrained"):
        adapter_dir = out_dir / f"{tag}_lora"
        model.lm.save_pretrained(adapter_dir)
        payload["lora_dir"] = str(adapter_dir)
    torch.save(model.vision_encoder.state_dict(), out_dir / f"{tag}_vision.pth")
    torch.save(payload, out_dir / f"{tag}_projector.pth")
    print(f"[S2] saved {tag} -> {out_dir}")


def make_weighted_loader(subset, ds_full, batch, workers, drop_last=True):
    # map subset indices -> source weights
    if hasattr(subset, "indices"):
        idxs = list(subset.indices)
        base = subset.dataset
        while hasattr(base, "indices"):
            idxs = [base.indices[i] for i in idxs]
            base = base.dataset
    else:
        idxs = list(range(len(subset)))
        base = ds_full
    weights = [SOURCE_WEIGHT.get(base.rows[i].get("source", "general"), 1.0)
               for i in idxs]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    return DataLoader(subset, batch_size=batch, sampler=sampler,
                      collate_fn=llava_collate_fn, num_workers=workers,
                      pin_memory=True, drop_last=drop_last,
                      persistent_workers=workers > 0)


def train(model, ds, device, args):
    n_val = max(128, int(len(ds) * 0.01))
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    if args.max_steps > 0:
        need = min(len(train_ds), args.max_steps * args.batch)
        train_ds = Subset(train_ds, list(range(need)))

    tl = make_weighted_loader(train_ds, ds, args.batch, args.workers)
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
            # 10%: shuffle images within batch to discourage caption-only copy
            if args.shuffle_aug > 0 and random.random() < args.shuffle_aug:
                if batch["pixel_values"].size(0) > 1:
                    # skip loss on shuffled — just don't train that step
                    # instead: keep real; shuffle_aug used as mix probability
                    # of replacing with a roll for a harder matched batch:
                    # actually train on real only; use roll as soft regularizer
                    # by mixing 50% pixels from rolled neighbor (breaks shortcut)
                    rolled = batch["pixel_values"].roll(1, dims=0)
                    batch = dict(batch)
                    batch["pixel_values"] = 0.5 * batch["pixel_values"] + 0.5 * rolled

            opt.zero_grad(set_to_none=True)
            loss = fwd(model, batch, device)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"[S2] step {step}/{total_steps} loss={float(loss):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {rate:.1f} smp/s",
                      flush=True)
                log_hist.append({"step": step, "loss": float(loss)})
            if step % args.val_every == 0 or step == total_steps:
                v = validate()
                print(f"[S2] step {step} VAL={v:.4f} (best {best_val:.4f})",
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
    print(f"[S2] done: {step} steps, best_val={best_val:.4f} -> {out_dir}")
    print("S2_EXIT=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl")
    ap.add_argument("--s0_dir", default="checkpoints/v2_stage0_full")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/v2_stage2_full")
    ap.add_argument("--vision_tail", type=int, default=4)
    ap.add_argument("--shuffle_aug", type=float, default=0.15)
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
            sys.exit("S2 smoke failed — refusing full train")
    train(model, ds, device, args)


if __name__ == "__main__":
    main()
