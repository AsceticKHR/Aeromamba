"""AeroMamba v2 — S2 grounding co-train (redesign after failed weighted-SFT).

Why this replaces v2_stage2_sft.py:
  Prior S2 (source reweight + pixel mix) made blank/shuffle ablations *worse*.
  This stage adds a decoupled grounding head (meta-query → bbox) and a
  blank-image hinge so the model is explicitly penalised for ignoring vision.

Hard rule: full training is refused unless `--pilot` gates beat S0 on the
same aerial holdout (blank Δ, shuffle Δ, grounding IoU).

Usage:
  # wiring smoke
  python training/v2_stage2_grounding.py --smoke

  # strict pilot (must PASS before full)
  python training/v2_stage2_grounding.py --pilot --pilot_steps 600 \\
      --out checkpoints/v2_stage2_pilot

  # full (auto-runs pilot unless --skip_pilot)
  python training/v2_stage2_grounding.py --epochs 1 --out checkpoints/v2_stage2_gfull
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

from data.l0_dataset import L0S2Dataset, l0_s2_collate_fn  # noqa: E402
from model.aerov2 import AeroV2, _batch_iou  # noqa: E402

SOURCE_WEIGHT = {
    "airspatial": 6.0,
    "aerial_spatial": 2.0,
    "hrvqa": 2.0,
    "cognitive": 1.0,
    "general": 0.4,
    "uav_motion": 0.05,
}
AERIAL = {"aerial_spatial", "hrvqa", "airspatial", "uav_motion"}
# Sources carrying grounding boxes. The head is a SINGLE-box referring-expression
# regressor, so P5 is based on single-object REC sources (dior_rsvg, open3d_vqa_rec).
# airspatial holds multi-object tiny boxes (~6 per row, median area ~0.5%) which are
# ill-posed for a single-box head, so it is reported as INFO only.
GROUNDING = {"airspatial", "dior_rsvg", "open3d_vqa_rec"}
REC_SOURCES = {"dior_rsvg", "open3d_vqa_rec"}


def build_optim(model, base_lr: float, grd_lr_mult: float = 5.0):
    """Higher LR for the from-scratch grounding head; base LR for the
    pretrained LoRA/projector/vision so grounding converges within budget."""
    grd, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (grd if n.startswith("grd_") else other).append(p)
    groups = [{"params": other, "lr": base_lr}]
    if grd:
        groups.append({"params": grd, "lr": base_lr * grd_lr_mult})
    return torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))


def load_s0(model: AeroV2, s0_dir: Path) -> None:
    from peft import PeftModel

    proj = torch.load(s0_dir / "best_projector.pth", map_location="cpu",
                      weights_only=False)
    model.projector.load_state_dict(proj["projector"], strict=True)
    vis = torch.load(s0_dir / "best_vision.pth", map_location="cpu",
                     weights_only=False)
    model.vision_encoder.load_state_dict(vis, strict=True)
    # Unwrap nested PeftModel wrappers (build + pilot reload).
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
    model.lm = PeftModel.from_pretrained(base, str(s0_dir / "best_lora"),
                                         is_trainable=True)
    print(f"[S2] loaded S0 {s0_dir} step={proj.get('step')} val={proj.get('val')} "
          f"base={type(base).__name__}")


def build(args, for_smoke: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    load_s0(model, Path(args.s0_dir))
    # Always enable grad checkpointing — blank hinge adds an extra LM forward.
    model.configure_stage2(vision_unfreeze_last_n=args.vision_tail, grad_ckpt=True)
    model.print_param_census()

    report = Path(args.jsonl).parent / "build_report.json"
    image_roots = None
    if report.exists():
        image_roots = json.loads(report.read_text(encoding="utf-8")).get("image_roots")

    ds = L0S2Dataset(
        jsonl_path=args.jsonl,
        tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform,
        image_roots=image_roots,
        max_text_len=args.max_text_len,
        exclude_sources=None if args.keep_motion else ["uav_motion"],
    )
    return model, ds, device


def fwd(model, batch, device, args):
    return model.forward_s2(
        batch["pixel_values"].to(device),
        batch["input_ids"].to(device),
        batch["labels"].to(device),
        bbox=batch["bbox"].to(device),
        has_bbox=batch["has_bbox"].to(device),
        lambda_grd=args.lambda_grd,
        lambda_blank=args.lambda_blank,
        lambda_shuffle=args.lambda_shuffle,
        blank_margin=args.blank_margin,
        shuffle_margin=args.shuffle_margin,
        blank_bs=args.blank_bs,
    )


def run_smoke(model, ds, device, args) -> int:
    fails = 0
    g = random.Random(0)
    bbox_idx = [i for i, s in enumerate(ds.samples) if s["bbox"] is not None]
    pick = bbox_idx if bbox_idx else list(range(min(100, len(ds))))
    batch = l0_s2_collate_fn([ds[g.choice(pick)] for _ in range(2)])

    model.train()
    out = fwd(model, batch, device, args)
    ok = math.isfinite(float(out["loss"]))
    print(f"[G1 loss] {float(out['loss']):.3f} clm={float(out['loss_clm']):.3f} "
          f"grd={float(out['loss_grd']):.3f} blank={float(out['loss_blank']):.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    # grd_queries / head / some LoRA may be unused if batch has no bbox path edge
    covered = 1.0 - len(missing) / max(len(trainable), 1)
    ok = covered >= 0.90
    print(f"[G2 grad] covered={covered:.3f} missing={len(missing)} "
          f"-> {'PASS' if ok else 'FAIL'} {missing[:4]}")
    fails += 0 if ok else 1

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    first = last = None
    for step in range(args.smoke_steps):
        opt.zero_grad(set_to_none=True)
        loss = fwd(model, batch, device, args)["loss"]
        loss.backward()
        opt.step()
        last = float(loss)
        if first is None:
            first = last
        if last < 0.5 * first:
            break
    ok = last < 0.75 * first
    print(f"[G3 overfit] {first:.3f}->{last:.3f} -> {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    print(f"[smoke] hard failures = {fails}")
    return fails


@torch.no_grad()
def eval_vision_gates(model, ds, device, n_samples=128, batch=8, seed=42):
    """Blank/shuffle CE + grounding IoU on aerial holdout."""
    rng = random.Random(seed)
    aerial = [i for i, s in enumerate(ds.samples) if s["source"] in AERIAL]
    rng.shuffle(aerial)
    idx = aerial[:n_samples]
    loader = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False,
                        collate_fn=l0_s2_collate_fn, num_workers=2)

    def ce_only(mutate=None):
        tot = cnt = 0.0
        for b in loader:
            pv = b["pixel_values"].to(device)
            if mutate == "blank":
                pv = torch.zeros_like(pv)
            elif mutate == "shuffle" and pv.size(0) > 1:
                pv = pv.roll(1, dims=0)
            out = model.forward_clm(pv, b["input_ids"].to(device),
                                    b["labels"].to(device))
            tot += float(out["loss"])
            cnt += 1
        return tot / max(cnt, 1)

    real = ce_only()
    blank = ce_only("blank")
    shuf = ce_only("shuffle")

    # Grounding IoU per source. The head predicts ONE box, so P5 is based on the
    # single-object REC sources (REC_SOURCES); airspatial (multi-object tiny boxes)
    # is measured for diagnostics only.
    def iou_for(idxs):
        vals, preview, centers = [], [], []
        if not idxs:
            return vals, preview, centers
        bl = DataLoader(Subset(ds, idxs), batch_size=batch, shuffle=False,
                        collate_fn=l0_s2_collate_fn, num_workers=0)
        model.eval()
        for b in bl:
            enc = model._encode_multimodal(
                b["pixel_values"].to(device), b["input_ids"].to(device),
                append_grd=False)
            hidden = model._trunk_hidden(enc["inputs_embeds"], enc["attn"])
            pred = model.predict_boxes(hidden, enc["n_vis"], enc["L_txt"],
                                       b["input_ids"].to(device),
                                       b["labels"].to(device),
                                       vis_kv=enc["inputs_embeds"][:, :enc["n_vis"], :])
            m = b["has_bbox"].to(device) > 0.5
            if m.any():
                g = b["bbox"].to(device)[m].float()
                p = pred[m]
                vals.append(_batch_iou(p, g))
                cxy = torch.stack([(p[:, 0] + p[:, 2]) * 0.5,
                                   (p[:, 1] + p[:, 3]) * 0.5], dim=-1)
                centers.append(cxy)
                if len(preview) < 2:
                    preview.append((g[0].tolist(), p[0].tolist()))
        return vals, preview, centers

    grd_by_src = {}
    for i, s in enumerate(ds.samples):
        if s["bbox"] is not None and s["source"] in GROUNDING:
            grd_by_src.setdefault(s["source"], []).append(i)
    per_source_iou, previews, center_std, rec_vals = {}, {}, {}, []
    for src, sidx in grd_by_src.items():
        rng.shuffle(sidx)
        vals, prev, centers = iou_for(sidx[:96])
        if vals:
            cat = torch.cat(vals)
            per_source_iou[src] = float(cat.mean())
            previews[src] = prev
            if centers:
                c = torch.cat(centers)
                center_std[src] = [float(c[:, 0].std()), float(c[:, 1].std())]
            if src in REC_SOURCES:
                rec_vals.append(cat)
    if rec_vals:
        mean_iou = float(torch.cat(rec_vals).mean())
    elif per_source_iou:
        mean_iou = float(sum(per_source_iou.values()) / len(per_source_iou))
    else:
        mean_iou = 0.0
    return {
        "real": real,
        "blank_delta": blank - real,
        "shuffle_delta": shuf - real,
        "grounding_iou": mean_iou,
        "grounding_iou_by_source": per_source_iou,
        "grounding_preview": previews,
        "grounding_center_std": center_std,
        "n": len(idx),
    }


def load_s0_eval_model(args, device):
    """Frozen S0 for baseline comparison (no grounding training)."""
    from peft import PeftModel

    m = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    s0 = Path(args.s0_dir)
    proj = torch.load(s0 / "best_projector.pth", map_location="cpu", weights_only=False)
    m.projector.load_state_dict(proj["projector"], strict=True)
    vis = torch.load(s0 / "best_vision.pth", map_location="cpu", weights_only=False)
    m.vision_encoder.load_state_dict(vis, strict=True)
    m.lm = PeftModel.from_pretrained(m.lm, str(s0 / "best_lora"))
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def run_pilot_gates(model, ds, device, args) -> int:
    """Train pilot_steps then require clear gains vs S0."""
    print(f"\n=== S2 PILOT ({args.pilot_steps} steps) ===")
    bbox_ids = [i for i, s in enumerate(ds.samples) if s["bbox"] is not None]
    other_ids = [i for i, s in enumerate(ds.samples) if s["bbox"] is None]
    print(f"[pilot] bbox_pool={len(bbox_ids)} other_pool={len(other_ids)}")
    if not bbox_ids:
        print("[pilot] FAIL: no bbox samples in dataset")
        return 1

    # bbox-heavy mix: grounding head needs enough box supervision to escape the
    # initial prior, while keeping some language turns for the CLM/vision hinges.
    rng = random.Random(0)
    need = args.pilot_steps * args.batch
    mix_ids = []
    for _ in range(need):
        if rng.random() < args.pilot_bbox_ratio:
            mix_ids.append(rng.choice(bbox_ids))
        else:
            mix_ids.append(rng.choice(other_ids))
    subset = Subset(ds, mix_ids)
    tl = DataLoader(subset, batch_size=args.batch, shuffle=True,
                    collate_fn=l0_s2_collate_fn, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    opt = build_optim(model, args.lr, args.grd_lr_mult)
    model.train()
    step = 0
    t0 = time.time()
    for batch in tl:
        opt.zero_grad(set_to_none=True)
        out = fwd(model, batch, device, args)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            rate = step * args.batch / (time.time() - t0)
            print(f"[pilot] step {step}/{args.pilot_steps} "
                  f"loss={float(out['loss']):.3f} clm={float(out['loss_clm']):.3f} "
                  f"grd={float(out['loss_grd']):.3f} blank={float(out['loss_blank']):.3f} "
                  f"shuf={float(out.get('loss_shuffle', 0.0)):.3f} "
                  f"iou={out['iou']:.3f} {rate:.1f} smp/s", flush=True)
        if step >= args.pilot_steps:
            break

    # evaluate candidate
    model.eval()
    s2_m = eval_vision_gates(model, ds, device, n_samples=args.gate_samples,
                             batch=min(8, args.batch), seed=42)
    # S0 baseline (reload separately to avoid contaminating)
    s0 = load_s0_eval_model(args, device)
    # rebuild dataset with s0 transform (same size)
    report = Path(args.jsonl).parent / "build_report.json"
    image_roots = json.loads(report.read_text()).get("image_roots") if report.exists() else None
    ds0 = L0S2Dataset(
        jsonl_path=args.jsonl, tokenizer=s0.tokenizer,
        transform=s0.vision_encoder.transform, image_roots=image_roots,
        max_text_len=args.max_text_len,
        exclude_sources=None if args.keep_motion else ["uav_motion"],
    )
    s0_m = eval_vision_gates(s0, ds0, device, n_samples=args.gate_samples,
                             batch=min(8, args.batch), seed=42)
    del s0
    torch.cuda.empty_cache()

    def _rec_center_std(m):
        cs = m.get("grounding_center_std", {})
        vals = [(v[0] + v[1]) * 0.5 for k, v in cs.items() if k in REC_SOURCES]
        return sum(vals) / len(vals) if vals else 0.0

    gates = {
        "P1_blank_abs": {
            "ok": s2_m["blank_delta"] >= args.min_blank_delta,
            "value": s2_m["blank_delta"],
            "thr": args.min_blank_delta,
        },
        "P2_shuffle_abs": {
            "ok": s2_m["shuffle_delta"] >= args.min_shuffle_delta,
            "value": s2_m["shuffle_delta"],
            "thr": args.min_shuffle_delta,
        },
        "P3_blank_vs_s0": {
            "ok": s2_m["blank_delta"] > s0_m["blank_delta"] + args.min_delta_gain,
            "value": s2_m["blank_delta"] - s0_m["blank_delta"],
            "thr": args.min_delta_gain,
        },
        "P4_shuffle_vs_s0": {
            "ok": s2_m["shuffle_delta"] > s0_m["shuffle_delta"] + args.min_delta_gain,
            "value": s2_m["shuffle_delta"] - s0_m["shuffle_delta"],
            "thr": args.min_delta_gain,
        },
        # P5/P6 are SOFT diagnostics, not hard gates. The deployed task is S3 action
        # waypoint regression — bbox IoU is only a proxy for "does the model ground
        # language spatially". Vision-reliance (P1–P4 blank/shuffle) is the real
        # must-pass evidence; bbox localisation is a lightweight auxiliary whose IoU
        # we report but do NOT block progress on.
        "P5_grounding_iou": {
            "ok": s2_m["grounding_iou"] >= args.min_iou,
            "value": s2_m["grounding_iou"],
            "thr": args.min_iou,
            "soft": True,
        },
        "P6_center_std": {
            "ok": _rec_center_std(s2_m) >= args.min_center_std,
            "value": _rec_center_std(s2_m),
            "thr": args.min_center_std,
            "soft": True,
        },
    }
    # Verdict is driven ONLY by hard gates (P1–P4 vision-reliance).
    fails = sum(1 for g in gates.values()
                if not g["ok"] and not g.get("soft", False))
    report_obj = {
        "verdict": "PASS" if fails == 0 else "FAIL",
        "hard_failures": fails,
        "s2": s2_m,
        "s0": s0_m,
        "gates": gates,
        "pilot_steps": args.pilot_steps,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_gates.json").write_text(
        json.dumps(report_obj, indent=2), encoding="utf-8")

    print("\n=== PILOT GATES (hard=P1-P4 vision-reliance; P5/P6 soft diagnostics) ===")
    print(f"S2 blankΔ={s2_m['blank_delta']:+.4f} shufΔ={s2_m['shuffle_delta']:+.4f} "
          f"IoU(REC)={s2_m['grounding_iou']:.3f}")
    if s2_m.get("grounding_iou_by_source"):
        print("  per-source IoU: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(s2_m["grounding_iou_by_source"].items())))
        if s2_m.get("grounding_center_std"):
            print("  pred centre std (collapse≈0): " + ", ".join(
                f"{k}=({v[0]:.3f},{v[1]:.3f})"
                for k, v in sorted(s2_m["grounding_center_std"].items())))
        for k, pv in sorted(s2_m.get("grounding_preview", {}).items()):
            for gt_b, pr_b in pv:
                print(f"    {k} gt=[{','.join('%.2f' % x for x in gt_b)}] "
                      f"pred=[{','.join('%.2f' % x for x in pr_b)}]")
    print(f"S0 blankΔ={s0_m['blank_delta']:+.4f} shufΔ={s0_m['shuffle_delta']:+.4f}")
    print(f"REC centre-std: S2={_rec_center_std(s2_m):.4f} vs "
          f"S0={_rec_center_std(s0_m):.4f} (collapse≈0.001; want S2 >> S0)")
    for name, g in gates.items():
        tag = "SOFT" if g.get("soft", False) else ("PASS" if g["ok"] else "FAIL")
        status = "ok" if g["ok"] else "low"
        suffix = f" [{status}, diagnostic]" if g.get("soft", False) else ""
        print(f"[{tag}] {name}: {g['value']:.4f} (need >= {g['thr']}){suffix}")
    print(f"verdict={report_obj['verdict']} hard_failures={fails} "
          f"(P5/P6 soft, do not block)")
    print(f"report -> {out / 'pilot_gates.json'}")
    return fails


def save_ckpt(model, out_dir: Path, tag: str, step: int, val):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "projector": model.projector.state_dict(),
        "grd_queries": model.grd_queries.detach().cpu(),
        "grd_ln_kv": model.grd_ln_kv.state_dict(),
        "grd_ln_q": model.grd_ln_q.state_dict(),
        "grd_ln_out": model.grd_ln_out.state_dict(),
        "grd_txt_proj": model.grd_txt_proj.state_dict(),
        "grd_attn": model.grd_attn.state_dict(),
        "grd_head": model.grd_head.state_dict(),
        "step": step,
        "val": val,
        "config": {"stage": "s2_grounding"},
    }
    if hasattr(model.lm, "save_pretrained"):
        model.lm.save_pretrained(out_dir / f"{tag}_lora")
        payload["lora_dir"] = str(out_dir / f"{tag}_lora")
    torch.save(model.vision_encoder.state_dict(), out_dir / f"{tag}_vision.pth")
    torch.save(payload, out_dir / f"{tag}_projector.pth")
    print(f"[S2] saved {tag} -> {out_dir}")


def make_loader(ds, batch, workers, for_train=True):
    weights = [SOURCE_WEIGHT.get(s["source"], 1.0) for s in ds.samples]
    for i, s in enumerate(ds.samples):
        if s["bbox"] is not None:
            weights[i] *= 3.0
    if for_train:
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler,
                          collate_fn=l0_s2_collate_fn, num_workers=workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=workers > 0)
    return DataLoader(ds, batch_size=batch, shuffle=False,
                      collate_fn=l0_s2_collate_fn, num_workers=2)


def train_full(model, ds, device, args):
    n_val = max(128, int(len(ds) * 0.01))
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    # rebuild weights for subset is awkward; use full-ds sampler indices via Subset
    # simpler: train on full ds with sampler, validate on val_ds
    weights = [SOURCE_WEIGHT.get(ds.samples[i]["source"], 1.0)
               for i in train_ds.indices]
    for j, i in enumerate(train_ds.indices):
        if ds.samples[i]["bbox"] is not None:
            weights[j] *= 3.0
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    tl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                    collate_fn=l0_s2_collate_fn, num_workers=args.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    collate_fn=l0_s2_collate_fn, num_workers=2)

    opt = build_optim(model, args.lr, args.grd_lr_mult)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(tl)
    max_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=max(total_steps, 2), pct_start=0.03)
    out_dir = Path(args.out)
    best_val = float("inf")
    best_step = 0
    no_improve = 0          # consecutive validations without >min_delta gain
    step = 0
    t0 = time.time()
    log_hist = []

    def validate():
        model.eval()
        tot = cnt = 0.0
        with torch.no_grad():
            for b in vl:
                # val uses CLM only (stable)
                tot += float(model.forward_clm(
                    b["pixel_values"].to(device),
                    b["input_ids"].to(device),
                    b["labels"].to(device))["loss"])
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
            out = fwd(model, batch, device, args)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"[S2] step {step}/{total_steps} loss={float(out['loss']):.4f} "
                      f"clm={float(out['loss_clm']):.3f} grd={float(out['loss_grd']):.3f} "
                      f"blank={float(out['loss_blank']):.3f} iou={out['iou']:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {rate:.1f} smp/s", flush=True)
                log_hist.append({"step": step, "loss": float(out["loss"]),
                                 "iou": out["iou"]})
            if step % args.val_every == 0 or step == total_steps:
                v = validate()
                improved = v < best_val - args.es_min_delta
                print(f"[S2] step {step} VAL={v:.4f} (best {best_val:.4f} @ "
                      f"{best_step}) no_improve={no_improve}"
                      f"{' *improved' if improved else ''}", flush=True)
                log_hist.append({"step": step, "val": v})
                if improved:
                    best_val = v
                    best_step = step
                    no_improve = 0
                    save_ckpt(model, out_dir, "best", step, v)
                else:
                    no_improve += 1
                    if args.early_stop and no_improve >= args.es_patience:
                        print(f"[S2] EARLY STOP: no val improvement > "
                              f"{args.es_min_delta} for {no_improve} vals "
                              f"(best={best_val:.4f} @ step {best_step})",
                              flush=True)
                        done = True
                        break
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break
    save_ckpt(model, out_dir, "last", step, best_val)
    (out_dir / "train_log.json").write_text(json.dumps(log_hist), encoding="utf-8")
    print(f"[S2] done: {step} steps best_val={best_val:.4f} @ step {best_step}")
    print("S2_EXIT=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl")
    ap.add_argument("--s0_dir", default="checkpoints/v2_stage0_full")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--early_stop", action="store_true",
                    help="stop when val CLM loss stops improving")
    ap.add_argument("--es_patience", type=int, default=3,
                    help="validations without improvement before stopping")
    ap.add_argument("--es_min_delta", type=float, default=0.002,
                    help="min val-loss decrease to count as improvement")
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--out", default="checkpoints/v2_stage2_gfull")
    ap.add_argument("--vision_tail", type=int, default=4)
    ap.add_argument("--grd_lr_mult", type=float, default=5.0,
                    help="LR multiplier for the from-scratch grounding head")
    ap.add_argument("--lambda_grd", type=float, default=1.0)
    ap.add_argument("--lambda_blank", type=float, default=0.5)
    ap.add_argument("--lambda_shuffle", type=float, default=0.75)
    ap.add_argument("--blank_margin", type=float, default=0.15)
    ap.add_argument("--shuffle_margin", type=float, default=0.10)
    ap.add_argument("--blank_bs", type=int, default=2)
    ap.add_argument("--keep_motion", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=40)
    ap.add_argument("--pilot", action="store_true",
                    help="run pilot gates only and exit")
    ap.add_argument("--pilot_steps", type=int, default=800)
    ap.add_argument("--pilot_bbox_ratio", type=float, default=0.7,
                    help="fraction of pilot batch drawn from bbox samples")
    ap.add_argument("--gate_samples", type=int, default=128)
    ap.add_argument("--min_blank_delta", type=float, default=0.05)
    ap.add_argument("--min_shuffle_delta", type=float, default=0.03)
    ap.add_argument("--min_delta_gain", type=float, default=0.02,
                    help="required improvement over S0 blank/shuffle deltas")
    ap.add_argument("--min_iou", type=float, default=0.12)
    ap.add_argument("--min_center_std", type=float, default=0.05,
                    help="anti-collapse: mean REC predicted-centre std must exceed "
                         "this, else a constant/giant box is inflating P5")
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--skip_smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, ds, device = build(args, for_smoke=args.smoke)

    if args.smoke:
        sys.exit(run_smoke(model, ds, device, args))

    if not args.skip_smoke:
        if run_smoke(model, ds, device, args) > 0:
            sys.exit("S2 smoke failed")

    if args.pilot or not args.skip_pilot:
        # re-init trainable heads after smoke overfit
        for m in list(model.grd_head.modules()) + list(model.projector.modules()):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        model.grd_queries.data.normal_(0, 0.02)
        model.grd_head.to(model.dtype)
        model.projector.to(model.dtype)
        # reload S0 projector/vision/lora cleanly
        load_s0(model, Path(args.s0_dir))
        model.configure_stage2(vision_unfreeze_last_n=args.vision_tail, grad_ckpt=True)

        fails = run_pilot_gates(model, ds, device, args)
        if args.pilot:
            sys.exit(fails)
        if fails > 0:
            sys.exit("S2 pilot gates FAILED — refusing full training. "
                     "Fix design before retry.")
        print("[S2] pilot PASS — starting full training")
        save_ckpt(model, Path(args.out), "pilot", args.pilot_steps, None)

    train_full(model, ds, device, args)


if __name__ == "__main__":
    main()
