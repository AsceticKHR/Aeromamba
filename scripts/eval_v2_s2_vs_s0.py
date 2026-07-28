"""Compare S2 (early-stopped) vs S0 on aerial holdout + vision ablations.

Reuses the same aerial slice / gates as eval_v2_s0.py so numbers are
directly comparable. Verdict: keep S2 only if blank & shuffle deltas beat S0.

Usage:
  python scripts/eval_v2_s2_vs_s0.py \
      --s2_dir checkpoints/v2_stage2_full \
      --s0_dir checkpoints/v2_stage0_full \
      --report reports/v2_s2_vs_s0_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.l0_dataset import L0CPTDataset  # noqa: E402
from data.llava_dataset import llava_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402

AERIAL = {"aerial_spatial", "hrvqa", "airspatial", "uav_motion"}


def load_stage(dir_path: Path, device, backbone, vision_type, tag: str):
    from peft import PeftModel

    m = AeroV2(backbone_id=backbone, vision_type=vision_type).to(device)
    proj = torch.load(dir_path / "best_projector.pth", map_location="cpu",
                      weights_only=False)
    m.projector.load_state_dict(proj["projector"], strict=True)
    vis = torch.load(dir_path / "best_vision.pth", map_location="cpu",
                     weights_only=False)
    m.vision_encoder.load_state_dict(vis, strict=True)
    m.lm = PeftModel.from_pretrained(m.lm, str(dir_path / "best_lora"))
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    print(f"[eval] loaded {tag} from {dir_path} "
          f"step={proj.get('step')} val={proj.get('val')}")
    return m, proj


@torch.no_grad()
def accumulate(model, loader, device, mutate=None):
    tot = cnt = 0.0
    n_tok = n_ok = 0
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        if mutate == "blank":
            pv = torch.zeros_like(pv)
        elif mutate == "shuffle" and pv.size(0) > 1:
            pv = pv.roll(1, dims=0)
        out = model.forward_clm(pv, ids, lbl)
        tot += float(out["loss"])
        cnt += 1
        pred = out["logits"].argmax(-1)
        mask = lbl != -100
        n_tok += int(mask.sum())
        n_ok += int(((pred == lbl) & mask).sum())
    return {
        "loss": tot / max(cnt, 1),
        "token_acc": n_ok / max(n_tok, 1),
        "n_tok": n_tok,
        "n_batches": int(cnt),
    }


@torch.no_grad()
def per_source(model, ds, indices_by_src, device, batch=8):
    out = {}
    for src, idxs in indices_by_src.items():
        if not idxs:
            continue
        loader = DataLoader(Subset(ds, idxs), batch_size=batch, shuffle=False,
                            collate_fn=llava_collate_fn, num_workers=0)
        out[src] = accumulate(model, loader, device)
    return out


def eval_model(model, ds, eval_idx, src_eval, device, batch):
    loader = DataLoader(Subset(ds, eval_idx), batch_size=batch, shuffle=False,
                        collate_fn=llava_collate_fn, num_workers=2)
    real = accumulate(model, loader, device)
    blank = accumulate(model, loader, device, mutate="blank")
    shuf = accumulate(model, loader, device, mutate="shuffle")
    src = per_source(model, ds, src_eval, device, batch=batch)
    return {
        "real": real,
        "blank": blank,
        "shuffle": shuf,
        "blank_delta": blank["loss"] - real["loss"],
        "shuffle_delta": shuf["loss"] - real["loss"],
        "per_source": src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_dir", default="checkpoints/v2_stage2_full")
    ap.add_argument("--s0_dir", default="checkpoints/v2_stage0_full")
    ap.add_argument("--jsonl", default="/root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--n_samples", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--min_blank_delta", type=float, default=0.05)
    ap.add_argument("--min_shuffle_delta", type=float, default=0.03)
    ap.add_argument("--report", default="reports/v2_s2_vs_s0_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    report_path = Path(args.jsonl).parent / "build_report.json"
    image_roots = None
    if report_path.exists():
        image_roots = json.loads(report_path.read_text(encoding="utf-8")).get(
            "image_roots")

    # ---- S2 first ----
    s2, s2_meta = load_stage(Path(args.s2_dir), device, args.backbone,
                             args.vision_type, "S2")
    ds = L0CPTDataset(
        jsonl_path=args.jsonl,
        tokenizer=s2.tokenizer,
        transform=s2.vision_encoder.transform,
        image_roots=image_roots,
        max_text_len=args.max_text_len,
    )
    aerial = [i for i, r in enumerate(ds.rows) if r.get("source") in AERIAL]
    rng.shuffle(aerial)
    eval_idx = aerial[: args.n_samples]
    by_src = {}
    for i in aerial:
        by_src.setdefault(ds.rows[i]["source"], []).append(i)
    src_eval = {s: idxs[:48] for s, idxs in by_src.items()}
    print(f"[eval] aerial n={len(eval_idx)}")

    s2_m = eval_model(s2, ds, eval_idx, src_eval, device, args.batch)
    del s2
    torch.cuda.empty_cache()

    # ---- S0 baseline (same indices) ----
    s0, s0_meta = load_stage(Path(args.s0_dir), device, args.backbone,
                             args.vision_type, "S0")
    ds0 = L0CPTDataset(
        jsonl_path=args.jsonl,
        tokenizer=s0.tokenizer,
        transform=s0.vision_encoder.transform,
        image_roots=image_roots,
        max_text_len=args.max_text_len,
    )
    s0_m = eval_model(s0, ds0, eval_idx, src_eval, device, args.batch)
    del s0
    torch.cuda.empty_cache()

    # Gates: S2 must improve vision use vs S0
    blank_gain = s2_m["blank_delta"] - s0_m["blank_delta"]
    shuf_gain = s2_m["shuffle_delta"] - s0_m["shuffle_delta"]
    gates = {
        "G1_s2_blank_absolute": {
            "ok": s2_m["blank_delta"] >= args.min_blank_delta,
            "value": s2_m["blank_delta"],
            "threshold": f">= {args.min_blank_delta}",
        },
        "G2_s2_shuffle_absolute": {
            "ok": s2_m["shuffle_delta"] >= args.min_shuffle_delta,
            "value": s2_m["shuffle_delta"],
            "threshold": f">= {args.min_shuffle_delta}",
        },
        "G3_blank_beats_s0": {
            "ok": blank_gain > 0.0,
            "value": blank_gain,
            "threshold": "> 0 vs S0",
            "s2": s2_m["blank_delta"],
            "s0": s0_m["blank_delta"],
        },
        "G4_shuffle_beats_s0": {
            "ok": shuf_gain > 0.0,
            "value": shuf_gain,
            "threshold": "> 0 vs S0",
            "s2": s2_m["shuffle_delta"],
            "s0": s0_m["shuffle_delta"],
        },
        "G5_airspatial_loss_not_worse": {
            "ok": (s2_m["per_source"].get("airspatial", {}).get("loss", 99)
                   <= s0_m["per_source"].get("airspatial", {}).get("loss", 0) + 0.05),
            "value": (
                s0_m["per_source"].get("airspatial", {}).get("loss", float("nan"))
                - s2_m["per_source"].get("airspatial", {}).get("loss", float("nan"))),
            "threshold": "S2 airspatial loss <= S0 + 0.05",
        },
    }
    fails = sum(1 for g in gates.values() if not g["ok"])
    # Keep S2 only if absolute vision gates pass OR both relative gains positive
    keep_s2 = (
        gates["G1_s2_blank_absolute"]["ok"] and gates["G2_s2_shuffle_absolute"]["ok"]
    ) or (
        gates["G3_blank_beats_s0"]["ok"] and gates["G4_shuffle_beats_s0"]["ok"]
        and blank_gain + shuf_gain >= 0.05
    )
    recommendation = "KEEP_S2" if keep_s2 else "REVERT_S0"

    report = {
        "recommendation": recommendation,
        "hard_failures": fails,
        "n_samples": len(eval_idx),
        "s2_meta": {"step": s2_meta.get("step"), "val": s2_meta.get("val")},
        "s0_meta": {"step": s0_meta.get("step"), "val": s0_meta.get("val")},
        "s2": s2_m,
        "s0": s0_m,
        "gates": gates,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== S2 vs S0 vision compare ===")
    print(f"S2 real loss={s2_m['real']['loss']:.4f}  blankΔ={s2_m['blank_delta']:+.4f}  "
          f"shufΔ={s2_m['shuffle_delta']:+.4f}  acc={s2_m['real']['token_acc']:.3f}")
    print(f"S0 real loss={s0_m['real']['loss']:.4f}  blankΔ={s0_m['blank_delta']:+.4f}  "
          f"shufΔ={s0_m['shuffle_delta']:+.4f}  acc={s0_m['real']['token_acc']:.3f}")
    print("\nper-source loss (s2 / s0):")
    for src in sorted(set(s2_m["per_source"]) | set(s0_m["per_source"])):
        a = s2_m["per_source"].get(src, {}).get("loss", float("nan"))
        b = s0_m["per_source"].get(src, {}).get("loss", float("nan"))
        print(f"  {src:16s}  s2={a:.3f}  s0={b:.3f}  Δ(s0-s2)={b-a:+.3f}")
    for name, g in gates.items():
        print(f"[{'PASS' if g['ok'] else 'FAIL'}] {name}: "
              f"value={g['value']:.4f} ({g['threshold']})")
    print(f"\nrecommendation={recommendation} hard_failures={fails}")
    print(f"report -> {out}")
    sys.exit(0 if keep_s2 else 2)


if __name__ == "__main__":
    main()
