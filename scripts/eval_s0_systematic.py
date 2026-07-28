"""Systematic evaluation of an S0 (aerial CPT) checkpoint.

Measures, on an eval set that is image-disjoint from training:
  1. Per-source CLM cross-entropy / perplexity (token-weighted).
  2. Vision-reliance ablation: Δ = CE(blank) - CE(real) and
     CE(shuffle) - CE(real). Large positive Δ => the model genuinely
     conditions on the image (alignment worked) rather than guessing blind.
  3. Grounding IoU on REC sources (dior_rsvg / open3d_vqa_rec) with the
     *untrained* grounding head — a floor baseline before S2.

Runs standalone; reuses the exact dataset / collate / loader as S2 so the
numbers are directly comparable to the pilot gates.

Usage:
  python scripts/eval_s0_systematic.py \
    --s0_dir checkpoints/v2_stage0_cradio --vision_type cradio_v3_b \
    --jsonl /root/autodl-tmp/datasets/l0_cpt/eval_subset_v2.jsonl \
    --per_source_n 96 --out reports/s0_cradio_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.l0_dataset import L0S2Dataset, l0_s2_collate_fn  # noqa: E402
from model.aerov2 import AeroV2, _batch_iou  # noqa: E402
from training.v2_stage2_grounding import load_s0, REC_SOURCES  # noqa: E402


def _loader(ds, idxs, batch, workers):
    return DataLoader(Subset(ds, idxs), batch_size=batch, shuffle=False,
                      collate_fn=l0_s2_collate_fn, num_workers=workers)


@torch.no_grad()
def ce_modes(model, ds, idxs, device, batch, workers):
    """Token-weighted mean CE for real / blank / shuffled image."""
    tot = {"real": 0.0, "blank": 0.0, "shuffle": 0.0}
    ntok = 0
    model.eval()
    for b in _loader(ds, idxs, batch, workers):
        pv = b["pixel_values"].to(device)
        ids = b["input_ids"].to(device)
        labels = b["labels"].to(device)
        n = int((labels != -100).sum().item())
        if n == 0:
            continue
        ntok += n
        variants = {
            "real": pv,
            "blank": torch.zeros_like(pv),
            "shuffle": pv[torch.randperm(pv.size(0), device=pv.device)],
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k, p in variants.items():
                out = model.forward_clm(p, ids, labels)
                tot[k] += float(out["loss"]) * n
    return {k: v / max(ntok, 1) for k, v in tot.items()}, ntok


@torch.no_grad()
def grounding_iou(model, ds, idxs, device, batch, workers):
    vals = []
    model.eval()
    for b in _loader(ds, idxs, batch, workers):
        m = b["has_bbox"].to(device) > 0.5
        if not m.any():
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model._encode_multimodal(
                b["pixel_values"].to(device), b["input_ids"].to(device),
                append_grd=False)
            hidden = model._trunk_hidden(enc["inputs_embeds"], enc["attn"])
            pred = model.predict_boxes(hidden, enc["n_vis"], enc["L_txt"],
                                       b["input_ids"].to(device),
                                       b["labels"].to(device),
                                       vis_kv=enc["inputs_embeds"][:, :enc["n_vis"], :])
        g = b["bbox"].to(device)[m].float()
        vals.append(_batch_iou(pred[m], g).float())
    if not vals:
        return None
    return float(torch.cat(vals).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0_dir", default="checkpoints/v2_stage0_cradio")
    ap.add_argument("--vision_type", default="cradio_v3_b")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--jsonl",
                    default="/root/autodl-tmp/datasets/l0_cpt/eval_subset_v2.jsonl")
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--per_source_n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/s0_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    load_s0(model, Path(args.s0_dir))
    model.eval()

    report = Path(args.jsonl).parent / "build_report.json"
    image_roots = None
    if report.exists():
        image_roots = json.loads(report.read_text(encoding="utf-8")).get("image_roots")
    ds = L0S2Dataset(
        jsonl_path=args.jsonl, tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform, image_roots=image_roots,
        max_text_len=args.max_text_len, exclude_sources=None)

    by_src = defaultdict(list)
    for i, s in enumerate(ds.samples):
        by_src[s["source"]].append(i)
    rng = random.Random(args.seed)

    results = {}
    all_idx = []
    for src in sorted(by_src):
        idxs = by_src[src][:]
        rng.shuffle(idxs)
        idxs = idxs[:args.per_source_n]
        all_idx += idxs
        ce, ntok = ce_modes(model, ds, idxs, device, args.batch, args.workers)
        row = {
            "n": len(idxs), "n_tok": ntok,
            "ce_real": round(ce["real"], 4),
            "ppl_real": round(math.exp(min(ce["real"], 20)), 2),
            "blank_delta": round(ce["blank"] - ce["real"], 4),
            "shuffle_delta": round(ce["shuffle"] - ce["real"], 4),
        }
        if src in REC_SOURCES:
            iou = grounding_iou(model, ds, idxs, device, args.batch, args.workers)
            row["grd_iou_untrained"] = None if iou is None else round(iou, 4)
        results[src] = row
        print(f"[{src:16s}] n={row['n']:3d} CE={row['ce_real']:.3f} "
              f"ppl={row['ppl_real']:7.2f} Δblank={row['blank_delta']:+.3f} "
              f"Δshuf={row['shuffle_delta']:+.3f}"
              + (f" IoU0={row.get('grd_iou_untrained')}"
                 if src in REC_SOURCES else ""), flush=True)

    # Overall (token-weighted across the sampled union)
    ce, ntok = ce_modes(model, ds, all_idx, device, args.batch, args.workers)
    overall = {
        "n": len(all_idx), "n_tok": ntok,
        "ce_real": round(ce["real"], 4),
        "ppl_real": round(math.exp(min(ce["real"], 20)), 2),
        "blank_delta": round(ce["blank"] - ce["real"], 4),
        "shuffle_delta": round(ce["shuffle"] - ce["real"], 4),
    }
    print(f"\n[OVERALL] CE={overall['ce_real']:.3f} ppl={overall['ppl_real']:.2f} "
          f"Δblank={overall['blank_delta']:+.3f} Δshuf={overall['shuffle_delta']:+.3f}")

    out = {
        "s0_dir": args.s0_dir, "vision_type": args.vision_type,
        "jsonl": args.jsonl, "per_source": results, "overall": overall,
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")
    print("S0_EVAL_EXIT=0")


if __name__ == "__main__":
    main()
