"""Systematic benchmark for a trained Stage-2 (VLM-SFT + grounding) checkpoint.

The deployed task is S3 action-waypoint regression, so bbox IoU is only a
*diagnostic proxy*. This benchmark therefore reports metrics tiered by how much
they matter for the downstream goal:

  TIER-A  Vision-reliance (PRIMARY): Δ = CE(blank)-CE(real) and CE(shuffle)-CE(real)
          per source. Large positive Δ ⇒ the model genuinely conditions on the
          image. This is the must-pass evidence that S2 learned to *use* vision.
  TIER-A  Answer-token accuracy (teacher-forced argmax on the answer span),
          overall / per-source / on the SPATIAL-relation subset (left/right/near/
          far/direction…). Spatial-QA accuracy is the capability we expect to
          transfer to action grounding.
  TIER-B  Language-model quality: per-source CLM CE / perplexity.
  TIER-C  Grounding diagnostics (soft): trained-head IoU per REC source, plus
          collapse probes — predicted-centre std and mean predicted vs GT area.

Runs on an eval set that is image-disjoint from training so the numbers are a
faithful held-out benchmark. Reuses the exact S2 dataset / collate so results
are directly comparable to the pilot gates and to eval_s0_systematic.py.

Usage:
  python scripts/eval_s2_systematic.py \
    --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \
    --vision_type cradio_v3_b \
    --jsonl /root/autodl-tmp/datasets/l0_cpt/eval_subset_v2.jsonl \
    --per_source_n 128 --out reports/s2_full_cradio_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.l0_dataset import L0S2Dataset, l0_s2_collate_fn  # noqa: E402
from model.aerov2 import AeroV2, _batch_iou  # noqa: E402
from training.v2_stage2_grounding import REC_SOURCES  # noqa: E402

# Spatial-relation cue words: samples whose question/answer mention these test
# the spatial reasoning we want to transfer to UAV action grounding.
SPATIAL_RE = re.compile(
    r"\b(left|right|above|below|top|bottom|near|nearer|nearest|far|farther|"
    r"farthest|closer|closest|behind|front|beside|between|adjacent|north|south|"
    r"east|west|upper|lower|corner|centre|center|middle|direction|toward|"
    r"towards|distance|metre|meter|meters|metres|row|column|side)\b", re.I)


def _loader(ds, idxs, batch, workers):
    return DataLoader(Subset(ds, idxs), batch_size=batch, shuffle=False,
                      collate_fn=l0_s2_collate_fn, num_workers=workers)


def load_s2(model: AeroV2, ckpt_dir: Path, tag: str = "best") -> dict:
    """Load a full S2 checkpoint: vision + projector + grounding heads + LoRA."""
    from peft import PeftModel

    payload = torch.load(ckpt_dir / f"{tag}_projector.pth", map_location="cpu",
                         weights_only=False)
    model.projector.load_state_dict(payload["projector"], strict=True)
    model.grd_queries.data.copy_(payload["grd_queries"].to(model.grd_queries.device))
    model.grd_ln_kv.load_state_dict(payload["grd_ln_kv"], strict=True)
    model.grd_ln_q.load_state_dict(payload["grd_ln_q"], strict=True)
    model.grd_ln_out.load_state_dict(payload["grd_ln_out"], strict=True)
    model.grd_txt_proj.load_state_dict(payload["grd_txt_proj"], strict=True)
    model.grd_attn.load_state_dict(payload["grd_attn"], strict=True)
    model.grd_head.load_state_dict(payload["grd_head"], strict=True)

    vis = torch.load(ckpt_dir / f"{tag}_vision.pth", map_location="cpu",
                     weights_only=False)
    model.vision_encoder.load_state_dict(vis, strict=True)

    # Unwrap any nested PeftModel wrappers before re-applying the S2 adapter.
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
def answer_acc(model, ds, idxs, device, batch, workers):
    """Teacher-forced argmax accuracy on answer tokens (labels != -100)."""
    correct = 0
    total = 0
    model.eval()
    for b in _loader(ds, idxs, batch, workers):
        pv = b["pixel_values"].to(device)
        ids = b["input_ids"].to(device)
        labels = b["labels"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward_clm(pv, ids, labels)
        logits = out["logits"].float()
        mask = labels != -100
        if mask.sum() == 0:
            continue
        pred = logits.argmax(dim=-1)
        correct += int((pred[mask] == labels[mask]).sum().item())
        total += int(mask.sum().item())
    return (correct / total) if total else None, total


@torch.no_grad()
def grounding_diag(model, ds, idxs, device, batch, workers):
    """Trained-head IoU + collapse probes (centre std, pred vs gt area)."""
    ious, cxs, cys, ap, ag = [], [], [], [], []
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
        p = pred[m].float()
        ious.append(_batch_iou(p, g))
        cxs.append(((p[:, 0] + p[:, 2]) * 0.5))
        cys.append(((p[:, 1] + p[:, 3]) * 0.5))
        ap.append((p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0))
        ag.append((g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0))
    if not ious:
        return None
    cx = torch.cat(cxs)
    cy = torch.cat(cys)
    return {
        "iou": round(float(torch.cat(ious).mean()), 4),
        "n": int(cx.numel()),
        "center_std": round(float((cx.std() + cy.std()) * 0.5), 4),
        "area_pred": round(float(torch.cat(ap).mean()), 4),
        "area_gt": round(float(torch.cat(ag).mean()), 4),
    }


def _is_spatial(sample) -> bool:
    # L0S2Dataset stores each turn as {"human": question, "gpt": answer};
    # fall back to conversation-style schemas if present.
    txt = " ".join(str(sample.get(k, "")) for k in ("human", "gpt"))
    if not txt.strip():
        convs = sample.get("conversations") or sample.get("messages") or []
        if isinstance(convs, list):
            txt = " ".join(str(c.get("value", "")) for c in convs)
        else:
            txt = str(convs)
    return bool(SPATIAL_RE.search(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/v2_stage2_full_cradio")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--vision_type", default="cradio_v3_b")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--jsonl",
                    default="/root/autodl-tmp/datasets/l0_cpt/eval_subset_v2.jsonl")
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--per_source_n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/s2_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    meta = load_s2(model, Path(args.ckpt_dir), args.tag)
    model.eval()
    print(f"[S2-EVAL] loaded {args.ckpt_dir}/{args.tag} "
          f"step={meta['step']} val={meta['val']}")

    report = Path(args.jsonl).parent / "build_report.json"
    image_roots = None
    if report.exists():
        image_roots = json.loads(report.read_text(encoding="utf-8")).get("image_roots")
    ds = L0S2Dataset(
        jsonl_path=args.jsonl, tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform, image_roots=image_roots,
        max_text_len=args.max_text_len, exclude_sources=None)

    by_src = defaultdict(list)
    spatial_idx = []
    for i, s in enumerate(ds.samples):
        by_src[s["source"]].append(i)
        if _is_spatial(s):
            spatial_idx.append(i)
    rng = random.Random(args.seed)

    results = {}
    all_idx = []
    for src in sorted(by_src):
        idxs = by_src[src][:]
        rng.shuffle(idxs)
        idxs = idxs[:args.per_source_n]
        all_idx += idxs
        ce, ntok = ce_modes(model, ds, idxs, device, args.batch, args.workers)
        acc, nacc = answer_acc(model, ds, idxs, device, args.batch, args.workers)
        row = {
            "n": len(idxs), "n_tok": ntok,
            "ce_real": round(ce["real"], 4),
            "ppl_real": round(math.exp(min(ce["real"], 20)), 2),
            "blank_delta": round(ce["blank"] - ce["real"], 4),
            "shuffle_delta": round(ce["shuffle"] - ce["real"], 4),
            "ans_acc": None if acc is None else round(acc, 4),
        }
        if src in REC_SOURCES:
            row["grounding"] = grounding_diag(
                model, ds, idxs, device, args.batch, args.workers)
        results[src] = row
        g = row.get("grounding")
        print(f"[{src:16s}] n={row['n']:3d} CE={row['ce_real']:.3f} "
              f"ppl={row['ppl_real']:7.2f} Δblank={row['blank_delta']:+.3f} "
              f"Δshuf={row['shuffle_delta']:+.3f} ans={row['ans_acc']}"
              + (f" | IoU={g['iou']} cstd={g['center_std']} "
                 f"area(p/g)={g['area_pred']}/{g['area_gt']}"
                 if g else ""), flush=True)

    # Spatial-relation subset (the transferable capability).
    sp = spatial_idx[:]
    rng.shuffle(sp)
    sp = sp[:max(args.per_source_n * 3, 256)]
    sp_acc, sp_n = answer_acc(model, ds, sp, device, args.batch, args.workers)
    sp_ce, _ = ce_modes(model, ds, sp, device, args.batch, args.workers)

    # Overall (token-weighted across the sampled union).
    ce, ntok = ce_modes(model, ds, all_idx, device, args.batch, args.workers)
    acc, _ = answer_acc(model, ds, all_idx, device, args.batch, args.workers)
    overall = {
        "n": len(all_idx), "n_tok": ntok,
        "ce_real": round(ce["real"], 4),
        "ppl_real": round(math.exp(min(ce["real"], 20)), 2),
        "blank_delta": round(ce["blank"] - ce["real"], 4),
        "shuffle_delta": round(ce["shuffle"] - ce["real"], 4),
        "ans_acc": None if acc is None else round(acc, 4),
    }
    spatial = {
        "n": len(sp),
        "ans_acc": None if sp_acc is None else round(sp_acc, 4),
        "blank_delta": round(sp_ce["blank"] - sp_ce["real"], 4),
        "shuffle_delta": round(sp_ce["shuffle"] - sp_ce["real"], 4),
    }
    print(f"\n[OVERALL] CE={overall['ce_real']:.3f} ppl={overall['ppl_real']:.2f} "
          f"Δblank={overall['blank_delta']:+.3f} Δshuf={overall['shuffle_delta']:+.3f} "
          f"ans_acc={overall['ans_acc']}")
    print(f"[SPATIAL] n={spatial['n']} ans_acc={spatial['ans_acc']} "
          f"Δblank={spatial['blank_delta']:+.3f} Δshuf={spatial['shuffle_delta']:+.3f}")

    out = {
        "ckpt_dir": args.ckpt_dir, "tag": args.tag,
        "vision_type": args.vision_type, "jsonl": args.jsonl,
        "step": meta["step"], "val": meta["val"],
        "per_source": results, "overall": overall, "spatial_subset": spatial,
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {op}")
    print("S2_EVAL_EXIT=0")


if __name__ == "__main__":
    main()
