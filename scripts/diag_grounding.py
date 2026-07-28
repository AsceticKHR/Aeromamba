"""Grounding diagnostic: can the head overfit a tiny fixed bbox set?

Isolates the grounding path from CLM/hinge/data-ratio confounds:
  1. dumps GT bbox values (sanity-check parsing / xyxy convention),
  2. trains ONLY the grounding loss on a fixed small batch with high LR,
  3. prints grd/IoU trend.

If grd → ~0 and IoU → high  → head + targets are fine (pilot issue is signal
balance / generalisation). If grd stays flat → target or gradient path is broken.
"""
import argparse
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.v2_stage2_grounding import build, l0_s2_collate_fn  # noqa: E402
from model.aerov2 import _batch_iou, _giou_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl")
    ap.add_argument("--s0_dir", default="checkpoints/v2_stage0_full")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--vision_tail", type=int, default=4)
    ap.add_argument("--keep_motion", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "l1iou", "l1giou"], default="l1iou")
    ap.add_argument("--distinct", action="store_true",
                    help="pick samples with distinct GT boxes")
    args = ap.parse_args()

    args.smoke = True  # build() expects this attr
    model, ds, device = build(args, for_smoke=True)

    bbox_idx = [i for i, s in enumerate(ds.samples) if s["bbox"] is not None]
    print(f"[diag] total bbox samples = {len(bbox_idx)}")
    if args.distinct:
        seen = set()
        fixed = []
        for i in bbox_idx:
            key = tuple(round(v, 3) for v in ds.samples[i]["bbox"].tolist())
            if key in seen:
                continue
            seen.add(key)
            fixed.append(i)
            if len(fixed) >= args.n:
                break
    else:
        fixed = bbox_idx[:args.n]
    items = [ds[i] for i in fixed]

    print("[diag] GT bbox values (xyxy in [0,1]) + source:")
    for i, it in zip(fixed, items):
        b = it["bbox"].tolist()
        src = ds.samples[i]["source"]
        x1, y1, x2, y2 = b
        valid = (x2 > x1) and (y2 > y1)
        print(f"  src={src:16s} bbox=[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] "
              f"valid_xyxy={valid} area={(x2-x1)*(y2-y1):.3f}")

    batch = l0_s2_collate_fn(items)
    pv = batch["pixel_values"].to(device)
    ids = batch["input_ids"].to(device)
    lbl = batch["labels"].to(device)
    bbox = batch["bbox"].to(device)
    hb = batch["has_bbox"].to(device)

    grd_params = (list(model.grd_head.parameters())
                  + list(model.grd_txt_proj.parameters())
                  + list(model.grd_attn.parameters())
                  + [model.grd_queries])
    opt = torch.optim.AdamW(grd_params, lr=args.lr)

    model.train()
    print("\n[diag] grounding-only overfit (grd params only, trunk frozen path):")
    for step in range(args.steps + 1):
        enc = model._encode_multimodal(pv, ids, append_grd=False)
        hidden = model._trunk_hidden(enc["inputs_embeds"], enc["attn"])
        n_vis, L_txt = enc["n_vis"], enc["L_txt"]
        pred = model.predict_boxes(hidden, n_vis, L_txt, ids)
        mask = hb > 0.5
        p, g = pred[mask], bbox[mask].float()
        l1 = torch.nn.functional.l1_loss(p, g)
        giou = _giou_loss(p, g).mean()
        iou_l = (1.0 - _batch_iou(p, g)).mean()
        if args.loss == "l1":
            loss = l1
        elif args.loss == "l1iou":
            loss = l1 + iou_l
        else:
            loss = l1 + giou
        with torch.no_grad():
            iou = _batch_iou(p, g).mean()
        if step % 20 == 0:
            pstd = p.std(dim=0).mean().item()
            print(f"  step {step:3d} grd={float(loss):.4f} l1={float(l1):.4f} "
                  f"giou={float(giou):.4f} iou={float(iou):.4f} "
                  f"pred_batch_std={pstd:.4f}")
            if step in (0, args.steps):
                for j in range(min(4, p.size(0))):
                    print(f"       s{j} pred=[{p[j,0]:.3f},{p[j,1]:.3f},"
                          f"{p[j,2]:.3f},{p[j,3]:.3f}] gt=[{g[j,0]:.3f},"
                          f"{g[j,1]:.3f},{g[j,2]:.3f},{g[j,3]:.3f}]")
        if step < args.steps:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    print("[diag] done")


if __name__ == "__main__":
    main()
