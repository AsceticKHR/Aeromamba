#!/usr/bin/env python3
"""Build an IMAGE-LEVEL held-out eval subset from a merged training JSONL.

Reserves whole images (seeded) per source for evaluation, removes every row on those
images from training, and emits both files so that the eval set is guaranteed
image-disjoint from training (no train/eval leakage). Includes a grounding split so
P5 IoU can be measured on held-out boxes.

Outputs:
  --out-eval  (JSON list, eval_subset.json schema): rows on held-out images
  --out-train (JSONL): merged rows with held-out images removed
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

# per-source held-out IMAGE counts
CAP_DEFAULT = {
    "general": 150, "aerial_spatial": 150, "cognitive": 150,
    "hrvqa": 150, "uav_motion": 150,
}
# grounding sources: reserve grounding images for the P5 IoU eval
GRD_CAP = {"dior_rsvg": 250, "open3d_vqa_rec": 120, "airspatial": 120}
# cap eval rows per source so the eval stays small
ROWS_CAP = {"__default__": 150, "dior_rsvg": 250, "open3d_vqa_rec": 150,
            "airspatial": 200}


def has_bbox(row):
    for t in row.get("conversations", []):
        if t.get("from") == "gpt" and "[" in t.get("value", ""):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    img_src = {}
    imgs_by_src = defaultdict(set)
    grd_imgs_by_src = defaultdict(set)
    with open(args.merged, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
            im = r.get("image", "")
            s = r.get("source", "?")
            img_src[im] = s
            imgs_by_src[s].add(im)
            if r.get("task") == "grounding":
                grd_imgs_by_src[s].add(im)

    rng = random.Random(args.seed)
    holdout = set()
    # 1) grounding holdout first (priority for P5 eval)
    for s, cap in GRD_CAP.items():
        pool = sorted(grd_imgs_by_src.get(s, []))
        rng.shuffle(pool)
        holdout.update(pool[:cap])
    # 2) capability holdout (skip images already held out)
    for s, cap in CAP_DEFAULT.items():
        pool = sorted(im for im in imgs_by_src.get(s, []) if im not in holdout)
        rng.shuffle(pool)
        holdout.update(pool[:cap])

    # split rows; cap eval rows per source
    eval_items, train_lines = [], []
    eval_src_count = defaultdict(int)
    for r in rows:
        if r.get("image", "") in holdout:
            s = r.get("source", "?")
            cap = ROWS_CAP.get(s, ROWS_CAP["__default__"])
            if eval_src_count[s] < cap:
                eval_items.append(r)
                eval_src_count[s] += 1
            # rows beyond the cap on held-out images are dropped from BOTH
            # sets to preserve image-level disjointness
        else:
            train_lines.append(r)

    # verify disjoint
    eval_imgs = {r.get("image", "") for r in eval_items}
    train_imgs = {r.get("image", "") for r in train_lines}
    inter = eval_imgs & train_imgs
    assert not inter, f"LEAKAGE: {len(inter)} images in both"

    with open(args.out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, ensure_ascii=False)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in train_lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[heldout] held-out images={len(holdout)}")
    print(f"[heldout] eval items={len(eval_items)} | train rows={len(train_lines)}")
    print("[heldout] eval per-source:")
    for s in sorted(eval_src_count):
        ng = sum(1 for r in eval_items if r.get("source") == s and r.get("task") == "grounding")
        print(f"    {s:16s} items={eval_src_count[s]:4d} grounding={ng}")
    print(f"[heldout] disjoint check OK (0 shared images)")


if __name__ == "__main__":
    main()
