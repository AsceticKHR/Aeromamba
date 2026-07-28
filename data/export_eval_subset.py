"""
Export the exact per-source evaluation subset used by
scripts/eval_stage2_capabilities.py, so the eval can run on another machine
without shipping the full 300MB json + 300k images.

Reproduces the selection deterministically:
  1. val split = torch.randperm(len(ds), seed=--split_seed)[n_train:]
  2. group val indices by item["source"] (in val-index order)
  3. per-source shuffle with random.Random(--seed)
  4. keep first max(--loss_per_source, --gen_per_source) items per source

Writes:
  <out_dir>/eval_subset.json      selected items (same schema, original order
                                  preserved per source)
  <out_dir>/images/...            referenced images, mirroring data_root paths

Memory-safe (streams the json twice with ijson), runs in a 2GB cgroup.

Usage:
  python data/export_eval_subset.py \
      --json /root/autodl-tmp/Aeromamba/data/stage2_mixed_data_v2.json \
      --data_root /root/autodl-tmp/Aeromamba/data \
      --out_dir /root/autodl-tmp/eval_subset_export \
      --loss_per_source 150 --gen_per_source 40
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss_per_source", type=int, default=150)
    ap.add_argument("--gen_per_source", type=int, default=40)
    args = ap.parse_args()

    try:
        import ijson
    except ImportError:
        sys.exit("pip install ijson")

    # pass 1: per-index source tags (mirrors LLaVADataset.sources)
    sources = []
    with open(args.json, "rb") as f:
        for item in ijson.items(f, "item"):
            sources.append(item.get("source", "general"))
    n = len(sources)
    print(f"[export] dataset size: {n}")

    # reproduce random_split(seed) -> val indices, in split order
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_indices = perm[n_train:]

    by_source = defaultdict(list)
    for i in val_indices:
        by_source[sources[i]].append(i)
    rng = random.Random(args.seed)
    n_keep = max(args.loss_per_source, args.gen_per_source)
    selected: dict = {}  # global idx -> (source, rank within source)
    for s in by_source:
        rng.shuffle(by_source[s])
        for rank, i in enumerate(by_source[s][:n_keep]):
            selected[i] = (s, rank)
    print(f"[export] selected {len(selected)} samples "
          f"({n_keep}/source x {len(by_source)} sources)")

    # pass 2: collect selected items
    picked = {}
    with open(args.json, "rb") as f:
        for idx, item in enumerate(ijson.items(f, "item")):
            if idx in selected:
                picked[idx] = item

    # per-source rank order must match eval-script iteration order
    out_items = []
    for s in sorted(by_source):
        ranked = sorted(
            (i for i in picked if selected[i][0] == s),
            key=lambda i: selected[i][1],
        )
        out_items.extend(picked[i] for i in ranked)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    for item in out_items:
        rel = item["image"]
        src = Path(args.data_root) / rel
        dst = img_dir / rel
        if not src.is_file():
            n_missing += 1
            print(f"[export] MISSING image: {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    with open(out_dir / "eval_subset.json", "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False)

    counts = defaultdict(int)
    for item in out_items:
        counts[item.get("source", "general")] += 1
    print(f"[export] wrote {len(out_items)} items, {n_missing} images missing")
    for s, c in sorted(counts.items()):
        print(f"  {s:16s} {c}")


if __name__ == "__main__":
    main()
