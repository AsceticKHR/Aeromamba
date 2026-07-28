#!/usr/bin/env python3
"""Convert DIOR-RSVG (PASCAL-VOC XML) to L0 grounding JSONL.

Source layout (LittleCollections/DIOR-RSVG on HF):
  Annotations/<idx>.xml   VOC XML: filename, size, N objects each with
                          <bndbox> + <description> (referring expression)
  JPEGImages/<name>.jpg   800x800 aerial images
  train.txt/val.txt/test.txt  expression-level indices (0..38319)

Enumeration: iterate XMLs in sorted filename order, and objects in file order,
assigning a running expression index e (0-based). Split membership comes from the
txt files. Emits one grounding row per object in the repo's L0 format with boxes
normalized to 0..1000.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

_INT_RE = re.compile(r"\d+")


def load_split(p: Path) -> set:
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.add(int(line))
    return out


def norm(v: float, size: int, scale: int) -> int:
    return max(0, min(scale, round(v / size * scale)))


def prompt_for(desc: str) -> str:
    return (f"Locate in the aerial image: {desc}. "
            "Answer with a bounding box [x1, y1, x2, y2] normalized to 0-1000.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-dir", required=True, help="dir with <idx>.xml")
    ap.add_argument("--split-dir", required=True, help="dir with train/val/test.txt")
    ap.add_argument("--img-prefix", default="images/dior_rsvg",
                    help="relative image path prefix stored in JSONL")
    ap.add_argument("--image-root", default="l0")
    ap.add_argument("--scale", type=int, default=1000)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)
    args = ap.parse_args()

    anno = Path(args.anno_dir)
    sd = Path(args.split_dir)
    train_set = load_split(sd / "train.txt")
    val_set = load_split(sd / "val.txt")
    test_set = load_split(sd / "test.txt")

    xmls = sorted(anno.glob("*.xml"), key=lambda p: int(_INT_RE.search(p.stem).group()))

    e = 0
    n_train = n_test = n_skip = 0
    counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    ft = open(args.out_train, "w", encoding="utf-8")
    fe = open(args.out_test, "w", encoding="utf-8")
    for xp in xmls:
        try:
            root = ET.parse(xp).getroot()
        except Exception:
            continue
        fname = (root.findtext("filename") or f"{xp.stem}.jpg").strip()
        size = root.find("size")
        w = int(size.findtext("width")) if size is not None else 800
        h = int(size.findtext("height")) if size is not None else 800
        for obj in root.findall("object"):
            idx = e
            e += 1
            desc = (obj.findtext("description") or "").strip()
            bb = obj.find("bndbox")
            if not desc or bb is None:
                n_skip += 1
                continue
            xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
            if not (xmin < xmax and ymin < ymax):
                n_skip += 1
                continue
            box = [norm(xmin, w, args.scale), norm(ymin, h, args.scale),
                   norm(xmax, w, args.scale), norm(ymax, h, args.scale)]
            if not (box[0] < box[2] and box[1] < box[3]):
                n_skip += 1
                continue
            split = ("test" if idx in test_set else
                     "val" if idx in val_set else
                     "train" if idx in train_set else "unknown")
            counts[split] += 1
            row = {
                "id": f"dior_rsvg_{idx:05d}",
                "source": "dior_rsvg",
                "task": "grounding",
                "image": f"{args.img_prefix}/{fname}",
                "image_root": args.image_root,
                "split": split,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + prompt_for(desc)},
                    {"from": "gpt", "value": json.dumps(box)},
                ],
            }
            if split == "test":
                fe.write(json.dumps(row, ensure_ascii=False) + "\n"); n_test += 1
            else:
                ft.write(json.dumps(row, ensure_ascii=False) + "\n"); n_train += 1
    ft.close(); fe.close()

    print(f"[dior] xml={len(xmls)} total_objects={e} skipped={n_skip}")
    print(f"[dior] split reconcile: built={counts} | "
          f"txt(train={len(train_set)},val={len(val_set)},test={len(test_set)})")
    print(f"[dior] wrote train={n_train} -> {args.out_train}")
    print(f"[dior] wrote test={n_test}  -> {args.out_test}")


if __name__ == "__main__":
    main()
