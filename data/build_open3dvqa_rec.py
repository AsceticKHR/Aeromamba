#!/usr/bin/env python3
"""Convert Open3DVQA chunk_*.pkl (mask-bearing scenes) to L0 grounding JSONL.

Each chunk is a pandas DataFrame. Mask-bearing scenes (RealworldUAV / UrbanScene /
WildUAV) carry columns: image, caption (list), masks (list of uint8 HxW 0/255),
valid_idx (list of bool). caption[i] <-> masks[i] <-> valid_idx[i].

Boxes are the tight bounding rect of each mask's nonzero pixels, normalized to
0..scale by the image (W, H). Masks are CLIPSeg+SAM pseudo-labels, so we drop empty,
degenerate, near-full, and within-image duplicate boxes (different captions mapped to
the same mask), keeping the first caption per distinct box.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_eval_exclusion(eval_json):
    """Build a set of (scene, stem) for open3d_vqa images used in the eval subset.

    Eval image paths look like: open3d_vqa/O3DVQA/<A>/<B>/rgb/<file>.png
    -> scene "<A>_<B>", stem = <file> without extension. Matches this script's
    scene (cp.parent relative to O3DVQA, '/'->'_') and image_filename stem.
    """
    excl = set()
    if not eval_json:
        return excl
    data = json.load(open(eval_json, encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("data", [])
    for it in items:
        img = it.get("image", "")
        parts = img.split("/")
        if "O3DVQA" not in parts or "rgb" not in parts:
            continue
        k = parts.index("O3DVQA")
        try:
            a, b = parts[k + 1], parts[k + 2]
            stem = Path(parts[-1]).stem
            excl.add((f"{a}_{b}", stem))
        except IndexError:
            continue
    return excl


def mask_to_box(m):
    m = np.asarray(m)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def norm(v, size, scale):
    return max(0, min(scale, round(v / size * scale)))


def prompt_for(desc):
    return (f"Locate in the aerial image: {desc}. "
            "Answer with a bounding box [x1, y1, x2, y2] normalized to 0-1000.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-dir", required=True, help="dir with O3DVQA/<scene>/chunk_*.pkl")
    ap.add_argument("--img-out", required=True, help="dir to save RGB images")
    ap.add_argument("--img-prefix", default="images/o3dvqa_rec")
    ap.add_argument("--image-root", default="l0")
    ap.add_argument("--scale", type=int, default=1000)
    ap.add_argument("--full-area-frac", type=float, default=0.92)
    ap.add_argument("--exclude-eval-json", default=None,
                    help="eval_subset.json; drop images used in eval to avoid leakage")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_out = Path(args.img_out)
    img_out.mkdir(parents=True, exist_ok=True)
    chunks = sorted(Path(args.ext_dir).glob("O3DVQA/*/*/chunk_*.pkl"))
    exclude = load_eval_exclusion(args.exclude_eval_json)

    n_img = n_rows = n_skip_nomask = n_empty = n_degen = n_full = n_dup = n_leak = 0
    fout = open(args.out, "w", encoding="utf-8")
    for cp in chunks:
        df = pickle.load(open(cp, "rb"))
        if "masks" not in df.columns:
            continue
        scene = cp.parent.relative_to(Path(args.ext_dir) / "O3DVQA").as_posix().replace("/", "_")
        for ridx, r in df.iterrows():
            masks = r.get("masks")
            caps = r.get("caption")
            valid = r.get("valid_idx")
            if masks is None or caps is None:
                continue
            orig_stem = Path(str(r["image_filename"])).stem
            if (scene, orig_stem) in exclude:
                n_leak += 1
                continue
            img = r["image"]
            W, H = img.size
            stem = f"{scene}_{orig_stem}"
            img_name = f"{stem}.jpg"
            img_path = img_out / img_name
            if not img_path.exists():
                img.convert("RGB").save(img_path, quality=92)
            n_img += 1

            seen_boxes = set()
            for i, cap in enumerate(caps):
                if valid is not None and i < len(valid) and not bool(valid[i]):
                    continue
                if i >= len(masks):
                    continue
                box_px = mask_to_box(masks[i])
                if box_px is None:
                    n_empty += 1
                    continue
                x1, y1, x2, y2 = box_px
                if not (x1 < x2 and y1 < y2):
                    n_degen += 1
                    continue
                area_frac = (x2 - x1) * (y2 - y1) / (W * H)
                if area_frac > args.full_area_frac:
                    n_full += 1
                    continue
                key = box_px
                if key in seen_boxes:
                    n_dup += 1
                    continue
                seen_boxes.add(key)
                box = [norm(x1, W, args.scale), norm(y1, H, args.scale),
                       norm(x2, W, args.scale), norm(y2, H, args.scale)]
                if not (box[0] < box[2] and box[1] < box[3]):
                    n_degen += 1
                    continue
                row = {
                    "id": f"o3dvqa_rec_{stem}_{i}",
                    "source": "open3d_vqa_rec",
                    "task": "grounding",
                    "image": f"{args.img_prefix}/{img_name}",
                    "image_root": args.image_root,
                    "scene": scene,
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + prompt_for(str(cap).strip())},
                        {"from": "gpt", "value": json.dumps(box)},
                    ],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    fout.close()
    print(f"[o3d-rec] mask chunks used; images={n_img} rows={n_rows}")
    print(f"[o3d-rec] dropped: empty={n_empty} degenerate={n_degen} "
          f"full(>{args.full_area_frac})={n_full} dup_box={n_dup} eval_leak_imgs={n_leak}")
    print(f"[o3d-rec] wrote -> {args.out}")


if __name__ == "__main__":
    main()
