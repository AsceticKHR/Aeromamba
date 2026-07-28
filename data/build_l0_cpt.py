"""Build the unified L0 aerial-domain CPT dataset (AeroMamba v2, design doc §2.1).

Merges three groups into one streaming-friendly JSONL + resized image trees:

  1. stage2_mixed_data_v2.json   (204k, LLaVA fmt, sources: general/aerial_spatial/
                                  uav_motion/cognitive)  -> referenced as-is
  2. HRVQA        (raw/hrvqa)     join Q+A by question_id, group per image into
                                  multi-turn conversations; 1024px PNG -> 512px JPEG
  3. AirSpatial   (raw/airspatial) rec_train -> grounding (bbox normalized 0-1000),
                                  qa_train  -> metric QA (distance / depth / type /
                                  color) generated from structured fields;
                                  DJI JPGs resized to max-side 1024 JPEG

Output layout (root = --out, default /root/autodl-tmp/datasets/l0_cpt):
  l0_mixed.jsonl          one sample per line:
      {id, source, task, image, image_root, conversations}
  images/hrvqa/<id>.jpg
  images/airspatial/<image_id>.jpg
  build_report.json       per-source counts + skip stats

All bboxes are emitted in Qwen-style normalized integer coords [0,1000).
CPU-only; image conversion uses a process pool (--workers).
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# NOTE: the no-GPU AutoDL mode enforces a 2GB cgroup memory cap. Keep peak
# memory low: reservoir-sample QA pairs per image, free parsed JSON blobs
# before spawning the image pool, and keep worker count modest.


# ---------------------------------------------------------------- image utils
def resize_save(args_tuple):
    """(src, dst, max_side) -> (ok, w, h) ; JPEG q90, RGB."""
    src, dst, max_side = args_tuple
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=90)
        return True, w, h
    except Exception:
        return False, 0, 0


def norm_box(bbox, w, h):
    """pixel [x1,y1,x2,y2] -> normalized int [0,1000), clipped."""
    x1, y1, x2, y2 = bbox
    def n(v, s):
        return max(0, min(999, round(v * 1000.0 / s)))
    return [n(x1, w), n(y1, h), n(x2, w), n(y2, h)]


# ------------------------------------------------------------------- group 1
def emit_stage2(src_json: Path, fout, image_root: str) -> dict:
    data = json.loads(src_json.read_text(encoding="utf-8"))
    task_map = {"general": "vqa", "aerial_spatial": "spatial_vqa",
                "uav_motion": "motion", "cognitive": "cognitive"}
    n = defaultdict(int)
    for row in data:
        out = {
            "id": row["id"],
            "source": row.get("source", "general"),
            "task": task_map.get(row.get("source", ""), "vqa"),
            "image": row["image"],
            "image_root": image_root,
            "conversations": row["conversations"],
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        n[out["source"]] += 1
    return dict(n)


# ------------------------------------------------------------------- group 2
HRVQA_Q_TEMPLATES = None  # questions come with the data


def emit_hrvqa(raw: Path, out_root: Path, fout, workers: int, max_side: int = 512,
               max_turns: int = 8) -> dict:
    ans = json.loads((raw / "jsons" / "train_answer.json").read_text(encoding="utf-8"))["annotations"]
    a_by_qid = {a["question_id"]: a["multiple_choice_answer"] for a in ans}
    del ans
    gc.collect()

    qs = json.loads((raw / "jsons" / "train_question.json").read_text(encoding="utf-8"))["questions"]
    by_img: dict = defaultdict(list)
    seen: dict = defaultdict(int)
    rng = random.Random(0)
    for q in qs:
        a = a_by_qid.get(q["question_id"])
        if a is None:
            continue
        bucket = by_img[q["image_id"]]
        seen[q["image_id"]] += 1
        # reservoir sampling: keep at most max_turns pairs per image
        if len(bucket) < max_turns:
            bucket.append((q["question"], str(a)))
        else:
            j = rng.randrange(seen[q["image_id"]])
            if j < max_turns:
                bucket[j] = (q["question"], str(a))
    del qs, a_by_qid, seen
    gc.collect()

    img_dir = raw / "images"
    jobs, kept_ids = [], []
    for img_id in sorted(by_img):
        src = img_dir / f"{img_id}.png"
        if src.exists():
            jobs.append((str(src), str(out_root / "images" / "hrvqa" / f"{img_id}.jpg"), max_side))
            kept_ids.append(img_id)

    ok_ids = set()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for img_id, (ok, _, _) in zip(kept_ids, ex.map(resize_save, jobs, chunksize=16)):
            if ok:
                ok_ids.add(img_id)

    n = 0
    for img_id in kept_ids:
        if img_id not in ok_ids:
            continue
        pairs = by_img[img_id]
        random.shuffle(pairs)
        conv = []
        for i, (q, a) in enumerate(pairs):
            prefix = "<image>\n" if i == 0 else ""
            conv.append({"from": "human", "value": prefix + q})
            conv.append({"from": "gpt", "value": str(a)})
        fout.write(json.dumps({
            "id": f"hrvqa_{img_id}",
            "source": "hrvqa",
            "task": "aerial_vqa",
            "image": f"images/hrvqa/{img_id}.jpg",
            "image_root": "l0",
            "conversations": conv,
        }, ensure_ascii=False) + "\n")
        n += 1
    return {"hrvqa": n, "hrvqa_images_failed": len(kept_ids) - len(ok_ids)}


# ------------------------------------------------------------------- group 3
def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def emit_airspatial(raw: Path, out_root: Path, fout, workers: int, max_side: int = 1024) -> dict:
    img_zip = raw / "images.zip"
    img_dir = raw / "images_unzipped"
    if img_zip.exists() and not img_dir.exists():
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(img_dir)
    # locate images (zip may nest a folder)
    idx = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            idx[p.name] = p

    # collect rows per image, with original sizes needed for bbox normalisation
    rec_rows = list(load_jsonl(raw / "airspatial_rec_train.jsonl"))
    qa_rows = list(load_jsonl(raw / "airspatial_qa_train.jsonl"))
    needed = sorted({r["image_id"] for r in rec_rows} | {r["image_id"] for r in qa_rows})

    jobs, kept = [], []
    for im_id in needed:
        src = idx.get(im_id)
        if src is None:
            continue
        dst = out_root / "images" / "airspatial" / (Path(im_id).stem + ".jpg")
        jobs.append((str(src), str(dst), max_side))
        kept.append(im_id)

    sizes = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for im_id, (ok, w, h) in zip(kept, ex.map(resize_save, jobs, chunksize=16)):
            if ok:
                sizes[im_id] = (w, h)

    n_rec = n_qa = 0
    # grounding: expression -> box
    by_img = defaultdict(list)
    for r in rec_rows:
        if r["image_id"] in sizes:
            by_img[r["image_id"]].append(r)
    for im_id, rows in by_img.items():
        w, h = sizes[im_id]
        random.shuffle(rows)
        conv = []
        for i, r in enumerate(rows[:6]):
            box = norm_box(r["bbox"], w, h)
            prefix = "<image>\n" if i == 0 else ""
            conv.append({"from": "human",
                         "value": prefix + f"Locate in the aerial image: {r['question']}. "
                                           "Answer with a bounding box [x1, y1, x2, y2] normalized to 0-1000."})
            conv.append({"from": "gpt", "value": json.dumps(box)})
        fout.write(json.dumps({
            "id": f"airspatial_rec_{im_id}",
            "source": "airspatial",
            "task": "grounding",
            "image": f"images/airspatial/{Path(im_id).stem}.jpg",
            "image_root": "l0",
            "conversations": conv,
        }, ensure_ascii=False) + "\n")
        n_rec += 1

    # metric QA from structured fields
    by_img_qa = defaultdict(list)
    for r in qa_rows:
        if r["image_id"] in sizes:
            by_img_qa[r["image_id"]].append(r)
    for im_id, rows in by_img_qa.items():
        w, h = sizes[im_id]
        random.shuffle(rows)
        conv = []
        for i, r in enumerate(rows[:6]):
            box = norm_box(r["bbox"], w, h)
            color, typ = r.get("color", ""), r.get("type", "vehicle")
            prefix = "<image>\n" if i == 0 else ""
            qtype = random.random()
            if qtype < 0.5 and r.get("distance"):
                conv.append({"from": "human",
                             "value": prefix + f"How far is the {color} {typ} at {json.dumps(box)} "
                                               "(bbox normalized 0-1000) from the camera, in meters?"})
                conv.append({"from": "gpt", "value": f"{float(r['distance']):.1f} meters"})
            elif r.get("depth"):
                conv.append({"from": "human",
                             "value": prefix + f"What is the depth of the {color} {typ} at {json.dumps(box)} "
                                               "(bbox normalized 0-1000), in meters?"})
                conv.append({"from": "gpt", "value": f"{float(r['depth']):.1f} meters"})
            else:
                conv.append({"from": "human",
                             "value": prefix + f"What object is at {json.dumps(box)} (bbox normalized 0-1000)?"})
                conv.append({"from": "gpt", "value": f"A {color} {typ}."})
        if not conv:
            continue
        fout.write(json.dumps({
            "id": f"airspatial_qa_{im_id}",
            "source": "airspatial",
            "task": "metric_vqa",
            "image": f"images/airspatial/{Path(im_id).stem}.jpg",
            "image_root": "l0",
            "conversations": conv,
        }, ensure_ascii=False) + "\n")
        n_qa += 1

    return {"airspatial_rec": n_rec, "airspatial_qa": n_qa,
            "airspatial_images": len(sizes), "airspatial_images_missing": len(needed) - len(kept)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/datasets/l0_cpt")
    ap.add_argument("--stage2_json", default="/root/autodl-tmp/Aeromamba/data/stage2_mixed_data_v2.json")
    ap.add_argument("--stage2_image_root", default="/root/autodl-tmp/Aeromamba/data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="stage2,hrvqa,airspatial",
                    help="comma list; rerun subsets after a crash")
    ap.add_argument("--append", action="store_true",
                    help="append to existing l0_mixed.jsonl instead of truncating")
    args = ap.parse_args()
    random.seed(args.seed)
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}

    out_root = Path(args.out)
    raw = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"image_roots": {"stage2": args.stage2_image_root, "l0": str(out_root)}}

    mode = "a" if args.append else "w"
    with (out_root / "l0_mixed.jsonl").open(mode, encoding="utf-8") as fout:
        if "stage2" in groups:
            print("[l0] group1: stage2 mixed ...", flush=True)
            report["stage2"] = emit_stage2(Path(args.stage2_json), fout, "stage2")
            gc.collect()
        if "hrvqa" in groups:
            print("[l0] group2: hrvqa ...", flush=True)
            report.update(emit_hrvqa(raw / "hrvqa", out_root, fout, args.workers))
            gc.collect()
        if "airspatial" in groups:
            print("[l0] group3: airspatial ...", flush=True)
            report.update(emit_airspatial(raw / "airspatial", out_root, fout, args.workers))

    rp = out_root / "build_report.json"
    if args.append and rp.exists():
        old = json.loads(rp.read_text(encoding="utf-8"))
        old.update(report)
        report = old
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[l0] DONE:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
