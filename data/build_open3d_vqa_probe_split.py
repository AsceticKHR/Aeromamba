"""
Build a paper-style Open3D-VQA probe split (RGB-only).

Paper (arXiv 2503.11094): 80% sim train / 10% sim val / remaining 10% sim +
all real-world -> sim-to-real test. This repo's Stage2 v2 already trained on
*all* Open3D-VQA samples, so the exported sets are labeled contaminated probes.

Deterministic assignment by md5(id + split_seed):
  sim ids with hash%10 == 0 -> test_sim_10pct
  sim ids with hash%10 == 1 -> val_sim_10pct
  remaining sim             -> train_sim_80pct (not exported for eval)
  all real                  -> test_real

Writes under --out_dir:
  test_official_probe.json   = test_real + test_sim_10pct  (LLaVA-style items)
  val_sim_probe.json         = val_sim_10pct
  split_manifest.json        = counts + scene breakdown + contamination flag

Image paths are relative to --o3d_root's parent when --image_prefix=open3d_vqa,
matching Stage2 training layout: open3d_vqa/O3DVQA/<scene>/rgb/<file>.

Usage:
  python data/build_open3d_vqa_probe_split.py \\
      --o3d_root "/path/to/open3d_vqa/O3DVQA" \\
      --out_dir ./data/open3d_vqa_probe \\
      --split_seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SIM_SCENES = {
    "EmbodiedCity/Wuhan",
    "UrbanScene/Campus",
    "UrbanScene/Residence",
}
REAL_PREFIXES = (
    "RealworldUAV/",
    "WildUAV/",
)


def scene_rel(merged_path: Path, o3d_root: Path) -> str:
    return merged_path.parent.relative_to(o3d_root).as_posix()


def is_sim(scene: str) -> bool:
    return scene in SIM_SCENES


def is_real(scene: str) -> bool:
    return any(scene.startswith(p) for p in REAL_PREFIXES)


def bucket_for(sample_id: str, scene: str, split_seed: int) -> str:
    if is_real(scene):
        return "test_real"
    if not is_sim(scene):
        return "other"
    h = hashlib.md5(f"{split_seed}:{sample_id}".encode()).hexdigest()
    r = int(h[:8], 16) % 10
    if r == 0:
        return "test_sim_10pct"
    if r == 1:
        return "val_sim_10pct"
    return "train_sim_80pct"


def rgb_filename(image_info: dict) -> str | None:
    # Official files store absolute Windows paths; keep the basename only.
    p = (image_info or {}).get("image_path") or ""
    if not p:
        return None
    return Path(p.replace("\\", "/")).name


def to_llava(item: dict, scene: str, image_prefix: str) -> dict | None:
    fname = rgb_filename(item.get("image_info") or {})
    if not fname:
        return None
    conv = item.get("conversation") or item.get("conversations")
    if not conv or len(conv) < 2:
        return None
    qa = item.get("qa_info") or {}
    return {
        "id": item.get("id"),
        "image": f"{image_prefix}/O3DVQA/{scene}/rgb/{fname}",
        "source": "open3d_vqa",
        "scene": scene,
        "qa_type": qa.get("type"),
        "question_name": qa.get("question_name"),
        "conversations": [
            {"from": t.get("from"), "value": t.get("value", "")} for t in conv
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--o3d_root", required=True, help=".../open3d_vqa/O3DVQA")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--image_prefix", default="open3d_vqa")
    args = ap.parse_args()

    o3d_root = Path(args.o3d_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list] = defaultdict(list)
    scene_counts: dict[str, Counter] = defaultdict(Counter)
    missing_rgb = 0
    n_total = 0

    for merged in sorted(o3d_root.glob("*/*/merged_qa.json")):
        scene = scene_rel(merged, o3d_root)
        with open(merged, "r", encoding="utf-8") as f:
            data = json.load(f)
        for raw in data:
            n_total += 1
            sid = str(raw.get("id", ""))
            b = bucket_for(sid, scene, args.split_seed)
            ll = to_llava(raw, scene, args.image_prefix)
            if ll is None:
                continue
            rgb = o3d_root / scene / "rgb" / Path(ll["image"]).name
            if not rgb.is_file():
                missing_rgb += 1
                continue
            ll["probe_split"] = b
            buckets[b].append(ll)
            scene_counts[b][scene] += 1

    test_official = buckets["test_real"] + buckets["test_sim_10pct"]
    val_sim = buckets["val_sim_10pct"]

    with open(out_dir / "test_official_probe.json", "w", encoding="utf-8") as f:
        json.dump(test_official, f, ensure_ascii=False)
    with open(out_dir / "val_sim_probe.json", "w", encoding="utf-8") as f:
        json.dump(val_sim, f, ensure_ascii=False)

    manifest = {
        "contaminated": True,
        "contamination_note": (
            "Stage2 v2 trained on all Open3D-VQA aerial_spatial samples; "
            "these splits are in-distribution probes, not a clean hold-out."
        ),
        "split_seed": args.split_seed,
        "o3d_root": str(o3d_root),
        "n_merged_qa_items": n_total,
        "missing_rgb_skipped": missing_rgb,
        "counts": {k: len(v) for k, v in buckets.items()},
        "test_official_probe": len(test_official),
        "val_sim_probe": len(val_sim),
        "scene_counts": {k: dict(v) for k, v in scene_counts.items()},
        "paper_protocol": "sim 80/10/10 + all real in test (id-hash approx)",
    }
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[split] total merged items: {n_total}, missing rgb: {missing_rgb}")
    for k, v in sorted(manifest["counts"].items()):
        print(f"  {k:20s} {v}")
    print(f"[split] test_official_probe={len(test_official)} "
          f"val_sim_probe={len(val_sim)}")
    print(f"[split] wrote {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
