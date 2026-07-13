"""
Build stage2_mixed_data_v2.json: a source-tagged, curated Stage-2 SFT mix.

Sources in the output (each sample gets a "source" field consumed by
Stage2Trainer's WeightedRandomSampler via --source_weights):

  general        COCO/LLaVA-Instruct subset, keyword-filtered to outdoor /
                 street / vehicle / building scenes (UAV-FPV-relevant).
  aerial_spatial Open3D-VQA aerial spatial QA (kept in full).
  uav_motion     Language->motion-semantics QA generated from UAV-Flow train
                 episodes (eval episodes excluded). Teaches the
                 "instruction -> which way to fly" binding under cheap CLM
                 supervision before Stage-3 action regression.
  cognitive      CognitiveDrone reasoning QA (built separately by
                 extract_cognitive_drone.py, merged here if present).

Memory-safe: the (300MB+) source JSON is streamed with ijson and the output
is written incrementally, so peak RSS stays low (runs inside a 2GB cgroup).

Sign conventions for uav_motion labels were verified empirically on 3000
episodes (see reports/stage2_data_report_20260713.md): body-frame dy>0 is
"right" (77-79% instruction agreement), dz>0 is "ascend" (97-100%). Net yaw
correlates only weakly with turn wording (39-55%), so rotation-based labels
are deliberately NOT generated.

Usage (on the training server):
  python data/build_stage2_v2.py \
      --old_json   /root/autodl-tmp/Aeromamba/data/stage2_mixed_data.json \
      --uavflow_root /root/autodl-tmp/datasets/uav-flow \
      --eval_episodes /root/autodl-tmp/Aeromamba/data/eval_episodes.txt \
      --cognitive_json /root/autodl-tmp/Aeromamba/data/cognitive_drone/cognitive_qa.json \
      --out        /root/autodl-tmp/Aeromamba/data/stage2_mixed_data_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

# ── general-data keyword filter ──────────────────────────────────────────────
# Indoor close-ups (kitchens, cats on sofas, food plates...) are dropped: they
# carry little value for UAV-FPV grounding and their epoch share is better
# spent on aerial data. A sample is kept if it hits one STRONG urban/traffic
# keyword, or at least two distinct WEAK outdoor keywords (single weak hits
# like "tree" or "sky" occur in far too many generic captions).
STRONG_KEYWORDS = [
    "street", "road", "highway", "intersection", "sidewalk", "crosswalk",
    "building", "skyscraper", "tower", "bridge", "rooftop", "city",
    "urban", "downtown", "traffic", "parking", "railway", "station",
    "airport", "airplane", "aircraft", "helicopter", "harbor", "plaza",
    "stadium", "construction", "aerial", "billboard", "monument",
]
WEAK_KEYWORDS = [
    "car", "cars", "truck", "bus", "vehicle", "van", "motorcycle",
    "bicycle", "bike", "train", "boat", "ship", "river", "lake", "beach",
    "mountain", "hill", "field", "tree", "trees", "forest", "grass", "roof",
    "house", "sky", "cloud", "outdoor", "outside", "crowd", "pedestrian",
    "square", "fountain", "statue", "playground", "fence", "pole",
    "lamppost", "park",
]


def sample_text(item: dict) -> str:
    return " ".join(
        turn.get("value", "") for turn in item.get("conversations", [])
    ).lower()


def keep_general(item: dict) -> bool:
    text = sample_text(item)
    if any(kw in text for kw in STRONG_KEYWORDS):
        return True
    weak_hits = sum(1 for kw in WEAK_KEYWORDS if kw in text)
    return weak_hits >= 2


# ── uav_motion QA generation ────────────────────────────────────────────────

def wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def motion_phrase(dx: float, dy: float, dz: float) -> str:
    """Body-frame net displacement (meters) -> short English description.

    dx: forward(+)/backward(-), dy: right(+)/left(-), dz: up(+)/down(-).
    Conventions empirically verified against instruction wording.
    """
    parts = []
    if abs(dx) >= 0.5:
        parts.append("forward" if dx > 0 else "backward")
    if abs(dy) >= 0.5 and abs(dy) >= 0.35 * max(abs(dx), 1e-6):
        parts.append("to the right" if dy > 0 else "to the left")
    horiz = " and ".join(parts) if parts else None

    vert = None
    if abs(dz) >= 0.3:
        vert = "ascending" if dz > 0 else "descending"

    if horiz and vert:
        return f"The UAV should move {horiz} while {vert}."
    if horiz:
        return f"The UAV should move {horiz} while roughly holding altitude."
    if vert:
        return f"The UAV should mainly {vert[:-3]} with little horizontal movement."
    return "The UAV should stay close to its current position."


def build_uav_motion(
    uavflow_root: str,
    eval_ids: set,
    rephrase_frac: float,
    rng: random.Random,
):
    """Yield source-tagged LLaVA-format samples from UAV-Flow episodes."""
    root = Path(uavflow_root)
    episodes = sorted(p.name for p in root.iterdir() if p.is_dir())
    n_skip_eval = n_skip_bad = 0
    for ep in episodes:
        if ep in eval_ids:
            n_skip_eval += 1
            continue
        log_path = root / ep / "log.json"
        img_rel = f"uav_flow_frames/{ep}/000000.jpg"
        if not log_path.is_file() or not (root / ep / "000000.jpg").is_file():
            n_skip_bad += 1
            continue
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            n_skip_bad += 1
            continue

        instruction = d.get("instruction") or d.get("instruction_unified")
        unified = d.get("instruction_unified")
        raw = d.get("raw_logs") or []
        if not instruction or len(raw) < 5:
            n_skip_bad += 1
            continue

        x0, y0, z0 = float(raw[0][0]), float(raw[0][1]), float(raw[0][2])
        yaw0 = float(raw[0][4]) if len(raw[0]) > 4 else 0.0
        c, s = math.cos(math.radians(yaw0)), math.sin(math.radians(yaw0))
        dxw = float(raw[-1][0]) - x0
        dyw = float(raw[-1][1]) - y0
        dx_body = c * dxw + s * dyw
        dy_body = -s * dxw + c * dyw
        dz = float(raw[-1][2]) - z0

        answer = motion_phrase(dx_body, dy_body, dz)
        yield {
            "id": f"uavmotion_dir_{ep}",
            "image": img_rel,
            "source": "uav_motion",
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nYou are piloting a UAV. Instruction: "
                        f"\"{instruction}\" Based on the current view, in "
                        "which direction should the UAV move overall?"
                    ),
                },
                {"from": "gpt", "value": answer},
            ],
        }

        # Canonical-rephrase QA: binds the LLM-diversified instruction to its
        # unified form, grounded on the same frame (cheap paraphrase signal).
        if (
            unified
            and unified.strip().lower() != instruction.strip().lower()
            and rng.random() < rephrase_frac
        ):
            yield {
                "id": f"uavmotion_rep_{ep}",
                "image": img_rel,
                "source": "uav_motion",
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\nA UAV pilot received this instruction: "
                            f"\"{instruction}\" Restate it as a single concise "
                            "flight command."
                        ),
                    },
                    {"from": "gpt", "value": unified.strip()},
                ],
            }
    print(f"[uav_motion] episodes skipped: eval={n_skip_eval} bad={n_skip_bad}")


# ── main build ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_json", required=True)
    ap.add_argument("--uavflow_root", required=True)
    ap.add_argument("--eval_episodes", required=True,
                    help="txt file, one eval episode id per line (leak guard)")
    ap.add_argument("--cognitive_json", default=None,
                    help="optional cognitive QA json from extract_cognitive_drone.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rephrase_frac", type=float, default=0.8,
                    help="fraction of episodes that also emit a rephrase QA")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import ijson
    except ImportError:
        sys.exit("pip install ijson (needed for memory-safe streaming)")

    rng = random.Random(args.seed)
    with open(args.eval_episodes, "r", encoding="utf-8") as f:
        eval_ids = {ln.strip() for ln in f if ln.strip()}
    print(f"[build] eval episodes to exclude: {len(eval_ids)}")

    stats = {"general_kept": 0, "general_dropped": 0,
             "aerial_spatial": 0, "uav_motion": 0, "cognitive": 0}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats["content_dups_skipped"] = 0
    import hashlib
    content_seen: set = set()
    emit_counter = [0]

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("[\n")
        first = True

        def emit(item: dict) -> bool:
            """Content-dedup + re-id (v1 reused COCO image ids across multiple
            conversations, producing 67k duplicate ids). Returns True if the
            item was written."""
            nonlocal first
            h = hashlib.md5(json.dumps(
                {"image": item.get("image"), "conv": item.get("conversations")},
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if h in content_seen:
                stats["content_dups_skipped"] += 1
                return False
            content_seen.add(h)
            orig = item.get("id")
            item["id"] = f"{item.get('source', 'x')}_{emit_counter[0]:07d}"
            if orig and orig != item["id"]:
                item["orig_id"] = str(orig)
            emit_counter[0] += 1
            if not first:
                out.write(",\n")
            out.write(json.dumps(item, ensure_ascii=False))
            first = False
            return True

        # 1. stream old mix: tag + filter
        with open(args.old_json, "rb") as f:
            for item in ijson.items(f, "item"):
                img = item.get("image", "")
                if img.startswith("open3d_vqa/"):
                    item["source"] = "aerial_spatial"
                    if emit(item):
                        stats["aerial_spatial"] += 1
                else:
                    if keep_general(item):
                        item["source"] = "general"
                        if emit(item):
                            stats["general_kept"] += 1
                    else:
                        stats["general_dropped"] += 1

        # 2. uav_motion QA from UAV-Flow
        for item in build_uav_motion(
            args.uavflow_root, eval_ids, args.rephrase_frac, rng
        ):
            if emit(item):
                stats["uav_motion"] += 1

        # 3. cognitive QA (already LLaVA format, source pre-tagged)
        if args.cognitive_json and Path(args.cognitive_json).is_file():
            with open(args.cognitive_json, "rb") as f:
                for item in ijson.items(f, "item"):
                    item.setdefault("source", "cognitive")
                    if emit(item):
                        stats["cognitive"] += 1
        else:
            print("[build] no cognitive_json provided/found — skipping")

        out.write("\n]\n")

    total = sum(v for k, v in stats.items()
                if k not in ("general_dropped", "content_dups_skipped"))
    print(f"[build] wrote {total} samples -> {out_path}")
    for k, v in stats.items():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
