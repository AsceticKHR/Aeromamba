"""
Extract CognitiveDrone RLDS tfrecord shards into LLaVA-format cognitive QA.

Source dataset: sonnt-vinu-cair/CognitiveDrone_rlds_dataset_REPRODUCTION
(community rebuild of ArtemLykov/CognitiveDrone_dataset, whose RLDS upload is
incomplete). Episodes carry:
    steps/observation/image            jpeg bytes per step (256x256)
    steps/language_instruction         reasoning-style instruction
    steps/language_instruction_simple  resolved instruction (correct gate)
    episode_metadata/type              task category (e.g. Reasoning/Math_desk)

For each episode we save the first frame and emit a QA pair that supervises
exactly the cognitive step: reasoning instruction -> resolved command.

Implementation reads tfrecords with a minimal pure-python parser (protobuf
tf.train.Example via the `tfrecord` pip package) instead of TensorFlow, so it
runs inside a 2GB-RAM cgroup.

Usage:
  python data/extract_cognitive_drone.py \
      --shards_dir /root/autodl-tmp/datasets/cognitive_drone_rlds/all/1.0.0 \
      --frames_dir /root/autodl-tmp/Aeromamba/data/cognitive_drone/frames \
      --out_json   /root/autodl-tmp/Aeromamba/data/cognitive_drone/cognitive_qa.json \
      --image_prefix cognitive_drone/frames
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
from pathlib import Path


def iter_tfrecord(path: str):
    """Yield raw record bytes from a TFRecord file (crc fields skipped)."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # length crc
            payload = f.read(length)
            f.read(4)  # payload crc
            if len(payload) < length:
                return
            yield payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--image_prefix", required=True,
                    help="image path prefix relative to stage2 data_root")
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from tfrecord import example_pb2  # lightweight, no TensorFlow

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.shards_dir, "*.tfrecord*")))
    if not shards:
        raise SystemExit(f"no tfrecord shards under {args.shards_dir}")
    print(f"[cogdrone] {len(shards)} shards")

    n_ep = n_bad = 0
    cat_counts: dict = {}
    with open(args.out_json, "w", encoding="utf-8") as out:
        out.write("[\n")
        first_item = True
        for shard in shards:
            for payload in iter_tfrecord(shard):
                if args.max_episodes and n_ep >= args.max_episodes:
                    break
                ex = example_pb2.Example()
                ex.ParseFromString(payload)
                feat = ex.features.feature

                def bytes_list(key):
                    return list(feat[key].bytes_list.value) if key in feat else []

                images = bytes_list("steps/observation/image")
                instrs = bytes_list("steps/language_instruction")
                simples = bytes_list("steps/language_instruction_simple")
                types = bytes_list("episode_metadata/type")
                if not images or not instrs:
                    n_bad += 1
                    continue

                instr = instrs[0].decode("utf-8", "replace").strip()
                simple = (
                    simples[0].decode("utf-8", "replace").strip()
                    if simples else ""
                )
                category = (
                    types[0].decode("utf-8", "replace").strip()
                    if types else "unknown"
                )
                if not instr or not simple or simple.lower() == instr.lower():
                    n_bad += 1
                    continue

                fname = f"ep{n_ep:06d}.jpg"
                with open(frames_dir / fname, "wb") as imf:
                    imf.write(images[0])

                item = {
                    "id": f"cogdrone_{n_ep:06d}",
                    "image": f"{args.image_prefix}/{fname}",
                    "source": "cognitive",
                    "category": category,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "<image>\nYou control a drone in front of "
                                "several gates. Task: \"" + instr + "\" "
                                "Resolve the task and state the exact flight "
                                "command."
                            ),
                        },
                        {"from": "gpt", "value": simple},
                    ],
                }
                if not first_item:
                    out.write(",\n")
                out.write(json.dumps(item, ensure_ascii=False))
                first_item = False
                cat_counts[category] = cat_counts.get(category, 0) + 1
                n_ep += 1
            print(f"[cogdrone] after {os.path.basename(shard)}: {n_ep} episodes")
        out.write("\n]\n")

    print(f"[cogdrone] done: {n_ep} QA samples, {n_bad} skipped")
    for k, v in sorted(cat_counts.items()):
        print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
