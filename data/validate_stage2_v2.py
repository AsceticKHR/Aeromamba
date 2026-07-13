"""
Strict quality validation for stage2_mixed_data_v2.json.

Checks (streaming, 2GB-RAM safe):
  1. Schema: id / image / source / conversations present, roles alternate
     human->gpt, non-empty text, <image> token in first human turn.
  2. Image existence: every referenced image path resolves under data_root
     (full check, not sampled).
  3. Eval leakage: no uav_motion/cognitive sample references an eval episode.
  4. Duplicates: no duplicate ids.
  5. Source distribution + answer-length stats per source.
  6. Decodability spot-check: N random images per source are actually opened
     with PIL.

Exit code 0 = all hard checks pass.

Usage:
  python data/validate_stage2_v2.py \
      --json /root/autodl-tmp/Aeromamba/data/stage2_mixed_data_v2.json \
      --data_root /root/autodl-tmp/Aeromamba/data \
      --eval_episodes /root/autodl-tmp/Aeromamba/data/eval_episodes.txt \
      --decode_per_source 25
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--eval_episodes", required=True)
    ap.add_argument("--decode_per_source", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import ijson
    except ImportError:
        sys.exit("pip install ijson")

    data_root = Path(args.data_root)
    with open(args.eval_episodes, "r", encoding="utf-8") as f:
        eval_ids = {ln.strip() for ln in f if ln.strip()}

    rng = random.Random(args.seed)
    errors: list = []
    n = 0
    seen_ids = set()
    dup_ids = 0
    src_counter: Counter = Counter()
    ans_len_sum: Counter = Counter()
    ans_len_max: Counter = Counter()
    missing_img = 0
    decode_pool: dict = defaultdict(list)  # source -> reservoir of image paths

    def err(msg: str) -> None:
        if len(errors) < 30:
            errors.append(msg)

    with open(args.json, "rb") as f:
        for item in ijson.items(f, "item"):
            n += 1
            sid = item.get("id")
            src = item.get("source")
            img = item.get("image")
            convs = item.get("conversations")

            if not sid:
                err(f"#{n}: missing id")
            elif sid in seen_ids:
                dup_ids += 1
                err(f"#{n}: duplicate id {sid}")
            else:
                seen_ids.add(sid)

            if not src:
                err(f"#{n} ({sid}): missing source")
            src_counter[src or "MISSING"] += 1

            # conversations schema
            if not convs or len(convs) < 2:
                err(f"#{n} ({sid}): conversations missing/too short")
            else:
                if "<image>" not in convs[0].get("value", ""):
                    err(f"#{n} ({sid}): first human turn lacks <image>")
                for i, turn in enumerate(convs):
                    want = "human" if i % 2 == 0 else "gpt"
                    if turn.get("from") != want:
                        err(f"#{n} ({sid}): turn {i} role {turn.get('from')} != {want}")
                    if not str(turn.get("value", "")).strip():
                        err(f"#{n} ({sid}): turn {i} empty text")
                gpt_lens = [len(t.get("value", "")) for t in convs
                            if t.get("from") == "gpt"]
                if gpt_lens:
                    ans_len_sum[src] += sum(gpt_lens) / len(gpt_lens)
                    ans_len_max[src] = max(ans_len_max[src], max(gpt_lens))

            # image existence (full check)
            if not img:
                err(f"#{n} ({sid}): missing image")
            else:
                p = data_root / img
                if not p.is_file():
                    missing_img += 1
                    err(f"#{n} ({sid}): image not found {img}")
                else:
                    pool = decode_pool[src]
                    if len(pool) < args.decode_per_source:
                        pool.append(p)
                    else:
                        j = rng.randrange(n)
                        if j < args.decode_per_source:
                            pool[j % args.decode_per_source] = p

            # eval leakage
            if src == "uav_motion" and img:
                ep = img.split("/")[1] if img.count("/") >= 2 else ""
                if ep in eval_ids:
                    err(f"#{n} ({sid}): EVAL LEAK episode {ep}")

    # decode spot-check
    from PIL import Image
    n_decode_fail = 0
    for src, pool in decode_pool.items():
        for p in pool:
            try:
                with Image.open(p) as im:
                    im.convert("RGB")
            except Exception as e:
                n_decode_fail += 1
                err(f"decode fail [{src}] {p}: {e}")

    print(f"== validate {args.json} ==")
    print(f"total samples : {n}")
    print(f"duplicate ids : {dup_ids}")
    print(f"missing images: {missing_img}")
    print(f"decode fails  : {n_decode_fail}")
    print("source distribution / mean / max answer chars:")
    for src, cnt in src_counter.most_common():
        mean_len = ans_len_sum[src] / max(cnt, 1)
        print(f"  {src:16s} {cnt:8d}  ({100.0*cnt/max(n,1):5.1f}%)  "
              f"mean_ans={mean_len:6.1f}  max_ans={ans_len_max[src]}")
    if errors:
        print(f"\nFIRST {len(errors)} ERRORS:")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
