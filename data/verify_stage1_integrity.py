"""
Stage1 数据完整性验证脚本
===========================
检查 blip_laion_cc_sbu_558k.json 中引用的所有图片是否能在 data_root 中找到。

图片查找顺序（与 LLaVADataset.__getitem__ 一致）:
  1. data_root / image
  2. data_root / images / image
  3. data_root / train2017 / image

用法:
  python data/verify_stage1_integrity.py \
      --data_root /root/Aeromamba/data/llava_pretrain \
      --json_name blip_laion_cc_sbu_558k.json \
      [--sample 5000]   # 只随机抽样验证N条，默认全量
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def find_image(data_root: Path, rel_path: str):
    """返回图片实际路径，找不到返回 None"""
    candidates = [
        data_root / rel_path,
        data_root / "images" / rel_path,
        data_root / "train2017" / rel_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def check_one(args):
    data_root, rel_path = args
    p = find_image(data_root, rel_path)
    return rel_path, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/Aeromamba/data/llava_pretrain")
    parser.add_argument("--json_name", default="blip_laion_cc_sbu_558k.json")
    parser.add_argument("--sample",    type=int, default=None,
                        help="随机抽样验证条数，默认全量")
    parser.add_argument("--workers",   type=int, default=8,
                        help="并发线程数")
    parser.add_argument("--show_missing", type=int, default=20,
                        help="最多打印多少条缺失样本路径")
    a = parser.parse_args()

    data_root = Path(a.data_root)
    json_path = data_root / a.json_name

    if not json_path.exists():
        print(f"[FATAL] JSON 不存在: {json_path}")
        sys.exit(1)

    print(f"[验证] 加载 JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[验证] 总样本数: {len(data):,}")

    # 抽样
    if a.sample and a.sample < len(data):
        data = random.sample(data, a.sample)
        print(f"[验证] 随机抽样: {len(data):,} 条")

    # 过滤出有 image 字段的样本
    samples_with_img = [(data_root, item["image"]) for item in data if "image" in item]
    no_img = len(data) - len(samples_with_img)
    print(f"[验证] 有 image 字段: {len(samples_with_img):,}  无 image 字段(文本Only): {no_img:,}")

    # 并发查找
    print(f"[验证] 开始并发验证 (workers={a.workers})...")
    missing = []
    found = 0
    total = len(samples_with_img)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(check_one, s): s for s in samples_with_img}
        for i, fut in enumerate(as_completed(futures), 1):
            rel_path, actual_path = fut.result()
            if actual_path is None:
                missing.append(rel_path)
            else:
                found += 1
            if i % 50000 == 0 or i == total:
                pct = i / total * 100
                print(f"  进度: {i:,}/{total:,} ({pct:.1f}%)  "
                      f"找到: {found:,}  缺失: {len(missing):,}", flush=True)

    # 报告
    print("\n" + "=" * 60)
    print(f"验证完成:")
    print(f"  总检查: {total:,}")
    print(f"  ✅ 找到: {found:,}  ({found/total*100:.2f}%)")
    print(f"  ❌ 缺失: {len(missing):,}  ({len(missing)/total*100:.2f}%)")

    if missing:
        print(f"\n前 {min(a.show_missing, len(missing))} 条缺失路径:")
        for p in missing[:a.show_missing]:
            print(f"  {p}")
        # 保存完整缺失列表
        miss_out = data_root / "missing_images.txt"
        with open(miss_out, "w") as f:
            f.write("\n".join(missing))
        print(f"\n完整缺失列表已保存: {miss_out}")
    else:
        print("\n✅ 所有图片均可找到，数据完整！")

    # 结论
    print("=" * 60)
    if len(missing) == 0:
        print("结论: 数据完整，可以直接启动 Stage1 训练。")
        sys.exit(0)
    elif len(missing) / total < 0.01:
        print(f"结论: 缺失率 < 1%，基本完整，可以启动训练（缺失样本会被跳过）。")
        sys.exit(0)
    else:
        print(f"结论: 缺失率 {len(missing)/total*100:.2f}%，建议补全后再训练。")
        sys.exit(1)


if __name__ == "__main__":
    main()
