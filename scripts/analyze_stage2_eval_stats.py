"""Analyze Open3D-VQA probe composition + write a detailed stats report.

Does not need GPU. Consumes:
  data/open3d_vqa_probe/*.json
  reports/open3d_vqa_probe_test.json
  reports/stage2_smoke_capability_eval.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def task_bucket(question_name, qa_type, gt: str) -> str:
    qn = (question_name or "").lower()
    gn = re.sub(r"\s+", " ", (gt or "").strip().lower())
    if "distance" in qn or "meter" in gn:
        if re.search(r"[-+]?\d*\.?\d+", gn) and "yes" not in gn and "no" not in gn:
            return "distance_quant"
    if "size" in qn or "width" in qn or "height" in qn:
        if re.search(r"[-+]?\d*\.?\d+", gn) and not re.search(r"\b(yes|no)\b", gn):
            return "size_quant"
        return "size_qual"
    if any(k in qn for k in ("direction", "left", "right", "clock", "o'clock")):
        return "direction"
    if re.search(r"\b(yes|no)\b", gn):
        return "tf_qual"
    if re.search(r"[-+]?\d*\.?\d+", gn):
        return "distance_quant"
    return "other_qual"


def analyze_probe(path: Path):
    items = json.loads(path.read_text(encoding="utf-8"))
    by_scene = Counter()
    by_qname = Counter()
    by_qtype = Counter()
    by_bucket = Counter()
    by_split = Counter()
    scene_bucket = defaultdict(Counter)
    domain_bucket = defaultdict(Counter)
    ans_len = defaultdict(list)

    for it in items:
        scene = it.get("scene", "?")
        qn = it.get("question_name") or "NONE"
        qt = it.get("qa_type") or "NONE"
        split = it.get("probe_split") or "?"
        gt = ""
        for t in it.get("conversations") or []:
            if t.get("from") == "gpt":
                gt = t.get("value") or ""
        b = task_bucket(qn, qt, gt)
        domain = "real" if str(scene).startswith(("Realworld", "Wild")) else "sim"

        by_scene[scene] += 1
        by_qname[qn] += 1
        by_qtype[qt] += 1
        by_bucket[b] += 1
        by_split[split] += 1
        scene_bucket[scene][b] += 1
        domain_bucket[domain][b] += 1
        ans_len[b].append(len(gt))

    return {
        "n": len(items),
        "by_scene": dict(by_scene),
        "by_qname": dict(by_qname),
        "by_qtype": dict(by_qtype),
        "by_bucket": dict(by_bucket),
        "by_split": dict(by_split),
        "scene_bucket": {k: dict(v) for k, v in scene_bucket.items()},
        "domain_bucket": {k: dict(v) for k, v in domain_bucket.items()},
        "ans_len_mean": {
            k: (sum(v) / max(len(v), 1)) for k, v in ans_len.items()
        },
    }


def pct(x, n):
    return 100.0 * x / max(n, 1)


def main():
    probe_dir = REPO / "data" / "open3d_vqa_probe"
    reports = REPO / "reports"
    test_comp = analyze_probe(probe_dir / "test_official_probe.json")
    val_comp = analyze_probe(probe_dir / "val_sim_probe.json")
    manifest = json.loads((probe_dir / "split_manifest.json").read_text(encoding="utf-8"))
    full = json.loads((reports / "open3d_vqa_probe_test.json").read_text(encoding="utf-8"))
    smoke100 = json.loads((reports / "open3d_vqa_probe_smoke100.json").read_text(encoding="utf-8"))
    cap = json.loads((reports / "stage2_smoke_capability_eval.json").read_text(encoding="utf-8"))

    lines = []
    a = lines.append
    a("# Stage2 评测详细统计")
    a("")
    a("> **污染声明**：Stage2 v2 已训练全部 Open3D-VQA；Open3D 分数为 **seen / in-distribution probe**。")
    a("")
    a(f"- Stage2 slim: `checkpoints/stage2_v2/best_slim.pth`")
    a(f"- 全量 probe n={full['n_eval']}，耗时 {full.get('elapsed_s', 0)/3600:.2f} h")
    a(f"- split_seed={manifest['split_seed']}")
    a("")

    # ── capability smoke ──
    a("## 1. 四源冒烟（v2 val，每源 20）")
    a("")
    a("| source | n | CLM loss | PPL | exact | partial | 解读 |")
    a("|---|---:|---:|---:|---:|---:|---|")
    gloss = {
        "uav_motion": "指令→运动绑定强",
        "cognitive": "模板对、细粒度弱",
        "aerial_spatial": "字面匹配严、空间定位一般",
        "general": "仅看语言遗忘（无准确率）",
    }
    for s, r in cap["results"].items():
        g = r.get("generation") or {}
        ex = f"{g['exact_acc']*100:.1f}%" if g else "—"
        pa = f"{g['partial_acc']*100:.1f}%" if g else "—"
        a(f"| {s} | {r['n_loss']} | {r['clm_loss']:.3f} | {r['ppl']:.2f} | {ex} | {pa} | {gloss.get(s,'')} |")
    a("")

    # ── Open3D full accuracy ──
    a("## 2. Open3D-VQA 全量 probe 准确率")
    a("")
    a("### 2.1 总体 / 域")
    a("")
    a("| 集合 | n | exact | partial |")
    a("|---|---:|---:|---:|")
    allb = full["by_bucket"]["ALL"]
    a(f"| ALL | {allb['n']} | {allb['exact_acc']*100:.1f}% | {allb['partial_acc']*100:.1f}% |")
    for d, v in full["by_domain"].items():
        a(f"| {d} | {v['n']} | {v['exact_acc']*100:.1f}% | {v['partial_acc']*100:.1f}% |")
    a("")
    a("### 2.2 任务桶")
    a("")
    a("| bucket | n | exact | partial | 相对总准确率 |")
    a("|---|---:|---:|---:|---|")
    for k, v in sorted(full["by_bucket"].items(), key=lambda x: -x[1]["n"]):
        if k == "ALL":
            continue
        delta = (v["exact_acc"] - allb["exact_acc"]) * 100
        a(
            f"| {k} | {v['n']} | {v['exact_acc']*100:.1f}% | {v['partial_acc']*100:.1f}% | "
            f"{delta:+.1f} pp |"
        )
    a("")
    a("### 2.3 前 100 条 Real 探针（开跑冒烟）")
    a("")
    a("| bucket | n | exact | partial |")
    a("|---|---:|---:|---:|")
    for k, v in sorted(smoke100["by_bucket"].items(), key=lambda x: -x[1]["n"]):
        a(f"| {k} | {v['n']} | {v['exact_acc']*100:.1f}% | {v['partial_acc']*100:.1f}% |")
    a("")
    a("说明：前 100 条按 JSON 顺序，几乎全是 RealworldUAV/Lab，故 domain=real only。")
    a("")

    # ── composition ──
    a("## 3. Test_official_probe 构成（n=%d）" % test_comp["n"])
    a("")
    a("### 3.1 Split / 场景")
    a("")
    a("| split | n | % |")
    a("|---|---:|---:|")
    for k, v in sorted(test_comp["by_split"].items(), key=lambda x: -x[1]):
        a(f"| {k} | {v} | {pct(v, test_comp['n']):.1f}% |")
    a("")
    a("| scene | n | % |")
    a("|---|---:|---:|")
    for k, v in sorted(test_comp["by_scene"].items(), key=lambda x: -x[1]):
        a(f"| {k} | {v} | {pct(v, test_comp['n']):.1f}% |")
    a("")
    a("### 3.2 QA type / question_name")
    a("")
    a("| qa_type | n | % |")
    a("|---|---:|---:|")
    for k, v in sorted(test_comp["by_qtype"].items(), key=lambda x: -x[1]):
        a(f"| {k} | {v} | {pct(v, test_comp['n']):.1f}% |")
    a("")
    a("| question_name | n | % |")
    a("|---|---:|---:|")
    for k, v in sorted(test_comp["by_qname"].items(), key=lambda x: -x[1]):
        a(f"| `{k}` | {v} | {pct(v, test_comp['n']):.1f}% |")
    a("")
    a("### 3.3 场景 × 任务桶（样本数）")
    a("")
    buckets = sorted({b for sb in test_comp["scene_bucket"].values() for b in sb})
    a("| scene | " + " | ".join(buckets) + " | total |")
    a("|---|" + "|".join(["---:"] * len(buckets)) + "|---:|")
    for scene, bc in sorted(test_comp["scene_bucket"].items()):
        row = [str(bc.get(b, 0)) for b in buckets]
        a(f"| {scene} | " + " | ".join(row) + f" | {sum(bc.values())} |")
    a("")
    a("### 3.4 域 × 任务桶（样本数）")
    a("")
    a("| domain | " + " | ".join(buckets) + " |")
    a("|---|" + "|".join(["---:"] * len(buckets)) + "|")
    for dom, bc in sorted(test_comp["domain_bucket"].items()):
        a(f"| {dom} | " + " | ".join(str(bc.get(b, 0)) for b in buckets) + " |")
    a("")
    a("### 3.5 答案平均长度（字符）")
    a("")
    a("| bucket | mean_ans_len |")
    a("|---|---:|")
    for k, v in sorted(test_comp["ans_len_mean"].items()):
        a(f"| {k} | {v:.1f} |")
    a("")

    # ── val sim composition ──
    a("## 4. Val_sim_probe 构成（未跑准确率，仅清单）")
    a("")
    a(f"- n={val_comp['n']}")
    a("| scene | n |")
    a("|---|---:|")
    for k, v in sorted(val_comp["by_scene"].items(), key=lambda x: -x[1]):
        a(f"| {k} | {v} |")
    a("")

    # ── interpretation ──
    a("## 5. 读数要点")
    a("")
    a("- 全量 exact **33.7%** 落在 seen 数据上，说明空间 QA 未被背熟。")
    a("- **direction 48.6%** 最好；**distance_quant 8.2%** 最差，拖低总分。")
    a("- Real(35.1%) 与 Sim(33.0%) 接近 → 未见差距小；但因污染，不能解读为 sim→real 泛化成功。")
    a("- Test 中 Sim 占 {:.1f}% / Real 占 {:.1f}%，总分更接近 Sim。".format(
        pct(sum(v for s, v in test_comp["by_scene"].items() if not s.startswith(("Realworld", "Wild"))), test_comp["n"]),
        pct(sum(v for s, v in test_comp["by_scene"].items() if s.startswith(("Realworld", "Wild"))), test_comp["n"]),
    ))
    a("- 冒烟 `uav_motion` 90% 与 Open3D 空间弱形成对比：Stage2 更擅长指令运动语义，而非精细测距。")
    a("- 当前全量结果**没有按 scene / question_name 存预测**，故准确率只能到 domain×bucket；构成表见上文。")
    a("")

    out = reports / "stage2_eval_detailed_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    detail_json = {
        "test_composition": test_comp,
        "val_composition": val_comp,
        "full_accuracy": full,
        "smoke100_accuracy": smoke100,
        "capability_smoke": cap["results"],
        "manifest": manifest,
    }
    (reports / "stage2_eval_detailed_stats.json").write_text(
        json.dumps(detail_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out}")
    print(f"test n={test_comp['n']} scenes={len(test_comp['by_scene'])} qnames={len(test_comp['by_qname'])}")


if __name__ == "__main__":
    main()
