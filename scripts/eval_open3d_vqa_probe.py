"""
Rule-based Open3D-VQA probe evaluation for a Stage-2 AeroMamba checkpoint.

Reads LLaVA-style JSON from data/build_open3d_vqa_probe_split.py.
Does NOT use GPT-4o: TF/MCQ exact; qualitative SAQ keyword overlap;
quantitative SAQ relative error in [0.75, 1.25] (paper numeric rule).

Contamination: results are probes when the ckpt trained on full Open3D-VQA.

Usage:
  python scripts/eval_open3d_vqa_probe.py \\
      --ckpt checkpoints/stage2_v2/best_slim.pth \\
      --data_root "/path/to/.../dataset/aeromamba" \\
      --probe_json data/open3d_vqa_probe/test_official_probe.json \\
      --max_samples 0 \\
      --out_json reports/open3d_vqa_probe_test.json \\
      --out_md reports/open3d_vqa_probe_test.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.uav_mamba_vla import AeroMambaVLA


# ── scoring helpers ──────────────────────────────────────────────────────────

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
YES_NO = re.compile(r"\b(yes|no)\b", re.I)
OPTION_RE = re.compile(r"\b([A-Da-d])\b")

DIR_WORDS = [
    "left", "right", "front", "behind", "back", "above", "below",
    "up", "down", "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "clockwise", "counterclockwise", "counter-clockwise",
]
CLOCK_RE = re.compile(r"\b(\d{1,2})\s*o'?clock\b", re.I)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".")


def extract_yes_no(s: str) -> str | None:
    m = YES_NO.search(s or "")
    return m.group(1).lower() if m else None


def extract_number(s: str) -> float | None:
    nums = NUM_RE.findall(s or "")
    if not nums:
        return None
    # Prefer the last number (answers often "... is 12.3 meters")
    try:
        return float(nums[-1])
    except ValueError:
        return None


def qualitative_tokens(s: str) -> set:
    t = norm_text(s)
    out = set()
    for w in DIR_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            out.add(w)
    for m in CLOCK_RE.finditer(t):
        out.add(f"{int(m.group(1))}oclock")
    yn = extract_yes_no(t)
    if yn:
        out.add(yn)
    return out


def task_bucket(question_name: str | None, qa_type: str | None, gt: str) -> str:
    qn = (question_name or "").lower()
    if "distance" in qn or "meter" in norm_text(gt):
        if extract_number(gt) is not None and "yes" not in norm_text(gt) and "no" not in norm_text(gt):
            return "distance_quant"
    if "size" in qn or "width" in qn or "height" in qn:
        if extract_number(gt) is not None and extract_yes_no(gt) is None:
            return "size_quant"
        return "size_qual"
    if "direction" in qn or "left" in qn or "right" in qn or "o'clock" in qn or "clock" in qn:
        return "direction"
    if (qa_type or "").lower() in ("qualitative", "tf", "true_false"):
        if extract_yes_no(gt):
            return "tf_qual"
    if extract_yes_no(gt):
        return "tf_qual"
    if extract_number(gt) is not None:
        return "distance_quant"
    return "other_qual"


def score_pair(pred: str, gt: str, bucket: str) -> tuple[bool, bool]:
    """Return (exact_or_numeric_ok, partial_ok)."""
    pn, gn = norm_text(pred), norm_text(gt)

    if bucket in ("distance_quant", "size_quant"):
        pv, gv = extract_number(pred), extract_number(gt)
        if pv is None or gv is None or abs(gv) < 1e-6:
            tok_ok = qualitative_tokens(pred) == qualitative_tokens(gt) and bool(qualitative_tokens(gt))
            return tok_ok, bool(qualitative_tokens(pred) & qualitative_tokens(gt))
        ratio = pv / gv
        ok = 0.75 <= ratio <= 1.25
        return ok, ok or (abs(ratio - 1.0) < 0.5)

    if bucket == "tf_qual":
        py, gy = extract_yes_no(pred), extract_yes_no(gt)
        if py and gy:
            ok = py == gy
            return ok, ok
        ok = pn == gn or gn in pn
        return ok, ok

    # MCQ-ish: letter match
    po, go = OPTION_RE.search(pred or ""), OPTION_RE.search(gt or "")
    if go and ("choice" in gn or len(gn) <= 3):
        if po:
            ok = po.group(1).upper() == go.group(1).upper()
            return ok, ok

    gpt = qualitative_tokens(gt)
    ppt = qualitative_tokens(pred)
    if gpt:
        ok = gpt == ppt
        partial = bool(gpt & ppt)
        if ok or partial:
            return ok, partial

    ok = pn == gn
    partial = (gn in pn) or ok
    return ok, partial


# ── model helpers (shared with eval_stage2_capabilities) ─────────────────────

def build_prefix(model, pixels, device):
    vis_patches = model._encode_vision(pixels)
    vis_tokens = model.projector(vis_patches)
    vis_tokens = model.token_resampler(vis_tokens)
    zero_state = torch.zeros(
        vis_tokens.size(0), model.proprio_encoder.proprio_dim,
        device=device, dtype=vis_tokens.dtype,
    )
    state_tokens = model.proprio_encoder.forward_pair(zero_state, zero_state)
    return torch.cat([state_tokens, vis_tokens], dim=1)


@torch.no_grad()
def greedy_generate(model, prefix, prompt_ids, device, max_new_tokens=48):
    input_ids = torch.tensor([prompt_ids], device=device)
    text_embs = model._embed_text(input_ids)
    curr = torch.cat([prefix, text_embs], dim=1)
    out_ids = []
    for _ in range(max_new_tokens):
        hidden = model._run_mamba(curr)
        logits = model.mamba.lm_head(hidden[:, -1:, :])
        tok = int(torch.argmax(logits[0, -1]))
        if tok == model.tokenizer.eos_token_id:
            break
        out_ids.append(tok)
        emb = model._embed_text(torch.tensor([[tok]], device=device))
        curr = torch.cat([curr, emb], dim=1)
    return model.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True,
                    help="parent of open3d_vqa/ (image paths in JSON are relative to this)")
    ap.add_argument("--probe_json", required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--dump_predictions", action="store_true",
                    help="also write per-sample preds next to out_json (large)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    use_autocast = args.dtype == "bf16" and device.type == "cuda"

    model = AeroMambaVLA(
        mamba_type="mamba-2-370m",
        vision_type="siglip2_base_384",
        chunk_size=8,
        proprio_dim=8,
        token_resampler="perceiver",
        num_visual_queries=64,
        resampler_layers=2,
        resampler_heads=8,
    )
    model.configure_stage2(lora_r=16, lora_alpha=32)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[o3d] ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    with open(args.probe_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if args.max_samples > 0:
        items = items[: args.max_samples]
    print(f"[o3d] evaluating {len(items)} samples from {args.probe_json}")

    tok = model.tokenizer
    data_root = Path(args.data_root)

    stats = defaultdict(lambda: {"n": 0, "ok": 0, "partial": 0})
    domain_stats = defaultdict(lambda: {"n": 0, "ok": 0, "partial": 0})
    examples = []
    per_sample = []
    scene_stats = defaultdict(lambda: {"n": 0, "ok": 0, "partial": 0})
    qname_stats = defaultdict(lambda: {"n": 0, "ok": 0, "partial": 0})
    t0 = time.time()

    for i, item in enumerate(items):
        convs = item["conversations"]
        human = gpt = ""
        for turn in convs:
            if turn["from"] == "human":
                human = turn["value"].replace("<image>", "").replace("\n", "").strip()
            elif turn["from"] == "gpt":
                gpt = turn["value"].strip()
        prompt = f"User: {human}\nAssistant: "
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"][: args.max_text_len]

        bucket = task_bucket(item.get("question_name"), item.get("qa_type"), gpt)
        domain = "real" if str(item.get("probe_split", "")).startswith("test_real") or (
            str(item.get("scene", "")).startswith("Realworld") or str(item.get("scene", "")).startswith("Wild")
        ) else "sim"

        img_path = data_root / item["image"]
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            tr = model.vision_encoder.transform(img)
            if isinstance(tr, dict):
                pixels = {k: v.unsqueeze(0).to(device) for k, v in tr.items()}
            else:
                pixels = tr.unsqueeze(0).to(device)
            with torch.autocast("cuda", torch.bfloat16, enabled=use_autocast):
                prefix = build_prefix(model, pixels, device)
                pred = greedy_generate(
                    model, prefix, prompt_ids, device, args.max_new_tokens
                )
        except Exception as e:
            print(f"[o3d] fail #{i} {item.get('id')}: {e}")
            continue

        ok, partial = score_pair(pred, gpt, bucket)
        for key in (bucket, "ALL"):
            stats[key]["n"] += 1
            stats[key]["ok"] += int(ok)
            stats[key]["partial"] += int(partial)
        domain_stats[domain]["n"] += 1
        domain_stats[domain]["ok"] += int(ok)
        domain_stats[domain]["partial"] += int(partial)
        scene = str(item.get("scene") or "UNKNOWN")
        qn = str(item.get("question_name") or "NONE")
        scene_stats[scene]["n"] += 1
        scene_stats[scene]["ok"] += int(ok)
        scene_stats[scene]["partial"] += int(partial)
        qname_stats[qn]["n"] += 1
        qname_stats[qn]["ok"] += int(ok)
        qname_stats[qn]["partial"] += int(partial)

        if len(examples) < 12:
            examples.append({
                "id": item.get("id"),
                "scene": item.get("scene"),
                "bucket": bucket,
                "gt": gpt[:120],
                "pred": pred[:120],
                "ok": bool(ok),
            })
        if args.dump_predictions:
            per_sample.append({
                "id": item.get("id"),
                "scene": scene,
                "domain": domain,
                "question_name": qn,
                "qa_type": item.get("qa_type"),
                "bucket": bucket,
                "ok": bool(ok),
                "partial": bool(partial),
                "gt": gpt,
                "pred": pred,
            })

        if (i + 1) % 50 == 0:
            acc = stats["ALL"]["ok"] / max(stats["ALL"]["n"], 1)
            print(f"[o3d] {i+1}/{len(items)} running_acc={acc:.1%} "
                  f"({time.time()-t0:.0f}s)")

    def pack(d):
        out = {}
        for k, v in d.items():
            n = max(v["n"], 1)
            out[k] = {
                "n": v["n"],
                "exact_acc": v["ok"] / n,
                "partial_acc": v["partial"] / n,
            }
        return out

    payload = {
        "ckpt": args.ckpt,
        "probe_json": args.probe_json,
        "contaminated": True,
        "contamination_note": (
            "Stage2 v2 trained on all Open3D-VQA; this is an in-distribution "
            "probe, not a clean official hold-out score."
        ),
        "n_eval": stats["ALL"]["n"],
        "by_bucket": pack(stats),
        "by_domain": pack(domain_stats),
        "by_scene": pack(scene_stats),
        "by_question_name": pack(qname_stats),
        "examples": examples,
        "elapsed_s": time.time() - t0,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[o3d] json -> {args.out_json}")
    if args.dump_predictions and per_sample:
        pred_path = Path(args.out_json).with_suffix(".preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[o3d] preds -> {pred_path}")

    if args.out_md:
        lines = [
            "# Open3D-VQA Official-style Probe",
            "",
            "> **污染声明**：Stage2 v2 训练已含全部 Open3D-VQA → 下表为 "
            "**in-distribution probe**，非干净官方 hold-out。",
            "",
            f"- checkpoint: `{args.ckpt}`",
            f"- probe: `{args.probe_json}`",
            f"- n={payload['n_eval']}, elapsed={payload['elapsed_s']:.0f}s",
            "",
            "## By domain",
            "",
            "| domain | n | exact | partial |",
            "|---|---|---|---|",
        ]
        for k, v in payload["by_domain"].items():
            lines.append(
                f"| {k} | {v['n']} | {v['exact_acc']:.1%} | {v['partial_acc']:.1%} |"
            )
        lines += [
            "",
            "## By task bucket",
            "",
            "| bucket | n | exact | partial |",
            "|---|---|---|---|",
        ]
        for k, v in sorted(payload["by_bucket"].items()):
            lines.append(
                f"| {k} | {v['n']} | {v['exact_acc']:.1%} | {v['partial_acc']:.1%} |"
            )
        lines.append("")
        lines.append("## Examples")
        for e in examples:
            mark = "OK" if e["ok"] else "MISS"
            lines.append(f"- [{mark}] `{e['bucket']}` {e.get('scene')} — GT: {e['gt']}")
            lines.append(f"  - Pred: {e['pred']}")
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[o3d] md -> {args.out_md}")

    overall = payload["by_bucket"].get("ALL", {})
    print(f"[o3d] DONE exact={overall.get('exact_acc', 0):.1%} "
          f"partial={overall.get('partial_acc', 0):.1%}")


if __name__ == "__main__":
    main()
