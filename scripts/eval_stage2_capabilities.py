"""
Systematic capability evaluation for a Stage-2 checkpoint on the v2 data mix.

Evaluates on the SAME validation split used in training (random_split with
--split_seed, default 42), grouped by data source:

  general         CLM loss / PPL (open-ended captioning, no accuracy metric)
  aerial_spatial  CLM loss / PPL + generation exact/contains accuracy
  uav_motion      CLM loss / PPL + direction-keyword accuracy
                  (forward/backward/left/right/ascend/descend/hold)
  cognitive       CLM loss / PPL + gate-descriptor accuracy
                  (colour + shape + size tokens of the target gate)

The forward pass replicates Stage2Trainer.compute_loss exactly:
  [zero-state pair tokens | resampled vision | text], loss on answer tokens.
Generation is greedy, using the same prefix, full re-forward per step
(Mamba2 without cache; fine at eval scale).

Designed to run WHILE Stage-3 training occupies the GPU: caps its own CUDA
allocation via set_per_process_memory_fraction so it can never destabilise
the training process (the eval OOMs first).

Usage:
  python scripts/eval_stage2_capabilities.py \
      --ckpt /root/autodl-tmp/Aeromamba/checkpoints/full_stage_20260710_125315/stage2_v2/best.pth \
      --data_root /root/autodl-tmp/Aeromamba/data \
      --json_name stage2_mixed_data_v2.json \
      --loss_per_source 150 --gen_per_source 40 \
      --out_json /root/autodl-tmp/stage2_v2_capability_eval.json \
      --out_md   /root/autodl-tmp/Aeromamba/reports/stage2_v2_capability_eval.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.uav_mamba_vla import AeroMambaVLA
from data.llava_dataset import LLaVADataset


# ── answer parsing for structured sources ────────────────────────────────────

DIRECTION_KEYS = [
    ("forward", r"\bforward\b"),
    ("backward", r"\bbackward\b"),
    ("left", r"\bleft\b"),
    ("right", r"\bright\b"),
    ("ascend", r"\bascend"),
    ("descend", r"\bdescend"),
    ("hold", r"holding altitude|hold altitude"),
    ("stay", r"stay close|current position"),
]

GATE_COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "white", "black"]
GATE_SHAPES = ["round", "square", "circular", "rectangular", "triangle"]
GATE_SIZES = ["small", "large", "big"]


def direction_labels(text: str) -> set:
    t = text.lower()
    return {k for k, pat in DIRECTION_KEYS if re.search(pat, t)}


def gate_descriptors(text: str) -> set:
    t = text.lower()
    out = set()
    for vocab in (GATE_COLORS, GATE_SHAPES, GATE_SIZES):
        out |= {w for w in vocab if re.search(rf"\b{w}\b", t)}
    return out


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


# ── model forward helpers (mirror Stage2Trainer.compute_loss) ────────────────

def build_prefix(model, pixels, device, dtype):
    """[state_pair | vision] prefix embeddings, matching Stage-2 training."""
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
def answer_loss(model, prefix, prompt_ids, response_ids, device):
    """CLM loss over response tokens only (prompt masked), like training."""
    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    text_embs = model._embed_text(input_ids)
    inputs_embeds = torch.cat([prefix, text_embs], dim=1)
    hidden = model._run_mamba(inputs_embeds)
    logits = model.mamba.lm_head(hidden).float()
    prefix_len = prefix.size(1)
    shift_logits = logits[:, prefix_len - 1 : -1, :]
    labels = torch.tensor(
        [[-100] * len(prompt_ids) + response_ids], device=device
    )
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    ).item()


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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--json_name", default="stage2_mixed_data_v2.json")
    ap.add_argument("--subset_json", default=None,
                    help="pre-exported eval subset (data/export_eval_subset.py); "
                         "items are used directly in file order per source, "
                         "skipping the full-dataset val-split reproduction")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--loss_per_source", type=int, default=150)
    ap.add_argument("--gen_per_source", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--mem_fraction", type=float, default=0.16,
                    help="hard cap on this process' share of GPU memory")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16",
                    help="bf16 halves memory so eval can share the GPU with a "
                         "running training job")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    print(f"[eval] device={device} mem_fraction={args.mem_fraction}")

    # ── model (aeromamba_opt config, mirrors trainer.run) ────────────────────
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
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] ckpt loaded: {len(missing)} missing / {len(unexpected)} unexpected")
    # Keep weights fp32 and rely on autocast: casting the whole model to bf16
    # breaks the sinusoidal proprio embedding (fp32 buffers) with
    # "expected scalar type BFloat16 but found Float".
    dtype = torch.float32
    use_autocast = args.dtype == "bf16" and device.type == "cuda"
    model.to(device).eval()

    if args.subset_json:
        # pre-exported subset: items already selected & ordered per source
        with open(args.subset_json, "r", encoding="utf-8") as f:
            subset_items = json.load(f)
        print(f"[eval] subset: {len(subset_items)} samples from {args.subset_json}")

        class _SubsetDS:
            data = subset_items
        ds = _SubsetDS()
        by_source = defaultdict(list)
        for i, item in enumerate(subset_items):
            by_source[item.get("source", "general")].append(i)
    else:
        # ── reproduce the training val split ─────────────────────────────────
        ds = LLaVADataset(
            data_root=args.data_root,
            tokenizer=model.tokenizer,
            transform=model.vision_encoder.transform,
            max_text_len=args.max_text_len,
            json_name=args.json_name,
        )
        n_val = max(1, int(len(ds) * args.val_frac))
        n_train = len(ds) - n_val
        gen = torch.Generator().manual_seed(args.split_seed)
        perm = torch.randperm(len(ds), generator=gen).tolist()
        val_indices = perm[n_train:]
        print(f"[eval] val split: {len(val_indices)} samples")

        by_source = defaultdict(list)
        for i in val_indices:
            by_source[ds.sources[i]].append(i)
        rng = random.Random(args.seed)
        for s in by_source:
            rng.shuffle(by_source[s])

    tok = model.tokenizer

    def encode_pair(item):
        convs = item["conversations"]
        human_text, gpt_text = "", ""
        for turn in convs:  # keep LAST pair — matches LLaVADataset.__getitem__
            if turn["from"] == "human":
                human_text = turn["value"].replace("<image>", "").replace("\n", "").strip()
            elif turn["from"] == "gpt":
                gpt_text = turn["value"].strip()
        prompt = f"User: {human_text}\nAssistant: "
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tok(gpt_text, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tok.eos_token_id]
        # truncate like training (max_text_len over prompt+response)
        total = (prompt_ids + response_ids)[: args.max_text_len]
        prompt_ids = total[: len(prompt_ids)]
        response_ids = total[len(prompt_ids):]
        return prompt, prompt_ids, response_ids, gpt_text

    def load_pixels(item):
        from PIL import Image
        img = Image.open(Path(args.data_root) / item["image"]).convert("RGB")
        tr = model.vision_encoder.transform(img)
        if isinstance(tr, dict):
            return {k: v.unsqueeze(0).to(device=device, dtype=dtype)
                    for k, v in tr.items()}
        return tr.unsqueeze(0).to(device=device, dtype=dtype)

    results = {}
    examples = defaultdict(list)

    for source in sorted(by_source):
        idxs = by_source[source]
        n_loss = min(args.loss_per_source, len(idxs))
        n_gen = min(args.gen_per_source, len(idxs))

        # 1. per-source CLM loss / PPL
        losses = []
        t0 = time.time()
        for i in idxs[:n_loss]:
            item = ds.data[i]
            try:
                pixels = load_pixels(item)
                _, p_ids, r_ids, _ = encode_pair(item)
                if not r_ids:
                    continue
                with torch.autocast("cuda", torch.bfloat16, enabled=use_autocast):
                    prefix = build_prefix(model, pixels, device, None)
                    losses.append(
                        answer_loss(model, prefix, p_ids, r_ids, device)
                    )
            except Exception as e:
                print(f"[eval] loss sample failed ({source} #{i}): {e}")
        mean_loss = sum(losses) / max(len(losses), 1)
        ppl = math.exp(min(mean_loss, 20.0))
        print(f"[eval] {source}: loss={mean_loss:.4f} ppl={ppl:.2f} "
              f"(n={len(losses)}, {time.time()-t0:.0f}s)")

        # 2. generation accuracy for structured sources
        gen_stats = None
        if source in ("aerial_spatial", "uav_motion", "cognitive"):
            n_ok = n_partial = n_tot = 0
            t0 = time.time()
            for i in idxs[:n_gen]:
                item = ds.data[i]
                try:
                    pixels = load_pixels(item)
                    _, p_ids, _, gt = encode_pair(item)
                    with torch.autocast("cuda", torch.bfloat16, enabled=use_autocast):
                        prefix = build_prefix(model, pixels, device, None)
                        pred = greedy_generate(
                            model, prefix, p_ids, device, args.max_new_tokens
                        )
                except Exception as e:
                    print(f"[eval] gen sample failed ({source} #{i}): {e}")
                    continue
                n_tot += 1
                if source == "uav_motion":
                    ok = direction_labels(pred) == direction_labels(gt)
                    partial = bool(direction_labels(pred) & direction_labels(gt))
                elif source == "cognitive":
                    gt_d, pr_d = gate_descriptors(gt), gate_descriptors(pred)
                    ok = gt_d == pr_d and len(gt_d) > 0
                    partial = bool(gt_d & pr_d)
                else:  # aerial_spatial
                    ok = norm_text(pred) == norm_text(gt)
                    partial = norm_text(gt) in norm_text(pred) or ok
                n_ok += ok
                n_partial += partial
                if len(examples[source]) < 5:
                    examples[source].append(
                        {"q": item["conversations"][0]["value"][:160],
                         "gt": gt[:120], "pred": pred[:120],
                         "exact": bool(ok)})
            gen_stats = {
                "n": n_tot,
                "exact_acc": n_ok / max(n_tot, 1),
                "partial_acc": n_partial / max(n_tot, 1),
            }
            print(f"[eval] {source}: exact={gen_stats['exact_acc']:.1%} "
                  f"partial={gen_stats['partial_acc']:.1%} "
                  f"(n={n_tot}, {time.time()-t0:.0f}s)")

        results[source] = {
            "n_loss": len(losses),
            "clm_loss": mean_loss,
            "ppl": ppl,
            "generation": gen_stats,
        }

    payload = {
        "ckpt": args.ckpt,
        "json_name": args.json_name,
        "val_split_seed": args.split_seed,
        "results": results,
        "examples": examples,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] json -> {args.out_json}")

    if args.out_md:
        lines = [
            "# Stage 2 能力系统性测评",
            "",
            f"- checkpoint: `{args.ckpt}`",
            f"- 数据: `{args.json_name}` 验证集（split_seed={args.split_seed}，与训练一致）",
            f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| source | 能力维度 | n | CLM loss | PPL | 生成精确率 | 部分正确率 |",
            "|---|---|---|---|---|---|---|",
        ]
        dim_name = {
            "general": "通用图文理解",
            "aerial_spatial": "俯视空间推理",
            "uav_motion": "指令→运动语义",
            "cognitive": "认知推理→指令解析",
        }
        for s, r in results.items():
            g = r["generation"]
            acc = f"{g['exact_acc']:.1%}" if g else "—"
            pacc = f"{g['partial_acc']:.1%}" if g else "—"
            lines.append(
                f"| {s} | {dim_name.get(s, s)} | {r['n_loss']} | "
                f"{r['clm_loss']:.3f} | {r['ppl']:.2f} | {acc} | {pacc} |"
            )
        lines.append("")
        for s, exs in examples.items():
            lines.append(f"## {s} 生成样例")
            for e in exs:
                mark = "✅" if e["exact"] else "❌"
                lines.append(f"- {mark} Q: {e['q']}")
                lines.append(f"  - GT: {e['gt']}")
                lines.append(f"  - Pred: {e['pred']}")
            lines.append("")
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[eval] md -> {args.out_md}")


if __name__ == "__main__":
    main()
