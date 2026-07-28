"""Strict small-scale eval for AeroMamba v2 Stage-1 projector alignment.

Gates (all hard by default):
  E1  trained val loss is finite and below ceiling (default 3.0)
  E2  trained vs random-projector loss gap (default >= 1.5 nats)
  E3  blank-image ablation: Δloss >= min_blank_delta
  E4  image-shuffle mismatch: Δloss >= min_shuffle_delta
  E5  teacher-forced token accuracy beats random by margin

Usage (remote):
  python scripts/eval_v2_s1.py \
      --ckpt checkpoints/v2_stage1_full/best_projector.pth \
      --data_root data/llava_pretrain \
      --json_name blip_laion_cc_sbu_558k.json \
      --n_samples 128 --batch 8 \
      --report reports/v2_s1_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.llava_dataset import LLaVADataset, llava_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402


def load_model(args, device):
    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    model.configure_stage1(grad_ckpt=False)
    model.eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.projector.load_state_dict(ckpt["projector"], strict=True)
    print(f"[eval] loaded projector from {args.ckpt} "
          f"(step={ckpt.get('step')} val={ckpt.get('val')}) "
          f"missing={missing} unexpected={unexpected}")
    return model, ckpt


@torch.no_grad()
def batch_metrics(model, batch, device, mutate=None):
    """Return mean CE loss + teacher-forced token accuracy on non-ignored labels."""
    pv = batch["pixel_values"].to(device)
    ids = batch["input_ids"].to(device)
    lbl = batch["labels"].to(device)
    if mutate == "blank":
        pv = torch.zeros_like(pv)
    elif mutate == "shuffle":
        # mismatch image↔caption within the batch
        if pv.size(0) > 1:
            pv = pv.roll(1, dims=0)
    out = model.forward_clm(pv, ids, lbl)
    loss = float(out["loss"])
    logits = out["logits"]  # [B, L, V]
    pred = logits.argmax(dim=-1)
    mask = lbl != -100
    n = int(mask.sum().item())
    correct = int(((pred == lbl) & mask).sum().item()) if n else 0
    return {"loss": loss, "n_tok": n, "n_correct": correct}


def accumulate(loader, model, device, mutate=None):
    tot_loss = 0.0
    n_batch = 0
    n_tok = 0
    n_correct = 0
    for batch in loader:
        m = batch_metrics(model, batch, device, mutate=mutate)
        tot_loss += m["loss"]
        n_batch += 1
        n_tok += m["n_tok"]
        n_correct += m["n_correct"]
    return {
        "loss": tot_loss / max(n_batch, 1),
        "token_acc": (n_correct / max(n_tok, 1)),
        "n_tok": n_tok,
        "n_batches": n_batch,
    }


@torch.no_grad()
def greedy_captions(model, ds, device, indices, max_new=48):
    """Minimal greedy caption samples for qualitative inspection."""
    samples = []
    tok = model.tokenizer
    for idx in indices:
        item = ds.data[idx]
        sample = ds[idx]
        pv = sample["pixel_values"].unsqueeze(0).to(device)
        # rebuild prompt only
        human = ""
        gpt = ""
        for turn in item["conversations"]:
            if turn["from"] == "human":
                human = turn["value"].replace("<image>", "").replace("\n", "").strip()
            elif turn["from"] == "gpt":
                gpt = turn["value"].strip()
        prompt = f"User: {human}\nAssistant: "
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)

        vis = model.vision_encoder(pv)
        vis_tok = model.projector(vis.to(model.dtype))
        emb = model.lm.get_input_embeddings()
        gen_ids = list(prompt_ids)
        for _ in range(max_new):
            txt = emb(ids) * getattr(model.lm.config, "embedding_multiplier", 1.0)
            embeds = torch.cat([vis_tok, txt.to(vis_tok.dtype)], dim=1)
            attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
            trunk = getattr(model.lm, "model", None) or model.lm.base_model
            hidden = trunk(inputs_embeds=embeds, attention_mask=attn,
                           use_cache=False).last_hidden_state
            # last text position predicts next token
            h = hidden[:, -1:, :]
            logits = model.lm.get_output_embeddings()(h)
            logits = logits * getattr(model.lm.config, "lm_head_multiplier", 1.0)
            next_id = int(logits[0, 0].argmax().item())
            if next_id == tok.eos_token_id:
                break
            gen_ids.append(next_id)
            ids = torch.tensor([gen_ids], device=device, dtype=torch.long)
        pred = tok.decode(gen_ids[len(prompt_ids):], skip_special_tokens=True)
        samples.append({
            "idx": idx,
            "image": item.get("image"),
            "prompt": human,
            "gt": gpt,
            "pred": pred.strip(),
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--json_name", default="blip_laion_cc_sbu_558k.json")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--max_loss", type=float, default=3.0)
    ap.add_argument("--min_gap_vs_random", type=float, default=1.5)
    ap.add_argument("--min_blank_delta", type=float, default=0.08)
    ap.add_argument("--min_shuffle_delta", type=float, default=0.04)
    ap.add_argument("--min_acc_margin", type=float, default=0.05)
    ap.add_argument("--n_captions", type=int, default=6)
    ap.add_argument("--report", default="reports/v2_s1_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model, ckpt = load_model(args, device)
    ds = LLaVADataset(
        data_root=args.data_root,
        tokenizer=model.tokenizer,
        transform=model.vision_encoder.transform,
        max_text_len=args.max_text_len,
        json_name=args.json_name,
    )
    # held-out slice matching training val seed convention (tail of shuffled)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    # use a mid-band slice to avoid overlapping the first N used by short pilots
    start = max(0, len(ds) // 2)
    eval_idx = indices[start: start + args.n_samples]
    if len(eval_idx) < args.n_samples:
        eval_idx = indices[: args.n_samples]
    loader = DataLoader(
        Subset(ds, eval_idx),
        batch_size=args.batch,
        shuffle=False,
        collate_fn=llava_collate_fn,
        num_workers=2,
    )
    print(f"[eval] n_samples={len(eval_idx)} batch={args.batch} device={device}")

    trained = accumulate(loader, model, device)
    blank = accumulate(loader, model, device, mutate="blank")
    shuffle = accumulate(loader, model, device, mutate="shuffle")

    # random projector baseline (same frozen vision/LM)
    with torch.no_grad():
        for m in model.projector.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        model.projector.to(model.dtype)
    random_m = accumulate(loader, model, device)

    # reload trained weights for captions
    model.projector.load_state_dict(ckpt["projector"], strict=True)
    model.projector.to(model.dtype)
    cap_idx = eval_idx[: args.n_captions]
    captions = greedy_captions(model, ds, device, cap_idx)

    gap = random_m["loss"] - trained["loss"]
    blank_delta = blank["loss"] - trained["loss"]
    shuffle_delta = shuffle["loss"] - trained["loss"]
    acc_margin = trained["token_acc"] - random_m["token_acc"]

    gates = {
        "E1_loss_ceiling": {
            "ok": math.isfinite(trained["loss"]) and trained["loss"] < args.max_loss,
            "value": trained["loss"],
            "threshold": f"< {args.max_loss}",
        },
        "E2_gap_vs_random": {
            "ok": gap >= args.min_gap_vs_random,
            "value": gap,
            "threshold": f">= {args.min_gap_vs_random}",
            "trained": trained["loss"],
            "random": random_m["loss"],
        },
        "E3_blank_ablation": {
            "ok": blank_delta >= args.min_blank_delta,
            "value": blank_delta,
            "threshold": f">= {args.min_blank_delta}",
            "blank_loss": blank["loss"],
        },
        "E4_shuffle_ablation": {
            "ok": shuffle_delta >= args.min_shuffle_delta,
            "value": shuffle_delta,
            "threshold": f">= {args.min_shuffle_delta}",
            "shuffle_loss": shuffle["loss"],
        },
        "E5_token_acc_margin": {
            "ok": acc_margin >= args.min_acc_margin,
            "value": acc_margin,
            "threshold": f">= {args.min_acc_margin}",
            "trained_acc": trained["token_acc"],
            "random_acc": random_m["token_acc"],
        },
    }
    fails = sum(1 for g in gates.values() if not g["ok"])
    verdict = "PASS" if fails == 0 else "FAIL"

    report = {
        "verdict": verdict,
        "hard_failures": fails,
        "ckpt": str(args.ckpt),
        "n_samples": len(eval_idx),
        "metrics": {
            "trained": trained,
            "random": random_m,
            "blank": blank,
            "shuffle": shuffle,
            "gap_vs_random": gap,
            "blank_delta": blank_delta,
            "shuffle_delta": shuffle_delta,
            "acc_margin": acc_margin,
        },
        "gates": gates,
        "captions": captions,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== S1 strict eval ===")
    for name, g in gates.items():
        flag = "PASS" if g["ok"] else "FAIL"
        print(f"[{flag}] {name}: value={g['value']:.4f} ({g['threshold']})")
    print(f"\nverdict={verdict} hard_failures={fails}")
    print(f"report -> {out}")
    for c in captions:
        print(f"\n--- sample {c['idx']} ---")
        print(f"GT  : {c['gt'][:160]}")
        print(f"PRED: {c['pred'][:160]}")
    sys.exit(fails)


if __name__ == "__main__":
    main()
