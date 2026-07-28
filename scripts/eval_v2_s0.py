"""Strict small-scale eval for AeroMamba v2 Stage-0 aerial CPT.

Compares S0 (projector+LoRA+vision-tail) vs S1-only projector on L0 aerial
holdout. Hard gates require CPT to beat S1 and retain vision use.

Usage:
  python scripts/eval_v2_s0.py \
      --s0_dir checkpoints/v2_stage0_full \
      --s1_ckpt checkpoints/v2_stage1_full/best_projector.pth \
      --jsonl /root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl \
      --n_samples 192 --report reports/v2_s0_eval.json
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

from data.l0_dataset import L0CPTDataset  # noqa: E402
from data.llava_dataset import llava_collate_fn  # noqa: E402
from model.aerov2 import AeroV2  # noqa: E402

AERIAL = {"aerial_spatial", "hrvqa", "airspatial", "uav_motion"}


def build_s1(args, device):
    m = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    m.configure_stage1(grad_ckpt=False)
    m.load_projector(args.s1_ckpt)
    m.eval()
    return m


def build_s0(args, device):
    from peft import PeftModel

    m = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type).to(device)
    s0 = Path(args.s0_dir)
    proj = torch.load(s0 / "best_projector.pth", map_location="cpu", weights_only=False)
    m.projector.load_state_dict(proj["projector"], strict=True)
    vis = torch.load(s0 / "best_vision.pth", map_location="cpu", weights_only=False)
    m.vision_encoder.load_state_dict(vis, strict=True)
    # Load LoRA onto the bare LM (do not call configure_stage0 — that would
    # re-init a random adapter).
    m.lm = PeftModel.from_pretrained(m.lm, str(s0 / "best_lora"))
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    print(f"[eval] loaded S0 from {s0} step={proj.get('step')} val={proj.get('val')}")
    return m


@torch.no_grad()
def accumulate(model, loader, device, mutate=None):
    tot = cnt = 0.0
    n_tok = n_ok = 0
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        if mutate == "blank":
            pv = torch.zeros_like(pv)
        elif mutate == "shuffle" and pv.size(0) > 1:
            pv = pv.roll(1, dims=0)
        out = model.forward_clm(pv, ids, lbl)
        tot += float(out["loss"])
        cnt += 1
        pred = out["logits"].argmax(-1)
        mask = lbl != -100
        n_tok += int(mask.sum())
        n_ok += int(((pred == lbl) & mask).sum())
    return {
        "loss": tot / max(cnt, 1),
        "token_acc": n_ok / max(n_tok, 1),
        "n_tok": n_tok,
        "n_batches": int(cnt),
    }


@torch.no_grad()
def per_source(model, ds, indices_by_src, device, batch=8):
    out = {}
    for src, idxs in indices_by_src.items():
        if not idxs:
            continue
        loader = DataLoader(Subset(ds, idxs), batch_size=batch, shuffle=False,
                            collate_fn=llava_collate_fn, num_workers=0)
        out[src] = accumulate(model, loader, device)
    return out


@torch.no_grad()
def captions(model, ds, device, indices, max_new=48):
    tok = model.tokenizer
    samples = []
    for idx in indices:
        row = ds.rows[idx]
        sample = ds[idx]
        pv = sample["pixel_values"].unsqueeze(0).to(device)
        human = gpt = ""
        for t in row["conversations"]:
            if t["from"] == "human":
                human = t["value"].replace("<image>", "").replace("\n", " ").strip()
            elif t["from"] == "gpt":
                gpt = t["value"].strip()
        prompt = f"User: {human}\nAssistant: "
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        gen = list(prompt_ids)
        ids = torch.tensor([prompt_ids], device=device)
        vis = model.vision_encoder(pv)
        vis_tok = model.projector(vis.to(model.dtype))
        emb = model.lm.get_input_embeddings()
        cfg = model._lm_config()
        emb_mult = float(getattr(cfg, "embedding_multiplier", 1.0) or 1.0)
        head_mult = float(getattr(cfg, "lm_head_multiplier", 1.0) or 1.0)
        for _ in range(max_new):
            txt = emb(ids) * emb_mult
            embeds = torch.cat([vis_tok, txt.to(vis_tok.dtype)], dim=1)
            attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
            base_lm = model.lm
            if hasattr(base_lm, "get_base_model"):
                try:
                    base_lm = base_lm.get_base_model()
                except Exception:
                    base_lm = getattr(base_lm, "base_model", base_lm)
                    base_lm = getattr(base_lm, "model", base_lm)
            trunk = getattr(base_lm, "model", None) or base_lm
            h = trunk(inputs_embeds=embeds, attention_mask=attn,
                      use_cache=False).last_hidden_state[:, -1:, :]
            logits = model.lm.get_output_embeddings()(h) * head_mult
            nid = int(logits[0, 0].argmax())
            if nid == tok.eos_token_id:
                break
            gen.append(nid)
            ids = torch.tensor([gen], device=device)
        samples.append({
            "idx": idx,
            "source": row.get("source"),
            "task": row.get("task"),
            "gt": gpt[:200],
            "pred": tok.decode(gen[len(prompt_ids):], skip_special_tokens=True).strip()[:200],
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0_dir", default="checkpoints/v2_stage0_full")
    ap.add_argument("--s1_ckpt", default="checkpoints/v2_stage1_full/best_projector.pth")
    ap.add_argument("--jsonl", default="/root/autodl-tmp/datasets/l0_cpt/l0_mixed.jsonl")
    ap.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_len", type=int, default=192)
    ap.add_argument("--min_gain_vs_s1", type=float, default=0.15,
                    help="S1_loss - S0_loss must be >= this")
    ap.add_argument("--max_s0_loss", type=float, default=1.6)
    ap.add_argument("--min_blank_delta", type=float, default=0.05)
    ap.add_argument("--min_shuffle_delta", type=float, default=0.03)
    ap.add_argument("--n_captions", type=int, default=6)
    ap.add_argument("--report", default="reports/v2_s0_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    report_path = Path(args.jsonl).parent / "build_report.json"
    image_roots = None
    if report_path.exists():
        image_roots = json.loads(report_path.read_text(encoding="utf-8")).get("image_roots")

    # Build S0 first (heavier), eval, free, then S1 baseline
    s0 = build_s0(args, device)
    ds = L0CPTDataset(
        jsonl_path=args.jsonl,
        tokenizer=s0.tokenizer,
        transform=s0.vision_encoder.transform,
        image_roots=image_roots,
        max_text_len=args.max_text_len,
    )
    aerial = [i for i, r in enumerate(ds.rows) if r.get("source") in AERIAL]
    rng.shuffle(aerial)
    eval_idx = aerial[: args.n_samples]
    # per-source stratified small slices for breakdown
    by_src = {}
    for i in aerial:
        src = ds.rows[i]["source"]
        by_src.setdefault(src, []).append(i)
    src_eval = {s: idxs[:48] for s, idxs in by_src.items()}

    loader = DataLoader(Subset(ds, eval_idx), batch_size=args.batch, shuffle=False,
                        collate_fn=llava_collate_fn, num_workers=2)
    print(f"[eval] aerial n={len(eval_idx)} device={device}")

    s0_m = accumulate(s0, loader, device)
    s0_blank = accumulate(s0, loader, device, mutate="blank")
    s0_shuf = accumulate(s0, loader, device, mutate="shuffle")
    s0_src = per_source(s0, ds, src_eval, device, batch=args.batch)
    caps = captions(s0, ds, device, eval_idx[: args.n_captions])

    del s0
    torch.cuda.empty_cache()

    s1 = build_s1(args, device)
    # rebuild dataset with s1 transform (same image size expected)
    ds1 = L0CPTDataset(
        jsonl_path=args.jsonl,
        tokenizer=s1.tokenizer,
        transform=s1.vision_encoder.transform,
        image_roots=image_roots,
        max_text_len=args.max_text_len,
    )
    loader1 = DataLoader(Subset(ds1, eval_idx), batch_size=args.batch, shuffle=False,
                         collate_fn=llava_collate_fn, num_workers=2)
    s1_m = accumulate(s1, loader1, device)
    s1_src = per_source(s1, ds1, src_eval, device, batch=args.batch)
    del s1
    torch.cuda.empty_cache()

    gain = s1_m["loss"] - s0_m["loss"]
    blank_d = s0_blank["loss"] - s0_m["loss"]
    shuf_d = s0_shuf["loss"] - s0_m["loss"]

    gates = {
        "E1_s0_loss_ceiling": {
            "ok": math.isfinite(s0_m["loss"]) and s0_m["loss"] < args.max_s0_loss,
            "value": s0_m["loss"],
            "threshold": f"< {args.max_s0_loss}",
        },
        "E2_gain_vs_s1": {
            "ok": gain >= args.min_gain_vs_s1,
            "value": gain,
            "threshold": f">= {args.min_gain_vs_s1}",
            "s0": s0_m["loss"],
            "s1": s1_m["loss"],
        },
        "E3_blank_ablation": {
            "ok": blank_d >= args.min_blank_delta,
            "value": blank_d,
            "threshold": f">= {args.min_blank_delta}",
        },
        "E4_shuffle_ablation": {
            "ok": shuf_d >= args.min_shuffle_delta,
            "value": shuf_d,
            "threshold": f">= {args.min_shuffle_delta}",
        },
        "E5_token_acc_beats_s1": {
            "ok": s0_m["token_acc"] > s1_m["token_acc"],
            "value": s0_m["token_acc"] - s1_m["token_acc"],
            "threshold": "> 0",
            "s0_acc": s0_m["token_acc"],
            "s1_acc": s1_m["token_acc"],
        },
    }
    fails = sum(1 for g in gates.values() if not g["ok"])
    verdict = "PASS" if fails == 0 else "FAIL"

    report = {
        "verdict": verdict,
        "hard_failures": fails,
        "n_samples": len(eval_idx),
        "metrics": {
            "s0": s0_m,
            "s1": s1_m,
            "s0_blank": s0_blank,
            "s0_shuffle": s0_shuf,
            "gain_vs_s1": gain,
            "blank_delta": blank_d,
            "shuffle_delta": shuf_d,
        },
        "per_source": {"s0": s0_src, "s1": s1_src},
        "gates": gates,
        "captions": caps,
        "train_val_curve_best": 1.0332,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== S0 strict aerial eval ===")
    for name, g in gates.items():
        print(f"[{'PASS' if g['ok'] else 'FAIL'}] {name}: "
              f"value={g['value']:.4f} ({g['threshold']})")
    print(f"\nper-source loss (s0 / s1):")
    for src in sorted(set(s0_src) | set(s1_src)):
        a = s0_src.get(src, {}).get("loss", float("nan"))
        b = s1_src.get(src, {}).get("loss", float("nan"))
        print(f"  {src:16s}  s0={a:.3f}  s1={b:.3f}  Δ={b-a:+.3f}")
    print(f"\nverdict={verdict} hard_failures={fails}")
    print(f"report -> {out}")
    for c in caps:
        print(f"\n--- {c['source']}/{c['task']} ---")
        print(f"GT  : {c['gt']}")
        print(f"PRED: {c['pred']}")
    sys.exit(fails)


if __name__ == "__main__":
    main()
