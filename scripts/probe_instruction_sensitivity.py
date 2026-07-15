"""
Instruction-sensitivity probe (AeroStream change plan §A6).

For a fixed set of evaluation frames and pairs of direction-opposed
instructions (left/right, ascend/descend, forward/stop), measure how much the
predicted action chunk changes when ONLY the instruction changes. During the
binding round the divergence for opposed pairs should rise monotonically over
checkpoints; a flat curve means language still does not reach the action
pathway (early abort signal).

Usage (local WSL / eval box, not the training GPU):
  python scripts/probe_instruction_sensitivity.py \
      --ckpt checkpoints/stage3_v3/best_slim.pth \
      --frames_dir <dir with .jpg frames>  [--n_frames 16] \
      [--use_lora] [--report probe_report.json]

Frames default to the first frame of the first N episodes under
--frames_dir (searched recursively), so the same probe set is reproducible
across checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.uav_mamba_vla import AeroMambaVLA

# 8 direction-opposed instruction pairs (fixed probe set).
INSTRUCTION_PAIRS = [
    ("Move to the left side.", "Move to the right side."),
    ("Pass the building from the left side.", "Pass the building from the right side."),
    ("Turn left.", "Turn right."),
    ("Rotate counterclockwise.", "Rotate clockwise."),
    ("Ascend to a higher altitude.", "Descend to a lower altitude."),
    ("Rise straight up.", "Lower yourself straight down."),
    ("Fly forward toward the target.", "Stop and hover in place."),
    ("Advance quickly ahead.", "Stay where you are."),
]


def build_model(args) -> AeroMambaVLA:
    model = AeroMambaVLA(
        mamba_type=args.mamba_type,
        vision_type=args.vision_type,
        chunk_size=args.chunk_size,
        token_resampler="perceiver",
        num_visual_queries=args.num_visual_queries,
    )
    if args.use_lora:
        model.configure_stage2(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[probe] loaded {args.ckpt} (missing={len(missing)} unexpected={len(unexpected)})")
    else:
        print("[probe] WARNING: no checkpoint — probing random weights")
    model.eval()
    return model


def collect_frames(frames_dir: str, n_frames: int) -> list:
    root = Path(frames_dir)
    frames = []
    seen_dirs = set()
    for jpg in sorted(root.rglob("000000.jpg")):
        if jpg.parent in seen_dirs:
            continue
        seen_dirs.add(jpg.parent)
        frames.append(jpg)
        if len(frames) >= n_frames:
            break
    if not frames:   # fall back to any jpgs
        frames = sorted(root.rglob("*.jpg"))[:n_frames]
    if not frames:
        raise SystemExit(f"no frames found under {frames_dir}")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--mamba_type", default="mamba-2-370m")
    ap.add_argument("--vision_type", default="siglip2_base_384")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--num_visual_queries", type=int, default=64)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    frames = collect_frames(args.frames_dir, args.n_frames)
    print(f"[probe] {len(frames)} frames x {len(INSTRUCTION_PAIRS)} instruction pairs")

    def tokenize(text: str) -> torch.Tensor:
        tokens = model.tokenizer(
            text, max_length=args.max_text_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return tokens["input_ids"].to(device)

    state = torch.zeros(1, 8, device=device)
    delta = torch.zeros(1, 8, device=device)

    # divergence[pair][frame] = ||action(instrA) - action(instrB)||
    matrix = []
    per_pair_mean = []
    with torch.inference_mode():
        for pair_idx, (instr_a, instr_b) in enumerate(INSTRUCTION_PAIRS):
            ids_a, ids_b = tokenize(instr_a), tokenize(instr_b)
            row = []
            for frame in frames:
                img = Image.open(frame).convert("RGB")
                pixels = model.vision_encoder.transform(img)
                if isinstance(pixels, dict):
                    pixels = {k: v.unsqueeze(0).to(device) for k, v in pixels.items()}
                else:
                    pixels = pixels.unsqueeze(0).to(device)
                act_a = model.predict_step(
                    pixel_values=pixels, input_ids=ids_a, state=state, delta_state=delta
                )["action"][0]
                act_b = model.predict_step(
                    pixel_values=pixels, input_ids=ids_b, state=state, delta_state=delta
                )["action"][0]
                row.append(float(torch.linalg.vector_norm(act_a - act_b)))
            matrix.append(row)
            mean_div = sum(row) / len(row)
            per_pair_mean.append(mean_div)
            print(f"  pair {pair_idx} [{instr_a[:30]!r} vs {instr_b[:30]!r}]: "
                  f"mean divergence {mean_div:.4f}")

    overall = sum(per_pair_mean) / len(per_pair_mean)
    print(f"[probe] OVERALL mean opposed-instruction divergence: {overall:.4f}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ckpt": str(args.ckpt),
                    "frames": [str(p) for p in frames],
                    "pairs": INSTRUCTION_PAIRS,
                    "divergence_matrix": matrix,
                    "per_pair_mean": per_pair_mean,
                    "overall_mean": overall,
                },
                f, indent=2, ensure_ascii=False,
            )
        print(f"[probe] report -> {args.report}")


if __name__ == "__main__":
    main()
