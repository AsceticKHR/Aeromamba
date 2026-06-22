"""
AeroMamba-VLA Inference Server.

Flask HTTP server compatible with the UAV-Flow-Eval evaluation harness.
Exposes the same /predict and /reset endpoints as the original OpenVLA-UAV
server, making it a drop-in replacement for batch_run_act_all.py.

Endpoints:
    POST /predict
        Request  : {"image": "<base64 PNG/JPG>",
                    "proprio": [dx, dy, dz, dyaw_deg],
                    "instr": "fly forward to the red building"}
        Response : {"action": [[dx,dy,dz,dyaw_rad], ...], "done": false}

    POST /reset
        Resets the temporal ensemble buffer.
        Response : {"status": "ok"}

    GET /health
        Response : {"status": "healthy", "model": "<mamba_type>"}

Usage:
    python inference/server.py --ckpt checkpoints/stage3/best.pth --port 5007
    python inference/server.py --dummy --port 5007   # quick test, random weights
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.uav_mamba_vla import AeroMambaVLA
from model.action_head   import TemporalEnsemble

logger = logging.getLogger(__name__)
app    = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global server state (initialised in main())
# ─────────────────────────────────────────────────────────────────────────────
_model:    AeroMambaVLA    | None = None
_ensemble: TemporalEnsemble | None = None
_device:   torch.device    | None = None
_args:     argparse.Namespace | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_image(b64_str: str) -> Image.Image:
    """Decode base64 PNG/JPG string → PIL Image (RGB)."""
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def preprocess_image(pil_img: Image.Image):
    """
    Apply model vision transform.

    For single encoders  : returns [1, 3, H, W] tensor on device.
    For DinoSigLIP dual  : returns dict {"dino": [1,3,H,W], "siglip": [1,3,H,W]}
                           where both tensors are on device.
    """
    result = _model.vision_encoder.transform(pil_img)
    if isinstance(result, dict):
        # DinoSigLIPTransform: move each stream to device
        return {k: v.unsqueeze(0).to(_device) for k, v in result.items()}
    # Single encoder: plain tensor
    return result.unsqueeze(0).to(_device)


def preprocess_proprio(proprio_list: list) -> torch.Tensor:
    """
    Convert UAV-Flow client format [dx, dy, dz, dyaw_deg] to model format.
    Positions divided by pos_scale (cm → normalised), yaw deg → rad.
    """
    pos_scale = getattr(_args, "pos_scale", 100.0)
    dx  = float(proprio_list[0]) / pos_scale if len(proprio_list) > 0 else 0.0
    dy  = float(proprio_list[1]) / pos_scale if len(proprio_list) > 1 else 0.0
    dz  = float(proprio_list[2]) / pos_scale if len(proprio_list) > 2 else 0.0
    yaw = float(np.radians(proprio_list[3]))  if len(proprio_list) > 3 else 0.0
    return torch.tensor([[dx, dy, dz, yaw]], dtype=torch.float32, device=_device)


def preprocess_instruction(instr: str) -> torch.Tensor:
    """Tokenise instruction string → [1, L] int64."""
    tokens = _model.tokenizer(
        instr,
        max_length=getattr(_args, "max_text_len", 64),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokens["input_ids"].to(_device)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    t0   = time.time()
    data = request.get_json(force=True)

    try:
        b64_img   = data.get("image",   "")
        proprio_l = data.get("proprio", [0, 0, 0, 0])
        instr     = data.get("instr",   "")

        # Pre-process inputs
        pixels    = preprocess_image(decode_image(b64_img))
        proprio   = preprocess_proprio(proprio_l)
        input_ids = preprocess_instruction(instr)

        # Model inference
        with torch.inference_mode():
            pred = _model.predict_step(
                pixel_values=pixels,
                input_ids=input_ids,
                proprio=proprio,
            )

        # pred["action"]: [1, K, 4]
        chunk = pred["action"][0]         # [K, 4]  on device

        # Temporal ensemble → smooth single action [4]
        smooth_action = _ensemble.update(chunk.cpu())

        # Build response
        if getattr(_args, "ensemble_mode", False):
            # Return one temporally ensembled action per step
            action_out = [smooth_action.tolist()]
        else:
            # Return full K-step chunk; client executes sequentially
            action_out = chunk.cpu().tolist()

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(f"Inference: {elapsed_ms:.1f} ms | action[0]={action_out[0]}")
        return jsonify({"action": action_out, "done": False})

    except Exception as exc:
        logger.exception(f"Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset episode state (clears temporal ensemble buffer)."""
    _ensemble.reset()
    logger.info("Reset: ensemble buffer cleared")
    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    model_name = getattr(_args, "mamba_type", "unknown") if _args else "unknown"
    return jsonify({"status": "healthy", "model": model_name})


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args) -> AeroMambaVLA:
    model = AeroMambaVLA(
        mamba_type=args.mamba_type,
        vision_type=args.vision_type,
        chunk_size=args.chunk_size,
        freeze_vision=True,
    )
    if args.ckpt and Path(args.ckpt).exists():
        logger.info(f"Loading checkpoint: {args.ckpt}")
        ckpt  = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    elif args.ckpt:
        logger.warning(f"Checkpoint not found: {args.ckpt} — using random weights")
    else:
        logger.info("No checkpoint — demo mode with random weights")

    model.eval()
    return model


def get_args():
    p = argparse.ArgumentParser(description="AeroMamba-VLA Inference Server")
    p.add_argument("--ckpt",            default=None)
    p.add_argument("--mamba_type",      default="mamba-370m")
    p.add_argument("--vision_type",     default="siglip_l_384")
    p.add_argument("--chunk_size",      type=int,   default=5)
    p.add_argument("--port",            type=int,   default=5007)
    p.add_argument("--host",            default="0.0.0.0")
    p.add_argument("--max_text_len",    type=int,   default=64)
    p.add_argument("--pos_scale",       type=float, default=100.0,
                   help="Position scaling divisor (cm → normalised)")
    p.add_argument("--ensemble_window", type=int,   default=5)
    p.add_argument("--ensemble_decay",  type=float, default=0.7)
    p.add_argument("--ensemble_mode",   action="store_true",
                   help="Return 1 ensembled action per step (vs full K-chunk)")
    p.add_argument("--dummy",           action="store_true",
                   help="Tiny model for quick testing")
    p.add_argument("--log_level",       default="INFO")
    return p.parse_args()


def main():
    global _model, _ensemble, _device, _args

    _args = get_args()
    logging.basicConfig(
        level=getattr(logging, _args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s — %(message)s",
    )

    if _args.dummy:
        _args.mamba_type  = "mamba-130m"
        _args.vision_type = "siglip_b_224"
        _args.chunk_size  = 3
        logger.info("Dummy mode: mamba-130m + siglip-b-224")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {_device}")

    logger.info("Loading model…")
    _model = load_model(_args).to(_device)
    logger.info(f"Model ready on {_device}")

    _ensemble = TemporalEnsemble(
        window=_args.ensemble_window,
        decay=_args.ensemble_decay,
    )

    logger.info(
        f"Server starting on {_args.host}:{_args.port} | "
        f"chunk_size={_args.chunk_size} | ensemble_mode={_args.ensemble_mode}"
    )
    app.run(host=_args.host, port=_args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
