"""AeroV2 (Stage-3) inference server — UAV-Flow-Eval compatible.

Serves the *new* v2 policy (C-RADIO vision + Falcon-H1 + S0/S2 LoRA + grounding
head, extended with ProprioEncoder + UAVActionHead) behind the SAME HTTP API the
legacy ``inference/server.py`` exposed, so ``batch_run_act_all.py`` can drive it
unchanged.

Endpoints:
    POST /predict  {"image": b64, "proprio": [x,y,z,yaw_deg], "instr": str}
                -> {"action": [[dx,dy,dz,yaw_rad], ...], "done": false}
    POST /reset    -> {"status": "ok"}
    GET  /health   -> {"status": "healthy", ...}

The frame / execution-chunk / clipping / rate-limit logic is ported verbatim
from the legacy server (it is model-agnostic). The only model-specific parts are
``load_v2_model`` and the ``forward_action`` call, which returns z-space actions
that we ``denormalize`` to physical m/rad before the shared post-processing.

Usage (WSL):
  HF_ENDPOINT=https://hf-mirror.com python inference/server_v2.py \
    --s2_ckpt_dir checkpoints/v2_stage2_full_cradio --s2_tag best \
    --s3_ckpt checkpoints/v2_stage3_cradio/best.pth --vision_type cradio_v3_b \
    --action_stats datasets/uav-flow/action_stats_k8.json \
    --chunk_size 8 --exec_mode chunk --exec_horizon 4 --bf16 --port 5007
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

from model.aerov2 import AeroV2  # noqa: E402
from training.v2_stage3_action import load_s2, load_action_stats  # noqa: E402
from scripts.eval_s3_systematic import load_s3  # noqa: E402

logger = logging.getLogger(__name__)
app = Flask(__name__)

_model: AeroV2 | None = None
_device: torch.device | None = None
_args: argparse.Namespace | None = None
_prev_pose4: torch.Tensor | None = None
_prev_vel4: torch.Tensor | None = None
_last_exec_steps: int = 1
_token_cache: dict = {}
_episode: int = 0
_step: int = 0


# ── pre-processing ──────────────────────────────────────────────────────────
def decode_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    return _model.vision_encoder.transform(pil_img).unsqueeze(0).to(_device)


def _wrap_yaw(delta_yaw: float) -> float:
    return float((delta_yaw + np.pi) % (2.0 * np.pi) - np.pi)


def preprocess_proprio(proprio_list: list) -> torch.Tensor:
    """Client 4-D episode-local pose [x,y,z,yaw_deg] -> 8-D state8 tensor.

    state8 = [pose4 | velocity4] matching UAVFlowDataset.parse_state8. Velocity
    is recovered from the inter-/predict pose delta divided by the number of
    waypoints the client executed (per-step velocity, anti-runaway).
    """
    global _prev_pose4, _prev_vel4
    pos_scale = float(getattr(_args, "pos_scale", 100.0))
    dx = float(proprio_list[0]) / pos_scale if len(proprio_list) > 0 else 0.0
    dy = float(proprio_list[1]) / pos_scale if len(proprio_list) > 1 else 0.0
    dz = float(proprio_list[2]) / pos_scale if len(proprio_list) > 2 else 0.0
    yaw = float(np.radians(proprio_list[3])) if len(proprio_list) > 3 else 0.0
    pose4 = torch.tensor([[dx, dy, dz, yaw]], dtype=torch.float32, device=_device)

    if _prev_pose4 is not None:
        vel4 = pose4 - _prev_pose4
        vel4[0, 3] = _wrap_yaw(float(vel4[0, 3].item()))
        if bool(getattr(_args, "vel_per_step", 1)):
            vel4 = vel4 / float(max(1, _last_exec_steps))
    else:
        vel4 = torch.zeros_like(pose4)

    state8 = torch.cat([pose4, vel4], dim=-1)
    _prev_pose4 = pose4.clone()
    _prev_vel4 = vel4.clone()
    return state8


def preprocess_instruction(instr: str) -> torch.Tensor:
    cached = _token_cache.get(instr)
    if cached is None:
        tokens = _model.tokenizer(
            instr, max_length=int(getattr(_args, "max_text_len", 64)),
            padding="max_length", truncation=True, return_tensors="pt")
        cached = tokens["input_ids"]
        if len(_token_cache) > 512:
            _token_cache.clear()
        _token_cache[instr] = cached
    return cached.to(_device)


# ── exec-chunk selection / clipping (ported from legacy server) ──────────────
def _clip_chunk_sigma(raw_chunk: torch.Tensor) -> torch.Tensor:
    clip_sigma = float(getattr(_args, "clip_sigma", 0.0))
    if clip_sigma <= 0:
        return raw_chunk
    head = getattr(_model, "action_head", None)
    mean = getattr(head, "action_mean", None)
    std = getattr(head, "action_std", None)
    if mean is None or std is None:
        return raw_chunk
    mean = mean.detach().float().cpu()
    std = std.detach().float().cpu()
    if mean.shape != raw_chunk.shape:
        return raw_chunk
    return torch.max(torch.min(raw_chunk, mean + clip_sigma * std),
                     mean - clip_sigma * std)


def _cap_increments(selected: np.ndarray) -> np.ndarray:
    max_step_m = float(getattr(_args, "max_step_cm", 0.0)) / 100.0
    max_yaw = float(np.radians(float(getattr(_args, "max_yaw_step_deg", 0.0))))
    if (max_step_m <= 0 and max_yaw <= 0) or len(selected) == 0:
        return selected
    out = selected.astype(np.float64).copy()
    prev = np.zeros(4, dtype=np.float64)
    acc = np.zeros(4, dtype=np.float64)
    for i in range(len(out)):
        inc = out[i] - prev
        prev = out[i].copy()
        if max_step_m > 0:
            n = float(np.linalg.norm(inc[:3]))
            if n > max_step_m:
                inc[:3] *= max_step_m / n
        if max_yaw > 0:
            inc[3] = float(np.clip(_wrap_yaw(inc[3]), -max_yaw, max_yaw))
        acc = acc + inc
        acc[3] = _wrap_yaw(acc[3])
        out[i] = acc
    return out.astype(selected.dtype)


def _select_exec_actions(raw_chunk: torch.Tensor) -> np.ndarray:
    global _last_exec_steps
    raw_chunk = _clip_chunk_sigma(raw_chunk)
    exec_mode = getattr(_args, "exec_mode", "chunk")
    if exec_mode == "chunk":
        horizon = max(1, int(getattr(_args, "exec_horizon", 4)))
        # Index 0 is the degenerate zero anchor only when the training chunks
        # started at the anchor itself. With chunk_offset >= 1 it is already the
        # first real waypoint, and skipping it would execute one step ahead of
        # the trained target every cycle.
        skip_anchor = int(getattr(_args, "chunk_offset", 0)) == 0
        start = 1 if (skip_anchor and raw_chunk.shape[0] > 1) else 0
        end = min(start + horizon, raw_chunk.shape[0])
        selected = _cap_increments(raw_chunk[start:end].numpy())
        _last_exec_steps = len(selected)
        return selected
    if exec_mode == "full_chunk":
        selected = _cap_increments(raw_chunk.numpy())
        _last_exec_steps = len(selected)
        return selected
    step_idx = max(0, min(int(getattr(_args, "exec_step_index", 1)),
                          raw_chunk.shape[0] - 1))
    _last_exec_steps = 1
    return _cap_increments(raw_chunk[step_idx].unsqueeze(0).numpy())


def postprocess_actions(actions: np.ndarray, proprio_list: list) -> list:
    """z-denormalised physical offsets -> UAV-Flow-Eval episode-local poses."""
    out_scale = getattr(_args, "output_pos_scale", None)
    if out_scale is None:
        out_scale = float(getattr(_args, "pos_scale", 100.0))
    cx = float(proprio_list[0]) if len(proprio_list) > 0 else 0.0
    cy = float(proprio_list[1]) if len(proprio_list) > 1 else 0.0
    cz = float(proprio_list[2]) if len(proprio_list) > 2 else 0.0
    cyaw = float(np.radians(float(proprio_list[3]))) if len(proprio_list) > 3 else 0.0
    frame = getattr(_args, "output_frame", "delta_local")
    processed = []
    for a in actions:
        x = float(a[0]) * out_scale
        y = float(a[1]) * out_scale
        z = float(a[2]) * out_scale
        yaw = float(a[3])
        if frame == "delta_local":
            cos_y, sin_y = float(np.cos(cyaw)), float(np.sin(cyaw))
            ex = cx + x * cos_y - y * sin_y
            ey = cy + x * sin_y + y * cos_y
            ez = cz + z
            eyaw = cyaw + yaw
        else:
            ex, ey, ez, eyaw = x, y, z, yaw
        eyaw = (eyaw + np.pi) % (2.0 * np.pi) - np.pi
        processed.append([ex, ey, ez, float(eyaw)])
    return processed


# ── endpoints ────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    global _step
    t0 = time.time()
    data = request.get_json(force=True)
    try:
        pixels = preprocess_image(decode_image(data.get("image", "")))
        proprio_l = data.get("proprio", [0, 0, 0, 0])
        instr = data.get("instr", "")
        state8 = preprocess_proprio(proprio_l)
        input_ids = preprocess_instruction(instr)

        # ProprioEncoder is sized to proprio_dim (=4 for v4); slice state8. In
        # proprio-free-query mode forward_action ignores the token anyway.
        state_in = state8[:, :int(getattr(_args, "proprio_dim", 8))]
        use_bf16 = bool(getattr(_args, "bf16", False)) and _device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = _model.forward_action(pixels, input_ids, state_in)
        chunk_z = out["action"][0].float()                    # [K,4] z-space
        chunk_phys = _model.action_head.denormalize(chunk_z)  # [K,4] physical
        if _device.type == "cuda":
            torch.cuda.synchronize()

        raw_chunk = chunk_phys.detach().float().cpu()
        raw_actions = _select_exec_actions(raw_chunk)
        action_out = postprocess_actions(raw_actions, proprio_l)

        _step += 1
        ms = (time.time() - t0) * 1000
        logger.info("ep=%s step=%s %.1fms mode=%s raw[0]=%s -> act[0]=%s",
                    _episode, _step, ms, getattr(_args, "exec_mode", "chunk"),
                    [round(v, 4) for v in raw_actions[0].tolist()],
                    [round(v, 2) for v in action_out[0]])
        return jsonify({"action": action_out, "done": False})
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    global _prev_pose4, _prev_vel4, _last_exec_steps, _episode, _step
    _prev_pose4 = None
    _prev_vel4 = None
    _last_exec_steps = 1
    _episode += 1
    _step = 0
    logger.info("reset -> episode=%s", _episode)
    return jsonify({"status": "ok", "episode": _episode})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "aerov2_stage3",
                    "vision": getattr(_args, "vision_type", None),
                    "exec_mode": getattr(_args, "exec_mode", None),
                    "episode": _episode, "step": _step})


# ── startup ──────────────────────────────────────────────────────────────────
def load_v2_model(args) -> AeroV2:
    # Head geometry and target scaling come from the checkpoint, not the CLI.
    # A readout/n_bins mismatch fails loudly on load_state_dict, but a norm_mode
    # or chunk_offset mismatch does not — it just serves wrong physical units,
    # which in closed loop looks like a bad policy rather than a bad config.
    payload = torch.load(args.s3_ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("policy_cfg", {})
    logger.info("policy_cfg from checkpoint: %s", cfg)
    no_proprio = cfg.get("no_proprio", args.no_proprio)
    # _select_exec_actions reads this to decide whether index 0 is an anchor.
    args.chunk_offset = cfg.get("chunk_offset", 0)

    model = AeroV2(backbone_id=args.backbone, vision_type=args.vision_type)
    load_s2(model, Path(args.s2_ckpt_dir), args.s2_tag)
    model.enable_action_head(chunk_size=args.chunk_size, proprio_dim=args.proprio_dim,
                             use_proprio=not no_proprio,
                             use_grounding_target=args.use_grounding_target,
                             readout=cfg.get("readout", "last"),
                             n_bins=cfg.get("n_bins", 0),
                             bin_range=cfg.get("bin_range", 1.5),
                             readout_layers=cfg.get("readout_layers", 2))
    load_action_stats(model, args.action_stats, args.chunk_size, args.pos_scale,
                      cfg.get("norm_mode", "zscore"), cfg.get("chunk_offset", 0))
    meta = load_s3(model, payload)
    logger.info("loaded S2=%s/%s  S3=%s (step=%s)", args.s2_ckpt_dir, args.s2_tag,
                args.s3_ckpt, meta.get("step"))
    model.eval()
    return model


def get_args():
    p = argparse.ArgumentParser(description="AeroV2 Stage-3 inference server")
    p.add_argument("--s2_ckpt_dir", default="checkpoints/v2_stage2_full_cradio")
    p.add_argument("--s2_tag", default="best")
    p.add_argument("--s3_ckpt", default="checkpoints/v2_stage3_cradio/best.pth")
    p.add_argument("--vision_type", default="cradio_v3_b")
    p.add_argument("--backbone", default="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
    p.add_argument("--action_stats",
                   default="datasets/uav-flow/action_stats_k8.json")
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--proprio_dim", type=int, default=8)
    p.add_argument("--no_proprio", action="store_true",
                   help="proprio-free-query policy (v4): pose token ignored, "
                        "action-query readout used instead")
    p.add_argument("--use_grounding_target", action="store_true",
                   help="v5: grounding-target-token action head (homes on the "
                        "S2-localised visual target)")
    p.add_argument("--pos_scale", type=float, default=100.0)
    p.add_argument("--output_pos_scale", type=float, default=None)
    p.add_argument("--output_frame", default="delta_local",
                   choices=["delta_local", "episode_local"])
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--exec_mode", default="chunk",
                   choices=["single_step", "full_chunk", "chunk"])
    p.add_argument("--exec_horizon", type=int, default=4)
    p.add_argument("--exec_step_index", type=int, default=1)
    p.add_argument("--vel_per_step", type=int, default=1, choices=[0, 1])
    p.add_argument("--clip_sigma", type=float, default=3.0)
    p.add_argument("--max_step_cm", type=float, default=35.0)
    p.add_argument("--max_yaw_step_deg", type=float, default=12.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--port", type=int, default=5007)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    global _model, _device, _args
    _args = get_args()
    logging.basicConfig(
        level=getattr(logging, _args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(message)s")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", _device)
    if _device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    _model = load_v2_model(_args).to(_device)
    _model.eval()
    logger.info("Model ready on %s", _device)

    n_warmup = max(0, int(getattr(_args, "warmup", 0)))
    for i in range(n_warmup):
        try:
            t0 = time.time()
            pixels = preprocess_image(Image.new("RGB", (384, 384), (127, 127, 127)))
            input_ids = preprocess_instruction("warmup")
            state8 = torch.zeros(1, _args.proprio_dim, device=_device)
            use_bf16 = bool(getattr(_args, "bf16", False)) and _device.type == "cuda"
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    _model.forward_action(pixels, input_ids, state8)
            if _device.type == "cuda":
                torch.cuda.synchronize()
            logger.info("warmup %d/%d: %.0f ms", i + 1, n_warmup,
                        (time.time() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("warmup failed (non-fatal): %s", exc)
    _token_cache.pop("warmup", None)

    logger.info("Server on %s:%s | chunk=%s exec=%s horizon=%s clip_sigma=%s "
                "max_step_cm=%s max_yaw=%s bf16=%s", _args.host, _args.port,
                _args.chunk_size, _args.exec_mode, _args.exec_horizon,
                _args.clip_sigma, _args.max_step_cm, _args.max_yaw_step_deg,
                getattr(_args, "bf16", False))
    app.run(host=_args.host, port=_args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
