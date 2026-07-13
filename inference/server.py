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
_prev_pose4:  torch.Tensor | None = None   # [1, 4] normalized pose for velocity
_prev_vel4:   torch.Tensor | None = None   # [1, 4] previous per-step velocity
_prev_state8: torch.Tensor | None = None   # [1, 8] pose + velocity, matches Stage-3 training
_last_exec_steps: int = 1                  # waypoints the client executed since last /predict
_token_cache: dict = {}                    # instruction → input_ids (CPU)
_diag_step: int = 0
_diag_episode: int = 0
_diag_fp = None  # optional JSONL handle


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


def _wrap_yaw_delta(delta_yaw: float) -> float:
    return float((delta_yaw + np.pi) % (2.0 * np.pi) - np.pi)


def preprocess_proprio(proprio_list: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert UAV-Flow client format to Stage-3 state8 / delta_state8 tensors.

    Matches UAVFlowHFDataset._state8_from_preprocessed and
    _state8_delta_from_preprocessed:
      state8       = [pose4 | velocity4]
      delta_state8 = [velocity4 | velocity4_t - velocity4_{t-1}]
    Positions are divided by pos_scale (cm → normalised metres).

    In chunk mode the client executes N waypoints between /predict calls, so
    the raw pose delta spans N control steps. With --vel_per_step the delta is
    divided by N to recover a per-frame velocity that matches the training
    distribution (this is the main anti-runaway fix).
    """
    global _prev_pose4, _prev_vel4, _prev_state8
    pos_scale = getattr(_args, "pos_scale", 100.0)
    dx  = float(proprio_list[0]) / pos_scale if len(proprio_list) > 0 else 0.0
    dy  = float(proprio_list[1]) / pos_scale if len(proprio_list) > 1 else 0.0
    dz  = float(proprio_list[2]) / pos_scale if len(proprio_list) > 2 else 0.0
    yaw = float(np.radians(proprio_list[3]))  if len(proprio_list) > 3 else 0.0
    pose4 = torch.tensor([[dx, dy, dz, yaw]], dtype=torch.float32, device=_device)

    if _prev_pose4 is not None:
        velocity4 = pose4 - _prev_pose4
        velocity4[0, 3] = _wrap_yaw_delta(float(velocity4[0, 3].item()))
        if bool(getattr(_args, "vel_per_step", True)):
            velocity4 = velocity4 / float(max(1, _last_exec_steps))
    else:
        velocity4 = torch.zeros_like(pose4)

    state8 = torch.cat([pose4, velocity4], dim=-1)
    # Training semantics: delta_state8[:4] = pose_t - pose_{t-1} = velocity_t,
    # delta_state8[4:] = velocity_t - velocity_{t-1}. Build directly from the
    # per-step velocity so chunk execution does not inflate the delta.
    if _prev_vel4 is not None:
        accel4 = velocity4 - _prev_vel4
        accel4[0, 3] = _wrap_yaw_delta(float(accel4[0, 3].item()))
        delta_state8 = torch.cat([velocity4, accel4], dim=-1)
    else:
        delta_state8 = torch.zeros_like(state8)

    _prev_pose4 = pose4.clone()
    _prev_vel4 = velocity4.clone()
    _prev_state8 = state8.clone()
    return state8, delta_state8


def _clip_chunk_sigma(raw_chunk: torch.Tensor) -> torch.Tensor:
    """
    Clamp each (k, dim) of the physical-unit chunk to mean ± clip_sigma·std of
    the training action distribution (buffers on the action head). Kills the
    runaway tail while leaving in-distribution predictions untouched.
    """
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
    lo = mean - clip_sigma * std
    hi = mean + clip_sigma * std
    return torch.max(torch.min(raw_chunk, hi), lo)


def _cap_increments(selected: np.ndarray) -> np.ndarray:
    """
    Rate-limit consecutive waypoint increments (smoothness guard).

    `selected` rows are cumulative offsets from the same anchor pose (metres /
    rad). Convert to per-step increments, cap the xyz norm at max_step_cm and
    |yaw| at max_yaw_step_deg, then re-accumulate.
    """
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
            inc[3] = float(np.clip(_wrap_yaw_delta(inc[3]), -max_yaw, max_yaw))
        acc = acc + inc
        acc[3] = _wrap_yaw_delta(acc[3])
        out[i] = acc
    return out.astype(selected.dtype)


def _select_exec_actions(raw_chunk: torch.Tensor) -> np.ndarray:
    """Pick model outputs according to exec_mode, with clipping + rate limit."""
    global _last_exec_steps
    exec_mode = getattr(_args, "exec_mode", "single_step")
    step_idx = int(getattr(_args, "exec_step_index", 1))
    step_idx = max(0, min(step_idx, raw_chunk.shape[0] - 1))

    raw_chunk = _clip_chunk_sigma(raw_chunk)

    if exec_mode == "chunk":
        # Return the first `exec_horizon` future waypoints. k=0 is the anchor
        # (always ~zero by construction), so start at k=1. All returned
        # waypoints are offsets from the SAME current pose (cumulative), and
        # the client executes them sequentially → 1 inference per N steps.
        horizon = max(1, int(getattr(_args, "exec_horizon", 4)))
        start = 1 if raw_chunk.shape[0] > 1 else 0
        end = min(start + horizon, raw_chunk.shape[0])
        selected = _cap_increments(raw_chunk[start:end].numpy())
        _last_exec_steps = len(selected)
        return selected

    if exec_mode == "full_chunk":
        selected = _cap_increments(raw_chunk.numpy())
        _last_exec_steps = len(selected)
        return selected

    _last_exec_steps = 1
    selected = raw_chunk[step_idx]
    if exec_mode == "ensemble":
        smooth = _ensemble.update(selected.unsqueeze(0))
        return _cap_increments(smooth.unsqueeze(0).detach().cpu().numpy())

    # single_step (default): one receding-horizon waypoint per predict call
    return _cap_increments(selected.unsqueeze(0).detach().cpu().numpy())


def preprocess_instruction(instr: str) -> torch.Tensor:
    """Tokenise instruction string → [1, L] int64 (cached per instruction)."""
    cached = _token_cache.get(instr)
    if cached is None:
        tokens = _model.tokenizer(
            instr,
            max_length=getattr(_args, "max_text_len", 64),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        cached = tokens["input_ids"]
        if len(_token_cache) > 512:
            _token_cache.clear()
        _token_cache[instr] = cached
    return cached.to(_device)


def postprocess_actions(
    actions: np.ndarray,
    proprio_list: list,
) -> list:
    """
    Convert model-normalised actions into UAV-Flow-Eval episode-local poses.

    Stage-3 training predicts normalised future offsets.  UAV-Flow-Eval,
    however, expects each returned action to be an episode-local pose:
        [relative_x, relative_y, relative_z, relative_yaw_rad]
    where the evaluator then rotates it by the episode initial yaw and writes
    the corresponding Unreal pose.
    """
    output_pos_scale = getattr(_args, "output_pos_scale", None)
    if output_pos_scale is None:
        output_pos_scale = getattr(_args, "pos_scale", 100.0)

    current_x = float(proprio_list[0]) if len(proprio_list) > 0 else 0.0
    current_y = float(proprio_list[1]) if len(proprio_list) > 1 else 0.0
    current_z = float(proprio_list[2]) if len(proprio_list) > 2 else 0.0
    current_yaw_deg = float(proprio_list[3]) if len(proprio_list) > 3 else 0.0
    current_yaw_rad = float(np.radians(current_yaw_deg))

    frame = getattr(_args, "output_frame", "delta_local")
    processed = []
    for action in actions:
        x = float(action[0]) * output_pos_scale
        y = float(action[1]) * output_pos_scale
        z = float(action[2]) * output_pos_scale
        yaw_rad = float(action[3])

        if frame == "delta_local":
            # NOTE: training chunks are cumulative offsets from the same anchor
            # pose (see UAVFlowDataset._extract_body_frame_chunk), so every
            # waypoint is applied to the SAME current pose — do not accumulate.
            cos_yaw = float(np.cos(current_yaw_rad))
            sin_yaw = float(np.sin(current_yaw_rad))
            episode_x = current_x + x * cos_yaw - y * sin_yaw
            episode_y = current_y + x * sin_yaw + y * cos_yaw
            episode_z = current_z + z
            episode_yaw = current_yaw_rad + yaw_rad
        else:
            episode_x = x
            episode_y = y
            episode_z = z
            episode_yaw = yaw_rad

        episode_yaw = (episode_yaw + np.pi) % (2.0 * np.pi) - np.pi
        processed.append([episode_x, episode_y, episode_z, float(episode_yaw)])

    return processed


def _output_pos_scale() -> float:
    scale = getattr(_args, "output_pos_scale", None)
    if scale is None:
        scale = getattr(_args, "pos_scale", 100.0)
    return float(scale)


def _build_step_diag(
    *,
    proprio_l: list,
    state: torch.Tensor,
    delta: torch.Tensor,
    raw_chunk: torch.Tensor,
    raw_actions: np.ndarray,
    action_out: list,
    instr: str,
    elapsed_ms: float,
) -> dict:
    """Structured per-/predict diagnostics for runaway / z-dive triage."""
    global _diag_step
    scale = _output_pos_scale()
    pose_m = state[0, :4].detach().float().cpu().tolist()
    vel_m = state[0, 4:8].detach().float().cpu().tolist()
    dstate = delta[0].detach().float().cpu().tolist()
    proprio_cm = [float(x) for x in (list(proprio_l) + [0, 0, 0, 0])[:4]]

    raw_full = raw_chunk.numpy()
    body_cm = (raw_actions.astype(np.float64) * np.array([scale, scale, scale, 1.0])).tolist()
    # Consecutive body-frame increments inside the returned chunk (cm / rad).
    body_inc = []
    if len(raw_actions) >= 1:
        prev = np.zeros(4, dtype=np.float64)
        for row in raw_actions.astype(np.float64):
            inc = row - prev
            body_inc.append(
                [float(inc[0] * scale), float(inc[1] * scale), float(inc[2] * scale), float(inc[3])]
            )
            prev = row

    xy_norms = [float(np.linalg.norm(a[:2])) for a in body_cm]
    z_vals = [float(a[2]) for a in body_cm]
    yaw_vals = [float(np.degrees(a[3])) for a in body_cm]
    ep_z = [float(a[2]) for a in action_out]
    pose_norm_m = float(np.linalg.norm(pose_m[:3]))
    vel_norm_m = float(np.linalg.norm(vel_m[:3]))

    flags = []
    if pose_norm_m > 5.0:
        flags.append("pose_ood")
    if vel_norm_m > 0.5:
        flags.append("vel_ood")
    if any(z < -20.0 for z in z_vals):
        flags.append("raw_z_dive")
    if any(n > 80.0 for n in xy_norms):
        flags.append("large_step")
    if abs(proprio_cm[2]) > 100.0:
        flags.append("cum_z_deep")

    return {
        "episode": _diag_episode,
        "step": _diag_step,
        "ms": round(elapsed_ms, 1),
        "exec_mode": getattr(_args, "exec_mode", "single_step"),
        "exec_horizon": int(getattr(_args, "exec_horizon", 4)),
        "instr": (instr or "")[:80],
        "proprio_cm": [round(v, 2) for v in proprio_cm],
        "pose_m": [round(v, 4) for v in pose_m],
        "vel_m": [round(v, 4) for v in vel_m],
        "delta_state8": [round(v, 4) for v in dstate],
        "pose_norm_m": round(pose_norm_m, 4),
        "vel_norm_m": round(vel_norm_m, 4),
        "raw_chunk_m": [[round(float(x), 4) for x in row] for row in raw_full.tolist()],
        "raw_selected_m": [[round(float(x), 4) for x in row] for row in raw_actions.tolist()],
        "body_offset_cm": [[round(float(x), 2) for x in row] for row in body_cm],
        "body_increment_cm": [[round(float(x), 2) for x in row] for row in body_inc],
        "xy_norm_cm": [round(v, 2) for v in xy_norms],
        "z_offset_cm": [round(v, 2) for v in z_vals],
        "yaw_offset_deg": [round(v, 2) for v in yaw_vals],
        "episode_pose": [[round(float(x), 2) for x in row] for row in action_out],
        "episode_z_cm": [round(v, 2) for v in ep_z],
        "flags": flags,
    }


def _emit_diag(diag: dict) -> None:
    every = max(1, int(getattr(_args, "diagnose_every", 1)))
    if diag["step"] % every != 0 and "vel_ood" not in diag["flags"] and "raw_z_dive" not in diag["flags"]:
        return
    logger.info(
        "DIAG ep=%s step=%s flags=%s proprio_cm=%s pose_norm=%.3fm vel_norm=%.3fm "
        "xy_cm=%s z_cm=%s yaw_deg=%s body_inc0=%s ep_z=%s",
        diag["episode"],
        diag["step"],
        diag["flags"] or ["ok"],
        diag["proprio_cm"],
        diag["pose_norm_m"],
        diag["vel_norm_m"],
        diag["xy_norm_cm"],
        diag["z_offset_cm"],
        diag["yaw_offset_deg"],
        diag["body_increment_cm"][0] if diag["body_increment_cm"] else None,
        diag["episode_z_cm"],
    )
    if _diag_fp is not None:
        import json

        _diag_fp.write(json.dumps(diag, ensure_ascii=False) + "\n")
        _diag_fp.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    global _diag_step
    t0   = time.time()
    data = request.get_json(force=True)

    try:
        b64_img   = data.get("image",   "")
        proprio_l = data.get("proprio", [0, 0, 0, 0])
        instr     = data.get("instr",   "")

        # Pre-process inputs
        pixels    = preprocess_image(decode_image(b64_img))
        state, delta = preprocess_proprio(proprio_l)
        input_ids = preprocess_instruction(instr)
        t_pre = time.time()

        # Model inference (predict_step denormalizes to physical m/rad units)
        use_bf16 = bool(getattr(_args, "bf16", False)) and _device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = _model.predict_step(
                    pixel_values=pixels,
                    input_ids=input_ids,
                    state=state,
                    delta_state=delta,
                )
        if _device.type == "cuda":
            torch.cuda.synchronize()
        t_model = time.time()

        # pred["action"]: [1, K, 4]
        chunk = pred["action"][0]         # [K, 4]  on device
        raw_chunk = chunk.detach().float().cpu()
        raw_actions = _select_exec_actions(raw_chunk)
        action_out = postprocess_actions(raw_actions, proprio_l)

        elapsed_ms = (time.time() - t0) * 1000
        pre_ms = (t_pre - t0) * 1000
        model_ms = (t_model - t_pre) * 1000
        post_ms = elapsed_ms - pre_ms - model_ms
        diag = None
        if bool(getattr(_args, "diagnose", False)):
            _diag_step += 1
            diag = _build_step_diag(
                proprio_l=proprio_l,
                state=state,
                delta=delta,
                raw_chunk=raw_chunk,
                raw_actions=raw_actions,
                action_out=action_out,
                instr=instr,
                elapsed_ms=elapsed_ms,
            )
            diag["pre_ms"] = round(pre_ms, 1)
            diag["model_ms"] = round(model_ms, 1)
            diag["post_ms"] = round(post_ms, 1)
            _emit_diag(diag)
        else:
            logger.info(
                f"Inference: {elapsed_ms:.1f} ms (pre={pre_ms:.0f} model={model_ms:.0f} post={post_ms:.0f}) "
                f"| mode={getattr(_args, 'exec_mode', 'single_step')} "
                f"raw[0]={raw_actions[0].tolist()} | action[0]={action_out[0]}"
            )

        payload = {"action": action_out, "done": False}
        if diag is not None and bool(getattr(_args, "diagnose_in_response", False)):
            payload["diag"] = diag
        return jsonify(payload)

    except Exception as exc:
        logger.exception(f"Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset episode state (ensemble buffer + state8 history)."""
    global _prev_pose4, _prev_vel4, _prev_state8, _last_exec_steps
    global _diag_step, _diag_episode
    _ensemble.reset()
    _prev_pose4 = None
    _prev_vel4 = None
    _prev_state8 = None
    _last_exec_steps = 1
    _diag_episode += 1
    _diag_step = 0
    logger.info(
        "Reset: ensemble buffer and state8 history cleared (diagnose episode=%s)",
        _diag_episode,
    )
    return jsonify({"status": "ok", "episode": _diag_episode})


@app.route("/health", methods=["GET"])
def health():
    model_name = getattr(_args, "mamba_type", "unknown") if _args else "unknown"
    return jsonify(
        {
            "status": "healthy",
            "model": model_name,
            "diagnose": bool(getattr(_args, "diagnose", False)) if _args else False,
            "exec_mode": getattr(_args, "exec_mode", None) if _args else None,
            "episode": _diag_episode,
            "step": _diag_step,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args) -> AeroMambaVLA:
    model = AeroMambaVLA(
        mamba_type=args.mamba_type,
        vision_type=args.vision_type,
        chunk_size=args.chunk_size,
        freeze_vision=True,
        token_resampler=args.token_resampler,
        num_visual_queries=args.num_visual_queries,
        resampler_layers=args.resampler_layers,
        resampler_heads=args.resampler_heads,
        action_head_type=args.action_head_type,
        action_bound=args.action_bound,
    )
    if args.use_lora:
        logger.info(
            f"Applying LoRA adapters before checkpoint load: "
            f"r={args.lora_r}, alpha={args.lora_alpha}"
        )
        model.configure_stage2(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    if args.ckpt and Path(args.ckpt).exists():
        logger.info(f"Loading checkpoint: {args.ckpt}")
        ckpt  = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
        if hasattr(model.action_head, "has_normalization"):
            if model.action_head.has_normalization():
                logger.info(
                    "  Action z-score stats loaded from checkpoint — "
                    "predict_step outputs physical units."
                )
            else:
                logger.warning(
                    "  Checkpoint has NO action normalization stats "
                    "(old-format checkpoint): outputs are raw head values."
                )
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
    p.add_argument("--token_resampler", default="none", choices=["none", "perceiver"])
    p.add_argument("--num_visual_queries", type=int, default=32)
    p.add_argument("--resampler_layers", type=int, default=2)
    p.add_argument("--resampler_heads", type=int, default=8)
    p.add_argument("--action_head_type", default="mlp", choices=["mlp", "dynamics"])
    p.add_argument("--action_bound",    type=float, default=1.0)
    p.add_argument("--use_lora",        action="store_true",
                   help="Build Mamba LoRA adapters before loading a LoRA checkpoint")
    p.add_argument("--lora_r",          type=int, default=16)
    p.add_argument("--lora_alpha",      type=int, default=32)
    p.add_argument("--port",            type=int,   default=5007)
    p.add_argument("--host",            default="0.0.0.0")
    p.add_argument("--max_text_len",    type=int,   default=64)
    p.add_argument("--pos_scale",       type=float, default=100.0,
                   help="Position scaling divisor (cm → normalised)")
    p.add_argument("--output_pos_scale", type=float, default=None,
                   help=(
                       "Multiplier for predicted xyz before returning to "
                       "UAV-Flow-Eval. Defaults to --pos_scale."
                   ))
    p.add_argument("--output_frame", default="delta_local",
                   choices=["delta_local", "episode_local"],
                   help=(
                       "delta_local: rotate/add predicted offsets to current "
                       "episode-local pose; episode_local: return scaled model "
                       "outputs directly."
                   ))
    p.add_argument("--ensemble_window", type=int,   default=5)
    p.add_argument("--ensemble_decay",  type=float, default=0.7)
    p.add_argument("--ensemble_mode",   action="store_true",
                   help="Deprecated alias for --exec_mode ensemble")
    p.add_argument("--exec_mode", default="chunk",
                   choices=["single_step", "full_chunk", "ensemble", "chunk"],
                   help=(
                       "chunk: return the first exec_horizon future waypoints "
                       "(client executes them sequentially, 1 inference per N steps); "
                       "single_step: return one receding-horizon waypoint per call; "
                       "full_chunk: return all K waypoints (legacy); "
                       "ensemble: temporally smooth the selected waypoint."
                   ))
    p.add_argument("--exec_horizon", type=int, default=4,
                   help="Number of future waypoints returned in 'chunk' mode.")
    p.add_argument("--exec_step_index", type=int, default=1,
                   help="Chunk index to execute (0=anchor, 1=first future step).")
    p.add_argument("--bf16", action="store_true",
                   help="Run inference under bfloat16 autocast (CUDA only).")
    p.add_argument("--vel_per_step", type=int, default=1, choices=[0, 1],
                   help=(
                       "Divide the inter-/predict pose delta by the number of "
                       "waypoints the client executed, recovering a per-frame "
                       "velocity that matches training (anti-runaway)."
                   ))
    p.add_argument("--clip_sigma", type=float, default=3.0,
                   help=(
                       "Clamp physical chunk outputs to mean ± sigma·std of the "
                       "training action stats (0 disables)."
                   ))
    p.add_argument("--max_step_cm", type=float, default=35.0,
                   help="Per-waypoint xyz increment cap in cm (0 disables).")
    p.add_argument("--max_yaw_step_deg", type=float, default=12.0,
                   help="Per-waypoint |yaw| increment cap in degrees (0 disables).")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup forward passes at startup (kernel autotune).")
    p.add_argument("--diagnose", action="store_true",
                   help="Per-/predict step diagnostics (proprio/vel/raw/z flags + JSONL).")
    p.add_argument("--diagnose_log", default="",
                   help="JSONL path for diagnostics (default: checkpoints/infer_diagnose.jsonl).")
    p.add_argument("--diagnose_every", type=int, default=1,
                   help="Log every N predict calls (flagged steps always logged).")
    p.add_argument("--diagnose_in_response", action="store_true",
                   help="Include diag dict in /predict JSON (heavier responses).")
    p.add_argument("--dummy",           action="store_true",
                   help="Tiny model for quick testing")
    p.add_argument("--log_level",       default="INFO")
    return p.parse_args()


def main():
    global _model, _ensemble, _device, _args, _diag_fp

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

    if getattr(_args, "ensemble_mode", False):
        _args.exec_mode = "ensemble"

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {_device}")

    if _device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    logger.info("Loading model…")
    _model = load_model(_args).to(_device)
    logger.info(f"Model ready on {_device}")

    _ensemble = TemporalEnsemble(
        window=_args.ensemble_window,
        decay=_args.ensemble_decay,
    )

    # Warmup: run the full predict pipeline so cuDNN autotune / lazy init
    # costs are not paid on the first real request.
    n_warmup = max(0, int(getattr(_args, "warmup", 0)))
    if n_warmup > 0:
        try:
            dummy_img = Image.new("RGB", (384, 384), (127, 127, 127))
            pixels = preprocess_image(dummy_img)
            input_ids = preprocess_instruction("warmup")
            state = torch.zeros(1, 8, device=_device)
            delta = torch.zeros(1, 8, device=_device)
            use_bf16 = bool(getattr(_args, "bf16", False)) and _device.type == "cuda"
            for i in range(n_warmup):
                t0 = time.time()
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        _model.predict_step(
                            pixel_values=pixels,
                            input_ids=input_ids,
                            state=state,
                            delta_state=delta,
                        )
                if _device.type == "cuda":
                    torch.cuda.synchronize()
                logger.info(f"Warmup {i + 1}/{n_warmup}: {(time.time() - t0) * 1000:.0f} ms")
            _token_cache.pop("warmup", None)
        except Exception as exc:
            logger.warning(f"Warmup failed (non-fatal): {exc}")

    if bool(getattr(_args, "diagnose", False)):
        log_path = Path(_args.diagnose_log) if _args.diagnose_log else Path("checkpoints/infer_diagnose.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _diag_fp = open(log_path, "a", encoding="utf-8")
        logger.info(
            "Diagnose ON → %s (every=%s, in_response=%s)",
            log_path,
            _args.diagnose_every,
            getattr(_args, "diagnose_in_response", False),
        )

    logger.info(
        f"Server starting on {_args.host}:{_args.port} | "
        f"chunk_size={_args.chunk_size} | exec_mode={_args.exec_mode} | "
        f"exec_horizon={getattr(_args, 'exec_horizon', 4)} | "
        f"exec_step_index={_args.exec_step_index} | bf16={getattr(_args, 'bf16', False)} | "
        f"vel_per_step={getattr(_args, 'vel_per_step', 1)} | "
        f"clip_sigma={getattr(_args, 'clip_sigma', 0)} | "
        f"max_step_cm={getattr(_args, 'max_step_cm', 0)} | "
        f"max_yaw_step_deg={getattr(_args, 'max_yaw_step_deg', 0)} | "
        f"diagnose={getattr(_args, 'diagnose', False)}"
    )
    app.run(host=_args.host, port=_args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
