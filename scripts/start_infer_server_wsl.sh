#!/bin/bash
# Launch AeroMamba-Opt inference server inside local WSL (Ubuntu-20.04).
#
# Uses the slim checkpoint (trainable weights only); base Mamba2/SigLIP2
# weights are pulled from HuggingFace on first start (~1.2 GB, cached in
# ~/.cache/huggingface afterwards).
#
# Usage (from Windows):
#   wsl -- bash "/mnt/c/Users/user/学习/UAV source code/Aeromamba/scripts/start_infer_server_wsl.sh"
set -euo pipefail

REPO="${AEROMAMBA_REPO:-/mnt/c/Users/user/学习/UAV source code/Aeromamba}"
PY="${AEROMAMBA_PY:-/home/khr/miniconda3/envs/aeromamba/bin/python}"
CKPT="${CKPT:-$REPO/checkpoints/stage3_v2/best_slim.pth}"
PORT="${PORT:-5007}"

EXEC_MODE="${EXEC_MODE:-chunk}"
EXEC_HORIZON="${EXEC_HORIZON:-4}"
VEL_PER_STEP="${VEL_PER_STEP:-1}"
CLIP_SIGMA="${CLIP_SIGMA:-3.0}"
MAX_STEP_CM="${MAX_STEP_CM:-35}"
MAX_YAW_STEP_DEG="${MAX_YAW_STEP_DEG:-12}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# All base models are already cached after the first successful launch;
# skip hub revalidation to avoid slow/flaky mirror handshakes on startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export AEROMAMBA_TOKENIZER=EleutherAI/gpt-neox-20b
export PYTHONUNBUFFERED=1

cd "$REPO"

LOG_FILE="${LOG_FILE:-/home/khr/aeromamba_infer_wsl.log}"
PID_FILE=/home/khr/aeromamba_infer_wsl.pid

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")" || true
  sleep 2
fi

nohup "$PY" inference/server.py \
  --ckpt "$CKPT" \
  --mamba_type mamba-2-370m \
  --vision_type siglip2_base_384 \
  --token_resampler perceiver \
  --num_visual_queries 64 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --chunk_size 8 \
  --action_head_type mlp \
  --use_lora --lora_r 16 --lora_alpha 32 \
  --pos_scale 100.0 --output_pos_scale 100.0 \
  --output_frame delta_local \
  --exec_mode "$EXEC_MODE" --exec_horizon "$EXEC_HORIZON" \
  --vel_per_step "$VEL_PER_STEP" \
  --clip_sigma "$CLIP_SIGMA" \
  --max_step_cm "$MAX_STEP_CM" \
  --max_yaw_step_deg "$MAX_YAW_STEP_DEG" \
  --warmup 2 \
  --bf16 \
  --port "$PORT" --host 0.0.0.0 --max_text_len 64 \
  >> "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") log=$LOG_FILE port=$PORT"
