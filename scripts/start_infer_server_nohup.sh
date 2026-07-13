#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/Aeromamba
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/torch_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export AEROMAMBA_TOKENIZER=EleutherAI/gpt-neox-20b
export PYTHONUNBUFFERED=1

PID_FILE=checkpoints/aeromamba_opt_infer_server.pid
LOG_FILE=checkpoints/aeromamba_opt_infer_server.log
CKPT=checkpoints/stage3_v2_20260712_005721/best.pth
EXEC_MODE="${EXEC_MODE:-chunk}"
EXEC_HORIZON="${EXEC_HORIZON:-4}"
DIAGNOSE="${DIAGNOSE:-0}"
DIAGNOSE_LOG="${DIAGNOSE_LOG:-checkpoints/infer_diagnose.jsonl}"
VEL_PER_STEP="${VEL_PER_STEP:-1}"
CLIP_SIGMA="${CLIP_SIGMA:-3.0}"
MAX_STEP_CM="${MAX_STEP_CM:-35}"
MAX_YAW_STEP_DEG="${MAX_YAW_STEP_DEG:-12}"

# Kill old server by PID file
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    kill "${OLD_PID}" || true
    sleep 2
  fi
fi
# Also kill any leftover server.py on 5007
for p in $(ps -eo pid,cmd | awk '/inference\/server\.py/ && !/awk/ {print $1}'); do
  kill "$p" 2>/dev/null || true
done
sleep 2

# Truncate diagnose JSONL when starting a fresh diagnose run
if [ "$DIAGNOSE" = "1" ]; then
  : > "$DIAGNOSE_LOG"
fi

EXTRA_ARGS=()
if [ "$DIAGNOSE" = "1" ]; then
  EXTRA_ARGS+=(--diagnose --diagnose_log "$DIAGNOSE_LOG" --diagnose_every 1)
fi

nohup python inference/server.py \
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
  --port 5007 --host 0.0.0.0 --max_text_len 64 \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "started pid=$(cat $PID_FILE) ckpt=$CKPT exec_mode=$EXEC_MODE diagnose=$DIAGNOSE"
sleep 3
ps -p "$(cat $PID_FILE)" -o pid,stat,etime,cmd || echo "server process missing"
