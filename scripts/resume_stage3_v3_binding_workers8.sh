#!/bin/bash
# Resume AeroStream Round A binding run with fewer DataLoader workers
# to reduce disk thrashing (GPU was starved at workers=16).
#
# Usage:
#   bash scripts/resume_stage3_v3_binding_workers8.sh \
#     [/path/to/latest.pth] [/path/to/save_dir]
set -u

RESUME_CKPT=${1:-/root/autodl-tmp/Aeromamba/checkpoints/stage3_v3_binding_20260716_024505/latest.pth}
CKPT_ROOT=${2:-/root/autodl-tmp/Aeromamba/checkpoints/stage3_v3_binding_20260716_024505}
STAGE2_CKPT=/root/autodl-tmp/Aeromamba/checkpoints/full_stage_20260710_125315/stage2_v2/best.pth
DATA_ROOT=/root/autodl-tmp/datasets/uav-flow
STATS_JSON=$DATA_ROOT/action_stats_k8.json

# Trainer resume starts at ckpt_epoch+1. Mid-Ep2 latest has epoch=2, so
# --epochs 3 enters Ep3. Prefer --max_steps to only makeup remaining Ep2
# length (48185-22000≈26185), not a full extra epoch.
EPOCHS=${EPOCHS:-3}
# Default: remaining Ep2 steps from latest@~22000, minus ~900 already run
# in the aborted full-Ep3 attempt. Override with MAX_STEPS=... if needed.
MAX_STEPS=${MAX_STEPS:-25285}

mkdir -p "$CKPT_ROOT"
echo "stage3_v3_binding_20260716_024505" > /root/autodl-tmp/Aeromamba/checkpoints/latest_stage3_v3_run.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/torch_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export AEROMAMBA_TOKENIZER=EleutherAI/gpt-neox-20b
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/Aeromamba

if [ ! -f "$RESUME_CKPT" ]; then
  echo "ABORT: resume ckpt missing: $RESUME_CKPT"
  exit 1
fi

LOG="$CKPT_ROOT/train_resume_w8_$(date +%Y%m%d_%H%M%S).log"
echo "=== STAGE3_V3_BINDING RESUME workers=8 START $(date -Is) ===" | tee -a "$LOG"
echo "resume=$RESUME_CKPT epochs=$EPOCHS max_steps=$MAX_STEPS save_dir=$CKPT_ROOT" | tee -a "$LOG"

python training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root "$DATA_ROOT" \
  --stage2_ckpt "$STAGE2_CKPT" \
  --resume "$RESUME_CKPT" \
  --stage3_train_lora \
  --action_stats "$STATS_JSON" \
  --oversample_turn_factor 3 --oversample_turn_deg 10.0 \
  --oversample_class_factor 3 \
  --aug_flip --aug_vision \
  --lambda_smooth 0.0 --lambda_endpoint 0.25 --lambda_direction 0.5 \
  --lambda_acc 0.25 --lambda_binding 0.3 \
  --channel_weight_z 2.5 --channel_weight_yaw 2.5 --magnitude_sample_weight 1 \
  --batch 56 --epochs "$EPOCHS" --lr 8.75e-5 \
  --workers 8 --max_text_len 64 \
  --chunk_size 8 --pos_scale 100.0 \
  --max_steps "$MAX_STEPS" \
  --max_val_steps 200 --log_every 100 --save_every_steps 5000 \
  --save_dir "$CKPT_ROOT" \
  >> "$LOG" 2>&1
RC=$?
echo "STAGE3_V3_BINDING RESUME rc=$RC $(date -Is)" | tee -a "$LOG"
echo "log: $LOG"
