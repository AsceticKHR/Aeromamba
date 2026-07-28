#!/bin/bash
# AeroStream Round A — Stage 3 v3 "binding" retrain (change plan §A8).
#
# Adds on top of run_stage3_v2.sh:
#   * instruction binding head (--lambda_binding 0.3): motion class + yaw
#     sign + dz sign + magnitude bin CE, training-only bypass
#   * whole-class oversampling (--oversample_class_factor 3) for
#     Move/Shift/Ascend/Descend/Surround/Rotate (max with turn oversampling)
#   * channel weighting dz/dyaw x2.5 + magnitude sample weighting
#   * acceleration loss (--lambda_acc 0.25, AnoleVLA)
#
# Start point: Stage-2 checkpoint (NOT stage3_v2) — over/re-weighting changes
# the data/loss distribution, so a clean restart keeps attribution clean.
#
# Usage (on remote):
#   bash scripts/run_stage3_v3_binding.sh [STAGE2_CKPT]
set -u

STAGE2_CKPT=${1:-/root/autodl-tmp/Aeromamba/checkpoints/full_stage_20260710_125315/stage2_v2/best.pth}
DATA_ROOT=/root/autodl-tmp/datasets/uav-flow
STATS_JSON=$DATA_ROOT/action_stats_k8.json

RUN=stage3_v3_binding_$(date +%Y%m%d_%H%M%S)
CKPT_ROOT=/root/autodl-tmp/Aeromamba/checkpoints/$RUN
mkdir -p "$CKPT_ROOT"
echo "$RUN" > /root/autodl-tmp/Aeromamba/checkpoints/latest_stage3_v3_run.txt

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

if [ ! -f "$STATS_JSON" ]; then
  echo "=== ACTION STATS START $(date -Is) ==="
  python data/compute_action_stats.py \
    --data_root "$DATA_ROOT" \
    --chunk_size 8 --pos_scale 100.0 \
    --symmetrize_lateral \
    --output "$STATS_JSON" \
    > "$CKPT_ROOT/action_stats.log" 2>&1
  RC=$?
  echo "ACTION STATS rc=$RC"
  if [ $RC -ne 0 ] || [ ! -f "$STATS_JSON" ]; then
    echo "ABORT: action stats failed — see $CKPT_ROOT/action_stats.log"
    exit 1
  fi
fi

echo "=== STAGE3_V3_BINDING START $(date -Is) ==="
python training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root "$DATA_ROOT" \
  --stage2_ckpt "$STAGE2_CKPT" \
  --stage3_train_lora \
  --action_stats "$STATS_JSON" \
  --oversample_turn_factor 3 --oversample_turn_deg 10.0 \
  --oversample_class_factor 3 \
  --aug_flip --aug_vision \
  --lambda_smooth 0.0 --lambda_endpoint 0.25 --lambda_direction 0.5 \
  --lambda_acc 0.25 --lambda_binding 0.3 \
  --channel_weight_z 2.5 --channel_weight_yaw 2.5 --magnitude_sample_weight 1 \
  --batch 56 --epochs 2 --lr 8.75e-5 \
  --workers 8 --max_text_len 64 \
  --chunk_size 8 --pos_scale 100.0 \
  --max_val_steps 200 --log_every 100 --save_every_steps 5000 \
  --save_dir "$CKPT_ROOT" \
  > "$CKPT_ROOT/train.log" 2>&1
RC=$?
echo "STAGE3_V3_BINDING rc=$RC $(date -Is)"
echo "checkpoints: $CKPT_ROOT"
