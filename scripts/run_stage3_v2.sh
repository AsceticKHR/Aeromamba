#!/bin/bash
# AeroMamba-Opt Stage 3 v2 — single merged stage from Stage-2 checkpoint.
#
# Changes vs the old S3a/S3b split:
#   * per-(k,dim) z-score action normalisation (loss in z-space, pure L1)
#   * endpoint (0.25) + direction-cosine (0.5) auxiliary terms
#   * turn-heavy chunk oversampling (3x for >10 deg windows; base turn
#     fraction is 17.1%, so 3x lifts the effective fraction to ~38%)
#   * appearance domain augmentation (color jitter / grayscale / blur)
#   * LoRA trainable from the start (merged 3a+3b), 2 epochs, cosine lr 7.5e-5
#     (linearly scaled for batch=48; one epoch ≈ 40k steps ≈ 5-6h)
#
# Usage (on remote):
#   bash scripts/run_stage3_v2.sh [STAGE2_CKPT]
set -u

STAGE2_CKPT=${1:-/root/autodl-tmp/Aeromamba/checkpoints/full_stage_20260710_125315/stage2/best.pth}
DATA_ROOT=/root/autodl-tmp/datasets/uav-flow
STATS_JSON=$DATA_ROOT/action_stats_k8.json

RUN=stage3_v2_$(date +%Y%m%d_%H%M%S)
CKPT_ROOT=/root/autodl-tmp/Aeromamba/checkpoints/$RUN
mkdir -p "$CKPT_ROOT"
echo "$RUN" > /root/autodl-tmp/Aeromamba/checkpoints/latest_stage3_v2_run.txt

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

cd /root/autodl-tmp/Aeromamba

# ── Step 0: action statistics (skip if already computed) ─────────────────
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

# ── Stage 3 v2: merged action training ───────────────────────────────────
echo "=== STAGE3_V2 START $(date -Is) ==="
python training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root "$DATA_ROOT" \
  --stage2_ckpt "$STAGE2_CKPT" \
  --stage3_train_lora \
  --action_stats "$STATS_JSON" \
  --oversample_turn_factor 3 --oversample_turn_deg 10.0 \
  --aug_flip --aug_vision \
  --batch 48 --epochs 2 --lr 7.5e-5 \
  --workers 32 --max_text_len 64 \
  --chunk_size 8 --pos_scale 100.0 \
  --lambda_smooth 0.0 --lambda_endpoint 0.25 --lambda_direction 0.5 --lambda_acc 0.0 \
  --max_val_steps 200 \
  --log_every 100 --save_every_steps 2000 \
  --save_dir "$CKPT_ROOT" \
  > "$CKPT_ROOT/train.log" 2>&1
RC=$?
echo "STAGE3_V2 rc=$RC $(date -Is)"
echo "checkpoints: $CKPT_ROOT"
