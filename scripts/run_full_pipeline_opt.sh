#!/bin/bash
# AeroMamba-Opt full-stage training pipeline: S1 -> S2 -> S3a -> S3b
set -u

RUN=full_stage_$(date +%Y%m%d_%H%M%S)
CKPT_ROOT=/root/autodl-tmp/Aeromamba/checkpoints/$RUN
LOG=$CKPT_ROOT/pipeline.log
mkdir -p "$CKPT_ROOT"

exec > "$LOG" 2>&1
echo "=== PIPELINE $RUN START $(date -Is) ==="
echo "$RUN" > /root/autodl-tmp/Aeromamba/checkpoints/latest_full_stage_run.txt

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

cd /root/autodl-tmp/Aeromamba

# ── Stage 1: projector + resampler alignment ────────────────────────────
echo "=== STAGE1 START $(date -Is) ==="
python training/stage1_align.py \
  --arch_preset aeromamba_opt \
  --data_root /root/autodl-tmp/Aeromamba/data/llava_pretrain \
  --json_name blip_laion_cc_sbu_558k.json \
  --batch 24 --epochs 1 --lr 1e-4 \
  --workers 16 --max_text_len 64 \
  --log_every 100 --save_every_steps 2000 \
  --save_dir "$CKPT_ROOT/stage1" \
  > "$CKPT_ROOT/stage1.log" 2>&1
S1_RC=$?
echo "STAGE1 rc=$S1_RC $(date -Is)"
if [ $S1_RC -ne 0 ] || [ ! -f "$CKPT_ROOT/stage1/best.pth" ]; then
  echo "PIPELINE ABORT: stage1 failed"
  exit 1
fi

# ── Stage 2: VLM SFT with LoRA ──────────────────────────────────────────
echo "=== STAGE2 START $(date -Is) ==="
python training/stage2_vlm.py \
  --arch_preset aeromamba_opt \
  --data_root /root/autodl-tmp/Aeromamba/data \
  --json_name stage2_mixed_data.json \
  --stage1_ckpt "$CKPT_ROOT/stage1/best.pth" \
  --lora_r 16 --lora_alpha 32 \
  --batch 12 --epochs 2 --lr 5e-5 \
  --workers 16 --max_text_len 128 \
  --log_every 100 --save_every_steps 2000 \
  --save_dir "$CKPT_ROOT/stage2" \
  > "$CKPT_ROOT/stage2.log" 2>&1
S2_RC=$?
echo "STAGE2 rc=$S2_RC $(date -Is)"
if [ $S2_RC -ne 0 ] || [ ! -f "$CKPT_ROOT/stage2/best.pth" ]; then
  echo "PIPELINE ABORT: stage2 failed"
  exit 1
fi

# ── Stage 3a: action head, LoRA frozen, no acc loss ─────────────────────
echo "=== STAGE3A START $(date -Is) ==="
python training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root /root/autodl-tmp/datasets/uav-flow \
  --stage2_ckpt "$CKPT_ROOT/stage2/best.pth" \
  --batch 32 --epochs 2 --lr 5e-5 \
  --workers 16 --max_text_len 64 \
  --chunk_size 8 --pos_scale 100.0 \
  --lambda_smooth 0.0 --lambda_endpoint 2.0 --lambda_direction 0.0 --lambda_acc 0.0 \
  --aug_flip \
  --log_every 100 --save_every_steps 2000 \
  --save_dir "$CKPT_ROOT/stage3a" \
  > "$CKPT_ROOT/stage3a.log" 2>&1
S3A_RC=$?
echo "STAGE3A rc=$S3A_RC $(date -Is)"
if [ $S3A_RC -ne 0 ] || [ ! -f "$CKPT_ROOT/stage3a/best.pth" ]; then
  echo "PIPELINE ABORT: stage3a failed"
  exit 1
fi

# ── Stage 3b: unfreeze LoRA + acc regularization ────────────────────────
echo "=== STAGE3B START $(date -Is) ==="
python training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root /root/autodl-tmp/datasets/uav-flow \
  --stage2_ckpt "$CKPT_ROOT/stage2/best.pth" \
  --resume "$CKPT_ROOT/stage3a/best.pth" \
  --resume_model_only \
  --stage3_train_lora \
  --batch 32 --epochs 1 --lr 2e-5 \
  --workers 16 --max_text_len 64 \
  --chunk_size 8 --pos_scale 100.0 \
  --lambda_smooth 0.0 --lambda_endpoint 2.0 --lambda_direction 0.0 --lambda_acc 0.5 \
  --aug_flip \
  --log_every 100 --save_every_steps 2000 \
  --save_dir "$CKPT_ROOT/stage3b" \
  > "$CKPT_ROOT/stage3b.log" 2>&1
S3B_RC=$?
echo "STAGE3B rc=$S3B_RC $(date -Is)"

echo "=== PIPELINE $RUN DONE $(date -Is) ==="
