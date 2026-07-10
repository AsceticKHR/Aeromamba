#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/Aeromamba}"
PYTHON="${PYTHON:-/root/miniconda3/envs/mamba2/bin/python}"
RUN_NAME="${RUN_NAME:-uavflow_mamba2_siglip2_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_DIR}/checkpoints/${RUN_NAME}}"

STAGE1_ROOT="${STAGE1_ROOT:-${REPO_DIR}/data/llava_pretrain}"
STAGE1_JSON="${STAGE1_JSON:-blip_laion_cc_sbu_558k.json}"
STAGE2_ROOT="${STAGE2_ROOT:-${REPO_DIR}/data}"
STAGE2_JSON="${STAGE2_JSON:-stage2_mixed_data.json}"
STAGE3_ROOT="${STAGE3_ROOT:-/root/autodl-tmp/datasets/uav-flow}"

MAMBA_TYPE="${MAMBA_TYPE:-mamba-2-370m}"
VISION_TYPE="${VISION_TYPE:-siglip2_base_384}"
TOKENIZER="${AEROMAMBA_TOKENIZER:-EleutherAI/gpt-neox-20b}"

export AEROMAMBA_TOKENIZER="${TOKENIZER}"
export STAGE1_ROOT STAGE1_JSON STAGE2_ROOT STAGE2_JSON STAGE3_ROOT
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "${SAVE_ROOT}"
cd "${REPO_DIR}"

echo "[pipeline] run=${RUN_NAME}"
echo "[pipeline] save_root=${SAVE_ROOT}"
echo "[pipeline] tokenizer=${AEROMAMBA_TOKENIZER}"
nvidia-smi || true

echo "[verify] checking datasets"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

checks = [
    ("stage1", Path(os.environ["STAGE1_ROOT"]), os.environ["STAGE1_JSON"]),
    ("stage2", Path(os.environ["STAGE2_ROOT"]), os.environ["STAGE2_JSON"]),
]
for name, root, json_name in checks:
    path = root / json_name
    assert path.exists(), f"missing {name} json: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data, f"empty {name} json: {path}"
    found = sum((root / item.get("image", "")).exists() for item in data[:1000])
    print(f"{name}: samples={len(data):,}, first1000_images={found}/1000")
    assert found > 900, f"{name} image check failed"

stage3 = Path(os.environ["STAGE3_ROOT"])
summary = stage3 / "metadata" / "summary.json"
assert summary.exists(), f"missing stage3 summary: {summary}"
print("stage3:", summary.read_text(encoding="utf-8"))
PY

echo "[stage1] projector/resampler alignment"
"${PYTHON}" training/stage1_align.py \
  --data_root "${STAGE1_ROOT}" \
  --json_name "${STAGE1_JSON}" \
  --mamba_type "${MAMBA_TYPE}" \
  --vision_type "${VISION_TYPE}" \
  --token_resampler perceiver \
  --num_visual_queries 64 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --batch "${STAGE1_BATCH:-12}" \
  --workers "${STAGE1_WORKERS:-16}" \
  --epochs "${STAGE1_EPOCHS:-1}" \
  --lr "${STAGE1_LR:-1e-4}" \
  --max_text_len "${STAGE1_MAX_TEXT_LEN:-64}" \
  --save_dir "${SAVE_ROOT}/stage1" \
  --log_every 50 \
  --max_val_steps "${STAGE1_MAX_VAL_STEPS:-200}" \
  2>&1 | tee "${SAVE_ROOT}/stage1.log"

echo "[stage2] VLM SFT with LoRA"
"${PYTHON}" training/stage2_vlm.py \
  --data_root "${STAGE2_ROOT}" \
  --json_name "${STAGE2_JSON}" \
  --stage1_ckpt "${SAVE_ROOT}/stage1/best.pth" \
  --mamba_type "${MAMBA_TYPE}" \
  --vision_type "${VISION_TYPE}" \
  --token_resampler perceiver \
  --num_visual_queries 64 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --batch "${STAGE2_BATCH:-8}" \
  --workers "${STAGE2_WORKERS:-16}" \
  --epochs "${STAGE2_EPOCHS:-2}" \
  --lr "${STAGE2_LR:-5e-5}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --max_text_len "${STAGE2_MAX_TEXT_LEN:-128}" \
  --save_dir "${SAVE_ROOT}/stage2" \
  --log_every 50 \
  --max_val_steps "${STAGE2_MAX_VAL_STEPS:-200}" \
  2>&1 | tee "${SAVE_ROOT}/stage2.log"

echo "[stage3] UAV-Flow action training"
"${PYTHON}" training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root "${STAGE3_ROOT}" \
  --stage2_ckpt "${SAVE_ROOT}/stage2/best.pth" \
  --mamba_type "${MAMBA_TYPE}" \
  --batch "${STAGE3_BATCH:-24}" \
  --workers "${STAGE3_WORKERS:-24}" \
  --epochs "${STAGE3_EPOCHS:-2}" \
  --lr "${STAGE3_LR:-5e-5}" \
  --chunk_size "${CHUNK_SIZE:-8}" \
  --pos_scale "${POS_SCALE:-100.0}" \
  --lambda_smooth "${LAMBDA_SMOOTH:-0.0}" \
  --lambda_endpoint "${LAMBDA_ENDPOINT:-2.0}" \
  --lambda_direction "${LAMBDA_DIRECTION:-0.0}" \
  --lambda_acc "${LAMBDA_ACC:-0.0}" \
  --max_text_len "${STAGE3_MAX_TEXT_LEN:-64}" \
  --save_dir "${SAVE_ROOT}/stage3" \
  --save_every_steps "${STAGE3_SAVE_EVERY_STEPS:-10000}" \
  --max_val_steps "${STAGE3_MAX_VAL_STEPS:-300}" \
  --no_amp \
  --log_every 50 \
  2>&1 | tee "${SAVE_ROOT}/stage3.log"

if [[ "${RUN_STAGE3B:-0}" == "1" ]]; then
  echo "[stage3b] UAV-Flow action refinement with LoRA + acceleration loss"
  "${PYTHON}" training/stage3_action.py \
    --arch_preset aeromamba_opt \
    --data_root "${STAGE3_ROOT}" \
    --stage2_ckpt "${SAVE_ROOT}/stage3/best.pth" \
    --mamba_type "${MAMBA_TYPE}" \
    --batch "${STAGE3B_BATCH:-${STAGE3_BATCH:-24}}" \
    --workers "${STAGE3B_WORKERS:-${STAGE3_WORKERS:-24}}" \
    --epochs "${STAGE3B_EPOCHS:-1}" \
    --lr "${STAGE3B_LR:-2e-5}" \
    --chunk_size "${CHUNK_SIZE:-8}" \
    --pos_scale "${POS_SCALE:-100.0}" \
    --lambda_smooth "${LAMBDA_SMOOTH:-0.0}" \
    --lambda_endpoint "${LAMBDA_ENDPOINT:-2.0}" \
    --lambda_direction "${LAMBDA_DIRECTION:-0.0}" \
    --lambda_acc "${LAMBDA_ACC:-0.5}" \
    --max_text_len "${STAGE3_MAX_TEXT_LEN:-64}" \
    --save_dir "${SAVE_ROOT}/stage3b" \
    --save_every_steps "${STAGE3_SAVE_EVERY_STEPS:-10000}" \
    --max_val_steps "${STAGE3_MAX_VAL_STEPS:-300}" \
    --stage3_train_lora \
    --no_amp \
    --log_every 50 \
    2>&1 | tee "${SAVE_ROOT}/stage3b.log"
fi

echo "[pipeline] complete: ${SAVE_ROOT}"
