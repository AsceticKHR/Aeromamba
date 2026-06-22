#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/Aeromamba}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE1_BATCH="${STAGE1_BATCH:-16}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_MAX_TEXT_LEN="${STAGE1_MAX_TEXT_LEN:-32}"
STAGE1_WORKERS="${STAGE1_WORKERS:-4}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_BATCH="${STAGE2_BATCH:-16}"
STAGE2_LR="${STAGE2_LR:-2e-4}"
STAGE2_MAX_TEXT_LEN="${STAGE2_MAX_TEXT_LEN:-128}"
STAGE2_WORKERS="${STAGE2_WORKERS:-4}"

PRETRAIN_DIR="${PRETRAIN_DIR:-${DATA_ROOT}/llava_pretrain}"
STAGE2_ROOT="${STAGE2_ROOT:-${DATA_ROOT}}"
STAGE2_JSON="${STAGE2_JSON:-stage2_mixed_data.json}"
COCO_DIR="${COCO_DIR:-${DATA_ROOT}/coco}"
COCO_PUBLIC_ZIP="${COCO_PUBLIC_ZIP:-/autodl-pub/data/COCO2017/train2017.zip}"

mkdir -p "${PRETRAIN_DIR}" "${COCO_DIR}" "${REPO_DIR}/checkpoints"
cd "${REPO_DIR}"

export HF_ENDPOINT
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

run_python() {
  if [[ -x /opt/conda/bin/python ]]; then
    /opt/conda/bin/python "$@"
  else
    python "$@"
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo "[data] exists: ${out}"
    return
  fi
  echo "[data] downloading: ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 1M --continue=true -o "$(basename "${out}")" -d "$(dirname "${out}")" "${url}"
  else
    wget -c -O "${out}" "${url}"
  fi
}

extract_zip_if_needed() {
  local zip_path="$1"
  local marker="$2"
  local dest="$3"
  if [[ -e "${marker}" ]]; then
    echo "[data] extracted marker exists: ${marker}"
    return
  fi
  echo "[data] extracting: ${zip_path}"
  unzip -q "${zip_path}" -d "${dest}"
}

prepare_stage1_data() {
  echo "[stage1-data] preparing LLaVA-Pretrain"
  download_file \
    "${HF_ENDPOINT}/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json" \
    "${PRETRAIN_DIR}/blip_laion_cc_sbu_558k.json"
  if [[ ! -e "${PRETRAIN_DIR}/00453/004539375.jpg" ]]; then
    download_file \
      "${HF_ENDPOINT}/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip" \
      "${PRETRAIN_DIR}/images.zip"
    extract_zip_if_needed "${PRETRAIN_DIR}/images.zip" "${PRETRAIN_DIR}/00453/004539375.jpg" "${PRETRAIN_DIR}"
    if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
      rm -f "${PRETRAIN_DIR}/images.zip"
    fi
  else
    echo "[stage1-data] LLaVA-Pretrain images already extracted, skipping download and extraction."
  fi
}

prepare_stage2_data() {
  echo "[stage2-data] preparing AeroMamba stage2 data + COCO train2017"
  if [[ ! -s "${STAGE2_ROOT}/${STAGE2_JSON}" ]]; then
    download_file \
      "${HF_ENDPOINT}/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json" \
      "${STAGE2_ROOT}/llava_instruct_150k.json"
    echo "[stage2-data] ERROR: ${STAGE2_ROOT}/${STAGE2_JSON} is project-generated and cannot be replaced by llava_instruct_150k.json."
    echo "[stage2-data] Copy stage2_mixed_data.json plus its COCO and Open3D-VQA images from the trusted AeroMamba dataset backup, then rerun."
    exit 1
  fi

  if [[ ! -d "${COCO_DIR}/train2017" ]]; then
    if [[ -s "${COCO_PUBLIC_ZIP}" ]]; then
      echo "[stage2-data] using public COCO zip: ${COCO_PUBLIC_ZIP}"
      extract_zip_if_needed "${COCO_PUBLIC_ZIP}" "${COCO_DIR}/train2017" "${COCO_DIR}"
    else
      download_file \
        "http://images.cocodataset.org/zips/train2017.zip" \
        "${COCO_DIR}/train2017.zip"
      extract_zip_if_needed "${COCO_DIR}/train2017.zip" "${COCO_DIR}/train2017" "${COCO_DIR}"
      if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
        rm -f "${COCO_DIR}/train2017.zip"
      fi
    fi
  fi
}

verify_data() {
  echo "[verify] checking dataset files"
run_python - <<'PY'
import json
import os
from pathlib import Path
data_root = Path(os.environ.get("DATA_ROOT", "/root/Aeromamba/data"))
pre = Path(os.environ.get("PRETRAIN_DIR", str(data_root / "llava_pretrain")))
stage2_root = Path(os.environ.get("STAGE2_ROOT", str(data_root)))
stage2_json = os.environ.get("STAGE2_JSON", "stage2_mixed_data.json")
assert (pre / "blip_laion_cc_sbu_558k.json").exists(), "missing LLaVA-Pretrain json"
assert (stage2_root / stage2_json).exists(), "missing stage2 json"
with open(pre / "blip_laion_cc_sbu_558k.json", "r", encoding="utf-8") as f:
    pre_data = json.load(f)
with open(stage2_root / stage2_json, "r", encoding="utf-8") as f:
    ins_data = json.load(f)
pre_existing = sum((pre / item["image"]).exists() for item in pre_data[:1000])
ins_images = {item["image"] for item in ins_data}
ins_existing = sum((stage2_root / name).exists() for name in ins_images)
print(f"LLaVA-Pretrain samples: {len(pre_data):,}; first-1000 images found: {pre_existing}/1000")
print(f"Stage2 samples: {len(ins_data):,}; unique images found: {ins_existing:,}/{len(ins_images):,}")
assert pre_existing > 0, "no LLaVA-Pretrain images found"
assert ins_existing == len(ins_images), "some Stage2 images are missing (COCO and/or Open3D-VQA)"
PY
}

train_stage1() {
  echo "[stage1] starting projector alignment"
  run_python run_stage1_train.py \
    --data_root "${PRETRAIN_DIR}" \
    --json_name blip_laion_cc_sbu_558k.json \
    --mamba_type mamba-130m \
    --vision_type dinosiglip_so_384 \
    --use_token_pooling \
    --pool_size 8 \
    --batch "${STAGE1_BATCH}" \
    --epochs "${STAGE1_EPOCHS}" \
    --lr "${STAGE1_LR}" \
    --workers "${STAGE1_WORKERS}" \
    --max_text_len "${STAGE1_MAX_TEXT_LEN}" \
    --save_dir checkpoints/stage1 \
    --log_every 20 \
    --max_val_steps 100
}

train_stage2() {
  echo "[stage2] starting visual SFT"
  stage1_ckpt="checkpoints/stage1/best.pth"
  if [[ ! -s "${stage1_ckpt}" && -s checkpoints/stage1/projector_only.pth ]]; then
    stage1_ckpt="checkpoints/stage1/projector_only.pth"
  fi
  run_python training/stage2_vlm.py \
    --data_root "${STAGE2_ROOT}" \
    --json_name "${STAGE2_JSON}" \
    --stage1_ckpt "${stage1_ckpt}" \
    --mamba_type mamba-130m \
    --vision_type dinosiglip_so_384 \
    --use_token_pooling \
    --pool_size 8 \
    --batch "${STAGE2_BATCH}" \
    --epochs "${STAGE2_EPOCHS}" \
    --lr "${STAGE2_LR}" \
    --workers "${STAGE2_WORKERS}" \
    --max_text_len "${STAGE2_MAX_TEXT_LEN}" \
    --save_dir checkpoints/stage2 \
    --log_every 20 \
    --max_val_steps 100
}

date
nvidia-smi || true
prepare_stage1_data
prepare_stage2_data
verify_data
train_stage1
train_stage2
date
echo "[done] stage1 checkpoint: ${REPO_DIR}/checkpoints/stage1/best.pth"
echo "[done] stage2 checkpoint: ${REPO_DIR}/checkpoints/stage2/best.pth"
