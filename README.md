# AeroMamba

AeroMamba is a lightweight staged vision-language-action (VLA) training stack for UAV navigation. The current full-stage recipe trains a SigLIP2 + Mamba2 model from Stage 1 alignment through Stage 3 UAV-Flow action learning.

This repository contains source code, launch scripts, and reproducibility notes only. Do not commit datasets, checkpoints, logs, credentials, or local evaluation outputs.

## Current Full-Stage Architecture

| Part | Current setting | Notes |
| --- | --- | --- |
| Vision encoder | `siglip2_base_384` | Frozen SigLIP2 384px visual backbone |
| Mamba backbone | `mamba-2-370m` | Frozen base backbone; LoRA is trained in Stage 2/3 |
| Tokenizer | `EleutherAI/gpt-neox-20b` | Explicit tokenizer for Mamba/Mamba2; GPT-2 fallback is disabled |
| Vision adapter | `MLPProjector` | Maps SigLIP2 hidden states into Mamba hidden space |
| Token compressor | `PerceiverResampler`, 32 queries | Keeps visual sequence short and stable across all stages |
| Stage 3 policy | `UAVDynamicsActionHead` + `action_context_fuser` | Uses language/state/vision/global summaries for UAV-Flow action chunks |
| Stage 3 loss | Smooth-L1 + endpoint + direction + smoothness | Reduces mean-trajectory collapse and emphasizes navigation target direction |

## Repository Layout

```text
configs/      Experiment/config files
data/         Dataset loaders and UAV-Flow preparation/validation scripts
docs/         Reports and environment/data guides
inference/    Inference server and evaluation helpers
model/        AeroMamba model modules
scripts/      Launch scripts
training/     Stage 1/2/3 training entry points
```

## Environment

Tested server setup:

```bash
conda activate mamba2
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, timm; print(transformers.__version__, timm.__version__)"
```

Known working baseline:

- Python env: `/root/miniconda3/envs/mamba2`
- PyTorch: `2.1.1+cu118`
- Transformers: `4.51.3`
- timm: `1.0.27`
- GPU: RTX 4090 24GB

Recommended cache variables:

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/torch_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export AEROMAMBA_TOKENIZER=EleutherAI/gpt-neox-20b
```

Cache the tokenizer before full training:

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
print(type(tok).__name__, tok.vocab_size)
PY
```

## Data Preparation

Recommended server layout:

```text
/root/autodl-tmp/Aeromamba/data/
  llava_pretrain/
    blip_laion_cc_sbu_558k.json
    <LLaVA pretrain image folders>
  stage2_mixed_data.json
  coco/train2017/
  open3d_vqa/O3DVQA/

/root/autodl-tmp/datasets/uav-flow/
  <trajectory_id>/
    000000.jpg
    000001.jpg
    ...
    log.json
  metadata/
    manifest.jsonl
    summary.json
```

### Stage 1 Data

Stage 1 uses the official LLaVA-Pretrain 558K image-text dataset:

- `blip_laion_cc_sbu_558k.json`
- corresponding LLaVA image folders

### Stage 2 Data

Stage 2 uses AeroMamba mixed VQA data:

- `stage2_mixed_data.json`
- COCO `train2017`
- Open3D-VQA under `open3d_vqa/O3DVQA`

Do not flatten or rename Open3D-VQA. The JSON references the original directory structure.

### Stage 3 Data

Stage 3 uses UAV-Flow converted into official folder-style trajectories. Build from parquet shards:

```bash
cd /root/autodl-tmp/Aeromamba

python data/prepare_uavflow_stage3.py \
  --parquet_glob "/root/autodl-tmp/datasets/uav-flow/train-*.parquet" \
  --output_dir "/root/autodl-tmp/datasets/uav-flow" \
  --split train \
  --chunk_size 5 \
  --hf_cache_dir "/root/autodl-tmp/hf_cache/datasets" \
  --verify_images
```

Validate:

```bash
python data/validate_uavflow_stage3.py \
  --data_root "/root/autodl-tmp/datasets/uav-flow" \
  --chunk_size 5 \
  --report "/root/autodl-tmp/Aeromamba/checkpoints/stage3_uavflow_validate.json"
```

A verified full conversion should report:

- `rows = 1785284`
- `written_trajectories = 26795`
- `bad_rows = 0`
- `incomplete_buffers = 0`

The Stage 3 loader uses `instruction_unified`/`instruction` and prefers official `preprocessed_logs` for body-frame action labels.

## One-Command Full Training

Use the full-stage script:

```bash
cd /root/autodl-tmp/Aeromamba

nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HOME=/root/autodl-tmp/hf_cache \
  AEROMAMBA_TOKENIZER=EleutherAI/gpt-neox-20b \
  RUN_NAME=uavflow_mamba2_siglip2_fast_$(date +%Y%m%d_%H%M%S) \
  STAGE1_BATCH=12 STAGE1_WORKERS=16 STAGE1_EPOCHS=1 \
  STAGE2_BATCH=8 STAGE2_WORKERS=16 STAGE2_EPOCHS=2 \
  STAGE3_BATCH=24 STAGE3_WORKERS=24 STAGE3_EPOCHS=2 \
  bash scripts/run_uavflow_three_stage_pipeline.sh \
  > checkpoints/full_stage_training.log 2>&1 < /dev/null &
```

Monitor:

```bash
RUN=$(cat checkpoints/latest_uavflow_pipeline_run.txt)
tail -f "checkpoints/${RUN}.log"
nvidia-smi
```

Outputs:

```text
checkpoints/<RUN>/
  stage1/
  stage2/
  stage3/
checkpoints/<RUN>.log
checkpoints/<RUN>.pid
```

## Manual Stage Commands

### Stage 1: Projector + Resampler Alignment

```bash
python training/stage1_align.py \
  --data_root /root/autodl-tmp/Aeromamba/data/llava_pretrain \
  --json_name blip_laion_cc_sbu_558k.json \
  --mamba_type mamba-2-370m \
  --vision_type siglip2_base_384 \
  --token_resampler perceiver \
  --num_visual_queries 32 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --batch 12 \
  --workers 16 \
  --epochs 1 \
  --lr 1e-4 \
  --max_text_len 64 \
  --save_dir checkpoints/full_stage/stage1
```

Trainable modules: `projector`, `token_resampler`.

### Stage 2: VLM SFT with LoRA

```bash
python training/stage2_vlm.py \
  --data_root /root/autodl-tmp/Aeromamba/data \
  --json_name stage2_mixed_data.json \
  --stage1_ckpt checkpoints/full_stage/stage1/best.pth \
  --mamba_type mamba-2-370m \
  --vision_type siglip2_base_384 \
  --token_resampler perceiver \
  --num_visual_queries 32 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch 8 \
  --workers 16 \
  --epochs 2 \
  --lr 5e-5 \
  --max_text_len 128 \
  --save_dir checkpoints/full_stage/stage2
```

Trainable modules: `projector`, `token_resampler`, Mamba2 LoRA adapters.

### Stage 3: UAV-Flow Action Training

```bash
python training/stage3_action.py \
  --arch_preset uav_lite_siglip \
  --data_root /root/autodl-tmp/datasets/uav-flow \
  --stage2_ckpt checkpoints/full_stage/stage2/best.pth \
  --mamba_type mamba-2-370m \
  --batch 24 \
  --workers 24 \
  --epochs 2 \
  --lr 5e-5 \
  --chunk_size 5 \
  --pos_scale 100.0 \
  --lambda_smooth 0.05 \
  --lambda_endpoint 0.7 \
  --lambda_direction 0.2 \
  --max_text_len 64 \
  --stage3_train_lora \
  --no_amp \
  --save_dir checkpoints/full_stage/stage3
```

Trainable modules: `token_resampler`, `proprio_encoder`, `action_context_fuser`, `UAVDynamicsActionHead`, Mamba2 LoRA adapters.

## Validation

Syntax check:

```bash
python -m py_compile \
  data/dataset.py \
  data/prepare_uavflow_stage3.py \
  data/validate_uavflow_stage3.py \
  model/action_head.py \
  model/uav_mamba_vla.py \
  training/stage1_align.py \
  training/stage2_vlm.py \
  training/stage3_action.py
```

Training sanity:

- Stage 1 loss should decrease in the first few hundred steps.
- Stage 2 validation CLM loss should remain finite and trend down.
- Stage 3 should log finite `main`, `endpoint`, `direction`, `smooth`, and `l1_err`.
- If Stage 3 OOMs, reduce `STAGE3_BATCH` first; if dataloader stalls, reduce `STAGE3_WORKERS`.

## Reports

Useful design and debugging notes are in:

- `docs/UAVFLOW_STAGE3_DATA_PREP.md`
- `docs/aeromamba_improved_architecture_report.md`
- `docs/aeromamba_stage3_action_collapse_diagnosis.md`
- `docs/aeromamba_uav_flow_eval_framework.md`

## Git Hygiene

Never commit:

- `checkpoints/`
- downloaded datasets
- model weights: `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
- SSH keys or tokens
- local eval outputs: `eval_results/`, `eval_smoke_jsons/`
- logs, caches, archives, temporary documents
