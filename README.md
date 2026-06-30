# AeroMamba

AeroMamba is a lightweight staged vision-language-action (VLA) training stack for UAV navigation. The current training path uses a frozen SigLIP2 vision encoder, a Mamba2 language/state-space backbone, a learned visual token resampler, and a dynamics-aware UAV action head.

This repository tracks source code, training scripts, and reproducibility notes only. Datasets, checkpoints, local credentials, logs, and cache files must stay outside Git.

## Current Recommended Architecture

| Component | Current default | Purpose |
| --- | --- | --- |
| Vision encoder | `siglip2_base_384` (`google/siglip2-base-patch16-384`) | Frozen visual feature extractor with 384px input and 576 patch tokens |
| Language backbone | `mamba-2-370m` (`state-spaces/mamba2-370m`) | Frozen base Mamba2 backbone with LoRA adapters in later stages |
| Visual adapter | `MLPProjector` | Maps SigLIP2 visual features into Mamba hidden space |
| Token compressor | `PerceiverResampler`, 32 queries | Reduces visual tokens before Mamba (`576 -> 32`) |
| Stage 3 head | `UAVDynamicsActionHead` | Predicts smooth UAV waypoint/action chunks |

Why the resampler is enabled from Stage 1: inserting it only at Stage 3 changes the visual distribution too late. Training `projector + resampler` together from Stage 1 lets Stage 2 and Stage 3 share the same visual interface.

## Repository Layout

```text
configs/      Experiment/config files
data/         Dataset loader source code
docs/         Environment and dataset notes
inference/    Inference and evaluation helpers
model/        AeroMamba model modules
scripts/      Utility and launch scripts
training/     Stage 1/2/3 training entry points
```

## Environment

The tested remote environment is:

```bash
conda activate mamba2
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, timm; print(transformers.__version__, timm.__version__)"
```

Known working versions on the current server:

- Python environment: `/root/miniconda3/envs/mamba2`
- PyTorch: `2.1.1+cu118`
- Transformers: `4.51.3`
- timm: `1.0.27`
- GPU: RTX 4090 24GB

Recommended Hugging Face cache variables:

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/torch_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
```

## Dataset Layout

Recommended server layout:

```text
/root/autodl-tmp/datasets/
  stage1_llava_pretrain/
    blip_laion_cc_sbu_558k.json
    images/
  stage2_aeromamba/
    stage2_mixed_data.json
    coco/train2017/
    open3d_vqa/O3DVQA/
  stage3_uavflow/
    train-00000-of-00054.parquet
    ...
```

Stage 1 uses the standard LLaVA-Pretrain 558K image-text alignment set.

Stage 2 uses AeroMamba mixed VQA data. It is not COCO-only: `stage2_mixed_data.json` references both COCO `train2017` images and Open3D-VQA images under `open3d_vqa/O3DVQA`.

Stage 3 uses UAV-Flow parquet shards. The current verified full split contains 54 shards and about 1.78M rows.

Before training on a new server, verify that every referenced image path exists and that the UAV-Flow parquet shards load cleanly. Do not flatten or rename the Open3D-VQA directory tree.

## Three-Stage Training

### Stage 1: Projector + Resampler Alignment

Train only the visual projector and Perceiver resampler. Vision and Mamba2 stay frozen.

```bash
python -u training/stage1_align.py \
  --data_root /root/autodl-tmp/datasets/stage1_llava_pretrain \
  --json_name blip_laion_cc_sbu_558k.json \
  --vision_type siglip2_base_384 \
  --mamba_type mamba-2-370m \
  --token_resampler perceiver \
  --num_visual_queries 32 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --batch 8 \
  --lr 1e-4 \
  --epochs 1 \
  --workers 6 \
  --max_text_len 64 \
  --log_every 500 \
  --max_val_steps 100 \
  --save_every_steps 10000 \
  --save_dir checkpoints/base384_mamba2_resampler/stage1
```

Expected trainable modules:

- `projector`
- `token_resampler`

### Stage 2: VLM SFT with LoRA

Load Stage 1 projector/resampler weights, then train projector, resampler, and Mamba2 LoRA adapters.

```bash
python -u training/stage2_vlm.py \
  --data_root /root/autodl-tmp/datasets/stage2_aeromamba \
  --json_name stage2_mixed_data.json \
  --stage1_ckpt checkpoints/base384_mamba2_resampler/stage1/best.pth \
  --vision_type siglip2_base_384 \
  --mamba_type mamba-2-370m \
  --token_resampler perceiver \
  --num_visual_queries 32 \
  --resampler_layers 2 \
  --resampler_heads 8 \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch 4 \
  --lr 5e-5 \
  --epochs 1 \
  --workers 6 \
  --max_text_len 64 \
  --log_every 500 \
  --max_val_steps 100 \
  --save_every_steps 10000 \
  --save_dir checkpoints/base384_mamba2_resampler/stage2
```

Expected trainable modules:

- `projector`
- `token_resampler`
- Mamba2 LoRA adapter parameters

### Stage 3: UAV-Flow Action Training

Load Stage 2 checkpoint, then train the UAV action stack. The `uav_lite_siglip` preset selects `siglip2_base_384`, `mamba-2-370m`, `PerceiverResampler`, and the dynamics action head.

```bash
python -u training/stage3_action.py \
  --arch_preset uav_lite_siglip \
  --hf_dataset parquet \
  --hf_data_files '/root/autodl-tmp/datasets/uav-flow/train-*.parquet' \
  --hf_cache_dir /root/autodl-tmp/hf_cache/datasets \
  --stage2_ckpt checkpoints/base384_mamba2_resampler/stage2/best.pth \
  --epochs 2 \
  --batch 4 \
  --workers 6 \
  --lr 5e-5 \
  --log_every 500 \
  --max_val_steps 100 \
  --save_every_steps 10000 \
  --save_dir checkpoints/base384_mamba2_resampler/stage3
```

Expected trainable modules:

- `token_resampler`
- `proprio_encoder`
- `UAVDynamicsActionHead`
- Mamba2 LoRA adapter parameters

## Current Remote Pipeline

The latest long-running server pipeline was launched under:

```text
/root/autodl-tmp/Aeromamba/checkpoints/base384_mamba2_resampler_<timestamp>/
```

It runs Stage 1, Stage 2, then Stage 3 sequentially and writes:

```text
pipeline.log
pipeline.pid
stage1/train.log
stage2/train.log
stage3/train.log
```

Monitor it with:

```bash
cd /root/autodl-tmp/Aeromamba
RUN=$(cat checkpoints/latest_light_pipeline_run.txt)
tail -f checkpoints/$RUN/stage1/train.log
nvidia-smi
```

## Validation

Basic syntax check:

```bash
python -m py_compile \
  model/vision.py \
  model/resampler.py \
  model/action_head.py \
  model/uav_mamba_vla.py \
  training/trainer.py \
  training/stage1_align.py \
  training/stage2_vlm.py \
  training/stage3_action.py
```

Training sanity criteria:

- Stage 1: loss should trend downward over the first few thousand steps.
- Stage 2: no NaN/OOM; validation CLM loss should not explode.
- Stage 3: `main`, `smooth`, and `l1_err` should remain finite; keep FP32/no AMP if Smooth-L1 becomes unstable.

## Git Hygiene

Do not commit:

- `checkpoints/`
- downloaded datasets
- model weights (`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`)
- SSH keys or tokens
- local cache/log artifacts
- draft papers or large PDF references

Use `.gitignore` as the source of truth for excluded artifacts.
