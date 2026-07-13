#!/bin/bash
# Stage 2 SFT on the v2 source-tagged data mix (stage2_mixed_data_v2.json).
#
# Differences vs the v1 run in run_full_pipeline_opt.sh:
#   - json: stage2_mixed_data_v2.json (general filtered to outdoor/urban,
#     + uav_motion language->motion QA, + CognitiveDrone reasoning QA)
#   - --source_weights makes the per-epoch mixing ratio explicit:
#       general        1.0  -> 25.0%  (language-ability retention)
#       aerial_spatial 1.5  -> 37.5%  (aerial spatial grounding, Stage-3 prior)
#       uav_motion     1.2  -> 30.0%  (instruction->motion binding)
#       cognitive      0.3  ->  7.5%  (object/symbol reasoning, small dose)
#     Tune via SOURCE_WEIGHTS env for A/B experiments.
#
# Usage: bash scripts/run_stage2_v2.sh [extra stage2_vlm.py args...]

set -e
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/Aeromamba/data}"
CKPT_ROOT="${CKPT_ROOT:-/root/autodl-tmp/Aeromamba/checkpoints/full_opt}"
SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-general=1.0,aerial_spatial=1.5,uav_motion=1.2,cognitive=0.3}"

python training/stage2_vlm.py \
  --arch_preset aeromamba_opt \
  --data_root "$DATA_ROOT" \
  --json_name stage2_mixed_data_v2.json \
  --source_weights "$SOURCE_WEIGHTS" \
  --stage1_ckpt "$CKPT_ROOT/stage1/best.pth" \
  --lora_r 16 --lora_alpha 32 \
  --batch 12 --epochs 2 --lr 5e-5 \
  --workers 16 --max_text_len 128 \
  --log_every 100 --save_every_steps 2000 \
  --save_dir "$CKPT_ROOT/stage2_v2" \
  "$@"
