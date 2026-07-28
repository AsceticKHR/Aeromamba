#!/bin/bash
set -e
cd /root/autodl-tmp/Aeromamba
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
PY=/root/autodl-tmp/envs/aerov2/bin/python
LOG=/root/autodl-tmp/s3_full_v2.log
: > "$LOG"
nohup "$PY" training/v2_stage3_action.py --mode full --train_lora \
  --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \
  --vision_type cradio_v3_b \
  --data_root /root/autodl-tmp/datasets/stage3_uavflow \
  --action_stats /root/autodl-tmp/datasets/uav-flow/action_stats_k8.json \
  --chunk_size 8 --batch 8 --workers 4 --lr 3e-4 --warmup_steps 200 \
  --no_proprio \
  --lambda_direction 1.0 --lambda_endpoint 0.5 --lambda_var 0.5 \
  --max_steps 1500 --val_every 250 --max_val_steps 40 --log_every 50 \
  --save_dir checkpoints/v2_stage3_cradio_v4 \
  >> "$LOG" 2>&1 &
echo "FULL_PID=$!"
