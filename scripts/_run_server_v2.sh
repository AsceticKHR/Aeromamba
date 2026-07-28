#!/bin/bash
cd /root/autodl-tmp/Aeromamba
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
PY=/root/autodl-tmp/envs/aerov2/bin/python
LOG=/root/autodl-tmp/server_v2.log
pkill -f server_v2.py 2>/dev/null || true
sleep 1
: > "$LOG"
nohup "$PY" inference/server_v2.py \
  --s2_ckpt_dir checkpoints/v2_stage2_full_cradio --s2_tag best \
  --s3_ckpt checkpoints/v2_stage3_cradio_v4/best_grounded.pth \
  --vision_type cradio_v3_b \
  --action_stats /root/autodl-tmp/datasets/uav-flow/action_stats_k8.json \
  --chunk_size 8 --proprio_dim 4 --no_proprio \
  --exec_mode chunk --exec_horizon 4 --bf16 --port 5007 \
  >> "$LOG" 2>&1 &
echo "SERVER_PID=$!"
