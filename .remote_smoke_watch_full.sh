#!/bin/bash
set -e
cd /root/autodl-tmp/Aeromamba
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY=/root/autodl-tmp/envs/aerov2/bin/python
DATA=/root/autodl-tmp/datasets/stage3_uavflow_sim
STATS=/root/autodl-tmp/datasets/uav-flow-sim/action_stats_k8_off1.json
LOGDIR=/root/autodl-tmp/logs
SMOKE_LOG=$LOGDIR/s3_v5_sim_smoke.log
FULL_LOG=$LOGDIR/s3_v5_sim.log
SAVE=checkpoints/v2_stage3_v5_sim
WATCH=$LOGDIR/smoke_watch.log
: > "$WATCH"

{
  echo "[watch] waiting for smoke verdict..."
  for i in $(seq 1 240); do
    if grep -q 'S3 SMOKE VERDICT:' "$SMOKE_LOG" 2>/dev/null; then
      echo "[watch] verdict found"
      grep -E 'G1 |G1v |G2 |G3 |G4 |G5 |VERDICT' "$SMOKE_LOG" | tail -n 40
      if grep -q 'S3 SMOKE VERDICT:.*ALL PASS' "$SMOKE_LOG"; then
        echo "[watch] SMOKE_ALL_PASS — launching full (L1 n_bins=0)"
        ps -eo pid,ppid,cmd | awk '/v2_stage3_action\.py/ && !/awk/ {print $1}' | while read pid; do
          kill "$pid" 2>/dev/null || true
        done
        sleep 3
        mkdir -p "$SAVE"
        : > "$FULL_LOG"
        nohup $PY training/v2_stage3_action.py --mode full --train_lora \
          --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \
          --vision_type cradio_v3_b \
          --data_root "$DATA" \
          --action_stats "$STATS" \
          --pos_unit auto \
          --chunk_size 8 --chunk_offset 1 --batch 16 --workers 4 \
          --readout xattn --n_bins 0 --norm_mode quantile \
          --no_proprio --aug_flip \
          --split_by trajectory --val_frac 0.03 \
          --lr 3e-4 --warmup_steps 300 --max_steps 20000 --sched_total_steps 20000 \
          --lambda_endpoint 0.5 --lambda_direction 0.5 \
          --lambda_var 1.0 --var_floor 0.35 \
          --channel_weight_z 1.0 --channel_weight_yaw 1.0 \
          --vis_share_min 0.15 --channel_std_min 0.05 --resp_min 0.08 \
          --epochs 1 --val_every 500 --max_val_steps 40 --log_every 100 \
          --save_dir "$SAVE" \
          >> "$FULL_LOG" 2>&1 < /dev/null &
        echo "[watch] FULL_PID=$! LOG=$FULL_LOG"
        exit 0
      else
        echo "[watch] SMOKE_FAILED — not launching full"
        exit 3
      fi
    fi
    if ! pgrep -f 'v2_stage3_action.py --mode smoke' >/dev/null; then
      # race: process may exit between writing VERDICT and our next poll
      sleep 2
      if grep -q 'S3 SMOKE VERDICT:' "$SMOKE_LOG" 2>/dev/null; then
        continue
      fi
      echo "[watch] smoke process gone without verdict"
      tail -n 50 "$SMOKE_LOG"
      exit 4
    fi
    sleep 30
  done
  echo "[watch] timeout"
  exit 5
} >> "$WATCH" 2>&1
