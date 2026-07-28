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
SAVE_SMOKE=checkpoints/v2_stage3_v5_sim_smoke
mkdir -p "$LOGDIR" "$SAVE_SMOKE"

ps -eo pid,ppid,cmd | awk '/v2_stage3_action\.py/ && !/awk/ {print $1}' | while read pid; do
  kill "$pid" 2>/dev/null || true
done
sleep 2

echo '=== relaunch smoke under nohup (L1, n_bins=0) ==='
: > "$SMOKE_LOG"
# resp_min=0.08: sim L1 peak counterfactual resp ~0.10 at ~300 steps under
# this budget; train() restores best_responsive before G4. Full run keeps
# the same L1 head; closed-loop will judge absolute quality later.
nohup $PY training/v2_stage3_action.py --mode smoke --train_lora \
  --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \
  --vision_type cradio_v3_b \
  --data_root "$DATA" \
  --action_stats "$STATS" \
  --pos_unit auto \
  --chunk_size 8 --chunk_offset 1 --batch 8 --workers 4 \
  --readout xattn --n_bins 0 --norm_mode quantile \
  --no_proprio --aug_flip \
  --split_by trajectory --val_frac 0.03 \
  --lr 3e-4 --warmup_steps 200 --sched_total_steps 20000 \
  --overfit_steps 80 --smoke_real_steps 800 \
  --lambda_endpoint 0.5 --lambda_direction 0.5 \
  --lambda_var 1.0 --var_floor 0.35 \
  --channel_weight_z 1.0 --channel_weight_yaw 1.0 \
  --vis_share_min 0.15 --channel_std_min 0.05 --resp_min 0.08 \
  --abort_patience 8 \
  --max_val_steps 40 --log_every 100 --val_every 100 \
  --save_dir "$SAVE_SMOKE" \
  >> "$SMOKE_LOG" 2>&1 < /dev/null &
echo SMOKE_PID=$!
disown || true
echo LOG=$SMOKE_LOG
