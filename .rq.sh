#!/bin/bash
set -u
REPO=/root/autodl-tmp/Aeromamba
PY=/root/autodl-tmp/envs/aerov2/bin/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

cd $REPO
git fetch -q origin aerov3-hugebench && git checkout -qf -B aerov3-hugebench FETCH_HEAD
git log --oneline -1

echo '=== qwen prefetch ==='
tail -n 2 /root/autodl-tmp/logs/qwen_dl.log | tr '\r' '\n' | tail -1

echo '=== data QC rerun ==='
nohup $PY -u scripts/hugebench_data_qc.py \
  --data_root /root/autodl-tmp/huge/dl/HUGE_Dataset_v0 \
  --anno_root /root/autodl-tmp/huge/HUGE-Bench/trajectory_generation/stage_annotations \
  --split train --sample 40 \
  > /root/autodl-tmp/logs/data_qc.log 2>&1 &
sleep 40
cat /root/autodl-tmp/logs/data_qc.log

echo '=== download ==='
df -h /root/autodl-tmp | tail -1
find /root/autodl-tmp/huge/dl/HUGE_Dataset_v0/train -name '*.parquet' | wc -l
