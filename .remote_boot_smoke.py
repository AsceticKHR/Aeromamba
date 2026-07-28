#!/usr/bin/env python3
"""Rewrite LF launch scripts and start sim smoke + watch."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Aeromamba")
LOGDIR = Path("/root/autodl-tmp/logs")
PY = "/root/autodl-tmp/envs/aerov2/bin/python"
DATA = "/root/autodl-tmp/datasets/stage3_uavflow_sim"
STATS = "/root/autodl-tmp/datasets/uav-flow-sim/action_stats_k8_off1.json"
SMOKE_LOG = LOGDIR / "s3_v5_sim_smoke.log"
FULL_LOG = LOGDIR / "s3_v5_sim.log"
WATCH_LOG = LOGDIR / "smoke_watch.log"

smoke_sh = f"""#!/bin/bash
set -e
cd {ROOT}
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY={PY}
DATA={DATA}
STATS={STATS}
LOGDIR={LOGDIR}
SMOKE_LOG=$LOGDIR/s3_v5_sim_smoke.log
SAVE_SMOKE=checkpoints/v2_stage3_v5_sim_smoke
mkdir -p "$LOGDIR" "$SAVE_SMOKE"
ps -eo pid,ppid,cmd | awk '/v2_stage3_action\\.py/ && !/awk/ {{print $1}}' | while read pid; do
  kill "$pid" 2>/dev/null || true
done
sleep 2
echo "=== relaunch smoke under nohup ==="
: > "$SMOKE_LOG"
nohup $PY training/v2_stage3_action.py --mode smoke --train_lora \\
  --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \\
  --vision_type cradio_v3_b \\
  --data_root "$DATA" \\
  --action_stats "$STATS" \\
  --pos_unit auto \\
  --chunk_size 8 --chunk_offset 1 --batch 8 --workers 4 \\
  --readout xattn --n_bins 128 --bin_range 1.5 --norm_mode quantile \\
  --no_proprio --aug_flip \\
  --split_by trajectory --val_frac 0.03 \\
  --lr 3e-4 --warmup_steps 200 --sched_total_steps 20000 \\
  --overfit_steps 80 --smoke_real_steps 1000 \\
  --lambda_endpoint 0.5 --lambda_direction 0.5 \\
  --vis_share_min 0.15 --channel_std_min 0.12 --resp_min 0.25 \\
  --max_val_steps 40 --log_every 100 --val_every 100 \\
  --save_dir "$SAVE_SMOKE" \\
  >> "$SMOKE_LOG" 2>&1 < /dev/null &
echo SMOKE_PID=$!
disown || true
echo LOG=$SMOKE_LOG
"""

watch_sh = f"""#!/bin/bash
set -e
cd {ROOT}
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY={PY}
DATA={DATA}
STATS={STATS}
LOGDIR={LOGDIR}
SMOKE_LOG=$LOGDIR/s3_v5_sim_smoke.log
FULL_LOG=$LOGDIR/s3_v5_sim.log
SAVE=checkpoints/v2_stage3_v5_sim
WATCH=$LOGDIR/smoke_watch.log
: > "$WATCH"
{{
  echo "[watch] waiting for smoke verdict..."
  for i in $(seq 1 240); do
    if grep -q 'S3 SMOKE VERDICT:' "$SMOKE_LOG" 2>/dev/null; then
      echo "[watch] verdict found"
      grep -E 'G1 |G1v |G2 |G3 |G4 |G5 |VERDICT' "$SMOKE_LOG" | tail -n 40
      if grep -q 'S3 SMOKE VERDICT:.*ALL PASS' "$SMOKE_LOG"; then
        echo "[watch] SMOKE_ALL_PASS — launching full"
        ps -eo pid,ppid,cmd | awk '/v2_stage3_action\\.py/ && !/awk/ {{print $1}}' | while read pid; do
          kill "$pid" 2>/dev/null || true
        done
        sleep 3
        mkdir -p "$SAVE"
        : > "$FULL_LOG"
        nohup $PY training/v2_stage3_action.py --mode full --train_lora \\
          --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \\
          --vision_type cradio_v3_b \\
          --data_root "$DATA" \\
          --action_stats "$STATS" \\
          --pos_unit auto \\
          --chunk_size 8 --chunk_offset 1 --batch 16 --workers 4 \\
          --readout xattn --n_bins 128 --bin_range 1.5 --norm_mode quantile \\
          --no_proprio --aug_flip \\
          --split_by trajectory --val_frac 0.03 \\
          --lr 3e-4 --warmup_steps 300 --max_steps 20000 --sched_total_steps 20000 \\
          --lambda_endpoint 0.5 --lambda_direction 0.5 \\
          --epochs 1 --val_every 500 --max_val_steps 40 --log_every 100 \\
          --save_dir "$SAVE" \\
          >> "$FULL_LOG" 2>&1 < /dev/null &
        echo "[watch] FULL_PID=$! LOG=$FULL_LOG"
        exit 0
      else
        echo "[watch] SMOKE_FAILED — not launching full"
        exit 3
      fi
    fi
    if ! pgrep -f 'v2_stage3_action.py --mode smoke' >/dev/null; then
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
}} >> "$WATCH" 2>&1
"""


def kill_matching(pattern: str) -> None:
    out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    for line in out.splitlines():
        if pattern in line and "awk" not in line and "boot_smoke" not in line:
            pid = line.strip().split(None, 1)[0]
            try:
                os.kill(int(pid), 15)
            except ProcessLookupError:
                pass


def main() -> None:
    stage3 = ROOT / "training" / "v2_stage3_action.py"
    stage3.write_bytes(
        stage3.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    )
    print("fixed", stage3, stage3.stat().st_size)

    for name, body in (
        (".remote_smoke_nohup.sh", smoke_sh),
        (".remote_smoke_watch_full.sh", watch_sh),
    ):
        path = ROOT / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        print("wrote", path)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    kill_matching("v2_stage3_action.py")
    kill_matching("smoke_watch")
    time.sleep(3)

    subprocess.check_call(["bash", str(ROOT / ".remote_smoke_nohup.sh")])
    time.sleep(5)
    outer = open(LOGDIR / "smoke_watch_outer.log", "ab")
    subprocess.Popen(
        ["bash", str(ROOT / ".remote_smoke_watch_full.sh")],
        stdout=outer,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(20)
    print("--- smoke markers ---")
    if SMOKE_LOG.exists():
        text = SMOKE_LOG.read_text(errors="replace")
        for line in text.splitlines():
            if any(k in line for k in ("SMOKE_PID", "scale=", "G1 ", "relaunch", "pos_unit")):
                print(line)
    print("--- procs ---")
    subprocess.call(
        "ps -eo pid,etime,cmd | awk '/v2_stage3_action|smoke_watch/ && !/awk/ {print}'",
        shell=True,
    )
    subprocess.call(
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",
        shell=True,
    )


if __name__ == "__main__":
    main()
