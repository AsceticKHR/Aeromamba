echo "=== proc ==="
ps -ef | grep "v2_stage1_align" | grep -v grep | head -2 || echo "NO_TRAIN_PROC"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "=== VAL ==="
grep "VAL=" /root/autodl-tmp/logs/s1_full_b48.log
echo "=== log tail ==="
tail -n 40 /root/autodl-tmp/logs/s1_full_b48.log
echo "=== done markers ==="
grep -E "done:|S1_FULL_EXIT|epoch|best_projector|complete|finished|EXIT|saved" /root/autodl-tmp/logs/s1_full_b48.log | tail -20
echo "=== ckpt ==="
ls -lh /root/autodl-tmp/Aeromamba/checkpoints/v2_stage1_full/ 2>/dev/null
wc -l /root/autodl-tmp/logs/s1_full_b48.log
