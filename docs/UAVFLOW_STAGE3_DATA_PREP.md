# UAV-Flow Stage 3 数据构建指南

本流程参考官方 `buaa-colalab/UAV-Flow` 的 `dataset_tools/prepare_data.py`：从 parquet 按 `id` 聚合完整轨迹，写成每条轨迹一个文件夹：

```text
stage3_uavflow/
  <trajectory_id>/
    000000.jpg
    000001.jpg
    ...
    log.json
  metadata/
    manifest.jsonl
    summary.json
```

相比官方脚本，AeroMamba 版本额外做了两件事：

- 保留 `instruction` / `instruction_unified`，Stage 3 训练会使用真实语言指令。
- loader 从 `raw_logs` 动态构造 body-frame action：`[forward, right, up, delta_yaw]`，避免世界坐标标签导致平均轨迹坍缩。

## 1. 从 parquet 构建完整目录

服务器示例：

```bash
cd /root/autodl-tmp/Aeromamba
conda activate mamba2

python data/prepare_uavflow_stage3.py \
  --parquet_glob "/root/autodl-tmp/datasets/uav-flow/train-*.parquet" \
  --output_dir "/root/autodl-tmp/datasets/stage3_uavflow" \
  --split train \
  --chunk_size 5 \
  --hf_cache_dir "/root/autodl-tmp/hf_cache/datasets" \
  --verify_images
```

如果数据盘空间不足，可以在确认 smoke test 正常后启用分片级安全删除：

```bash
python data/prepare_uavflow_stage3.py \
  --parquet_glob "/root/autodl-tmp/datasets/uav-flow/train-*.parquet" \
  --output_dir "/root/autodl-tmp/datasets/stage3_uavflow" \
  --split train \
  --chunk_size 5 \
  --hf_cache_dir "/root/autodl-tmp/hf_cache/datasets" \
  --verify_images \
  --delete_parquet_after_success
```

注意：这个选项只会在某个 parquet 分片中出现过的轨迹都已经写完或安全跳过后删除该分片，不会读到一半就删。

快速小样本测试：

```bash
python data/prepare_uavflow_stage3.py \
  --parquet_glob "/root/autodl-tmp/datasets/uav-flow/train-*.parquet" \
  --output_dir "/root/autodl-tmp/datasets/stage3_uavflow_smoke" \
  --split train \
  --chunk_size 5 \
  --hf_cache_dir "/root/autodl-tmp/hf_cache/datasets" \
  --max_trajectories 100 \
  --overwrite \
  --verify_images
```

## 2. 验证目录完整性

```bash
python data/validate_uavflow_stage3.py \
  --data_root "/root/autodl-tmp/datasets/stage3_uavflow" \
  --chunk_size 5 \
  --report "/root/autodl-tmp/Aeromamba/checkpoints/stage3_uavflow_validate.json"
```

报告中应满足：

- `bad = 0`
- `bad_images = 0`
- `missing_instruction = 0`
- `zero_action_trajectories` 尽量接近 0

## 3. 用 folder 数据启动 Stage 3

```bash
python training/stage3_action.py \
  --arch_preset uav_lite_siglip \
  --data_root "/root/autodl-tmp/datasets/stage3_uavflow" \
  --stage2_ckpt "/root/autodl-tmp/Aeromamba/checkpoints/base384_mamba2_resampler_20260630_142642/stage2/best.pth" \
  --save_dir "/root/autodl-tmp/Aeromamba/checkpoints/stage3_uavflow_bodyframe" \
  --batch 16 \
  --workers 16 \
  --epochs 2 \
  --chunk_size 5 \
  --pos_scale 100.0 \
  --lr 5e-5 \
  --no_amp \
  --stage3_train_lora \
  --max_val_steps 200
```

## 4. 重要语义

- 原始 UAV-Flow / UE 坐标单位按 cm 处理。
- 训练标签内部归一化为 `position_cm / pos_scale`，默认 `pos_scale=100.0`，即约等于米。
- yaw 标签从 degree 转成 rad。
- 当前实现输出 body-frame waypoint，更适合无人机动力学控制和跨场景泛化。
