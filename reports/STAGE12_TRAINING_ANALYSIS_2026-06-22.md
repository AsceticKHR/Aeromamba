# AeroMamba Stage 1/2 训练配置与效果分析

生成日期：2026-06-22  
远端目录：`/root/Aeromamba`  
原始日志：`reports/stage12_pipeline_2026-06-22.log`  
统计数据：`reports/stage12_metrics_2026-06-22.json`

## 1. 训练结果概览

| 项目 | Stage 1 | Stage 2 |
|---|---:|---:|
| 状态 | 完成 | 完成 |
| 目标 | 视觉 projector 对齐 | Projector + Mamba LoRA 视觉指令微调 |
| 数据量 | 558,128 | 231,036 |
| 训练/验证 | 502,316 / 55,812 | 207,933 / 23,103 |
| Epoch | 3 | 5 |
| 最终 train loss | 3.2928 | 1.5432 |
| 最佳 val loss | 3.3559 | 1.6494 |
| 最终 val PPL | 29.1997 | 5.2348 |
| Checkpoint | 3,505,621,522 bytes | 3,528,385,198 bytes |
| SHA-256 | `573b93d42b8f0fe99e71766226f496c9675328f95dc38a1e6c0f87abdbd5841b` | `2c2eb5df7e8337e617cc6e2639e208cedb7630d03be468776dca8d78a6b00584` |

流水线从 2026-06-21 17:19:11 运行到 2026-06-22 05:24:33，总耗时约 12 小时 5 分钟。按 checkpoint 时间估算，Stage 1 约 6 小时 32 分钟，Stage 2 约 5 小时 33 分钟。

## 2. 实际超参数

### 公共配置

| 参数 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| PyTorch | 2.1.2，CUDA runtime 11.8 |
| Mamba | `mamba-130m` |
| 视觉编码器 | `dinosiglip_so_384` |
| 视觉组成 | DINOv2-L-reg4 + SigLIP-SO400M/14 |
| 输入分辨率 | 384×384 |
| 视觉输出 | 729 patches，DINO 1024 + SigLIP 1152 = 2176 维 |
| Token pooling | 开启，`pool_size=8`，输出 64 个视觉 token |
| 优化器 | AdamW |
| Weight decay | `1e-4` |
| LR scheduler | CosineAnnealingLR |
| 最低 LR | 初始 LR 的 10% |
| 混合精度 | CUDA AMP + GradScaler |
| CUDA allocator | `max_split_size_mb:128` |
| Train DataLoader | shuffle，drop_last，pin_memory，persistent workers |
| 验证上限 | 每个 epoch 最多 100 steps |

### Stage 1

| 参数 | 值 |
|---|---|
| 数据 | LLaVA-Pretrain `blip_laion_cc_sbu_558k.json` |
| Batch size | 32 |
| Epochs | 3 |
| 初始 LR | `1e-4` |
| 最低 LR | `1e-5` |
| Max text length | 32 |
| Workers | 4 |
| 每 epoch steps | 15,697 |
| 实际每 epoch 样本 | 502,304，drop_last 丢弃 12 条 |
| 可训练模块 | MLP projector |
| 可训练参数 | 4,525,824 / 1,599,204,052（0.28%） |
| 冻结模块 | DINO、SigLIP、Mamba、proprio encoder、action head |
| Loss | 自回归 next-token cross entropy，`ignore_index=-100` |

注意：部分代码注释和 README 将 Stage 1 写成 InfoNCE，但当前实际执行的 `training/stage1_align.py` 使用的是 masked causal language modeling cross entropy。日志中的 `loss` 与 `clm_loss` 完全相同，也证明本次没有使用 InfoNCE。

### Stage 2

| 参数 | 值 |
|---|---|
| 数据 | `stage2_mixed_data.json` |
| 数据组成 | COCO + Open3D-VQA |
| Batch size | 16 |
| Epochs | 5 |
| 初始 LR | `2e-4` |
| 最低 LR | `2e-5` |
| Max text length | 128 |
| Workers | 4 |
| 每 epoch steps | 12,995 |
| 实际每 epoch 样本 | 207,920，drop_last 丢弃 13 条 |
| Stage 1 初始化 | `checkpoints/stage1/best.pth` projector |
| 可训练模块 | MLP projector + Mamba LoRA |
| 可训练参数 | 7,229,184 / 1,601,907,412（0.45%） |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| LoRA targets | `in_proj`, `x_proj`, `dt_proj` |
| LoRA bias | none |
| 冻结模块 | DINO、SigLIP、proprio encoder、action head |
| Loss | 自回归 next-token cross entropy，`ignore_index=-100` |

## 3. Epoch Loss 记录

### Stage 1

| Epoch | Train loss | Val loss | Val PPL | Val-Train gap | Epoch 后 LR |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.9053 | 3.6121 | 37.6893 | -0.2932 | 7.75e-5 |
| 2 | 3.4662 | 3.4246 | 31.2480 | -0.0416 | 3.25e-5 |
| 3 | 3.2928 | **3.3559** | **29.1997** | +0.0631 | 1.00e-5 |

Stage 1 从 Epoch 1 到 3：

- Train loss 下降 15.68%。
- Val loss 下降 7.09%。
- Val PPL 下降 22.53%。
- Epoch 1 日志前 50 个窗口平均 loss 为 4.5750，后 50 个窗口为 3.6447，下降 20.33%。
- Epoch 2、3 内部首尾改善仅 2.81% 和 1.07%，说明主要对齐收益集中在首个 epoch，后续进入细化阶段。

### Stage 2

| Epoch | Train loss | Val loss | Val PPL | Val-Train gap | Epoch 后 LR |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.8209 | 1.7244 | 5.6392 | -0.0965 | 1.83e-4 |
| 2 | 1.6858 | 1.6784 | 5.3861 | -0.0074 | 1.38e-4 |
| 3 | 1.6261 | 1.6580 | 5.2789 | +0.0319 | 8.22e-5 |
| 4 | 1.5791 | 1.6510 | 5.2428 | +0.0719 | 3.72e-5 |
| 5 | 1.5432 | **1.6494** | **5.2348** | +0.1062 | 2.00e-5 |

Stage 2 从 Epoch 1 到 5：

- Train loss 下降 15.25%。
- Val loss 下降 4.35%。
- Val PPL 下降 7.17%。
- Val loss 每轮改善依次约为 0.0460、0.0204、0.0070、0.0016，收益快速递减。
- Epoch 4 到 Epoch 5 的 val loss 只改善 0.097%，但 train loss 继续下降 2.27%。
- Val-Train gap 从 -0.0965 增长到 +0.1062，表明 Epoch 3 后出现轻微但持续增加的过拟合趋势。

## 4. 批次 Loss 稳定性

日志每 20 steps 记录一次 loss。

| 阶段/Epoch | 记录点 | Mean | Std | Min | Max | 前 50 均值 | 后 50 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1-E1 | 784 | 3.9083 | 0.3270 | 3.2484 | 6.2930 | 4.5750 | 3.6447 |
| S1-E2 | 784 | 3.4732 | 0.1943 | 2.9072 | 4.1204 | 3.5205 | 3.4217 |
| S1-E3 | 784 | 3.3077 | 0.1902 | 2.8044 | 3.9063 | 3.2970 | 3.2619 |
| S2-E1 | 649 | 1.8235 | 0.1368 | 1.4742 | 2.3829 | 2.0420 | 1.7494 |
| S2-E2 | 649 | 1.6868 | 0.1000 | 1.2957 | 1.9746 | 1.6910 | 1.6867 |
| S2-E3 | 649 | 1.6287 | 0.1010 | 1.2683 | 1.9121 | 1.6259 | 1.6397 |
| S2-E4 | 649 | 1.5869 | 0.1017 | 1.2139 | 1.8434 | 1.5939 | 1.5764 |
| S2-E5 | 649 | 1.5453 | 0.0955 | 1.2392 | 1.8545 | 1.5455 | 1.5532 |

没有出现 loss 爆炸、NaN 或训练后期波动放大的迹象。Stage 2 的标准差从 0.1368 降至约 0.10，说明优化过程稳定；Epoch 3 和 5 内部后 50 个窗口略高于前 50 个窗口，但幅度不足 1%，更像混合数据难度差异造成的批次噪声。

## 5. 训练效率

| 指标 | Stage 1 | Stage 2 |
|---|---:|---:|
| 每 20 steps 平均耗时 | 约 9.91 秒 | 约 6.10 秒 |
| 平均 step/s | 约 2.02 | 约 3.28 |
| 估算 sample/s | 约 64.6 | 约 52.5 |
| 典型显存 | 约 8.3 GB | 约 8.0 GB |

Stage 1 batch 更大但文本更短、只训练 projector；Stage 2 文本长度增加到 128，并保存 Mamba LoRA 反向图，因此 sample/s 略低是合理的。

## 6. 当前效果判断

### 正面信号

1. 两个阶段的 train loss、val loss 和 val PPL 均持续下降，没有发散。
2. Stage 1 的 PPL 改善明显，说明视觉 projector 已学到能支持语言预测的有效对齐。
3. Stage 2 将 val PPL 从 5.64 降到 5.23，LoRA SFT 在指令数据上继续产生稳定收益。
4. Batch loss 方差逐步下降，训练过程稳定。
5. Stage 2 最终 loss 没有突然反弹，checkpoint 可正常保存。

### 限制与风险

1. **不能只凭 loss 判断视觉问答质量。** CLM loss 下降不直接等价于描述准确、减少幻觉或具备空间理解，必须做真实图片推理。
2. **Stage 2 已接近平台期。** Epoch 4→5 的验证收益只有 0.0016，继续增加 epoch 很可能只扩大泛化差距。
3. **验证只跑 100 batches。** Stage 1 最多覆盖 3,200/55,812 个验证样本（约 5.7%）；Stage 2 最多覆盖 1,600/23,103（约 6.9%）。当前 val loss 是方向性指标，不是完整验证集指标。
4. **随机划分没有固定 seed。** `random_split` 未传 generator，重跑时 train/val 划分会变化，无法严格复现实验数值。
5. **验证集来自同一混合数据随机划分。** 它衡量同分布泛化，不代表对新场景、不同无人机视角或 OOD 图像的效果。
6. **Checkpoint 体积过大。** 只有 0.28%/0.45% 参数可训练，但保存的是完整模型状态，单个 checkpoint 超过 3.5 GB。后续建议额外保存 projector-only 与 LoRA adapter-only 权重。

## 7. 结论与建议

当前结果可以判断为：**优化成功、对齐有效、Stage 2 已收敛，但真实视觉语言质量仍需推理验证。**

建议下一步：

1. 使用 Stage 2 `best.pth` 对 COCO 和 Open3D-VQA 各抽取至少 20 张未见图片，检查描述正确性、目标识别、空间关系与幻觉。
2. 报告 BLEU/CIDEr 等文本指标前，优先增加 BERTScore 或人工评分，因为开放式描述存在多种正确答案。
3. 下一次训练设置固定 Python、NumPy、PyTorch seed，并保存 train/val 索引。
4. 将 `max_val_steps` 设为 `None` 或使用固定、覆盖更广的验证子集。
5. Stage 2 可考虑 early stopping patience=1；本次 Epoch 4 已非常接近 Epoch 5 最优值。
6. 增加 adapter-only checkpoint，减少归档与部署时间。

本地真实推理将在 Stage 2 checkpoint 完成下载及 SHA-256 校验后执行，推理输出另存到 `checkpoints/remote_backup/inference/stage2_vqa.txt`。
