# 改良版 AeroMamba 架构设计报告与旧版对比

## 1. 设计背景

AeroMamba 的目标是面向 UAV-Flow 这类无人机视觉-语言-动作任务，在较低显存和较高推理效率下完成：

- 第一视角图像理解；
- 自然语言导航指令理解；
- 当前无人机状态建模；
- 未来局部轨迹 waypoint 预测；
- 与 UAV-Flow-Eval / UnrealCV 仿真闭环对接。

旧版 AeroMamba 已经能跑通多阶段训练，但存在若干问题：视觉侧较重、token 序列较长、动作头缺少动力学先验、Stage3 训练慢且 GPU 利用率低、checkpoint 与数据路径容易混淆。改良版围绕“轻量视觉、Mamba2 语义融合、动力学动作头、标准三阶段训练、标准评估接口”进行重构。

## 2. 改良版总体结构

改良版整体数据流如下：

```text
FPV Image
  -> SigLIP2-Base-384 frozen vision encoder
  -> MLP Projector
  -> Perceiver Resampler: 576 visual tokens -> 32 visual queries

Instruction
  -> tokenizer
  -> text tokens

Proprio [x, y, z, yaw]
  -> ProprioEncoder
  -> proprio token

[text tokens | proprio token | visual tokens]
  -> Mamba2-370M backbone + LoRA adapters
  -> last hidden global token
  -> Dynamics Action Head
  -> K future waypoints [K, 4]
```

核心配置：

| 模块 | 改良版配置 | 作用 |
|---|---|---|
| Vision encoder | `siglip2_base_384` | 冻结视觉特征提取 |
| Language backbone | `mamba-2-370m` | 多模态序列融合 |
| Projector | MLP projector | 映射视觉 hidden 到 Mamba hidden |
| Resampler | Perceiver, 32 queries | 压缩视觉 token，降低序列长度 |
| State encoder | ProprioEncoder | 编码 UAV 当前状态 |
| Action head | DynamicsActionHead | 预测连续 waypoint 轨迹块 |
| Adaptation | LoRA r=16, alpha=32 | 低成本适配 Mamba2 |
| Output | `[K, 4]` | `[x, y, z, yaw]` |

## 3. 关键模块设计

### 3.1 SigLIP2-Base-384

旧版使用过 DINO + SigLIP 或更复杂的双视觉编码思路。改良版改为单 SigLIP2-Base-384，主要原因是：

- 单视觉 encoder 更简单，部署和训练更稳；
- SigLIP2 本身具备较好的图文对齐能力；
- frozen 使用可以节省显存并避免视觉 backbone 被 UAV-Flow 过拟合；
- 相比双编码器，减少了 projector 输入复杂度和数据加载开销。

SigLIP2-Base-384 会产生约 576 个视觉 patch tokens。直接送入 Mamba2 会带来较长序列，因此后续加入 Perceiver Resampler。

### 3.2 MLP Projector

Projector 的任务是：

```text
SigLIP2 hidden space -> Mamba2 hidden space
```

它不是动作模块，而是视觉-语言对齐模块。Stage1 主要训练 projector，使 frozen 的 Mamba2 能读懂 frozen 的 SigLIP2 输出。

### 3.3 Perceiver Resampler

Resampler 将大量视觉 patch tokens 压缩为固定数量的 visual queries：

```text
576 visual tokens -> 32 visual queries
```

收益：

- 缩短 Mamba 输入序列；
- 降低显存；
- 提高训练速度；
- 固定视觉 token 数量；
- 更适合实时 VLA 推理。

代价是细粒度空间信息可能被压缩，但 UAV-Flow 主要评估导航与轨迹预测，并非像素级分割，因此该取舍合理。

### 3.4 ProprioEncoder

无人机动作不仅取决于图像和语言，还取决于当前相对状态：

```text
proprio = [x, y, z, yaw]
```

ProprioEncoder 将这个连续状态编码为一个 Mamba token，并插入序列：

```text
[text tokens | proprio token | visual tokens]
```

这种顺序让最后的视觉 token hidden state 同时聚合语言、状态和视觉上下文，适合 Mamba 的顺序扫描机制。

### 3.5 Mamba2-370M + LoRA

旧版使用较小的 Mamba-130M。改良版升级为 Mamba2-370M，以获得更强的序列建模和语言理解能力。

但 Mamba2 base weights 不全量微调，而是使用 LoRA：

```text
W' = W + B A
r = 16
alpha = 32
alpha / r = 2
```

这样可以在较低显存和较少参数更新的情况下，让 Mamba2 适配视觉问答和 UAV action 任务。

### 3.6 Dynamics Action Head

旧版普通 MLP action head 容易把 K 个 waypoint 当作彼此独立的点，缺少轨迹连续性。改良版使用 DynamicsActionHead：

```text
global token + proprio -> recurrent hidden state
for step in K:
    GRUCell(prev_action, step_context)
    predict delta action
```

优势：

- 结构上建模时间连续性；
- 前一个动作会影响后一个动作；
- 更符合 UAV 的速度、加速度、yaw-rate 连续变化；
- 减少 waypoint 抖动；
- 比 token-by-token 自回归解码更快。

## 4. 三阶段训练流程

### 4.1 Stage 1：Projector / Resampler Alignment

目标：让 frozen SigLIP2 的视觉特征进入 frozen Mamba2 可理解的语言空间。

数据：

```text
LLaVA-Pretrain / blip_laion_cc_sbu_558k.json
```

训练模块：

```text
trainable:
- MLP Projector
- Perceiver Resampler

frozen:
- SigLIP2
- Mamba2 base
- ProprioEncoder
- ActionHead
```

主要参数：

```text
vision_type          siglip2_base_384
mamba_type           mamba-2-370m
token_resampler      perceiver
num_visual_queries   32
resampler_layers     2
resampler_heads      8
batch                8
lr                   1e-4
epochs               1
workers              6
max_text_len         64
```

Loss：

```text
CLM Cross Entropy
```

也就是让模型根据图像上下文预测 caption token。因为 SigLIP2 和 Mamba2 都 frozen，loss 下降主要说明 projector / resampler 正在把视觉特征对齐到 Mamba2 语言空间。

当前结果：

```text
train loss = 8.0051
val loss   = 7.6157
val ppl    = 2171.38
```

评价：Stage1 loss 偏高，但它只做 1 epoch 粗对齐，后续 Stage2 能明显下降，说明该阶段起到了 warmup 作用。

### 4.2 Stage 2：VLM SFT with LoRA

目标：让模型具备视觉问答和指令理解能力。

数据：

```text
COCO VQA + Open3D-VQA mixed data
stage2_mixed_data.json
```

训练模块：

```text
trainable:
- Projector
- Perceiver Resampler
- Mamba2 LoRA adapters

frozen:
- SigLIP2
- Mamba2 base weights
- ProprioEncoder
- ActionHead
```

主要参数：

```text
stage1_ckpt          Stage1 best.pth
lora_r               16
lora_alpha           32
batch                4
lr                   5e-5
epochs               1
workers              6
max_text_len         64
```

Loss：

```text
Masked CLM Cross Entropy
```

prompt / user token 被 mask 为 `-100`，只对 assistant answer token 计算 loss。这样模型学习“如何回答”，而不是学习复述问题。

当前结果：

```text
train loss = 4.1250
val loss   = 3.4880
val ppl    = 43.6083
```

评价：Stage2 明显收敛，验证 loss 低于训练平均，没有明显过拟合。它为 Stage3 提供了可用的视觉指令理解初始化。

### 4.3 Stage 3：UAV-Flow Action Training

目标：根据图像、语言指令和当前 UAV 状态，预测未来 K 个局部 waypoint。

数据：

```text
UAV-Flow full parquet dataset
约 1,606,756 train / 178,528 val
```

训练模块：

```text
trainable:
- Perceiver Resampler
- ProprioEncoder
- DynamicsActionHead
- Mamba2 LoRA adapters

frozen:
- SigLIP2
- Mamba2 base weights
```

最终实际训练参数：

```text
arch_preset          uav_lite_siglip
stage2_ckpt          Stage2 best.pth
epochs               2
batch                24
workers              24
lr                   5e-5
max_text_len         64
max_val_steps        100
save_every_steps     10000
no_amp               true
```

Loss：

```text
total_loss = SmoothL1(pred_action, gt_action)
           + lambda_smooth * smoothness_loss
```

其中：

- `SmoothL1` 是主动作回归损失；
- `smoothness_loss` 是二阶差分轨迹平滑项；
- `l1_err` 是日志指标，用于观察绝对误差。

为什么用 Smooth-L1：

- 比 MSE 更抗异常值；
- 比 L1 在小误差附近更平滑；
- 适合 UAV-Flow 中急转、长尾动作、目标变化等样本。

为什么加 smoothness：

- UAV 轨迹应连续；
- 避免相邻 waypoint 抖动；
- 与 DynamicsActionHead 的结构先验互补。

当前结果：

```text
train loss = 0.0024
val loss   = 0.0022
val main   = 0.0022
val smooth = 0.0000
val l1 err = 0.0063
```

评价：Stage3 已稳定收敛。Epoch 1 和 Epoch 2 的验证 loss 基本一致，说明继续增加 epoch 的收益有限，下一步重点应转向闭环仿真评估。

## 5. 与旧版 AeroMamba 的对比

### 5.1 架构对比

| 维度 | 旧版 AeroMamba | 改良版 AeroMamba | 改进意义 |
|---|---|---|---|
| 视觉编码器 | DINO + SigLIP 或更重视觉组合 | SigLIP2-Base-384 单编码器 | 降低复杂度和显存开销 |
| 视觉 token | 大量 patch tokens 直接参与融合 | Perceiver 压缩为 32 queries | 缩短序列，提高速度 |
| 主干模型 | Mamba-130M | Mamba2-370M | 更强语义与序列建模 |
| 主干适配 | 较弱或不统一 | LoRA r=16 alpha=32 | 低成本任务适配 |
| 状态输入 | 状态利用较弱 | Proprio token 显式输入 | 更符合 UAV 控制 |
| 动作头 | MLP / 独立 waypoint 回归 | Dynamics GRU rollout | 轨迹更连续 |
| Loss | 简单回归或不稳定设置 | Smooth-L1 + smoothness | 更稳健、更符合动力学 |
| AMP | 曾有 NaN 风险 | Stage3 关闭 AMP | 稳定优先 |
| 评估接口 | 未标准化 | 对齐 UAV-Flow-Eval HTTP 协议 | 可闭环评测 |

### 5.2 训练效率对比

| 项目 | 旧版设置 | 改良后设置 | 效果 |
|---|---|---|---|
| Stage3 batch | 4 或更低 | 24 | 吞吐显著提升 |
| Stage3 workers | 6 或更低 | 24 | 降低数据加载瓶颈 |
| GPU 利用率 | 经常不足 | 可达 90-100% | 资源利用更充分 |
| 视觉序列长度 | 长 | 32 visual queries | Mamba 输入更短 |
| 训练稳定性 | 曾出现 NaN / 慢速 | 稳定完成 2 epoch | 可复现性更好 |

### 5.3 效果对比

| 指标 | 旧版问题 | 改良版结果 |
|---|---|---|
| Stage1 | 对齐较粗、视觉配置不统一 | 完成 projector/resampler warmup |
| Stage2 | 指令理解能力有限 | `val_loss=3.4880` |
| Stage3 | 慢、动作不稳定、动力学先验弱 | `val_loss=0.0022`, `val_l1_err=0.0063` |
| 推理评估 | 服务协议不清晰 | 已设计 Flask / HTTP 评估框架 |

## 6. 改良版的核心收益

### 6.1 更轻量

从双视觉编码器转向单 SigLIP2 + resampler，减少视觉计算和序列长度。

### 6.2 更稳定

Stage3 使用 FP32 / no AMP，结合 Smooth-L1，降低 NaN 和异常梯度风险。

### 6.3 更适合 UAV dynamics

DynamicsActionHead 通过 recurrent rollout 预测 waypoint，使轨迹天然具有时间连续性。

### 6.4 更容易复现

三阶段数据、checkpoint、训练参数和评估接口都更加清晰。

## 7. 当前限制与后续工作

当前仍有几个待完善点：

1. Stage1 只跑 1 epoch，视觉语言对齐仍偏粗；
2. Stage3 训练验证只使用 `max_val_steps=100`，完整验证还需要补；
3. 训练 loss 很低不代表仿真闭环一定最优；
4. 需要正式实现 AeroMamba HTTP 推理服务；
5. 需要用 UAV-Flow-Eval 计算完整 per-class nDTW；
6. 需要和 OpenVLA baseline 做同任务对比。

建议后续流程：

```text
1. 离线单样本推理，检查 checkpoint 和输出 shape。
2. 启动 WSL AeroMamba Flask server。
3. Windows 端跑 UAV-Flow-Eval smoke test。
4. 完整跑 test_jsons。
5. 用 metric.py 计算 per-class nDTW。
6. 对失败轨迹做可视化分析。
```

## 8. 总结

改良版 AeroMamba 的核心变化可以概括为：

```text
SigLIP2 单视觉编码器
+ Perceiver visual token compression
+ Mamba2-370M LoRA adaptation
+ Proprio token fusion
+ Dynamics action head
+ Smooth-L1 action loss
+ 标准化三阶段训练
+ UAV-Flow-Eval 推理协议对齐
```

相比旧版，它更轻、更稳定、更适合 UAV dynamics，也更容易进行闭环仿真评测。当前训练结果显示 Stage2 具备可用的视觉指令理解能力，Stage3 动作回归已经稳定收敛，下一步应重点推进真实仿真评估和 failure case 分析。
