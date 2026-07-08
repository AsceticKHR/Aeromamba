# AeroMamba Stage3 轨迹相似问题诊断报告

## 1. 现象概述

修复 UE 厘米单位输出后，UAV-Flow-Eval 的轨迹不再原地不动，但出现了新的明显问题：不同任务、不同指令生成的轨迹高度相似。

当前观察到的典型轨迹特征：

- 每个任务基本都跑满 100 step；
- 每步移动约 13.4–13.7 cm；
- 最终 yaw 基本都在 -70° 左右；
- 不同任务的 step direction cosine 大多数高于 0.99；
- 指令包括 turn、ascend、left、orbit、navigate 等，但轨迹形态仍接近固定弧线。

这说明当前问题不是简单的后处理尺度错误，而是 Stage3 策略出现了动作模式塌缩。

## 2. 已执行的诊断测试

### 2.1 轨迹相似度检查

对 `eval_results/aeromamba_full_fixed_20260708_095433` 中已有轨迹进行统计，发现：

| 指标 | 结果 |
|---|---|
| 已检查轨迹 | 11 条 |
| 每条轨迹长度 | 100 step |
| 单步位移 | 约 13.5 cm |
| 最终 yaw | 约 -70° |
| 方向相似度 | 多数 `cos > 0.99` |

代表性轨迹：

```text
Turn to person        -> last ≈ (1263.9, -57.4, 171.5), yaw≈-70°
Move left             -> last ≈ (1272.4, -52.1, 79.9), yaw≈-70°
Ascend 5m             -> last ≈ (1257.9, -72.4, 52.5), yaw≈-70°
Rotate 105° right     -> last ≈ (1261.5, -72.0, 99.0), yaw≈-70°
Orbit person          -> last ≈ (1272.0, -55.1, 8.9), yaw≈-70°
```

结论：轨迹具有少量 z 差异，但 x/y/yaw 主趋势高度一致。

### 2.2 干净离线消融测试

为了避免 Flask server 的 temporal ensemble 污染，直接在远端加载 checkpoint：

```text
/root/autodl-tmp/Aeromamba/checkpoints/base384_mamba2_resampler_20260630_142642/stage3_b24_w24/best.pth
```

并分别改变文本、图像、proprio 输入，观察模型原始动作输出。

#### 文本消融

不同指令会改变 Mamba global token，但 action 变化极小：

| 输入变化 | `d_global` | `d_action` |
|---|---:|---:|
| Move left | 3.04 | 0.000256 |
| Ascend | 3.35 | 0.000368 |
| Rotate right | 3.10 | 0.000252 |
| Orbit | 3.63 | 0.000418 |

说明文本信息确实进入了 Mamba 表征，但 ActionHead 几乎没有利用这些差异。

#### 图像消融

改变图像颜色块、左右区域、噪声图后：

| 输入变化 | `d_global` | `d_action` |
|---|---:|---:|
| red left | 0.0054 | 0.000000 |
| blue right | 0.0184 | 0.000000 |
| green top | 0.0045 | 0.000000 |
| noise | 0.0087 | 0.000000 |

说明当前 Stage3 策略几乎完全不使用视觉差异。

#### Proprio 消融

改变 proprio 后 action 有变化，但依然很小：

| 输入变化 | `d_global` | `d_action` |
|---|---:|---:|
| x + 1 | 0.010 | 0.000700 |
| y + 1 | 0.011 | 0.000461 |
| z + 1 | 0.009 | 0.000715 |
| yaw + 90° | 0.026 | 0.000974 |

说明 proprio 有轻微作用，但不足以形成任务级策略。

### 2.3 ActionHead 单独测试

进一步单独测试 `UAVDynamicsActionHead`：

- 输入 zero global token 时，输出与真实模型输出不同；
- 输入 random global token 时，动作会显著变化；
- 说明 ActionHead 本身不是完全死掉的 bias head。

因此更准确的判断是：

```text
ActionHead 有表达能力，
但 Stage3 训练后真实 multimodal global token 落在很窄的流形上，
并且 ActionHead 学到的是该流形上的平均动作。
```

## 3. 模块级归因

### 3.1 不是单纯后处理问题

后处理尺度问题已经修复：

- 训练输出是归一化动作；
- UE 使用厘米；
- server 已加 `output_pos_scale=10000`；
- 输出已从 `0.001` 级变成 `10 cm` 级。

修复后模型能移动，但轨迹仍高度相似，因此不是单纯单位问题。

### 3.2 视觉分支没有被 Stage3 有效使用

图像消融显示：

```text
image change -> global token change 极小 -> action change 约 0
```

可能原因：

1. Stage3 训练目标并不强制依赖图像；
2. UAV-Flow parquet 中的 `log` 轨迹可能更多反映预记录运动，而不是当前图像条件下的任务决策；
3. Vision encoder frozen，projector/resampler 虽可训练，但 loss 可以通过学习平均轨迹快速下降；
4. `SmoothL1` 对小尺度标签很容易被平均解满足。

### 3.3 文本分支进入了 Mamba，但没有进入动作决策

文本消融显示 `d_global≈3`，但 `d_action≈0.0003`。这说明：

```text
文本差异存在于 hidden state，
但 ActionHead 对这些差异不敏感。
```

最直接原因是 Stage3 数据加载器使用固定 instruction：

```text
Navigate the UAV along the planned trajectory.
```

因此 Stage3 从未学习：

- ascend 对应 z 增大；
- left 对应 y 方向改变；
- rotate 对应 yaw 改变；
- face person / dog 需要视觉目标定位；
- orbit 需要圆周轨迹。

Stage2 学过 VQA/指令理解，但 Stage3 的 action loss 没有把真实任务指令和动作监督绑定起来，所以动作头无法把语言语义映射成控制策略。

### 3.4 ActionHead 学到平均短期运动

训练日志中 Stage3 loss 很低：

```text
val_loss=0.0022
val_l1_err=0.0063
```

但低 loss 不等于任务成功。当前 loss 的本质是拟合局部 waypoint，而不是达成评估任务。

由于数据标签短期、平滑、尺度小，`DynamicsActionHead + SmoothL1` 很容易学到平均动作：

```text
dx ≈ 0.0012
dy ≈ 0.00075
dyaw ≈ -0.012 rad
```

映射到 UE 后就是：

```text
每步前进约 12 cm，偏移约 7 cm，右转约 0.7°
```

这正好解释了评估中的固定弧线。

## 4. 当前问题定位结论

优先级从高到低：

| 模块 | 问题程度 | 结论 |
|---|---:|---|
| Stage3 数据/监督定义 | 极高 | 核心根因 |
| ActionHead 训练目标 | 高 | 学到平均轨迹而非任务策略 |
| 文本到动作绑定 | 高 | Stage3 固定 instruction 导致语言无效 |
| 视觉到动作绑定 | 高 | 图像变化对动作几乎无影响 |
| 后处理单位 | 已修复 | 不是当前主要问题 |
| ActionHead 结构本身 | 中低 | 有表达能力，不是完全坏掉 |
| Mamba2 backbone | 中低 | global token 能响应文本，但动作头未利用 |

一句话结论：

```text
当前 Stage3 checkpoint 的主要问题不是模型跑不动，
而是 Stage3 训练目标把任务条件策略退化成了平均局部轨迹预测。
```

## 5. 优化建议

### 5.1 立即停止当前 full eval

当前评估结果只会证明模型塌缩，不建议继续消耗时间。

建议保留前 10–20 条轨迹作为 failure case，后续训练后对比。

### 5.2 重构 Stage3 训练样本

Stage3 样本必须包含真实任务语义：

```text
image
真实 instruction
current pose / proprio
target object or target pose
future action / waypoint
```

不能继续使用固定 instruction。

建议 prompt 模板：

```text
Current UAV state: x={x}, y={y}, z={z}, yaw={yaw}.
Instruction: {instruction}
Predict the next local waypoint.
```

### 5.3 对齐标签坐标系

当前 eval 期望 server 返回 episode-local waypoint。训练期最好也直接监督同一含义：

```text
gt_action[t] = episode-local future pose relative to initial pose
```

或者如果训练 current-local delta，则 server 必须严格执行：

```text
current-local delta -> rotate by current yaw -> add current episode pose
```

建议长期统一为 episode-local waypoint，避免反复转换出错。

### 5.4 加强条件依赖，避免平均动作

训练中加入以下辅助损失：

1. Direction classification loss  
   预测主方向类别：left/right/forward/back/up/down/rotate/orbit。

2. Delta magnitude loss  
   单独约束水平距离、垂直距离、yaw 变化。

3. Endpoint loss  
   对 chunk 最后一个 waypoint 加更高权重。

4. Contrastive action loss  
   同一图像不同 instruction 的动作必须不同。

5. Stop / done head  
   让模型知道什么时候到达，而不是固定跑满。

### 5.5 改 ActionHead 输入

当前 ActionHead 只吃 `global_token + proprio`。建议加入显式条件 token：

```text
action_context = concat(
    last_hidden,
    pooled_text_hidden,
    pooled_visual_hidden,
    proprio_embedding
)
```

再送入 dynamics head。

这样可以避免最后一个 visual token 把文本差异压掉。

### 5.6 分阶段重新训练 Stage3

建议不要立刻全量长训，先做小闭环：

1. 取 2k–10k 条带真实 instruction 的样本；
2. 训练 1–2 小时；
3. 每 1000 step 固定跑 10 条 eval smoke；
4. 检查轨迹是否按指令分化；
5. 再扩大到全量。

### 5.7 增加训练期监控指标

除了 loss，必须记录：

```text
action_std_by_batch
action_mean_by_instruction_type
text_ablation_delta
image_ablation_delta
endpoint_error_cm
yaw_error_deg
```

如果 `action_std` 很低，即使 loss 下降也应判定为塌缩。

## 6. 推荐下一步

建议下一步直接修改代码：

1. 新增 `UAVFlowEvalStyleDataset`；
2. 从 `test_jsons` / UAV-Flow metadata 中读取真实 instruction；
3. 统一 episode-local waypoint 标签；
4. 修改 ActionHead 输入为 text/vision/proprio 显式融合；
5. 加入 endpoint loss 和 direction auxiliary loss；
6. 先用小数据训练一个 `stage3_evalstyle_probe`；
7. 跑 10 条 smoke eval，确认轨迹不再同质化。

如果 probe 显示不同指令开始产生明显不同轨迹，再启动全量 Stage3。
