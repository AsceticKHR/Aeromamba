# AeroStream 执行改动清单与训练计划（绑定轮 + 流式轮）

- 日期：2026-07-15
- 依据：`docs/AEROSTREAM_TOPTIER_DESIGN_PROPOSAL.md` v1.5 §14/§15；273 条轨迹诊断（`UAV-Flow-Eval/scripts/diagnose_trajectories.py`）
- 原则：**Stage1/Stage2 checkpoint、动作统计、数据集全部复用，不重训**；只重训 Stage3（两轮）。

---

## 0. 复用/新增总览

| 资产 | 处置 | 说明 |
|---|---|---|
| Stage1 checkpoint | **复用** | 不涉及 |
| Stage2 checkpoint（远端 `full_stage_20260710_125315/stage2/best.pth`） | **复用** | 绑定轮的起点，不重训 |
| Stage3 v2 checkpoint（`stage3_v2_20260714_082604/best.pth`） | 保留为对照基线 | 绑定轮**不从它续训**（见 §2.6 理由） |
| `action_stats_k8.json` | **复用** | z-score 统计不变（通道加权在 loss 侧做，不动统计量） |
| 训练数据 `/root/autodl-tmp/datasets/uav-flow` | **复用** | 绑定标签在线提取，无需重新导出 |
| `UAVFlowDataset`（`data/dataset.py`） | **修改** | 加绑定标签 + 按类过采样 |
| `aero_collate_fn` | **修改** | 透传新标签字段 |
| `aero_action_loss`（`model/action_head.py`） | **修改** | 加通道权重 + 样本幅度权重；`lambda_acc` 已实现仅需启用 |
| `AeroMambaVLA`（`model/uav_mamba_vla.py`） | **修改** | 挂绑定头旁路 + 流式 step 前向 |
| `Stage3Trainer`（`training/stage3_action.py`） | **修改** | 绑定损失合并 + CLI |
| `inference/server.py` | **修改**（流式轮） | cache_params 增量推理 |
| 绑定分类头 | **新增** | `model/binding_head.py`，仅训练期 |
| 轨迹连续采样 + TBPTT | **新增**（流式轮） | 新 Dataset/Sampler + Trainer 钩子 |
| 诊断/评测脚本（diagnose_trajectories / run_metric / eval_success_rate / restart_eval.ps1 / start_infer_server_wsl.sh） | **复用** | 验收闭环不变 |
| `UAVFlowHFDataset` 全零时间通道 | **卫生修复** | 半天，非阻塞（防未来 `--hf_dataset` 误用） |

---

## 1. 绑定轮（Round A）——改动明细

### A1【新增】`model/binding_head.py`：指令绑定分类头

- 输入：与动作头相同的 `h_last`（[B, D_m]）。
- 四组输出（独立 linear 或共享一层 MLP 后分叉）：
  - `motion_class`：10 类（move/shift/turn/rotate/ascend/descend/land/approach/pass/surround）
  - `yaw_sign`：3 类（左 / 右 / 不转，阈值建议 |Δyaw_chunk| > 2°）
  - `dz_sign`：3 类（升 / 降 / 平，阈值建议 |Δz_chunk| > 0.05 m）
  - `magnitude_bin`：8 箱对数分箱，chunk 末端位移范数，边界 [0.1, 0.2, 0.5, 1, 2, 5, 10, 30] m
- 输出层零初始化；模块参数量 < 1M。
- **仅训练期前向**；推理路径（`predict_step`）完全不碰，零部署开销。

### A2【修改】`data/dataset.py`：标签在线提取 + 按类过采样

标签提取（`UAVFlowDataset.__getitem__` 内，零标注成本）：

- `yaw_sign` / `dz_sign` / `magnitude_bin`：直接从当前样本的 `gt_action`（[K,4] 物理单位）差分/求和得出——**flip 增强后 gt_action 已翻转，标签自动一致，无需额外镜像逻辑**。
- `motion_class`：从指令正则模板推断（UAV-Flow 指令模板性强：orbit/circle→surround、land→land、ascend/rise/lower/descend→ascend_descend、turn to face→turn、rotate/degrees clockwise→rotate、pass/through→pass、approach/toward→approach、move back/retreat→retreat、shift/move…meters at…angle→shift/move）。落不进模板的样本给 `ignore_index=-100`，不参与该项损失。**注意：模板必须在 `mirror_instruction_lr` 之后应用**（镜像不改变类别，只改方向词，安全）。
- 按类过采样：现有 turn 过采样机制（`self.index` 复制）扩展——构造期先对每条轨迹推断 motion_class，对 Move/Shift/Ascend/Descend/Surround/Rotate 类的全部窗口按 `--oversample_class_factor`（建议 3）复制索引。与现有 `--oversample_turn_factor` 叠加取 max，不叠乘。

`aero_collate_fn`：透传 `binding_labels`（dict of LongTensor）。

### A3【修改】`model/action_head.py`：损失加权

`aero_action_loss` 增加两个可选参数：

- `channel_weights`：[4] 张量，z-space L1 主项与 endpoint 项按维度加权，默认 `[1, 1, 1, 1]`，绑定轮用 `[1, 1, 2.5, 2.5]`（dz、dyaw 加权）。
- `sample_weights`：[B] 张量，主项按样本加权，来自 `1 + log1p(‖chunk 末端位移‖_m)`（大位移样本梯度更大，对抗均值坍缩）。在 dataset 侧随 batch 给出。

### A4【修改】`model/uav_mamba_vla.py` + `training/stage3_action.py`

- `AeroMambaVLA.forward(return_loss=True)` 路径：`h_last` 分叉进绑定头，四项交叉熵（各自 `ignore_index=-100`）加权合入总损失；`loss_detail` 增加四项精度即时打印（yaw_sign_acc 等，训练日志直接可见验收指标趋势）。
- `Stage3Trainer.compute_loss`：透传 `binding_labels` 与新 CLI 参数。
- 新 CLI：`--lambda_binding 0.3`（四项交叉熵总权重，内部均分）、`--oversample_class_factor 3`、`--channel_weight_z 2.5 --channel_weight_yaw 2.5`、`--magnitude_sample_weight 1`。
- checkpoint 兼容性：绑定头是新增模块，`load_state_dict(strict=False)` 下从 Stage2 checkpoint 加载时按 missing 处理，与现有动作头/proprio 的加载惯例一致，**无需改加载逻辑**。

### A5【启用】加速度损失

`aero_action_loss` 的 `lambda_acc` 分支已实现（动作序列一阶差分 L1；动作本身是每步位移，其差分即加速度，与 AnoleVLA 语义一致）。只改启动脚本：`--lambda_acc 0.25`（AnoleVLA 两阶段可简化为单阶段直接启用，若 loss 不稳再降 0.1）。

### A6【新增】`scripts/probe_instruction_sensitivity.py`：训练期探针

固定 16 张评测帧 × 8 条方向对立指令（左/右、升/降、前/停），对每个 checkpoint 输出动作差异范数矩阵。绑定轮训练中每 save_every 跑一次（本地 4060/WSL 执行，不占 4090）。判据：对立指令对的输出差异应随训练单调上升；不动说明语言仍未进入动作通路，提前止损。

### A7【卫生修复】`UAVFlowHFDataset`

`_state8_delta_from_preprocessed` 与 `_velocity4_from_preprocessed(idx=0)` 的全零行为：导出侧为每 row 附 anchor 前一帧后合法计算，或在类构造时打印显著 WARNING + docstring 更正（删除"server 发零 delta"的过时表述）。不阻塞主线。

### A8 绑定轮训练命令（新增 `scripts/run_stage3_v3_binding.sh`，从 v2 脚本复制修改）

```bash
python training/stage3_action.py \
  --arch_preset aeromamba_opt \
  --data_root /root/autodl-tmp/datasets/uav-flow \
  --stage2_ckpt /root/autodl-tmp/Aeromamba/checkpoints/full_stage_20260710_125315/stage2/best.pth \
  --stage3_train_lora \
  --action_stats $DATA_ROOT/action_stats_k8.json \
  --oversample_turn_factor 3 --oversample_class_factor 3 \
  --aug_flip --aug_vision \
  --lambda_smooth 0.0 --lambda_endpoint 0.25 --lambda_direction 0.5 \
  --lambda_acc 0.25 --lambda_binding 0.3 \
  --channel_weight_z 2.5 --channel_weight_yaw 2.5 --magnitude_sample_weight 1 \
  --batch 48 --epochs 2 --lr 7.5e-5 --workers 32 \
  --chunk_size 8 --pos_scale 100.0 --max_text_len 64 \
  --max_val_steps 200 --log_every 100 --save_every_steps 2000
```

**起点选择说明**：从 **Stage2 checkpoint 重训**（不从 stage3_v2 续训）。理由：过采样与加权改变了数据/损失分布，续训会让 v2 已坍缩的先验（前飞、dz≈0）与新分布对抗，归因混乱；从 Stage2 起点重训与 v2 完全同构，改动可干净归因（也是论文消融叙事需要）。成本 ~11-13h（2 epoch，与 v2 相同量级，过采样后样本数增约 15-25%）。

---

## 2. 流式轮（Round B）——改动明细

### B1【新增，可立即做】`scripts/test_streaming_smoke.py`：cache_params 冒烟

纯工程验证，不训练：370M 模型上用 `mamba_ssm` 的 `InferenceParams`（或 transformers `cache_params`）做两帧续推，比对与全序列前向的输出数值一致性（容差 1e-3）。验证增量 token 喂入 `[state|delta|vision]`（首帧额外含 text）的链路。**已确认无前置阻塞**（proprio 语义正确，见 proposal §15）。

### B2【修改】`model/uav_mamba_vla.py`：流式前向接口

- 新增 `stream_reset()` / `stream_step(pixel_values, state, delta_state, input_ids=None)`：内部持有 cache_params；`input_ids` 仅首帧（或周期刷新时）传入。
- 现有 `predict_step` 保留不动（回退路径 + 对照基线）。

### B3【修改】`inference/server.py`

- `--stream 1` 开关：`/reset` 清 cache_params；`/predict` 走 `stream_step`；`exec_horizon` 强制 1（每步重查询，μVLA 纪律）。
- 三重防御（vel_per_step / clip_sigma / max_step）保留不动。
- 关闭 `--stream` 时行为与现状完全一致（A/B 对照零成本）。

### B4【新增】轨迹连续采样 + TBPTT 训练

- `data/dataset.py` 新增 `UAVFlowStreamDataset`（或 `UAVFlowDataset` 的 sequential 模式）：按轨迹顺序产出连续帧，batch 槽位与轨迹绑定（`torch.utils.data.Sampler` 实现槽位调度）。
- `training/stage3_action.py` 新增 TBPTT 分支：K=2 截断——每 K 步 detach 状态继续；梯度检查点 + bf16；batch 48→16。
- 起点：**绑定轮 best checkpoint 续训**（流式是能力叠加不是分布修正，续训成本低且增益可归因），1~2 epoch。

### B5 验收

复跑同一诊断脚本，新增两个指标：末段速度衰减（停止判断，基线：末段 15~26cm/步无衰减）、Surround 角度覆盖（曲率，基线≈0°）。加上绑定轮三指标共五项。

---

## 3. 训练计划与排期（4090 训练 ∥ 本地 4060/WSL 评测）

| 时间 | 4090（远端） | 本地（并行） | 产出/验收 |
|---|---|---|---|
| D1 | — | A1-A4 编码 + `py_compile` + 单元冒烟（DummyUAVDataset 过一遍 forward/loss）；B1 流式冒烟；A7 卫生修复 | 代码就绪、流式链路确认 |
| D2-D3 | **绑定轮训练**（~13h，2 epoch） | A6 探针每 2000 步跑一次；中期看 yaw_sign_acc/探针散度，异常提前止损 | 绑定轮 best.pth |
| D4 | （空闲/短程消融 ±绑定头） | 抽 slim → WSL 推理服务 → 273 条闭环评测 → 诊断脚本 | **验收 1**：冻结率 <60/273、yaw 符号 >20/27、dz 非零 >12/19（未过线则调 λ/权重做第二次短程） |
| D5-D6 | **流式轮训练**（TBPTT K=2 续训，~10h） | B3 server 流式改造 + 本地 A/B（stream 0/1） | 流式轮 best.pth |
| D7 | 短程消融（K∈{1,2}、exec_horizon∈{1,4}） | 273 条闭环评测 ×2（stream on/off） | **验收 2**：末段速度衰减出现、Surround 角度覆盖 >90°、nDTW 整体对比 v2 基线（0.0771） |

里程碑判据：
- 绑定轮达标 → 流式轮直接叠加；绑定轮部分达标（如仅 dz 改善）→ 流式轮照常进行（两者独立），未达标项回炉调权重；
- 流式轮若训练不稳（loss 尖峰）→ K=2 降 K=1（等价于带状态的单步训练）先稳住，再升 K。

## 4. 明确不做的（本计划范围外）

- Stage1/Stage2 重训、Stage2 数据改动（度量分箱模板等归入后续 S2 轮）；
- 终端检索补丁、视觉塔尾 LoRA、576 token 直通、proprio 后置 A/B——全部留在消融队列，等两轮主线出结果；
- 等参数 Transformer 对照自训、双塔、K=8（proposal 已裁）。
