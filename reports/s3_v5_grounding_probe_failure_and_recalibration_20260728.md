# S3-v5 接地探针失效与重新标定

日期：2026-07-28
对象：`checkpoints/v2_stage3_v5_sim`（Falcon-H1-1.5B-Deep + C-RADIO + xattn readout + L1，sim 数据，step 14000/20000 手动停止）
结论：**接地探针一直失效；修复后重测显示策略同时接地语言与视觉，且 `last.pth` 标定最好。此前基于 `val loss` 的选点系统性地挑走了最视觉盲的 checkpoint。**

---

## 1. 探针失效

`grounding_metrics` 用 `ids.roll(1, 0)` 在批次内互换指令。但探针批次取自
`next(iter(va_loader))`，而 `va_loader` 不打乱、`va_idx` 按轨迹顺序构建。验证集
281 条轨迹 / 7965 chunk（约 28 chunk/轨迹），因此首批 8 行全部来自同一条轨迹、
共享同一条指令 —— roll 成为恒等操作，`instr_sens` 恒为位精确 0。

证据：全量日志中 `instr=` 去重后只有 `0.0000` 一个取值，而同一函数算出的 `vis=`
正常波动于 0.0064–0.3638。

**连带污染：**

| 受影响处 | 后果 |
|---|---|
| `is_vision_grounded(0, v)` | 退化为 `v >= 0.15·v`，恒真 —— G4 接地门控全程空过，`vis_share` 恒为 1.000 |
| `resp = (i+v)/gt_spread` | 退化为纯视觉敏感度，且探针批次同轨迹，接近噪声 |
| `best_responsive.pth` | 由 step 9000 的 `resp=2.254` 噪声尖峰选出 |

代码注释中原有的合理化解释（"on sim the short smoke often routes via vision
alone (instr~0)"）是错的，已改正。

**修复**（`training/v2_stage3_action.py`）：

- 新增 `build_probe_batches()`，扫描验证批次收集指令互异的样本。
- `grounding_metrics` 只在指令确实改变的行上取均值；整批同指令时返回 `NaN`
  而非 0，使失效变响而非静默通过。
- G4 改用同一构造。
- 新增 `--mode grounding_eval`：离线重测已存 checkpoint（探针是测量，探针 bug
  不需要重训）。
- 新增 `instruction_only_spread()`：固定一张图像、遍历全部互异指令，测语言单独
  贡献的端点散布。

---

## 2. 数据侧对照：这个基准需要多少视觉

按指令分组统计验证集端点方差（228 组，无模型参与）：

| 量 | 值 |
|---|---|
| 组间散布（指令可解释） | 1.5283 m |
| 组内散布（只有视觉可解释） | 0.4198 m |
| **视觉可解释份额** | **0.215** |

即 UAV-Flow-Sim 上约 **78.5% 的端点方差由指令决定，21.5% 需要视觉**。这个数字
是判断 `vis_sens` 高低是否合理的标尺 —— 没有它，低 `vis_sens` 无法区分"模型
视觉盲"与"数据本就不需要视觉"。

---

## 3. 修复后的重测结果

8 组独立探针批次 × 8 条互异指令，均值 ± 标准差：

| checkpoint | step | val pos_err | instr_sens | vis_sens | 模型 vis_share | 语言单独散布 / GT |
|---|---|---|---|---|---|---|
| `best.pth` | 1000 | **0.2095** | 1.659±0.473 | 0.041±0.017 | **0.024** | 1.224 / 1.200 |
| `best_grounded.pth` | 1000 | 0.2095 | 1.659±0.473 | 0.041±0.017 | 0.024 | 1.224 / 1.200 |
| `best_responsive.pth` | 9000 | 0.2800 | 1.301±0.551 | 0.442±0.308 | 0.254 | 1.054 / 1.200 |
| **`last.pth`** | **14000** | 0.2389 | **1.539±0.342** | **0.506±0.252** | **0.247** | **1.190 / 1.200** |

参照系：`gt_spread = 1.6066`（换指令时 GT 端点实际差多远），数据 vis_share = 0.215。

**读法：**

1. **策略强接地语言。** `last.pth` 的 `instr_sens=1.539` 对 `gt_spread=1.607`，
   达到理想值的 96%。固定图像只变指令时预测散布 1.190 对 GT 1.200，比值 0.99 ——
   语言→动作的映射幅度标定几乎精确。
2. **视觉使用在整个训练过程中单调上升**：0.041（step 1000）→ 0.442（9000）→
   0.506（14000）。`last.pth` 的 vis_share 0.247 对数据要求的 0.215，略微超配，
   基本对齐。
3. **`best.pth` 是最差的 checkpoint。** vis_share 0.024，只有数据要求的 1/9 ——
   近乎视觉盲，却拥有最低的 val pos_err。

---

## 4. 两个必须改的结论

### 4.1 val loss 选点系统性地选走视觉盲策略

step 1000（pos_err 0.2095，vis_share 0.024）比 step 14000（pos_err 0.2389，
vis_share 0.247）"更好" —— 但前者不看世界。原因是数据里语言已解释 78.5% 的方差，
拟合视觉只带来剩余 21.5% 的收益却引入方差，因此**最小化开环 ADE 主动惩罚视觉使用**。

这直接解释了闭环失败（0/273 自主停止，SR@3m 30.1）：部署的正是这类视觉盲
checkpoint，而"何时到达"只能由视觉决定。

**选点判据必须改为**：在 pos_err 可接受的 checkpoint 中，选 `vis_share` 最接近
数据 0.215 者，而非 val loss 最低者。

### 4.2 "验证曲线见顶 = 算力浪费"的读法是错的

val pos_err 从 step 1000 到 14000 基本持平，据此曾判断训练早已饱和。实际上模型
在这 13000 步里一直在学视觉接地（vis_sens 0.041→0.506），只是被监控的指标看不见。
停在 14000 时 `vis_sens` 仍在上升，剩余 6000 步可从 `last.pth` 续跑。

---

## 5. 对后续路线的影响

- **`last.pth` 可以作为合格的 1.5B 能力上界参照**，用于与后续 0.5B 对比。此前
  的 `best.pth` 不合格。
- 缩小模型之前不需要先修"绑定问题" —— 绑定是好的。真正的缺口仍是**停止能力**，
  而现在已确认策略确实在看图像，停止头有可用的信号。
- 训练监控需加入 `vis_share` 与数据 0.215 的偏差作为一等指标。

## 6. 基准饱和：开环 ADE 无法区分 VLA 与查表

`scripts/trivial_action_baselines.py`，同一 seed/val_frac 的轨迹级切分，
9092 train / 281 val，7965 个去重验证 chunk：

| 预测器 | pos_err_m | end_pos_err_m | yaw_err_deg |
|---|---|---|---|
| zeros | 0.3752 | 0.6653 | 2.462 |
| global-mean | 0.3762 | 0.6648 | 2.499 |
| **class-mean**（仅按指令的运动类别查表） | **0.2398** | **0.4213** | 2.467 |
| oracle-traj（看到了留出轨迹自身均值，作弊） | 0.0944 | 0.1614 | 0.015 |

对照模型：

| checkpoint | pos_err_m | 相对 class-mean | end_pos_err_m | 相对 class-mean | yaw_err_deg |
|---|---|---|---|---|---|
| `best.pth` (1000) | 0.2095 | **−12.6%** | 0.2784 | **−33.9%** | 1.034 |
| `best_responsive` (9000) | 0.2800 | +16.8%（更差） | 0.4428 | +5.1%（更差） | 1.391 |
| `last.pth` (14000) | 0.2389 | −0.4% | 0.3754 | −10.9% | 1.151 |

**读法：**

1. **`last.pth` 在 pos_err 上与查表持平**（0.2389 vs 0.2398）。端点好 10.9%，
   yaw 好 53%（1.151° vs 2.467°）——**yaw 是唯一明确的胜利**。
2. `best_responsive.pth` 在两个位置指标上都**输给**查表。
3. 唯一明确赢过查表的是 `best.pth`，而它恰恰是视觉盲的那个（vis_share 0.024）。
4. `oracle-traj` 0.0944 说明剩余空间很大：一个"知道自己在哪条轨迹上"的常量
   预测器能到 0.094，而我们在 0.239。class-mean → oracle-traj 之间这段
   （0.240 → 0.094）正是需要视觉与相位信息才能拿到的部分，我们几乎没拿到。

**结论：UAV-Flow-Sim 的开环 ADE 已被指令先验饱和，不能单独作为动作阶段的
主指标。** 任何只报 `pos_err_m` 的表格都会被审稿人用这张平凡基线表击穿。

这与 §2 的方差分解一致：指令解释了 78.5% 的端点散布，所以按指令查表自然接近
最优；模型能加的只有那 21.5%，而现有损失（L1 + 端点 + 方向）不足以逼它去拿。

## 7. 复现

```bash
python training/v2_stage3_action.py --mode grounding_eval \
  --train_lora --ckpt_dir checkpoints/v2_stage2_full_cradio --tag best \
  --vision_type cradio_v3_b --data_root <sim> --action_stats <stats> \
  --pos_unit auto --chunk_size 8 --chunk_offset 1 --batch 16 \
  --readout xattn --n_bins 0 --norm_mode quantile --no_proprio --aug_flip \
  --split_by trajectory --val_frac 0.03 --probe_groups 8 \
  --save_dir checkpoints/v2_stage3_v5_sim
```

输出：`checkpoints/v2_stage3_v5_sim/grounding_eval.json`
