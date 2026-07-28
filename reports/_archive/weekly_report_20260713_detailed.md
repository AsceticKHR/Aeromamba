# AeroMamba 周会报告（2026-07-06 ～ 2026-07-13）【详细版】

> 主题：Stage-3 v2 重训 → Unreal 闭环评测 → 失控根因逐步诊断 → 推理侧系统性优化
>
> 模型：**AeroMamba-Opt** = SigLIP2-Base-384（视觉，冻结）+ Perceiver Resampler（64 查询）+ Mamba-2-370M（LoRA r=16/α=32）+ ProprioEncoder（state8）+ MLP ActionHead（chunk_size=8）
>
> 推荐 ckpt：`checkpoints/stage3_v2_20260712_005721/best.pth`（远端 `/root/autodl-tmp/Aeromamba/`）

---

## 0. TL;DR

1. Stage-3 v2（动作 z-score 归一化 + z-space L1 loss + 转向过采样）训练完成，val_loss **0.2474**，离线 pos_err ≈ **4.5cm**。
2. 闭环评测首批轨迹严重失控：过冲 10 倍以上、z 系统性钻地至 **-17.7m**、转向任务不转。
3. 自建逐步诊断（server 端 per-step JSONL），把根因拆成三层：**数据「前进+下潜」先验 × chunk 执行导致速度 OOD 正反馈 × 指令条件化弱**。
4. 推理侧修复后：单次推理 **1852ms → 300ms**，z 最深 **-17.7m → -0.6m**，路径中位数 **30.9m → 1.7m**，失控 flags 清零。
5. 关键新发现：导航类任务其实已能把最近距离压到 **9–20cm**（方向/尺度准），核心缺陷是**不会停**；转向缺陷集中在**语义类指令**（"Turn to the person"），显式角度指令（"Rotate 165 degrees"）有部分响应。

---

## 1. 时间线

| 日期 | 事项 |
|---|---|
| 07-06 ~ 07-08 | 设计并落地动作归一化 / loss / 数据管线改造；全量统计动作分布（160 万 chunk 样本） |
| 07-09 ~ 07-10 | Stage-3 v2 远端训练（含一次 OOM 排障、一次 SIGKILL 事故与验证补做） |
| 07-11 | 推理协议对齐（chunk 执行）、评测基建修复（隧道守护、断点续评） |
| 07-12 | 首批闭环轨迹分析 → 发现三类失控；上线逐步诊断；定位根因 |
| 07-12 深夜 ~ 07-13 | 推理侧优化（速度归一/裁剪/限幅 + JPEG/预热）；对照评测 46 任务；停机保留现场 |

---

## 2. 训练侧改动明细（Stage-3 v2）

### 2.1 动作归一化（`model/action_head.py`）

- 新增 `ActionNormalizerMixin`：`action_mean` / `action_std`（形状 `[K=8, 4]`，per-horizon per-dim）注册为 buffer，**随 checkpoint 保存与加载**，推理端 `predict_step` 自动反归一化到物理单位（米/弧度）。
- 动机：原始物理空间中 x 方向位移量级远大于 z/yaw，直接回归时梯度被 x 主导。

### 2.2 动作统计（`data/compute_action_stats.py`，新增）

- 与训练完全一致的 chunk 提取逻辑（同锚点累计偏移、body-frame），扫全量 UAV-Flow：**约 160 万样本 / 1.8 万条轨迹**，转向样本（>10°）占比 **17%**。
- 产出 `action_stats_k8.json`（远端 `/root/autodl-tmp/datasets/uav-flow/`）。关键数字（后文根因分析的证据）：

| horizon k | x 均值 (m) | z 均值 (m) | yaw 均值 | z 标准差 |
|---|---|---|---|---|
| 1 | +0.113 | -0.0037 | 0（对称化） | 0.025 |
| 4 | +0.466 | -0.0152 | 0 | 0.077 |
| 7 | **+0.833** | **-0.0267** | 0 | 0.125 |

→ 数据先验就是「持续前进 + 轻微下潜」；yaw/y 因 `symmetrize_lateral` 均值归零。

### 2.3 Loss 重设计（`model/action_head.py::aero_action_loss`）

- **z-space 纯 L1**（归一化空间回归）+ **endpoint 项**（权重 0.25，约束轨迹末端）+ **方向余弦项**（权重 0.5，约束运动方向）。
- 物理单位误差（pos_err/yaw_err）仅作诊断指标，不回传梯度。

### 2.4 数据与训练管线（`data/dataset.py`、`training/stage3_action.py`、`training/trainer.py`）

- 转向过采样：yaw 变化 >10° 的样本 **3×** 过采样。
- 视觉增广 `aug_vision`：color jitter / 随机灰度 / 高斯模糊。
- 新参数 `--action_stats`；DataLoader `prefetch_factor=4`。
- `scripts/run_stage3_v2.sh`：合并 3a/3b 为单阶段，从 Stage-2 ckpt 直训。最终配置 **batch=48, workers=8, lr=7.5e-5, epochs=2**。

### 2.5 训练过程记录（远端 run `stage3_v2_20260712_005721`）

| 事件 | 详情 |
|---|---|
| OOM | batch=64 CUDA OOM → 降 48 |
| **SIGKILL(137)** | epoch1 末验证阶段被杀：容器 cgroup 内存 ~90GB 上限 + workers=32 的验证内存尖峰 |
| 恢复 | 用 `epoch1_end.pth` 补做验证：val_loss **0.2631**；epoch2 降 workers=8 跑完 |
| 最终 | val_loss **0.2474**，pos_err ≈ **4.5cm**，yaw_err ≈ **1.2°** |

> 教训：验证阶段的 DataLoader worker 内存尖峰是被杀主因，远端训练一律 workers≤8 并在验证前显式释放。

---

## 3. 评测系统与推理协议

### 3.1 chunk 执行协议

- `inference/server.py --exec_mode chunk --exec_horizon 4`：一次 `/predict` 返回 k=1..4 共 4 个未来路点。**语义关键点**：8 个路点全部是相对**同一次预测时刻位姿（锚点）**的累计偏移（对齐训练 `_extract_body_frame_chunk`），客户端对同一锚点依次落位，不能逐路点累加（上周曾因误改成累加导致指数放大，本周确认回滚正确）。
- `UAV-Flow-Eval/batch_run_act_all.py`：顺序执行整段 chunk；请求重试；服务器失联的任务不落盘（下轮重试）；`task_output_complete` 跳过已完成任务（断点续评）。

### 3.2 基建

- SSH 隧道守护 `keep_tunnel.ps1` + 一键重启 `scripts/restart_eval.ps1`（健康检查复用隧道，避免反复重建 plink——历史上评测中断的最大来源）。

---

## 4. 问题发现：首批闭环轨迹全面失控

结果目录：`UAV-Flow-Eval/results/aeromamba_opt_stage3_v2_chunk/`（10 任务，max_steps=100）

### 4.1 定量总表（10 条）

| 指标 | 中位数 | 最差 | 训练分布参考 |
|---|---|---|---|
| 路径长度 | **30.9m** | 61.2m | 任务 GT 通常 0.7~9m |
| 单步步长 | 31cm/步，max 105cm | — | 单帧 ~10cm |
| z 最低 | **-5.7m** | **-17.7m** | GT 几乎全程 z≈0 |
| 终点距目标（5 个有 target 任务） | **10.1m** | 62.5m | — |

### 4.2 案例一：Move 3.5m —— 「到了却不停」+ 越飞越快

图：`results/aeromamba_opt_stage3_v2_chunk/2025-03-30_11-49-28_2d.png`

- **第 10 步已到目标 13.5cm 内**（目标 local (284, 0, 0)cm，此时位姿 (272, -6, -2)）——说明前段方向与尺度都学对了。
- 之后不减速：步长按四分位 Q1→Q4 = 50 → 62 → 62 → 49 cm/步，z 每步下潜从 -2.5 恶化到 **-21.8cm**，最终飞到 (44.1, 16.1, **-14.7**)m，距目标 46.7m。
- 图上可见：直线穿过目标 → 1000cm 处开始之字漂移 → 后段大幅右偏狂奔。

### 4.3 案例二：Ascend 5m@65° —— 指令与行为相反

轨迹 `2025-03-30_11-52-19`：要求爬升 5m，实际 z 一路下潜到 **-17.7m**，路径 33m。「不管指令说什么都前进+下潜」→ 指向共同的先验行为，而非任务理解错误。

### 4.4 案例三：Turn to the person —— 不转、不动、被 early-stop

轨迹 `2025-03-30_11-49-14`：GT 参考 yaw 应转 ~30°；实际 11 步、总位移 7cm、**yaw 仅 -0.7°**，触发「连续 10 步位移 <3cm」提前终止。

---

## 5. 逐步诊断基建（本周新增）

### 5.1 实现

- `inference/server.py --diagnose`：每次 `/predict` 落一行 JSONL（`checkpoints/infer_diagnose.jsonl`），字段包括：`proprio_cm`、`pose_m`/`vel_m`（模型实际输入）、`delta_state8`、`raw_chunk_m`（8 路点裸输出）、`body_increment_cm`（相邻路点增量）、`episode_z_cm`、分段耗时 `pre/model/post_ms`，并自动打标：

| flag | 触发条件 | 含义 |
|---|---|---|
| `vel_ood` | ‖vel‖>0.5m/帧 | 速度输入超训练分布 |
| `large_step` | 路点 xy 范数 >80cm | 输出异常大 |
| `raw_z_dive` | 路点 z < -20cm | 单 chunk 深潜 |
| `pose_ood` | ‖pose‖>5m | 位姿超分布 |

- `scripts/analyze_infer_diagnose.py`：按 episode 汇总 flags、速度/步长的 Q1–Q4 增长曲线、首个 flag 触发点。

### 5.2 诊断数据（修复前，节选）

| episode | 指令 | 速度 Q1→Q4 (m/帧) | z 输出 Q1→Q4 (cm) | 首个 flag |
|---|---|---|---|---|
| ep1 Turn to person | 0.004 → **1.095** | -0.0 → -3.6 | step5 `large_step`（proprio 跳变后一拍） |
| ep3 Move 3.5m | 0.083 → **2.908** | -0.3 → **-32.2** | step3 `vel_ood` |
| ep4 Climb 8m | 0.091 → **2.414** | -0.3 → -10.7 | step2 `large_step` |

规律非常一致：**失控永远发生在 proprio 大跳变的下一拍**，速度越滚越大、z 越压越深。

---

## 6. 根因分析（三层，证据链）

```
[第 1 层·方向] 数据先验 = 前进+下潜
    action_mean: x@k7=+0.83m, z 全负 → 模型 OOD 时退化输出 ≈ 先验均值
    证据: Ascend/Descend/Turn 全部表现为同一行为模式（前飞+下潜）

[第 2 层·放大器] chunk 执行 → 速度 OOD 正反馈
    客户端 1 次执行 4 路点、只回报末路点 → 服务器速度 = 4 步位移 ≈ 4×训练单帧
    证据: first_flag 全部在 proprio 跳变次拍; vel 0.02→3.5m 滚雪球
    ⇒ 输入 OOD → 输出更大 → 下一拍更 OOD（闭环发散）

[第 3 层·触发条件] 指令条件化弱
    证据: 同输入换指令输出几乎不变; "Turn to X" 类 yaw≈0
    背景: Stage-3 仅 2 epoch; yaw 被对称化归零、量纲小、监督信号弱
```

三层缺一不可：先验决定失控**方向**，速度 OOD 决定失控**幅度**，指令弱决定**谁先失控**。

---

## 7. 推理侧优化（实现 + 实测效果）

### 7.1 效率：单次推理 1852ms → 300ms（6.2×）

分段计时先破除「模型慢」的错误假设：

| 阶段 | 优化前中位 | 优化后中位 | 手段 |
|---|---|---|---|
| pre（图像传输+base64+解码） | **1754ms** | **199ms** | 评测端 PNG → **JPEG q88**（payload ~8×缩小，瓶颈在 SSH 隧道带宽） |
| model（前向） | 100ms | 106ms | 本来就达标，无需动 |
| post（后处理） | 0.5ms | 0.5ms | — |
| 首次请求 | 5043ms | **63ms** | 启动预热 2 次完整前向（cuDNN autotune 提前吃掉） |

另有：指令 tokenize LRU 缓存、`cudnn.benchmark=True`、`float32_matmul_precision('high')`。chunk 模式下等效 **~75ms/控制步**，满足 10Hz 实时目标。

### 7.2 平滑度：三重防线（`inference/server.py` 新参数）

| 防线 | 参数 | 机制 | 针对根因 |
|---|---|---|---|
| **速度归一** | `--vel_per_step 1` | 帧间位姿差 ÷ 上次实际执行路点数，恢复训练单帧速度；`delta_state8` 直接由单帧速度构造 | 第 2 层（主修复，掐断正反馈） |
| **分布裁剪** | `--clip_sigma 3` | 物理输出逐 (k,dim) 裁到训练 mean±3σ | 第 1 层（先验尾部无法爆） |
| **增量限幅** | `--max_step_cm 35` / `--max_yaw_step_deg 12` | 相邻路点增量 xyz 范数/偏航角硬限幅后重新累计 | 兜底 |

### 7.3 修复前后全量对比

**旧 = `results/aeromamba_opt_stage3_v2_chunk`（10 任务，100 步）；新 = `results/aeromamba_opt_stage3_v2_smooth`（46 任务，60 步）**

| 指标 | 旧 | 新 |
|---|---|---|
| 路径长度中位 | 30.9m | **1.7m** |
| 单步最大步长中位 | 69.8cm | **9.3cm**（上限 35 生效） |
| z 最深（中位 / 最差） | -5.7m / **-17.7m** | **-1.1cm / -0.6m** |
| `vel_ood`/`raw_z_dive` flags | 每条失控轨迹都有 | **0** |
| 有 target 任务：全程最近距离中位 | 121cm（n=5） | 194cm（n=15，含大量 turn 类 target 远的任务） |
| 到过目标 1m 内的比例 | 2/5 | 6/15 |

**分任务类型（新，46 条）**：

| 类型 | n | 路径中位 | z 最低中位 | yaw 变化中位 |
|---|---|---|---|---|
| turn/rotate | 13 | 1.07m | -1.1cm | 0.5° |
| navigate | 6 | 9.7m | -16.5cm | 0.2° |
| ascend | 2 | 7.8m | -2.3cm | 0.6° |
| descend | 2 | 4.9m | -30cm | 0.2° |
| circle | 3 | 3.0m | -2.3cm | 2.7° |
| move/other | 20 | 1.2m | -0.5cm | 0.3° |

### 7.4 案例复检（同任务前后对比，图）

**Move 3.5m（`2025-03-30_11-49-28_2d.png`）**
- 旧图（`..._v2_chunk/`）：之字漂移、44m 失控、钻地 14.7m。
- 新图（`..._v2_smooth/`）：**笔直穿过目标点**，min_dist **9cm**（第 36 步），无横向失稳、z 全程 |z|<0.4cm；唯一缺陷是继续匀速前飞到 6.5m 外（不会停）。

**Rotate 105° right（`..._v2_smooth/2025-03-30_11-52-11_2d.png`）**
- yaw 箭头全程朝前、纹丝不转，只有 2m 缓慢左移。这张图是「语义转向失效」最直接的证据。

**Ascend 5m@65°**
- 旧：z 一路 -17.7m；新：z 终值 **+0.39m**（首次方向正确，但幅度远不足 5m）。

### 7.5 优化后的新发现（对下轮训练最有价值）

1. **导航精度已经很好，只缺「停」**：5 个 navigate 任务全程最近距离 9 / 14 / 18 / 19 / 20 cm（都在第 27–37 步到达），随后继续前飞到 3.6–6.7m 外。→ 策略方向与尺度已学对，「到达检测/停止」是唯一短板。
2. **yaw 并非全死，分指令类型**：
   - 显式角度：`Rotate 30°`→ 转 59.8°、`Rotate 165° left`→ 转 148.3°、`Rotate 90°`→ 转 28.1°（有响应但精度差）；
   - 语义指向：`Turn to the person` / `Face toward` → yaw ≈ 0（完全无响应）。
   → 语言侧能解析数字角度，但「对着某物体转」需要视觉-语言联合推理，这部分没学会。

---

## 8. 遗留问题清单

| # | 问题 | 层面 | 严重度 |
|---|---|---|---|
| 1 | 到点不停（匀速穿过目标） | 数据/监督缺失 | 高（直接决定成功率） |
| 2 | 语义转向失效（Turn to/Face toward yaw≈0） | 指令-视觉对齐 | 高 |
| 3 | 显式角度转向精度差（误差 30–60°） | 监督信号弱 | 中 |
| 4 | 高度类任务幅度不足（Ascend 5m 只升 0.4m） | z 信号弱/被裁剪压制 | 中 |
| 5 | OOD 退化到先验的倾向仍在（被裁剪兜底，未根治） | 域差距 | 中 |
| 6 | 隧道稳定性（评测中断的主因） | 基建 | 低（有守护脚本） |

---

## 9. 后续优化方向

### 高优先级（下一轮训练直接对着遗留问题 1/2）

1. **到点停止监督**：
   - 数据侧：轨迹末端已含「减速-停止」段，检查 chunk 采样是否系统性偏向轨迹中段；给轨迹尾部提高采样权重。
   - 结构侧：加 done/stop 头，或把「剩余距离」编码进监督（chunk 后段标签按剩余距离衰减）。
   - 评测侧：加 `dist(target)<阈值` 成功判定并终止（同时使指标可比 UAV-Flow 论文）。
2. **修 yaw 监督**：
   - 审查 `symmetrize_lateral`：确认翻转增广只做镜像、没有把 yaw 标签本身弱化；
   - yaw 维单独提高 loss 权重或单独归一化尺度；
   - 转向过采样从 3× 提到 5×+，并按「语义转向 vs 数字转向」分层；
   - 针对 "Turn to X" 类：确认训练集中该类指令的 GT yaw 分布非零（排除数据处理阶段就丢了信号）。
3. **z 幅度**：ascend/descend 样本过采样；确认 `clip_sigma=3` 对大幅爬升是否过紧（k=7 z 3σ≈±37cm/chunk，爬 5m 需要 >13 个 chunk——可对 ascend 类放宽或按指令自适应）。

### 中优先级

4. **域适应**：Unreal 渲染图小规模 finetune（用本周评测收集的图像即可起步），或更强域随机化。
5. **训练更充分**：epochs 2→4–6（epoch2 val 仍在降）；建立分任务类型验证集，替代单一 val_loss 作为选型标准。
6. **推理协议 A/B**（诊断基建已支持量化）：`single_step` vs `chunk` vs 自适应 horizon（转向指令或接近目标时 horizon=1）。

### 远期

7. **Stage-4 RL**：先离线偏好优化（现成评测轨迹的 target 距离即 reward），再 Unreal 在线 RL。
8. **跨 chunk temporal ensemble**：把上一 chunk 重叠路点变换到当前锚点做加权融合，进一步消除 chunk 边界不连续。

---

## 10. 附录

### 10.1 关键产物路径

| 产物 | 路径 |
|---|---|
| v2 checkpoint | 远端 `checkpoints/stage3_v2_20260712_005721/best.pth` |
| 动作统计 | 远端 `/root/autodl-tmp/datasets/uav-flow/action_stats_k8.json` |
| 失控轨迹（修复前） | `UAV-Flow-Eval/results/aeromamba_opt_stage3_v2_chunk/`（10 任务 + 2d/3d 图） |
| 修复后轨迹 | `UAV-Flow-Eval/results/aeromamba_opt_stage3_v2_smooth/`（46 任务 + 图） |
| 诊断 JSONL | 远端 `checkpoints/infer_diagnose.jsonl`（本地有同步副本 `Aeromamba/checkpoints/`） |
| 诊断分析脚本 | `Aeromamba/scripts/analyze_infer_diagnose.py` |

### 10.2 本周改动文件

- `model/action_head.py` — ActionNormalizerMixin、aero_action_loss 重设计
- `model/uav_mamba_vla.py` — loss 接 head、predict_step 反归一化
- `data/compute_action_stats.py`（新增）、`data/dataset.py` — 过采样/增广
- `training/stage3_action.py`、`training/trainer.py` — --action_stats 等
- `inference/server.py` — chunk 协议、--diagnose、vel_per_step/clip_sigma/max_step 限幅、JPEG 兼容、预热、tokenize 缓存、分段计时
- `scripts/run_stage3_v2.sh`、`scripts/start_infer_server_nohup.sh`（环境变量化）、`scripts/analyze_infer_diagnose.py`（新增）
- `UAV-Flow-Eval/batch_run_act_all.py` — chunk 执行、JPEG payload、断点续评

### 10.3 可复现命令

```bash
# 远端起服务（诊断 + 平滑防线，环境变量可调）
DIAGNOSE=1 EXEC_MODE=chunk EXEC_HORIZON=4 CLIP_SIGMA=3.0 MAX_STEP_CM=35 \
  bash scripts/start_infer_server_nohup.sh

# 诊断分析
python scripts/analyze_infer_diagnose.py checkpoints/infer_diagnose.jsonl

# 本地评测（断点续评，跳过已完成）
conda run -n unrealcv python batch_run_act_all.py -p 5007 \
  -o results/aeromamba_opt_stage3_v2_smooth --img_size 384 --no_early_stop -m 60
```

*当前状态：本地评测与远端推理服务均已停止，结果与日志已保留。*
