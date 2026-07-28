# AeroMamba 周会报告（2026-07-06 ～ 2026-07-13）

> 主题：Stage-3 v2 重训 → 闭环失控根因诊断 → 推理侧修复与加速 → 本地化部署 → 首个全量评测基线
>
> 模型：**AeroMamba-Opt** = SigLIP2-Base-384（冻结）+ Perceiver Resampler(64q) + Mamba-2-370M(LoRA r=16) + ProprioEncoder + MLP ActionHead(chunk=8)
>
> 更多案例图与逐步证据见 `weekly_report_20260713_detailed.md`

---

## 0. 一页总览

```
训练:  Stage-3 v2 重训完成 (动作 z-score 归一化 + 新 loss + 转向过采样)
          └─ val_loss 0.2474, 离线 pos_err ≈ 4.5cm
诊断:  闭环失控 → 自建 per-step 诊断 → 三层根因
          └─ 数据先验(前进+下潜) × 速度OOD正反馈 × 指令条件化弱
修复:  推理三重防线 → 失控清零 (z 最深 -17.7m → -0.6m, 路径中位 30.9m → 1.7m)
提速:  单次推理 1852ms → 289ms → 125ms (JPEG传输 + 本地化 + Mamba-2 CUDA内核)
落地:  推理迁移到本地 WSL (8GB 4060), 免隧道; Unreal 无头+低显存跑评测
评测:  首次完成全量 273 任务 × 2 轮 (46分钟/轮), nDTW 基线 = 0.044
```

| 日期 | 事项 |
|---|---|
| 07-06 ~ 08 | 动作归一化 / loss / 数据管线改造；全量统计动作分布（160 万 chunk） |
| 07-09 ~ 10 | Stage-3 v2 远端训练（一次 OOM、一次验证阶段 SIGKILL 排障） |
| 07-11 ~ 12 | chunk 推理协议对齐；首批闭环失控 → 逐步诊断 → 定位三层根因 |
| 07-12 ~ 13 | 推理三重防线 + 效率优化；权重瘦身迁本地 WSL；全量评测 ×2 轮 + nDTW 基线 |

---

## 1. 训练侧：Stage-3 v2 重训

| 改动 | 内容 | 文件 |
|---|---|---|
| 动作归一化 | per-(k,dim) z-score，mean/std 随 ckpt 保存，推理自动反归一化 | `model/action_head.py` |
| Loss 重设计 | z-space 纯 L1 + endpoint(0.25) + 方向余弦(0.5)；物理误差仅作诊断 | 同上 |
| 动作统计 | 与训练一致的 chunk 逻辑扫全量：160 万样本，转向(>10°)占 17% | `data/compute_action_stats.py`（新增） |
| 数据 | 转向样本 3× 过采样；视觉增广（color jitter/灰度/模糊） | `data/dataset.py` |
| 结果 | val_loss **0.2474**，pos_err ≈ **4.5cm**，yaw_err ≈ 1.2° | run `stage3_v2_20260712_005721` |

动作统计里埋着后面根因分析的关键证据——**数据先验就是「持续前进 + 轻微下潜」**：

| horizon k | x 均值 (m) | z 均值 (m) | yaw 均值 |
|---|---|---|---|
| 1 | +0.113 | -0.004 | 0（对称化归零） |
| 4 | +0.466 | -0.015 | 0 |
| 7 | **+0.833** | **-0.027** | 0 |

> 训练事故记录：batch=64 OOM → 降 48；epoch1 末验证被 SIGKILL（cgroup 90GB 上限 + workers=32 内存尖峰）→ 补做验证并降 workers=8 跑完。教训：远端验证一律 workers≤8。

---

## 2. 问题发现与根因诊断

### 2.1 首批闭环轨迹全面失控（10 任务）

| 指标 | 中位数 | 最差 | 参考 |
|---|---|---|---|
| 路径长度 | **30.9m** | 61.2m | GT 通常 0.7~9m |
| z 最低 | **-5.7m** | **-17.7m** | GT 几乎全程 z≈0 |
| 终点距目标 | 10.1m | 62.5m | — |

三个典型案例：

- **Move 3.5m**：第 10 步已到目标 13.5cm 内（方向/尺度学对了），之后不减速反而越飞越快，最终 44m 外钻地 14.7m —— **「到了却不停」+ 正反馈发散**；
- **Ascend 5m**：指令要求爬升，实际一路下潜 -17.7m —— **不管指令说什么都执行先验行为**；
- **Turn to the person**：11 步位移 7cm、yaw 仅 -0.7° —— **语义转向完全无响应**。

### 2.2 逐步诊断基建（新增）

`inference/server.py --diagnose`：每次预测落一行 JSONL（模型实际输入 pose/vel、裸输出 chunk、增量、分段耗时），自动打 OOD 标记（`vel_ood`/`large_step`/`raw_z_dive`/`pose_ood`）；`scripts/analyze_infer_diagnose.py` 按 episode 汇总。

诊断数据规律高度一致：**失控永远发生在 proprio 大跳变的下一拍**，速度从 0.08 滚雪球到 2.9m/帧。

### 2.3 三层根因（证据链）

```
[方向]   数据先验 = 前进+下潜 (action_mean: x@k7=+0.83m, z 全负)
            → 模型 OOD 时退化输出 ≈ 先验均值
[放大器] chunk 执行一次走 4 路点、只回报末路点
            → 服务器算出的速度 ≈ 4× 训练单帧 → 输入 OOD → 输出更大 → 闭环发散
[触发]   指令条件化弱 (换指令输出几乎不变)
            → 决定「谁先失控」; yaw 被对称化归零、监督信号弱
```

---

## 3. 推理侧修复与加速

### 3.1 平滑度：三重防线（`inference/server.py` 新参数）

| 防线 | 参数 | 机制 | 针对 |
|---|---|---|---|
| 速度归一 | `--vel_per_step 1` | 帧间位姿差 ÷ 实际执行路点数，恢复单帧速度 | 放大器（主修复） |
| 分布裁剪 | `--clip_sigma 3` | 输出逐 (k,dim) 裁到训练 mean±3σ | 先验尾部 |
| 增量限幅 | `--max_step_cm 35` / `--max_yaw_step_deg 12` | 相邻路点硬限幅 | 兜底 |

修复前后（10 任务 vs 46 任务对照）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 路径长度中位 | 30.9m | **1.7m** |
| z 最深（最差） | **-17.7m** | **-0.6m** |
| 单步最大步长中位 | 69.8cm | **9.3cm** |
| OOD flags | 每条失控轨迹都有 | **0** |

### 3.2 效率：1852ms → 125ms（14.8×）

| 阶段 | 单次推理 | 关键手段 |
|---|---|---|
| 初始（远端+隧道） | 1852ms | 瓶颈是 PNG 图像走 SSH 隧道（pre=1754ms） |
| JPEG q88 + 预热 + 缓存 | 300ms | payload 缩小 8×；启动预热吃掉 cuDNN autotune |
| 迁移本地 WSL | 289ms | 传输 8ms，剩下全是模型 |
| + Mamba-2 CUDA 内核 | **125ms**（空载） | `mamba_ssm 2.2.4 + causal_conv1d` fused kernel，≈8Hz |

评测并发时（Unreal 同卡抢 GPU）约 250~320ms，全程平稳。

### 3.3 本地化部署（新）

远端实例降配（2GB RAM）后，推理闭环整体迁到本地：

- **权重瘦身**：3.3GB 全量 ckpt → 只取可训练部分（LoRA/resampler/action head/统计量）**188MB**，基座权重走 HF 缓存自动下载，并行 SCP 2 分钟传完；
- **WSL 推理**：Ubuntu-20.04 conda `aeromamba` 环境，RTX 4060 8GB，`scripts/start_infer_server_wsl.sh` 一键启动（`HF_HUB_OFFLINE=1` 免网络校验）；
- **评测省显存**：Unreal `-RenderOffScreen` 无头 + 纹理/阴影降档（`--offscreen --ue_lowmem`，相机画面不受影响），显存 7.2GB → **6.1GB**，8GB 卡稳定同时跑 Unreal + 模型；
- 期间顺手修了两个兼容问题：transformers 5.x 拒载 .bin 权重（CVE-2025-32434，回退 safetensors 分支）、peft 0.19 禁止 Mamba out_proj LoRA（按训练配置关闭该检查）。

---

## 4. 评测结果：首个全量 nDTW 基线（273 任务 × 2 轮）

本周首次跑完**全部 273 个测试任务**（此前最多 99 子集）。nDTW = exp(−DTW/路径长)，越高越好，1 = 完全重合。两轮结果一致，基线可靠：

| 类别 | n | 第 1 轮 | 第 2 轮 |
|---|---|---|---|
| Turn | 15 | 0.101 | **0.105** |
| Rotate | 15 | 0.085 | **0.101** |
| Shift | 49 | 0.082 | 0.072 |
| Retreat | 12 | 0.046 | **0.083** |
| Move | 15 | 0.032 | 0.051 |
| Approach | 42 | 0.043 | 0.033 |
| Ascend/Descend | 19 | 0.048 | 0.031 |
| Pass | 40 | 0.015 | 0.019 |
| Land | 54 | 0.015 | 0.010 |
| Surround | 12 | 0.002 | 0.001 |
| **总体** | **273** | **0.0440** | **0.0427** |

**解读**：

- 轨迹已稳定（无跑飞/俯冲），但 0.044 对应约 3× 路径长度的累积偏差——推理防线解决了「发散」，**精度上限在模型侧**；
- 转向类相对最好（≈0.10）→ 转向过采样 + 方向余弦 loss 有效；
- Surround ≈ 0 最差 → 位置+朝向持续协调的复合动作未学会；Land/Pass 低与「到点不停」直接相关。

**修复后的两个关键新发现**（对下轮训练最有价值）：

1. **导航精度已经很好，只缺「停」**：navigate 任务全程最近距离 9–20cm（第 27–37 步到达），随后匀速穿过目标继续飞——方向与尺度已学对，「到达检测/停止」是唯一短板；
2. **yaw 分指令类型**：显式角度（"Rotate 165°"→转 148°）有响应但精度差；语义指向（"Turn to the person"）完全无响应——语言侧能解析数字，「对着某物转」的视觉-语言联合推理没学会。

---

## 5. 遗留问题与下周计划

| # | 问题 | 证据 | 改进方向 | 优先级 |
|---|---|---|---|---|
| 1 | 到点不停 | 最近距离 9–20cm 后穿过目标 | 轨迹尾部（减速-停止段）过采样 / done 头 / 剩余距离监督 | 高 |
| 2 | 语义转向失效 | "Turn to X" yaw≈0 | 指令-视觉对齐监督；语义/数字转向分层过采样；核查该类 GT yaw 分布 | 高 |
| 3 | 显式角度精度差 | 误差 30–60° | yaw 维单独加权/归一化 | 中 |
| 4 | z 幅度不足 | Ascend 5m 只升 0.4m | ascend/descend 过采样；clip_sigma 按指令自适应放宽 | 中 |
| 5 | Surround 复合动作 | nDTW≈0 | 专项数据/监督设计 | 中 |
| 6 | 域差距 | Unreal 渲染 vs 训练图 | 评测图像小规模 finetune | 中 |

下周计划：

1. **训练侧对着问题 1/2**：尾部过采样 + yaw 监督审查（重点核查 `symmetrize_lateral` 是否弱化了 yaw 标签）+ 语义转向专项；
2. **评测侧**：加 `dist(target)<阈值` 成功率指标（对齐 UAV-Flow 论文口径）；找 OpenVLA-UAV 同测试集 nDTW 对照；
3. **训练更充分**：epochs 2→4+（epoch2 val 仍在降）；建分任务类型验证集替代单一 val_loss 选型；
4. 本地闭环已就绪（46 分钟/全量轮），迭代周期从「远端训练+隧道评测」缩短为**本地小时级**。

---

## 6. 关键产物与复现

| 产物 | 路径 |
|---|---|
| v2 checkpoint（远端全量） | `checkpoints/stage3_v2_20260712_005721/best.pth` |
| 瘦身权重（本地） | `Aeromamba/checkpoints/stage3_v2/best_slim.pth`（188MB） |
| 动作统计 | 远端 `/root/autodl-tmp/datasets/uav-flow/action_stats_k8.json` |
| 全量评测结果 ×2 | `UAV-Flow-Eval/results/aeromamba_opt_stage3_v2_wsl_local/`、`..._wsl_fast/` |
| 失控/修复对照轨迹 | `results/aeromamba_opt_stage3_v2_chunk/`（前）、`..._v2_smooth/`（后） |
| WSL 推理启动脚本 | `Aeromamba/scripts/start_infer_server_wsl.sh` |
| 诊断分析脚本 | `Aeromamba/scripts/analyze_infer_diagnose.py` |

```bash
# 本地闭环复现（WSL 推理 + Windows 无头评测）
wsl -- bash Aeromamba/scripts/start_infer_server_wsl.sh
powershell UAV-Flow-Eval/scripts/restart_eval.ps1 -Port 5007 -OutputDir results/xxx -Offscreen -UeLowMem

# nDTW 指标
conda run -n unrealcv python -c "from metric import evaluate_by_classification; \
  evaluate_by_classification('./classified_instr.json', './results/xxx', './test_jsons', default_step=5)"
```
