# AeroMamba 周会报告（2026-07-13 ～ 2026-07-19）

> 主题：训练侧缺陷修复 → v2 重训新基线 → **绑定轮（Stage3 v3 Round A）训练与全量评测** → 失败模式深挖 → AeroMamba v2 白板重设计启动
>
> 模型：**AeroMamba-Opt** = SigLIP2-Base-384（冻结）+ Perceiver Resampler(64q) + Mamba-2-370M(LoRA r=16) + ProprioEncoder + MLP ActionHead(chunk=8)，本轮新增**训练期绑定分类头**
>
> 上周报告见 `weekly_report_20260713.md`

---

## 0. 一页总览

```
修复:  flip 增广同步镜像指令文本 + HF delta_state8 标签泄漏 (07-13)
基线:  v2 重训 (带修复) 全量评测 nDTW 0.0771 —— 当前最强对照
改动:  绑定轮 = 绑定分类头(4任务) + 通道/幅度加权 + 原语类过采样 + 加速度损失
训练:  stage3_v3_binding 远端 4090 训完 (Ep1-2 + 补缺 + Ep4), val_loss 0.2960
       └─ 训练期绑定指标全达标: motion=1.0, yaw≈0.98, dz≈0.85, mag≈0.80
评测:  273 任务全量闭环 nDTW 0.0493 —— 比最早基线 +12%, 但比 v2 重训回退 36%
       └─ 局部成功: Pass 0.015→0.151; 局部恶化: 过冲加重、转向回退、升降近乎失效
结论:  "训练期指标好 ≠ 闭环好"; 增量修补收益见顶, 启动 v2 白板重设计
进行中: 重设计文档已定稿; 远端数据准备 (L0 空中域 CPT + L1 索引) 并行推进
```

| 日期 | 事项 |
|---|---|
| 07-13 ~ 14 | flip 镜像/标签泄漏修复；Stage2 数据管线 v2；v2 重训（新对照基线） |
| 07-15 | v2 重训全量评测（nDTW 0.0771）；AeroStream 改动计划定稿；绑定头/流式推理代码合入 |
| 07-16 ~ 18 | 绑定轮远端训练（含一次卡顿排障 + Ep4 续训）；权重瘦身迁本地 |
| 07-18 | 273 任务全量闭环评测 + nDTW/成功率 + 273 张逐任务 GT-vs-pred 对比图 |
| 07-18 ~ 19 | 失败模式根因分析；与 OpenVLA-UAV 差距六层拆解；v2 白板重设计文档；远端数据准备启动 |

---

## 1. 本周改动清单

### 1.1 训练侧缺陷修复（绑定轮的前置，07-13 ~ 14 合入）

| 改动 | 内容 | 意义 |
|---|---|---|
| flip 增广镜像指令 | `mirror_instruction_lr`：镜像图像时同步交换指令中的方向词（left↔right 等） | 修复"方向语义被增广精确清零"这一 v1 最大单点事故 |
| HF 标签泄漏 | 修复 `UAVFlowHFDataset` 的 `delta_state8` 泄漏未来信息 | 卫生修复 |
| Stage2 数据管线 v2 | 按来源加权混合（filtered general + uav_motion + cognitive，共 204k） | 为后续 S2 轮备好数据 |

带上述修复的 **v2 重训**（`stage3_v2_20260714_082604`）在 07-15 本地全量评测拿到 **nDTW 0.0771**，成为当前最强对照基线（此前 WSL 两轮为 0.044/0.043）。

### 1.2 绑定轮（Round A）——本次实验的核心改动

背景：v2 诊断出的核心问题不是"飞得准不准"，而是**动作和指令脱钩**（换指令输出几乎不变）。绑定轮的思路是用分类监督把"指令说了什么"硬压进共享表征——回归损失对"符号错了"惩罚太软，交叉熵梯度更尖锐。

| 改动 | 内容 | 文件 |
|---|---|---|
| 绑定分类头（新增） | 与动作头共享 `h_last`，四组输出：`motion_class`(10类) / `yaw_sign`(3类) / `dz_sign`(3类) / `magnitude_bin`(8箱对数分箱)；零初始化，<1M 参数；**仅训练期前向，推理零开销** | `model/binding_head.py` |
| 标签在线提取 | yaw/dz/幅度从 `gt_action` 差分得出（flip 后自动一致）；motion 类从指令正则模板推断，落不进模板给 ignore_index | `data/dataset.py` |
| 原语类过采样 | 对 Move/Shift/Ascend/Descend/Surround/Rotate 全部窗口 3× 复制索引，与转向过采样取 max | `data/dataset.py` |
| 损失加权 | z/yaw 通道 ×2.5；样本按 chunk 末端位移 `1+log1p(‖d‖)` 加权（对抗均值坍缩） | `model/action_head.py` |
| 加速度损失 | 启用 `--lambda_acc 0.25`（动作序列一阶差分 L1，AnoleVLA 做法） | 启动脚本 |
| 流式推理路径 | `stream_reset`/`stream_step` + server `--stream 1`（Round B 预埋，本轮未启用） | `model/uav_mamba_vla.py`、`inference/server.py` |

训练配置：**从 Stage2 checkpoint 重训**（不从 v2 续训，保证改动可干净归因）；`--lambda_binding 0.3`、batch 64→56、lr 1e-4→8.75e-5、2 epoch + Ep4 续训。

> 训练事故记录：Ep2 中途 DataLoader 卡顿（workers=16），从 `latest.pth` 以 workers=8 + `--max_steps 25285` 补齐等量 steps；随后续训 Ep4。教训重申：远端训练/验证 workers≤8。

训练收敛情况（`stage3_v3_binding_20260716_024505`）：

| 指标 | 结果 |
|---|---|
| val_loss | **0.2960**（上一轮 0.3039） |
| 验证绑定精度 | motion=**1.00**，yaw_sign≈**0.98**，dz_sign≈0.85，magnitude≈0.80 |

**训练期四项验收指标全部达标**——这让下面的闭环结果更有诊断价值。

---

## 2. 评测结果：绑定轮全量闭环（273 任务，07-18）

权重瘦身（`scripts/make_stage3_slim.py`，含绑定头/LoRA/resampler/动作头）→ 本地 WSL 4060 推理服务 → Unreal 无头低显存跑全量，约 1 小时 54 分完成。

### 2.1 四次全量评测横向对比

| 评测 | 权重 | 总体 nDTW | SR@5m | 到 target 中位距离 |
|---|---|---|---|---|
| 07-13 WSL 基线 | Stage3 v2 | 0.0440 | 14.7% | 8.3m |
| 07-13 WSL fast | Stage3 v2（复测） | 0.0427 | 20.0% | 7.9m |
| **07-15 本地** | **v2 重训（带修复）** | **0.0771** | 14.7% | 13.2m |
| **07-18 本次** | **v3 绑定轮 ep4** | **0.0493** | 12.1% | 16.5m |

**一句话结论：比最早基线略好（+12%），但比最强的 v2 重训明显回退（−36%）；到点成功率没有提升，终点反而更远（过冲加重）。**

### 2.2 分类别：谁涨谁跌（vs v2 重训 0.0771）

| 类别 | v2 重训 | 本次绑定轮 | 变化 |
|---|---|---|---|
| **Pass** | 0.113 | **0.151** | **明显变好**（最早基线仅 0.015） |
| Surround | 0.001 | 0.005 | 略好，仍≈0 |
| Approach | 0.035 | 0.035 | 持平 |
| Retreat | 0.063 | 0.052 | 略差 |
| Land | 0.060 | 0.041 | 变差 |
| Rotate | **0.199** | 0.078 | **大回退** |
| Turn | **0.223** | 0.058 | **大回退** |
| Shift | 0.067 | 0.020 | 回退 |
| Ascend/Descend | 0.056 | 0.003 | **近乎崩** |
| Move | 0.023 | **0.000** | **崩** |

### 2.3 轨迹级证据（新增可视化基建）

本周新增两套绘图脚本：`scripts/plot_gt_vs_pred.py`（每类一条的 2D/3D 汇总对比，含 target 标注）和 `scripts/plot_gt_vs_pred_per_task.py`（**273 张逐任务对比图**，完全对齐官方画法：yaw quiver 箭头、target 红 X、equal aspect）。产物在 `reports/plots/`。

从图上看到的失败模式（比 nDTW 数字更有信息量）：

- **过冲不停是第一失分项**：Move（"navigate to a point 3.5m away"）target 在前方 2.8m，GT 到点即停，模型一路冲到 15m 还在飞；Approach 同样穿过 target 不停。终点中位距离 16.5m 由此而来；
- **旋转被译成平移**：Rotate 105° 任务 GT 原地转向，模型却向右平移 6m+——"right" 被绑到平移通道而非旋转通道，转向角度数字没被用上；
- **z 幅度不足**："lower altitude by 9.0m" 模型最多下潜 0.2–0.5m；Land 水平走到一半就停、没有完成下降段；
- **Surround 完全没有环绕行为**：在起点附近徘徊后向外漂移——需要相位记忆的任务单帧策略学不会；
- **做得好的**：Pass 的 S 形绕行在三维空间与 GT 基本贴合（仅尾段过冲）；Retreat 方向距离都对（误差 0.76m）。

---

## 3. 怎么解读：绑定轮的实验结论

1. **绑定监督本身有效，但只到"表征"为止**：训练期 motion=1.0、yaw≈0.98 说明指令信息确实被压进了共享特征；Pass 从 0.015→0.151 说明"语言→轨迹形状"的绑定在复杂路径上真实起效。
2. **但动作头照样可以绕开它**：分类头只整形共享特征，回归头在闭环里仍坍缩到先验行为——幅度标定（magnitude_bin 监督没变成"到点即停"）和"转向≠平移"的解耦都没有落到动作输出上。
3. **v2 已学到的转向能力被过采样+加权破坏**（Turn/Rotate 0.20→0.06）：改变数据/损失分布的代价比预期大，"局部补丁修一个坑、塌另一个坑"的模式再次出现。
4. **上线决策**：绑定轮权重**不替换 v2 重训作为对照/演示权重**；其价值是作为诊断证据和（原计划中）Round B 流式训练的起点。

### 与 OpenVLA-UAV 的差距拆解（六层，按影响排序）

结合 UAV-Flow 原论文（NeurIPS'25）训练配置核对：① 骨干 370M vs 7B + 海量预训练（论文结论：**空间 grounding 是 Flow 任务决定因素**）；② 算力差 1–2 个数量级（8×A100 × 20 万步 vs 单卡 2 epoch）；③ 离散 bin 天然有界 vs 连续回归无界（直接对应过冲）；④ 协议偏离（chunk-8 执行 + proprio 含速度的正反馈回路，官方单步 + 无速度）；⑤ 自身缺陷（已修一部分）；⑥ nDTW 朝向占一半权重，恰好惩罚我们最弱的 yaw。

**判断：①②是先天差距，③④⑤的修补空间上周已基本用完——增量修补收益见顶，需要结构性方案。**

---

## 4. 决策：启动 AeroMamba v2 白板重设计

基于两轮实验（v2、v3 绑定轮）的失败证据 + 对 RoboMamba / AnoleVLA / CosFly-VLA / OpenVLA-OFT / FAST / Falcon-H1 / Mamba-3 的系统调研，完成重设计方案 `docs/MAMBA_UAV_VLA_REDESIGN_20260719.md`。与现架构的关键差异：

| 维度 | 现状 (v1) | 重设计 (v2) | 依据 |
|---|---|---|---|
| 空中域预训练 | 无 | S0 CPT 300–500k（AirSpatial/Open3DVQA/HRVQA 等） | CosFly 消融最大增益项 |
| 骨干 | Mamba2-370M（Pile 语料，弱底座） | **Falcon-H1-1.5B-Deep**（hybrid SSM，对标 7B–10B；Mamba-3 留作消融） | 小尺寸 SSM 预训练 SOTA |
| 视觉 token | 64-query resampler 压缩 | 全量 729 不压缩 + 编码器参与微调 | Cobra/AnoleVLA |
| 动作读出 | 最后一个视觉 token | meta-query 解耦三头：动作 / grounding / **进度-停止** | CosFly；停止是我们第一失分项 |
| 动作输出 | 无界回归 + 事后裁剪 | tanh×q99 **有界**回归 | 本次过冲证据 + OFT |
| proprio | 位姿+速度（正反馈隐患） | 位姿+Δ位姿，历史交给 SSM 流式状态 | 官方协议对照 |
| 训练 | 3 阶段模仿 | 5 阶段：CPT→对齐→SFT+grounding→动作→**流式 TBPTT**→**DAgger+RL** | CosFly 闭环 RL +30% SR |

原计划的 Round B（流式轮）不再基于当前 370M 底座继续，其核心思想（SSM 状态跨步携带、TBPTT）升级并入 v2 方案的 S4 阶段。

### 已在进行的落地动作（07-19）

远端换用 128 核 CPU 实例做数据准备（无卡模式省钱），当前状态：

- **L1 动作数据索引完成**：26,795 个 episode（帧数与 parquet 行数精确对账 1,785,284）、42,932 个流式训练窗口；发现 254GB parquet 原始分片与已解包 episode 完全冗余，待确认后删除释放磁盘；
- **L0 空中域 CPT 数据**：Open3DVQA-v2（4GB）、AirSpatial（1.1GB）已下完，HRVQA（26GB）下载中，链式任务会在下载完成后自动构建统一 JSONL（`data/build_l0_cpt.py`，复用 204k Stage2 混合数据 + 新增两源，CosFly 式 schema）。

---

## 5. 遗留问题与下周计划

| # | 事项 | 说明 | 优先级 |
|---|---|---|---|
| 1 | L0/L1 数据管线收尾 | HRVQA 转换 + 统一 JSONL 校验 + 数据清单报告；确认后删除 254GB 冗余 parquet | 高 |
| 2 | v2 骨干落地 | Falcon-H1-1.5B-Deep 下载 + S1 对齐代码适配（hybrid 架构接入现有管线） | 高 |
| 3 | S0 空中域 CPT | 4090 全参数可训（bf16+8bit Adam+梯度检查点），预计 4–8h/epoch | 高 |
| 4 | 验收探针前置 | 指令置换 ΔL1 作为 S3 硬门禁（v1 教训：训练期指标好 ≠ 闭环好） | 高 |
| 5 | 绑定轮消融记录 | 本次"分类头达标但闭环回退"是有价值的负结果，留档供论文消融叙事 | 中 |

---

## 6. 关键产物与复现

| 产物 | 路径 |
|---|---|
| 绑定轮 checkpoint（远端） | `checkpoints/stage3_v3_binding_20260716_024505/best.pth` |
| 瘦身权重（本地） | `Aeromamba/checkpoints/stage3_v3/best_slim.pth` |
| 全量评测结果 | `UAV-Flow-Eval/results/aeromamba_stage3_v3_binding_ep4_clean_20260718/` |
| v2 重训对照（最强基线） | `UAV-Flow-Eval/results/aeromamba_opt_stage3_v2_local/`（nDTW 0.0771） |
| 汇总对比图（2D/3D） | `reports/plots/gt_vs_pred[_3d]_aeromamba_stage3_v3_binding_ep4_clean_20260718.png` |
| 273 张逐任务对比图 | `reports/plots/per_task_aeromamba_stage3_v3_binding_ep4_clean_20260718/` |
| 绑定头 / 瘦身 / 绘图脚本 | `model/binding_head.py`、`scripts/make_stage3_slim.py`、`scripts/plot_gt_vs_pred*.py` |
| 改动计划 / 重设计文档 | `docs/AEROSTREAM_CHANGE_PLAN.md`、`docs/MAMBA_UAV_VLA_REDESIGN_20260719.md` |

```bash
# 复现绑定轮评测（WSL 推理 + Windows 无头评测）
wsl -- bash Aeromamba/scripts/start_infer_server_wsl.sh   # 指向 stage3_v3/best_slim.pth
conda run -n unrealcv python batch_run_act_all.py -p 5007 \
  -o results/aeromamba_stage3_v3_binding_ep4_clean_20260718 \
  --img_size 384 --no_early_stop --offscreen --ue_lowmem

# nDTW 指标（支持多目录横向对比）
python scripts/run_metric.py aeromamba_stage3_v3_binding_ep4_clean_20260718 aeromamba_opt_stage3_v2_local

# 逐任务对比图
python scripts/plot_gt_vs_pred_per_task.py --run aeromamba_stage3_v3_binding_ep4_clean_20260718
```
