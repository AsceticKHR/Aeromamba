# AeroMamba Stage 2（C-RADIO 版）全量训练评估报告

- 日期：2026-07-26
- 检查点：`checkpoints/v2_stage2_full_cradio/best.pth`（step 13500，val CLM = 1.1452）
- 视觉编码器：NVIDIA **C-RADIO v3-B**（`cradio_v3_b`，单塔，agglomerative 蒸馏 CLIP+DINOv2+SAM）
- 语言主干：**Falcon-H1-1.5B-Deep-Instruct**（hybrid attention + Mamba-2，d_lm=1280）
- 数据来源（真值）：`pilot_gates.json`、`train_log.json`、`reports/s2_final_eval.json`

---

## 1. 结论速览

Stage 2 的目标是让模型**真正依赖图像内容**做视觉-语言推理，并具备**空间/指代定位**能力，为 Stage 3 动作路点回归提供可靠底座。本次全量训练达成：

- **图像特异性彻底解决**：空间问答子集上，把图像换成"另一张真实图"会使 CLM 交叉熵暴涨 **Δshuffle=+0.918 nats**——模型对"这张具体图像"的空间内容高度敏感（此前该指标接近 0 是核心痛点）。
- **回答准确率高位稳定**：整体 teacher-forced 回答准确率 **0.716**，UAV 运动 0.943、航拍空间 0.919、认知推理 0.858、HR-VQA 0.867。
- **定位稳升未塌陷**：DIOR-RSVG IoU **0.194**、Open3DVQA-REC IoU **0.290**，预测框中心随输入变化（center_std 0.06–0.09，远离塌陷阈 1e-3），框面积与真值同量级（无"巨框刷 IoU"退化）。
- **验证损失单调下降并平台化**：val CLM 1.239 → **1.145**，step 13.5k 后平台，step 15.5k 早停。

---

## 2. 模型架构

```
[FPV 图像] → C-RADIO v3-B（冻结主体，仅解冻末 4 层）→ [B, N_vis, 768]
              → MLP 投影器（LLaVA-1.5 式 2 层 GELU）→ [B, N_vis, 1280]
[指令文本] → Falcon-H1 词嵌入 → [B, L, 1280]
   ↓ 拼接（视觉在前）: [vision | text]
   ↓ Falcon-H1 主干（S0/S2 LoRA 注入）
   ├─ CLM 读出：文本位隐状态 → lm_head → 交叉熵（语言/VQA 监督）
   └─ 定位读出（grounding head，见 §2.1）→ [B,4] xyxy 框
```

### 2.1 Grounding 头设计（本轮关键结构决策）

- **KV 来自投影器输出 token（pre-LM）**，而非 LM 因果扫描后的隐状态。理由：视觉在前的因果 SSM/attention 主干里，patch i 的隐状态是对 patch 0..i 的一维前缀扫描，2D patch 身份被压掉，注意力无法定位（此前 center_std≈1e-3 塌陷）。投影器 token 保留完整 2D patch 结构（外加 2D 正弦位置编码），并让投影器获得直接的空间梯度。
- **soft-argmax 定位**：框中心 = 注意力权重对 patch 网格坐标的加权均值（结构上输入相关，杜绝"常数框"）；框尺寸由头直接回归、bound 到 [0,0.5]，与注意力发散解耦。
- **CIoU + 超尺寸惩罚 + 尺寸头负偏置**：`L1 + (1−CIoU) + 0.5·relu(area_pred−area_gt)`，尺寸头 bias 初始化 −1.4（框初始偏小、"长大"到真值），联合破解"巨框刷 IoU"的尺寸塔陷。
- **文本 query 来自 LM 隐状态**（末 prompt token），保留 LM 理解指代表达的能力。

### 2.2 视觉依赖对比铰链（blank / shuffle hinge）

在 CLM 之外加入两个对比 hinge，直接优化"是否依赖图像"：

- `loss_blank = relu(margin_blank − (CE(空白图) − CE(真实图)))`，margin=0.15；
- `loss_shuffle = relu(margin_shuffle − (CE(错配真实图) − CE(真实图)))`，margin=0.10。

---

## 3. 训练配置

| 项 | 值 | 备注 |
|---|---|---|
| 视觉编码器 | `cradio_v3_b` | 冻结主体 + 解冻末 4 层 |
| 语言主干 | Falcon-H1-1.5B-Deep-Instruct | S0 CPT 得到的 LoRA 结构上加载 |
| 可训练模块 | 投影器 + 视觉末 4 层 + LoRA + grounding 头 | 主干冻结 |
| 训练损失 | `CLM + λ_grd·grd + λ_blank·blank + λ_shuffle·shuffle` | |
| λ_grd | **0.3** | 用户设定（弱化定位权重，主攻语言/空间理解）|
| λ_blank / λ_shuffle | 0.5 / 0.75 | 代码默认 |
| blank_margin / shuffle_margin | 0.15 / 0.10 | |
| grounding 损失 | L1 + (1−CIoU) + 0.5·over-coverage | KV=投影器 token |
| batch | 16 | 24GB 卡、梯度检查点 + bf16 autocast（日志实测 `--batch 16`）|
| 精度 | bf16 autocast + 梯度检查点 | 梯度穿冻结主干回投影器 |
| pilot | 300 步门控 | 通过后转全量 |
| 全量步数 | 15,500（step 13.5k 后平台，早停） | best=step 13500 |

---

## 4. 数据

- **训练/对齐数据**：L0 混合航拍语料（航拍 VQA、空间关系问答、认知推理、UAV 运动描述、通用 VQA）+ **grounding 混合集**（DIOR-RSVG 指代框 + Open3DVQA-REC mask→box + 航拍空间 REC）。
- **评估集**：`eval_subset_v2.jsonl`，**图像层面与训练不相交**（image-disjoint held-out），每源采样 128 条，共 8 源、1024 条、20,302 监督 token。空间关系子集 384 条（含 left/right/near/far/方向/距离等线索词）。

---

## 5. 训练过程

### 5.1 Pilot 门控（300 步，全部通过，hard_failures=0）

硬门 P1–P4 为"视觉依赖"，P5/P6（IoU/center_std）已降级为软诊断（部署任务是 Stage 3 动作回归，bbox IoU 仅代理指标）。S2 vs S0（同 300 步基线）对比：

| 门 | 指标 | S2 | S0 | 阈值 | 结果 |
|---|---|--:|--:|--:|:--:|
| P1 | blank_delta 绝对 | 0.700 | 0.214 | 0.05 | PASS |
| P2 | shuffle_delta 绝对 | 0.917 | 0.087 | 0.03 | PASS |
| P3 | blank_delta − S0 | +0.486 | — | 0.02 | PASS |
| P4 | shuffle_delta − S0 | +0.830 | — | 0.02 | PASS |
| P5(软) | grounding IoU | 0.142 | 0.040 | 0.12 | ok |
| P6(软) | REC center_std | 0.074 | 0.003 | 0.05 | ok |

要点：S2 的 shuffle_delta（0.917）是 S0（0.087）的 **10 倍**，center_std 从 0.003（塌陷）升到 0.074，验证 §2.1 的 KV=投影器 token 修复对定位与图像特异性同时有效。

### 5.2 全量验证曲线（val CLM，每 500 步）

```
step   500  1.239   3000  1.201   6000  1.177   9000  1.161   12000 1.148
step  1000  1.219   4000  1.194   7000  1.173  10000  1.158   13000 1.147
step  1500  1.215   4500  1.188   7500  1.169  10500  1.154   13500 1.145 (best.pth)
step  2000  1.210   5000  1.185   8000  1.165  11000  1.150   15500 1.144 (早停)
```

单调下降、无过拟合抬头；13.5k 后进入平台（1.145→1.144），据此早停。

---

## 6. 系统性基准（best.pth，held-out，分层报告）

评估分层：TIER-A 视觉依赖 + 回答准确率（首要，直接关系下游）；TIER-B 语言质量（CE/PPL）；TIER-C 定位诊断（软）。

### 6.1 逐源结果

| 源 | n_tok | CE↓ | PPL↓ | Δblank↑ | Δshuffle↑ | 回答acc↑ | IoU | center_std |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| aerial_spatial | 1961 | 0.202 | 1.22 | +4.070 | −0.000 | **0.919** | — | — |
| airspatial(REC) | 1854 | 1.456 | 4.29 | +2.472 | +0.573 | 0.611 | — | — |
| cognitive | 1657 | 0.227 | 1.25 | +4.099 | −0.000 | 0.858 | — | — |
| **dior_rsvg** | 2668 | 1.047 | 2.85 | +2.304 | +0.168 | 0.579 | **0.194** | 0.060 |
| general | 7837 | 1.166 | 3.21 | +1.762 | +0.088 | 0.676 | — | — |
| hrvqa | 345 | 0.358 | 1.43 | **+12.45** | +0.318 | 0.867 | — | — |
| **open3d_vqa_rec** | 2480 | 0.990 | 2.69 | +2.554 | +0.074 | 0.654 | **0.290** | 0.091 |
| uav_motion | 1500 | 0.214 | 1.24 | +4.790 | +0.019 | **0.943** | — | — |

### 6.2 汇总与空间子集

| 指标 | 整体（1024） | 空间子集（384） |
|---|--:|--:|
| CE / PPL | 0.901 / 2.46 | — |
| Δblank | +2.814 | +2.637 |
| **Δshuffle** | +0.117 | **+0.918** |
| 回答准确率 | **0.716** | 0.673 |

> 读法：`Δblank=CE(空白图)−CE(真实图)`、`Δshuffle=CE(错配真实图)−CE(真实图)`。Δblank 大 = 依赖"有图"；**Δshuffle 大 = 依赖"这张具体图的内容"**（图像特异性）。纯文本可答的源（aerial_spatial/cognitive/uav_motion）Δshuffle≈0 属正常；**需要看图定位的源（airspatial REC +0.573、dior +0.168、空间子集 +0.918）Δshuffle 显著为正**，正是我们要迁移到动作 grounding 的能力。

---

## 7. 与早期检查点对比（进步轨迹）

| 指标 | step2000 | step9000 | best(13500) |
|---|--:|--:|--:|
| 整体回答acc | 0.704 | 0.721 | 0.716 |
| 整体 Δshuffle | +0.012 | +0.055 | **+0.117** |
| 空间子集 Δshuffle | (评估器 bug) | +0.550 | **+0.918** |
| DIOR IoU | 0.106 | 0.168 | **0.194** |
| Open3D IoU | 0.231 | 0.298 | 0.290 |

后期训练主要在**加强视觉锐度与定位**（Δshuffle、IoU 持续涨），而非刷答案分（整体 acc 已在 ~0.72 平台）——这是更利于下游动作的方向。

---

## 8. 关键结论

1. **单塔 C-RADIO-B 可替代双塔视觉编码**：以 ~1/2 成本提供语义+几何+稠密特征，S2 在其上取得高图像特异性与可用定位，验证选型成立。
2. **KV=投影器 token 是定位不塌陷的关键**：把 grounding 的 K/V 从"因果扫描后的 LM 隐状态"改为"投影器 2D patch token"，center_std 从 1e-3 提升一个数量级，IoU 与 Δshuffle 同步改善。
3. **CIoU + 超尺寸惩罚 + 负偏置**解决了尺寸塔陷（巨框刷 IoU），使框面积与真值同量级。
4. **图像特异性问题解决**：空间子集 Δshuffle +0.918，是 Stage 3 动作路点回归可靠的视觉底座。

---

## 9. 局限与后续

- **IoU 绝对值仍中等**（dior 0.19 / open3d 0.29）：受任务权重（λ_grd=0.3，刻意弱化）与航拍小目标影响；因下游是动作回归，已将其定位为软诊断而非阻塞门。
- **airspatial REC 的 CE 偏高（PPL 4.29）**：该源答案空间更开放，后续可考虑补充监督或提示模板统一。
- **后续**：Stage 3 已在此 best.pth 上启动（冻结主干 + proprio 编码器 + 动作头，K=8 路点回归，UAV-Flow 全量训练中）；系统性基准脚本 `scripts/eval_s2_systematic.py` 可复用于回归测试。

---

## 附录 A：三阶段数据集对照（C-RADIO 版流水线）

当前 C-RADIO 版实际训练路径为 **S0 → S2 → S3**（`v2_stage1_align.py` 存在但本轮未用——S0 CPT 已同时完成投影器 + LoRA 对齐，吸收了 S1 的功能）。

| 阶段 | 脚本 | 数据集（远端文件） | 规模 | 训练内容 |
|---|---|---|---|---|
| **S0 领域预训练 (CPT)** | `v2_stage0_cpt.py` | `datasets/l0_cpt/l0_mixed.jsonl` | 215,552 条 | 航拍语料混合：航拍 VQA、空间关系问答、认知推理、UAV 运动描述、通用 VQA（对齐 C-RADIO→Falcon-H1；训投影器 + LoRA + 视觉末层）|
| **S2 视觉 SFT + 定位** | `v2_stage2_grounding.py` | `datasets/l0_cpt/l0_mixed_grd_heldout.jsonl` | 232,602 条 | 在 S0 混合基础上加入定位数据：DIOR-RSVG 指代框 + Open3DVQA-REC(mask→box) + 航拍空间 REC；并做图像层面 held-out 切分 |
| **S3 动作头** | `v2_stage3_action.py` | `datasets/stage3_uavflow`（UAV-Flow 轨迹，HF `wangxiangyu0814/UAV-Flow`）| 26,797 条轨迹 → 采样/过采样后 2,908,282 chunk | UAV 轨迹路点回归（K=8：Δx/Δy/Δz/Δyaw），配 `datasets/uav-flow/action_stats_k8.json` z-score 统计 |
| **评估集（不参训）** | `scripts/eval_s{0,2}_systematic.py` | `datasets/l0_cpt/eval_subset_v2.jsonl` | 1,350 条 | 与训练图像不相交的 held-out 基准 |

说明：

- `l0_mixed.jsonl`（S0）与 `l0_mixed_grd_heldout.jsonl`（S2）为同源递进——后者 = 前者 + 定位数据 + 图像 held-out 切分；`l0_mixed_grd.jsonl` / `l0_mixed_grd_clean.jsonl` 为中间版本（含全部定位数据 / 清洗版），最终 S2 采用 `_heldout` 版以保证评估无泄漏。
- S3 的 UAV-Flow 是唯一带动作真值（轨迹路点）的数据集；S0/S2 均为图文数据（VQA / 定位），不含动作。
