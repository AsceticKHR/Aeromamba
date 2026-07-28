# AsyncVLA 精读报告：异步架构对 AeroMamba / AeroStream 的研究启发

- 日期：2026-07-16
- 文献对象（同名两篇，均精读）：
  1. **Jiang et al.** — *AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models*（arXiv:2511.14148）
  2. **Hirose et al.** — *AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge*（arXiv:2602.13476）
- 对照基线：AeroMamba-Opt（`[state8|delta|vision|text]` / SigLIP2 / 64q resampler / Mamba-2-370M / chunk=8）与 AeroStream 设计书（`docs/AEROSTREAM_TOPTIER_DESIGN_PROPOSAL.md` v1.5）
- 方法：原文 HTML 精读 + 消融/附录交叉核验；非综述堆砌
- AI 披露：AI 辅助检索与撰写；关键数字与机制均回原文核对

---

## 0. 一句话结论

两篇 AsyncVLA 回答的是**两个不同层级的“异步”问题**：Jiang 版解决**动作 chunk 内部**“同步去噪 → 错误级联”；Hirose 版解决**系统控制环**“大模型延迟 → 导航失控”。对 AeroMamba 而言，后者与 UAV 边缘部署更同构，前者则提示：**chunk 级自信度感知与选择性重生成**可成为 AeroStream 流式记忆之外的第二层可靠性机制。二者都不应原样照搬（AeroMamba 当前是 MLP 直接回归而非 flow matching；骨干是 Mamba 而非 8B Transformer），但抽象出的三条原则——**部分更新、置信门控、训练-推理延迟对齐**——可直接进入下一轮实验设计。

---

## 1. 两篇同名论文：问题定义对照

| 维度 | Jiang AsyncVLA（AFM） | Hirose AsyncVLA（Edge） |
|---|---|---|
| 核心痛点 | SFM 对全部 action token 用同一时间表去噪，无上下文、无自纠错；单步误差在长程任务级联 | 大 VLA 推理/通信延迟破坏控制环；移动机器人需高频反应动态障碍 |
| “异步”含义 | **生成过程异步**：不同 action token 可处在不同 FM 时间 $\tau$，低置信 token 被 remask 再生成 | **控制频率异步**：慢外环（远程大 VLA）+ 快内环（机载 Edge Adapter） |
| 域 | 桌面操作（LIBERO / WidowX / Google Robot / 真机 PiPER） | 地面导航（姿态/语言条件导航，含行人动态） |
| 骨干 | Qwen2.5-VL-3B + FM action head | OmniVLA ~8.26B（远程）+ Edge Adapter 76M（Orin） |
| 动作表示 | 连续 action chunk，velocity 参数化 FM | 2D pose chunk（N=8，3Hz → 2.4s 视界） |
| 与 UAV 距离 | 机制可迁移，任务域远 | 任务同为移动导航，部署形态近 |

**精读立场**：对 AeroMamba/UAV 研究，**应以 Hirose 版为系统架构主参照，以 Jiang 版为动作可靠性辅参照**；后文分两节精读，再合并启发。

---

## 2. Jiang AsyncVLA 精读（异步流匹配 + 置信自纠错）

### 2.1 问题诊断（WHY）

主流连续动作 VLA（$\pi_0$ / $\pi_{0.5}$ / WALL-OSS / EO-1）用 flow matching，但采用 **synchronous FM（SFM）**：chunk 内所有 action token 共享同一噪声时间 $\tau$，同步从噪声推到动作。作者论断：

1. SFM **不利用已生成动作的上下文**；
2. 无 **自纠错** 机制；
3. 长程/高精度任务中，一次错误动作可导致不可恢复失败。

这与“离散 diffusion VLA 的 remask”形成对照：后者对离散 token 可做二次掩码，但对连续动作尚缺统一框架。

### 2.2 方法（HOW）——三件套

**（a）SFM → Confidence rater → AFM**

```
观测 o_t + 指令 ℓ
        │
        ▼
   SFM（均匀时间表，全 mask 特例）──► â^SFM
        │
        ▼
   Confidence rater（4 层 Transformer + sigmoid，308M ≈ 总参 7.56%）
        │  p_l ∈ (0,1)，m_l = 1{p_l < T}，默认 T=0.5
        ▼
   AFM：高置信 token 保留为上下文；低置信 token 重新加噪并异步去噪
```

AFM 更新（原文式 1）：仅对 mask 位置做 Euler 步进

$$\hat{\boldsymbol{a}}^{\tau-\delta}=\hat{\boldsymbol{a}}^{\tau}-\delta\,V_\theta(\cdot)\odot\boldsymbol{m}$$

起始噪声（式 4）：未 mask 位置用 SFM 结果，mask 位置用高斯噪声。于是 AFM 在“半干净上下文”条件下修正可疑步。

**（b）异步时间嵌入**

对每个 token 用 $\mathcal{S}(\tau\boldsymbol{m})$ 的正弦时间嵌入，与投影后的噪声动作拼接再经 MLP，使骨干能区分“正在去噪”与“已锁定为上下文”的 token。动作生成用 **full attention**（引用 CoT-VLA）。

**（c）统一训练（关键工程贡献）**

- 把 SFM 视为 AFM 的 **全 mask 特例**；
- 训练时 mask 的每个元素 i.i.d. Bernoulli($y$), $y\sim U(0,1)$；
- 未 mask 的上下文注入小噪声 $\sigma_c=0.05$（式 7–8），缓解 train–test 的 exposure bias（推理时上下文是不完美的 SFM 而非 GT）；
- 损失只在 mask 位置上算 velocity MSE（式 6）；
- $\tau\sim\mathrm{Beta}(1.5,1)$（沿用 $\pi_0$）。

置信伪标签（式 9）：对 SFM chunk 的 token-wise MSE 做 **chunk 内 min-max 相对归一化**，映射到约 $[0.01,0.99]$，再训 rater（骨干冻结）。作者发现这优于轨迹级 TSI（成功/失败）硬标签。

### 2.3 主要结果（WHAT）——已核验数字

| 基准 | AsyncVLA | 强基线 | 备注 |
|---|---|---|---|
| LIBERO 四套平均 SR | **97.4** | $\pi_{0.5}$ 96.9 / dVLA 96.4 | 全套联合微调，非分套模型 |
| WidowX（SimplerEnv）平均 | **70.8** | UD-VLA 62.5 / $\pi_{0.5}$ 57.1 | |
| Google Robot Matching / Variant | **75.3 / 64.3** | $\pi_0$ 71.4 / 54.7 | |
| 真机 PiPER 四任务平均 | **87.0** | $\pi_{0.5}$ 77.0 | 各 50 trial |

**消融（WidowX，Table 4）——对机制归因极关键**：

| 变体 | Avg SR |
|---|---|
| w/o Unified Training | 7.3 |
| w/o AFM（10 / 20 SFM steps） | 47.9 / 51.1 |
| w/o Confidence rater（随机 mask 0.5） | 62.5 |
| Delta / Direct refinement 头 | 61.5 / 62.5 |
| TSI 标签训 rater | 64.6 |
| **完整 AsyncVLA** | **70.8** |

解读：

1. **统一训练几乎是生死条件**（7.3 vs 70.8）——训练目标必须覆盖异步推理模式；
2. **多跑几步 SFM ≠ AFM**（20 步只到 51.1）——需要的是选择性重生成，不是更长同步去噪；
3. **收益主要来自“置信门控的部分更新”**，不是“再加一个 refinement 网络”。

**效率（Appendix B，RTX 4090，双相机，10 FM steps）**：总 95.9 ms；SFM 86.8%，AFM 因复用 VL KV-cache 仅 10.5%，rater 2.7%。

**数据效率**：1/4 LIBERO-Spatial、200 epoch 时，统一训练持续降损并最终 SR 95.8% vs SFM ~86.2%。

### 2.4 作者自承局限（Appendix A.1）

相对置信在 **SFM 全局失败** 时会把“相对最好但仍很差”的 token 当作可信上下文——这是相对伪标签的结构性角落案例。未来需绝对误差校准或 chunk 级可靠性估计。

### 2.5 与离散 diffusion VLA 的边界（Appendix D）

| | Discrete Diffusion VLA | Jiang AsyncVLA |
|---|---|---|
| 动作 | 离散 | 连续 |
| 参数化 | 多为 $x$-pred | $v$-pred |
| Remask | mask token | 高斯噪声 |
| 置信 | logits | 独立 confidence rater |

---

## 3. Hirose AsyncVLA 精读（边缘异步导航）

### 3.1 问题诊断（WHY）

研究问题原文表述得很干净：

> How can large robotic foundation models be deployed on the edge without being constrained by their computational cost?

移动机器人额外难点：

1. 自我中心视角 FOV 有限，必须**持续更新观测**；
2. 行人等动态障碍要求**高频反应**；
3. 机载常无高端 GPU → 远程推理引入 **WiFi 延迟**（实验测得 **0.28–6.0 s**）；
4. 即便机载可跑大模型，功耗也会显著缩短续航。

作者类比经典分层控制：外环慢规划、内环快扰动抑制。

### 3.2 方法（HOW）——三件套

**（1）Edge Adapter（76M）**

输入：

- 远程 base VLA 的 **延迟 action token embeddings**（经 token projector：每 token $4\times4096\to1024$，便于 WiFi 传输）；
- 当前低分辨率图 $I^s_t$（96×96，EfficientNet-B0）；
- **当前与延迟观测的六通道差** $[I^s_t; I^s_{t-k}]$，显式编码“延迟窗口内发生了什么”。

设计纪律：动作头**只吃当前图对应的 transformer 输出 token**，避免被陈旧 base embedding 主导而输出延迟动作。

**（2）Reactive trajectory up-weighting**

比较同轨迹在 $t$ 与 $t-k$ 两套参考 action chunk；若终点位姿距离 $>d_{\mathrm{th}}$（1.0 m），判定为“chunk 内行为突变”（避障/让行），对该样本升权。SACSoN 另优先含行人片段。

**（3）两阶段端到端训练**

1. 冻住 base VLA 主体 $\psi$，从零训 Edge Adapter $\theta$ + token projector $\phi$；
2. 再 E2E 微调 $\{\theta,\phi,\psi\}$（base 侧 LoRA，可训约 5%），对齐异步流。

损失：$J_{\mathrm{im}}$（局部坐标 pose + 相对增量）+ $J_{\mathrm{sm}}$（平滑）。

### 3.3 系统推理协议（Algorithm 1）

- Orin：高频环——采图、缓冲、跑 Edge Adapter、PD 跟航点出速度；
- 工作站：低频环——收图跑 OmniVLA，回传 embeddings + 时间戳；
- 机载用时间戳从图像缓冲取回 $I_{t-k}$，与最新 $I_t$ 一起 refine。

硬件：Vizbot + Jetson Orin 30W；工作站 RTX 4090。

### 3.4 主要结果（WHAT）——已核验数字（Table I）

| 方法 | 部署 | Pose SR↑ | 到达时间↓ | 静/动碰撞↓ | 语言跟随↑ |
|---|---|---|---|---|---|
| OmniVLA-edge 108M | Edge 6Hz | 0.25 | 80.07 | 0.60 / 1.00 | 0.50 |
| OmniVLA 8.26B | WS 5Hz | 0.45 | 70.73 | 0.30 / 1.05 | 0.83 |
| Ours w/o E2E | WS+Edge | 0.25 | 82.78 | 0.60 / 1.05 | 0.75 |
| Ours（全在工作站） | WS | 0.30 | 89.79 | 0.70 / 0.50 | 0.67 |
| **Ours（AsyncVLA）** | **WS 5Hz + Edge 8Hz** | **0.85** | **59.18** | **0.10 / 0.10** | **0.75** |

要点：

1. 相对最强基线约 **+40% SR**（摘要声称与表一致量级）；
2. **E2E 对齐不可或缺**（无 E2E 掉到 0.25）；
3. **仅把 Edge 放到机载**才真正兑现收益（全在工作站仅 0.30）——异步部署本身就是贡献，不是附带工程；
4. 人为加工作站延迟 0.2 / 2.0 / 5.0 s 时，AsyncVLA 退化远慢于纯 OmniVLA；附录称 0.2 Hz 外环仍有约 50% 可达目标。

局限（作者 Discussion）：依赖可 E2E 微调的开源 base；动态交互数据稀缺；未来希望只训 Edge Adapter。

---

## 4. 与 AeroMamba 最新设计的结构性对照

### 4.1 现行 AeroMamba-Opt / AeroStream 状态（对照锚）

| 组件 | AeroMamba-Opt / AeroStream | Jiang AFM | Hirose Edge |
|---|---|---|---|
| 骨干 | Mamba-2-370M（SSM 递归） | Transformer VLM 3B | LLaMA2-7B 级导航 VLA |
| 动作头 | MLP 直接回归 chunk=8 | FM velocity 多步去噪 | Edge Adapter 再出 pose chunk |
| 记忆 | 规划中的 **跨步 cache 流式状态（C1）**；代码已有 `stream_step` | 无跨控制步记忆；chunk 内 AFM 上下文 | 无 SSM 状态；靠延迟差图像补偿 |
| 部署叙事 | C4：Orin 级恒定延迟闭环 | 单机 4090 ~96 ms | **远程大模型 + 机载小模型** |
| 执行协议 | 设计书要求 receding-horizon（exec_horizon=1）；现状 server 仍支持 chunk 执行 | chunk 生成后执行 | Edge 高频重算 chunk + PD |
| 自纠错 | TemporalEnsemble / 损失平滑；无 token 级置信 | **置信 remask** | **用最新观测改写陈旧指导** |
| Token 序 | `[state\|delta\|vision\|text]` | VL + 异步动作 token | base embeddings + 当前/延迟图 |

### 4.2 同构点（可直接借用的抽象）

1. **部分更新原则**  
   Jiang：只更新低置信 action token。  
   Hirose：只让快环用最新观测改写慢环指导。  
   AeroStream：`stream_step` 每步只增量喂新 token，状态 O(1) 更新——已是“部分更新”的 SSM 版本。

2. **训练必须对齐异步推理**  
   Jiang 无统一训练 → SR 崩溃；Hirose 无 E2E → SR 崩溃。  
   对应 AeroStream S4：**TBPTT + 推理端 receding-horizon** 必须成对出现（μVLA 纪律已写入设计书；AsyncVLA 两篇从不同角度再次强化）。

3. **延迟不是测试后补丁，而是训练分布**  
   Hirose 随机抽延迟观测训 Edge；Jiang 对上下文加 $\sigma_c$ 噪声。  
   AeroMamba 若做云边协同或长 `exec_horizon`，必须在 Stage3/4 **显式采样延迟/陈旧指导**，否则上线必掉点。

4. **快慢双系统 ≠ 单纯蒸馏小模型**  
   Hirose 表中 OmniVLA-edge  alone SR 仅 0.25；大模型语义 + 小模型反应缺一不可。  
   AeroMamba 370M 已偏“可机载”，但仍可拆：**语义/记忆外环（完整 Mamba+VLM）** vs **反应内环（轻量 adapter）**。

### 4.3 异构点（不可硬搬）

| 不可硬搬之处 | 原因 |
|---|---|
| 原样 AFM + confidence rater | AeroMamba 动作头是单次 MLP 回归，无 FM 多步去噪与 KV-cache 复用叙事 |
| 8B 远程 + 76M Edge 的默认拓扑 | 370M Mamba 目标是整机机载；远程拓扑是**可选扩展**，不是主叙事 |
| Full-attention 动作 token 交互 | Mamba 因果扫描；chunk 内双向依赖需另做（小窗注意力 / 动作头内交互） |
| 相对 MSE 置信伪标签 | 需可定义的“第一次预测误差”；MLP 一次前向没有天然 SFM 第一轮 |

---

## 5. 对 AeroMamba 研究的具体启发（可执行提案）

按与现有路线的耦合度排序。**A/B 为可立即做；C 为中期架构扩展。**

### A. 强化已有 AeroStream 主线（最高优先级，与设计书 C1/C4 同向）

**启发来源**：Hirose 的“快环必须吃最新观测”+ Jiang/μVLA 的“训练-推理节奏一致”。

| ID | 提案 | 做法 | 验收 |
|---|---|---|---|
| A1 | **把异步执行写进训练** | Stage4：以概率 $p$ 将“指导帧”设为 $t-k$（$k\sim\{0..K_{\mathrm{delay}}\}$），当前帧仍为 $t$；损失对应当前帧动作。模拟通信/算力抖动 | 人为延迟 0/2/5 步下的 SR/nDTW；对照无延迟增强 |
| A2 | **强制 receding-horizon 为默认推理** | `inference/server.py`：`exec_horizon=1` 为 AeroStream 默认；chunk>1 仅作平滑候选 | 复现 μVLA 式崩溃曲线：horizon∈{1,4,8} |
| A3 | **延迟差作为显式输入（轻量）** | 在 proprio/delta 旁增加 `obs_age` 或 `Δt_since_last_backbone` 标量嵌入；或拼接上一关键帧低分辨率差（Hirose 六通道思想的 1/10 成本版） | 动态避障子集碰撞率 |

A1–A3 不改变“SSM 状态即飞行记忆”的主贡献，而是让 C4 边缘效率叙事在**真实延迟分布**下站得住。

### B. Chunk 级置信与选择性重生成（中优先级，移植 Jiang 精神而非公式）

AeroMamba 无 FM，但可构造**一次前向 + 廉价二次修正**：

| ID | 提案 | 做法 | 备注 |
|---|---|---|---|
| B1 | **动作置信头** | 在 `global_token` 上挂小 MLP，预测每步/每维不确定度（可用 MC dropout、异方差头，或对 SFM 式“两遍预测”的差分作伪标签） | 伪标签避免轨迹级成功/失败（Jiang 消融已否 TSI） |
| B2 | **选择性重查询** | 若未来步置信低：仅对低置信步加大噪声重采样 / 或触发一次额外 `stream_step`；高置信步锁定 | 对应 AFM 的部分更新；计算预算可控 |
| B3 | **与 TemporalEnsemble 分工** | Ensemble 做时间平滑；置信门控做**内容级**纠错——二者正交 | 避免“再加平滑当自纠错” |

建议实验顺序：先 B1 校准（置信 vs 真实 endpoint 误差相关），再 B2 闭环增益。

### C. 云边 / 双频异步拓扑（中长期，对齐 Science Robotics C4）

当 370M 仍不够语义、或需上更大视觉塔时，Hirose 拓扑几乎可直接映射：

```
机载（Orin）                              云端 / 地面站
----------------                          ----------------
高频：最新 FPV + Mamba cache 步进          低频：更大 VLM / 规划 VLA
      Edge / Action Adapter  <--- 压缩指导 embedding + 时间戳 ---┘
      （token projector 接收端 + 图像时间戳缓冲）
```

与 AeroStream 的差异化：**内环不只是 CNN adapter，而是携带 SSM 飞行记忆的轻量步进**——这是 Hirose（无跨步状态）与 Jiang（无部署分层）都未占据的位置。

论文叙事锚点：

> AsyncVLA（Hirose）证明远程语义 + 机载反应在导航上必要；μVLA/AeroStream 证明记忆必须与控制频率对齐；**AeroMamba 把二者收束为“可延迟的语义外环 + 带状态记忆的机载内环”。**

### D. 数据侧：反应轨迹升权（低成本，可立刻做）

Hirose 的 $A^t$ vs $A^{t-k}$ 终点距离阈值，在 UAV-Flow 上可改写为：

- 同一轨迹窗口内 **yaw / 高度突变**、或 **与匀速外推偏差大** 的 chunk 升权；
- 与现行“转向 3× 过采样 / 运动原语过采样”合并，而不是另起炉灶。

这直接服务 stage3_v2 诊断中的失败模式（直线惯性、Surround 失效、终点刹不住）。

### E. 明确不建议作为近期主线的方向

1. **为异步而改成 flow-matching 动作头**：与现行 MLP chunk、Stage3 损失栈、checkpoint 兼容成本过高；除非单独开“生成式动作头”子课题。  
2. **308M 级 confidence rater**：对 370M 总预算不成比例；置信头应 <5–10M。  
3. **用 AFM 替代跨步状态记忆**：二者解决不同时间尺度问题（chunk 内 vs 飞行全程），替代会丢 C1。

---

## 6. 建议的研究问题（可写进下一轮 RQ）

**RQ-Async（建议）**：在 SSM-based UAV VLA 中，将控制环拆成“可延迟的语义/记忆外环”与“高频机载内环”时，如何在训练中显式对齐观测陈旧分布与状态更新节奏，才能在通信/推理延迟下保持语言条件飞行的成功率与安全性？

子问题：

1. 仅流式状态（AeroStream C1）在人为延迟下能撑多久？缺 Edge 式当前观测修正时退化曲线如何？  
2. Chunk 置信门控（Jiang 精神）能否在不引入 FM 的前提下降低长程误差级联？  
3. 反应轨迹升权对 Surround/避障类原语的边际收益是否大于单纯类过采样？

---

## 7. 与现有报告的衔接（避免重复劳动）

| 已有结论 | AsyncVLA 增量 |
|---|---|
| μVLA：exec_horizon 与记忆更新必须同频 | Hirose 用**秒级网络延迟**把同命题推到更极端；A1/A2 应升级为“延迟鲁棒”主实验而非附属 |
| AeroStream C4：Orin 10Hz | Hirose 给出可引用的双机部署协议与评测表格式（SR / 碰撞 / 延迟扫描） |
| TemporalEnsemble | 定位为平滑；自纠错应另建置信/重生成（B1–B2） |
| 指令绑定头（v1.5 P0） | 仍是绑定问题主修复；异步机制是**部署与长程可靠性**层，不替代绑定头 |

推荐排期插入点（相对 AeroStream §9）：

- **W1–2（本地）**：A2 推理协议 + D 反应升权统计脚本；  
- **绑定轮 Stage3 之后**：A1 延迟增强小规模消融；  
- **S4 流式训练并行**：B1 置信头零初始化挂上；  
- **实机/Science Robotics 冲刺前**：C 云边原型（可先同机双进程模拟 WiFi 延迟）。

---

## 8. 精读总评

**Jiang AsyncVLA** 的学术贡献在于：把离散 diffusion 的 remask 思想落到连续 FM，并用统一训练 + 相对置信证明“选择性异步修正 > 更长同步去噪 / 额外 refinement 头”。对操作域 SOTA 数字强，但对 UAV 的直接迁移面窄。

**Hirose AsyncVLA** 的学术贡献在于：把分层控制真正做成可训练的 VLA 系统，并在**最高 6 s 延迟**的真实导航中给出 +40% 级收益；E2E 与机载部署两项消融把贡献钉死在“异步系统”而非“又一个小模型”。对 AeroMamba 的边缘飞行叙事几乎是同题作文。

对 AeroMamba：**异步不是要改成另一套动作生成器，而是要承认——飞行 VLA 同时存在“chunk 内误差级联”与“控制环延迟”两类异步问题；SSM 流式记忆解决第三类（跨步历史），三者正交叠加才构成完整可靠性栈。**

---

## 参考文献（精读源）

1. Jiang, Y., Cheng, S., Ding, Y., Gao, F., & Qi, B. (2025). *AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models*. arXiv:2511.14148. https://arxiv.org/abs/2511.14148  
2. Hirose, N., Glossop, C., Shah, D., & Levine, S. (2026). *AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge*. arXiv:2602.13476. https://arxiv.org/abs/2602.13476  
3. 项目内对照：`docs/AEROSTREAM_TOPTIER_DESIGN_PROPOSAL.md`；`reports/aeromamba_mamba_structural_optimization_20260714.md`；`model/uav_mamba_vla.py`（`stream_step`）
