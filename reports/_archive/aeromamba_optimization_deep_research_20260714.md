# AeroMamba 空间能力优化：文献深度调研与可信设计提案

- 日期：2026-07-14
- 方法：deep-research（three-way-scan + lit-review 混合）；全部关键论断经原文核验
- 动机：Stage2 v2 在 Open3D-VQA seen probe 上 exact 33.7%（direction 48.6% / distance 8.2%），需要有据可依的优化设计
- AI 披露：本报告由 AI 辅助检索与撰写，所有引用均经独立核验存在性与原文表述

---

## 一、证据基础（WHY / HOW / WHAT + 来源分级）

分级标准：T1 = 已通过同行评审；T2 = arXiv 预印本（作者署名）；T3 = 匿名在审稿件（仅参考，不作为设计依据）。

### 1. SpatialVLM（CVPR 2024）— T1

- **WHY**：VLM 缺定量空间推理（距离/尺寸），根因是训练数据缺 3D 度量知识。
- **HOW**：自动生成 20 亿条度量空间 VQA（检测+度量深度+分割管线），与通用数据共训。
- **WHAT**（已核验原文 Table 3）：微调后定量距离命中率 —— 解冻 ViT：**[0.5,2]× 37.2% / [0.67,1.5]× 10.7% / [0.9,1.1]× 8.4%**；冻结 ViT 分别 34.9/9.3/5.6。结论①**米制距离对所有模型都极难**；②**解冻视觉塔对细粒度距离"considerably better"**。
- **对 AeroMamba 的含义**：我们 distance_quant 8.2%@[0.75,1.25] 与该领域水平持平，不是独有缺陷；视觉塔全程冻结是可疑设计点。

### 2. SpatialRGPT（NeurIPS 2024）— T1

- **WHY**：RGB-only VLM 几何推理弱；直接拼接 RGB+深度特征会损害性能。
- **HOW**：**相对深度图（离线单目估计）走同一个视觉编码器，另配独立 depth-to-language connector**（从 RGB connector 初始化），仅在空间 QA 上训练该 connector。
- **WHAT**：SpatialRGPT-Bench 上显著超过 GPT-4V（区域级空间任务）；证明"深度插件"式融合可行且不需要重训主干。
- **含义**：AeroMamba 的 `MLPProjector` 可以按同款思路复制一份做 depth connector。

### 3. AutoFly（arXiv 2602.09657，UAV VLA）— T2

- **WHY**：UAV 3D 飞行强依赖深度做避障/高度控制，纯 RGB VLA 空间推理不足；真实深度传感器噪声大、加重 sim-to-real gap。
- **HOW**：**Depth Anything V2 伪深度 + Siamese MLP projector（与视觉分支共享参数）**；两阶段训练（对齐→空间感知动作微调）。消融显示 Siamese 共享参数优于独立 projector 和直接注入（已核验原文）。
- **WHAT**：伪深度分支带来 SR +3.9% / 碰撞率 −2.6%（二手来源 papernotes 摘录，量级供参考）。
- **含义**：与 AeroMamba 同为 UAV 域、同为 LLaVA 式投影结构，迁移路径最短。

### 4. UAV-Track VLA（arXiv 2604.02241）— T2

- **WHY**：通用 VLA 高层语义特征与低层连续控制失配，缺精确空间几何先验。
- **HOW**（已核验原文）：**旁路空间辅助 grounding head**——backbone 后挂小型 transformer，训练期预测目标相对 UAV 的 3D 位置 + yaw 偏差，与 flow-matching 动作损失加权联合；**仅训练期反传，推理零开销**；双分支共享 cross-modal 特征避免表征冲突。
- **WHAT**：辅助空间监督把几何先验压进 encoder token，为连续控制提供特征基础。
- **含义**：AeroMamba Stage3 每条 UAV-Flow 轨迹自带相对位移标签，辅助头监督信号免费。

### 5. Lost in Space（NAACL 2024）— T1

- **WHY**：Perceiver/Q-Former 类 resampler 压缩视觉 token，但是否保留细粒度空间信息未被检验。
- **HOW**：对 BLIP-2/InstructBLIP 的 resampler 输出做诊断线性探针（RefCOCOg 视觉 grounding）。
- **WHAT**（已核验原文）：**冻结 resampler 时空间信息基本缺失；resampler 与探针联合训练则显著提升**——压缩机制本身能编码空间信息，但对比学习/CLM 目标不足以促成，需要 object-aware 目标。
- **含义**：AeroMamba 的 64-query PerceiverResampler 是潜在空间瓶颈；解法不是丢弃 resampler，而是给它 object-aware 训练信号 + 位置保持。

### 6. PPE（arXiv 2510.22936）+ Nüwa（arXiv 2602.02951）— T2

- **WHY**：视觉 token 压缩破坏空间布局，grounding 类任务依赖全局空间参考系。
- **HOW**：PPE 给每个压缩 token 保留多重时空位置 ID（免参数、即插即用）；Nüwa 两阶段空间感知剪枝（保持空间均匀覆盖 + 文本引导细筛）。
- **WHAT**：PPE 在 MMBench/TextVQA 等 +2~5%；Nüwa 揭示「保空间拓扑」三原则。
- **含义**：给 resampler 的输入/queries 显式注入 27×27 网格位置信息是低成本高置信改动。

### 7. RoboMamba（NeurIPS 2024）— T1

- **WHY**：VLA 推理不足 + 训练/推理开销大。
- **HOW**：视觉编码器 + Mamba LLM 共训对齐建立推理能力；随后**只训一个简单 policy head（0.1% 参数）**习得 SE(3) 位姿预测。
- **WHAT**（已核验原文）：一旦推理能力足够，动作能力可用极小代价获得；推理速度 3× 于现有 VLA。
- **含义**：AeroMamba「Stage2 建推理、Stage3 轻量动作头」的总设计与该结论一致，**架构路线不需要推翻**；瓶颈定位在 Stage2 空间推理质量，而非动作头容量。

### 8. μVLA（arXiv 2606.12497）— T2

- **WHY**：VLA 从单帧观测预测动作 chunk，部分可观测场景（信息消失后需回忆）失效。
- **HOW**（已核验原文）：最小递归记忆——可学习 memory token 跨步携带，**TBPTT 端到端训练（K=2 已够强，K=8 更稳）**；推理用 **receding-horizon：每步重查询、只执行 chunk 第一步**，保证记忆更新节奏与训练一致。
- **WHAT**：MIKASA-Robo 训练任务 SR 0.42→0.84；关键警告——**若推理端每 H 步才更新一次记忆（chunk 整段执行），cue-recall 任务性能崩溃**。
- **含义**：直接约束 AeroMamba 流式闭环方案：Mamba 递归状态 + `exec_horizon=4` chunk 执行存在 μVLA 指出的**训练/推理节奏失配**，需要配套调整。

### 9. Open3D-VQA（ACM MM / arXiv 2503.11094）— T1

- **WHAT**（已核验原文）：微调 LLaVA-1.5/Qwen2-VL 后 **size +10%以上、direction +5%以上，distance 提升 marginal**；SAQ 定量评分规则 [0.75,1.25]。
- **含义**：投入产出比排序应为 方向/尺寸 > 距离；距离不值得作为主攻目标。

### 10. SpatialVLA-Mamba（ICLR 2026 在审，匿名）— T3

- 声称显式几何编码 + Mamba 解码器提升长时程任务。**在审匿名稿，仅作方向参考，不作为设计依据。**

### 跨文献综合

- **共同 WHY**：RGB-only + 压缩投影 + 无 3D 监督 → 空间（尤其度量）推理弱。这正是 AeroMamba 的配置。
- **分歧 HOW**：数据路线（SpatialVLM 海量度量 QA）vs 结构路线（深度分支/辅助头/位置保持）。小模型（370M）容量有限，**结构注入几何先验比堆数据更划算**。
- **最强 WHAT**：辅助空间监督（训练期）与深度插件在多篇独立工作中重复验证有效——可信度最高。
- **未解 Gap**：Mamba 主干（非 Transformer）上的空间 grounding 系统研究仍空白（仅 T3 在审稿），AeroMamba 的改动需自行 A/B 验证。

---

## 二、优化设计提案（按证据强度 × 实施成本排序）

### P1. Stage3 旁路空间辅助头 【证据 T1/T2 双源；成本 ~1-2 天】

- 做法：Mamba 最后 hidden 后挂 2 层 MLP 辅助头，预测「轨迹终点相对当前位姿的 body-frame 方向角 + 对数距离」，损失 `L_total = L_action + λ_aux * L_spatial`（λ 从 0.1 起调）。标签由 UAV-Flow `raw_logs` 直接计算，零标注成本。
- 依据：UAV-Track VLA 旁路设计（推理零开销）+ SpatialVLM「辅助度量监督塑形表征」。
- 验证：Stage3 后跑 UAV-Flow-Eval，对比 nDTW 与转向类任务成功率。

### P2. Resampler 位置保持 + object-aware 目标 【证据 T1；成本 ~0.5-2 天】

- 做法（两步）：
  1. 给 resampler 输入 token 加 27×27 二维位置编码，queries 加可学习位置嵌入（PPE 思路的最小实现）；
  2. Stage2 数据混入 region-level QA（"图像左上/右下象限有什么"、粗略 bbox 坐标问答），用 O3DVQA 管线的 bbox 元数据自产，给 resampler 提供 Lost in Space 证明必需的 object-aware 信号。
- 验证：重跑 Open3D probe，重点看 direction / other_qual 桶。

### P3. 伪深度 Siamese 分支 【证据 T1/T2 双源；成本 ~3-5 天】

- 做法：Depth Anything V2 离线生成 FPV 深度图 → 深度图复用 SigLIP2 编码 → **与 RGB 分支共享 MLPProjector**（AutoFly 消融证明共享优于独立）→ 深度 token 拼在 vision token 后。O3DVQA 仿真场景自带 GT depth `.npy` 可校准伪深度质量。
- 风险控制：先只在 Stage2 引入（VQA 收益可直接量化），Stage3 是否携带视推理延迟预算而定（4060 8GB 需实测双份视觉编码开销）。

### P4. 流式闭环必须配 receding-horizon 【证据 T2（μVLA 强消融）；成本：改推理协议】

- 现状风险：已计划的 Mamba 递归状态闭环若保持 `exec_horizon=4`（每次执行 4 步才回传新观测），等价于 μVLA 证明会崩溃的「每 H 步更新一次记忆」模式。
- 做法：启用递归状态后，**exec_horizon 降为 1（每步重查询、只执行 chunk 首步）**；Mamba 增量推理快（125ms 级），10Hz 预算内可行。训练侧 Stage3 改按轨迹连续采样 + TBPTT（K=2 起步，μVLA 证明 K=2 已捕获大部分收益）。
- 这是「递归状态」计划从可能有害变为可靠收益的关键配套。

### P5. 距离题改分箱 + 视觉塔尾部 LoRA 【证据 T1；成本 ~1 天 + 训练开销】

- 分箱：距离答案模板化为对数分箱区间（<5m / 5-10m / 10-20m / 20-40m / >40m），从生成数值回归改为分类式选择。依据：SpatialVLM 显示自由数值最难；Open3D-VQA 显示距离微调收益 marginal——**用低成本方式止损，不作主攻**。
- 视觉塔：SigLIP2 最后 2-4 block 加 LoRA 在 Stage2 放开（SpatialVLM Table 3：解冻 ViT 对细粒度距离 8.4% vs 5.6%）。
- 预期校准：距离 exact 从 8% 提到 20-35%（分箱后与 SpatialVLM [0.5,2]× 37.2% 可比口径），不应预期更高。

### P6. 干净评测协议（所有改动的前提）【成本：数据重建】

- 下一轮 Stage2 训练集剔除 Open3D-VQA Real 全量 + Sim 10%（按论文 80/10/10），已有 `build_open3d_vqa_probe_split.py` 的 id-hash split 可直接复用为剔除清单。
- 没有这一条，P1-P5 的任何收益都无法可信量化。

### 明确不建议做的

- **重造 Mamba 主干加 cross-scan**（VMamba/2D-CrossScan 思路）：改造成本高，且 AeroMamba 视觉 token 经 resampler cross-attention（无方向偏置）进入 Mamba，2D 失配已被部分缓解；P2 优先。
- **把距离定量当主攻**：三篇 T1 文献一致表明收益边际。
- **依赖 SpatialVLA-Mamba 的设计**：T3 在审稿，等录用后再评估。

---

## 三、实施顺序建议

```
第 1 周   P6 数据重建（后台） + P1 辅助头 + P2a 位置编码 → Stage3 增量重训 A/B
第 2 周   P2b region QA + P5 分箱 → Stage2 增量微调（干净 hold-out 上量化）
第 3-4 周 P3 深度分支（Stage2 先行）＋ P4 流式闭环改造（与递归状态计划合并）
```

每步用固定三件套量化：Open3D 干净 hold-out（direction/tf/other/distance 分桶）→ Stage2 四源能力面板 → UAV-Flow-Eval 闭环（nDTW + SR）。

---

## 四、局限性声明

1. UAV 域三篇（AutoFly/AerialVLA/UAV-Track VLA）均为预印本，数字未经同行评审；AutoFly 增益引自二手笔记站点。
2. 所有文献均基于 Transformer 主干（除 RoboMamba/VL-Mamba），结论迁移到 Mamba2-370M 存在不确定性，效果需自行 A/B。
3. 效果预估（如距离 20-35%）是跨论文口径换算，非承诺值。
4. 本报告检索时间窗截至 2026-07；ICLR 2026 在审工作可能后续更新。

## 参考文献

1. Chen, B. et al. (2024). SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. *CVPR 2024*. arXiv:2401.12168
2. Cheng, A.-C. et al. (2024). SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models. *NeurIPS 2024*. arXiv:2406.01584
3. AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild. arXiv:2602.09657
4. UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models. arXiv:2604.02241
5. Pantazopoulos, G. et al. (2024). Lost in Space: Probing Fine-grained Spatial Understanding in Vision and Language Resamplers. *NAACL 2024*. arXiv:2404.13594
6. PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models. arXiv:2510.22936
7. Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning. arXiv:2602.02951
8. Liu, J. et al. (2024). RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation. *NeurIPS 2024*. arXiv:2406.04339
9. μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models. arXiv:2606.12497
10. Zhang, W. et al. (2025). Open3D-VQA: A Benchmark for Comprehensive Spatial Reasoning with Multimodal Large Language Model in Open Space. *ACM MM*. arXiv:2503.11094
11. Qiao, Y. et al. (2024). VL-Mamba: Exploring State Space Models for Multimodal Learning. arXiv:2403.13600
