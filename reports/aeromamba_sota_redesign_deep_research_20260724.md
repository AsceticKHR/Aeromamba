# AeroMamba 选型 / 训练 / 数据的 SOTA 改进研究报告

> Deep-Research（full 模式，applied engineering research）· 2026-07-24
> 目标：调研相关研究，改进当前 Mamba-VLA 的模型选型、训练流程与数据；要求充分利用 Mamba 的结构优势，训练设计尽量高效优雅。
> AI 披露：本报告由 AI 辅助研究工具完成检索与综合，所有事实性论断均附一手来源，作者应在采用前二次核验。

---

## 执行摘要（Executive Summary）

一手文献（RoboMamba NeurIPS'24、Cobra AAAI'25、SmolVLA、TinyVLA、Discrete-Diffusion-VLA、SpatialVLA RSS'25、VLM2VLA 2025、Mamba/Mamba-2 原论文）给出五条对 AeroMamba 直接可用的结论：

1. **架构方向被验证**：Cobra 用 `DINOv2+SigLIP → MLP projector → Mamba` 的完全相同管线，比同规模 Transformer VLM 快 3–4×、以 ~43% 参数比肩 LLaVA-7B。我们的 DinoSigLIP + Mamba 路线是正确的（Chen et al., 2024）。
2. **"先推理、后动作"是高效范式**：RoboMamba 证明，只要 Mamba 主干具备足够推理能力，一个占 **0.1% 参数**的极简 MLP 策略头就能以极低成本习得动作，推理快 3×（Liu et al., 2024）。→ 不该把预算耗在重头设计上。
3. **分离式回归/扩散头会破坏预训练能力**：多项工作发现，外挂连续扩散/flow-matching 头"过度依赖视觉"并引发灾难性遗忘；把动作**离散化成 token 走语言头**、或**梯度隔离**是当前主流解（Discrete Diffusion VLA, 2025；NVIDIA WAM Blog, 2025）。→ 我们外挂、正在塌缩的 bbox/动作回归头与此教训一致。
4. **空间能力来自"位置编码注入"，不是外挂框头**：SpatialVLA 把深度反投影成 Ego3D 位置编码，正弦+MLP 后**加到 SigLIP 视觉特征上**；去掉它成功率从 81.6% 掉到 68.9%（Qu et al., 2025）。→ 直接指向我们 grounding 的正解，且让 AirZoo 深度/Open3DVQA 有了真正用途。
5. **优雅训练配方已成型**：冻结视觉编码器 + LoRA-only（VLM2VLA 避免遗忘）、动作头只读**中间层**特征（SmolVLA 省算力）、机器人:视觉语言 **50/50 共训**、跳过 OpenX 级昂贵预训练（TinyVLA 数据高效）、异步推理 + action chunking 支撑实时控制。

**一句话建议**：保留 Mamba+DinoSigLIP 主干，**把"外挂回归头"范式换成"语言头 token 化输出 + 位置编码注入"**，训练收敛到"冻结编码器 + Mamba-LoRA + 50/50 共训 + 极简/中间层头"的两阶段流程，并把 Mamba 的**恒定显存流式递归**用在时序观测与 10Hz 实时控制上——这是我们相对 Transformer-VLA 最大的、尚未兑现的结构红利。

---

## Phase 1 — 立题与方法蓝图（Scoping）

**主研究问题 (RQ)**：在冻结视觉编码器、显存受限、需 10Hz 实时控制的约束下，如何改进 AeroMamba 的（a）视觉编码器/主干选型、（b）动作与 grounding 输出头、（c）训练流程与数据配方，使其**充分利用 Mamba 的线性/递归/恒定显存结构优势**，并在工程上高效优雅？

**子问题**：
- SQ1：SSM/Mamba 系 VLM/VLA 如何组织视觉 token 与主干？其相对 Transformer 的结构优势能为 UAV 实时控制带来什么？
- SQ2：动作头设计（离散 token / L1 回归 / 扩散 / flow-matching / 语言即动作）在精度、OOD、遗忘、效率上的权衡？
- SQ3：grounding/空间能力在 SOTA VLA 中如何实现（外挂框头 vs 位置编码 vs pointing/token）？
- SQ4：如何在不灾难性遗忘、不做昂贵预训练的前提下，设计高效共训配方与数据？

**范围**：聚焦架构/训练/数据方法论迁移；不含具体超参搜索与全量复现实验（列为后续消融）。
**方法**：结构化文献检索（arXiv/会议/官方实现）+ 证据分级（同行评审>预印本>博客/官方文档）+ 跨源综合 + 反方检查。FINER 自评：可行、有趣、对本项目高度相关、无伦理风险、具新颖工程整合价值。

**DA 检查点 1（通过）**：RQ 可答、方法与问题匹配、范围适中（限定为"方法迁移+选型建议"，不越界到全量实验）。风险：证据多来自机械臂 manipulation 域，需在"UAV 导航航点"域做外推标注（见局限）。

---

## Phase 2/3 — 证据综合（Investigation + Synthesis）

### 主题 A：Mamba 结构优势与主干（SQ1）

- **恒定显存、恒定单步延迟、线性训练**：Mamba 把历史压缩进**固定大小隐状态**并递归更新，推理每步开销与序列长度无关，无 KV cache；适合流式/长时序/实时，per-token 延迟恒定（Gu & Dao, 2023；Mamba-2/SSD：Dao & Gu, 2024；General Compute, 2025）。**这对 10Hz UAV 控制是决定性红利**：可把多帧观测+本体历史当序列喂入而延迟不膨胀。
- **短板**：精确检索/少样本 in-context learning 弱于注意力，常用混合架构补偿（IBM, 2025）。对我们影响小（我们不做长检索）。
- **Cobra**（AAAI'25）：`DINOv2+SigLIP 通道拼接 → MLP projector → Mamba` 的 MLLM，比 MobileVLM-v2-3B/TinyLLaVA-3B 快 3–4×，~43% 参数比肩 LLaVA-7B，并降低幻觉（Zhao et al., 2024）。**与我们 DinoSigLIP 路线完全一致，是最直接的架构背书**。
- **视觉编码器选型**（详见 20260724 选型清单，另表）：单塔聚合式 **NVIDIA C-RADIO-B(90M)** 一次前向即得 CLIP 语义+DINO 几何+SAM 稠密，避免双塔 2× 开销；或 **DINOv3(B/卫星-L) + SigLIP2** 双塔取质量上限；robotics 专用 **Theia** 最省但 224px 偏粗（Meta DINOv3, 2025；NVIDIA RADIOv2.5, CVPR'25；Theia, CoRL'24；SigLIP2, 2025）。

### 主题 B：动作/输出头设计（SQ2）

证据一致指向**"离散 token / 语言化输出 + （必要时）梯度隔离"**优于"外挂连续头"：

| 方案 | 精度(ID) | OOD/长时序 | 遗忘风险 | 效率 | 代表 |
|---|---|---|---|---|---|
| L1 回归 (ACT/极简 MLP) | 高 (LIBERO 97.1%) | 较弱 | 中 | 极高 | RoboMamba, OpenVLA-OFT(L1) |
| 连续扩散 / flow-matching | 最高精度、平滑 | 视觉退化最重(↓29%) | **高**（头"过度依赖视觉"） | 中 | π0, Diffusion Policy, SmolVLA |
| 离散动作 token (FAST) | 高 | 好、保留 VLM 先验 | 低 | 高（可并行） | OpenVLA-OFT(Discrete), π0-FAST |
| 离散扩散 (统一 transformer) | 96.4% | **OOD 最佳**(语言退化仅0.8%) | 低 | 高（并行解码） | Discrete Diffusion VLA (2025) |
| 语言即动作 (数字串) | 高 | 好 | **最低**（LoRA 即可） | 高 | VLM2VLA (2025) |

关键机理：VLM/Mamba 预训练是**离散 next-token 交叉熵**；连续 flow-matching 目标与之分歧，朴素微调导致**灾难性遗忘**；对策=离散 token 化 + 隔离梯度（NVIDIA WAM Blog, 2025）。**RoboMamba 的启示**：只要主干推理够强，0.1% 参数的极简头即可，无需复杂头（Liu et al., 2024）。

### 主题 C：grounding / 空间（SQ3）

- **SpatialVLA**（RSS'25）：Ego3D Position Encoding——ZoeDepth 估深度→反投影到相机系 3D 坐标→正弦 γ(·)+MLP→**加到 SigLIP 2D 视觉特征**；消融显示去掉后成功率 81.6%→68.9%（Qu et al., 2025）。**这解释了我们 grounding 头塌缩的根因**：视觉特征缺空间/3D 信号，我们此前只手工加了 2D 正弦编码作拐杖。
- **RoboPoint**：VLM 指令微调预测 **2D 指向点**→经深度转 3D，pointing 而非 bbox。
- **通用趋势**：定位能力从**位置编码注入 + 语言头 token/point 输出**获得，而非外挂坐标回归头（与主题 B 呼应）。

### 主题 D：训练流程与数据（SQ4）

- **RoboMamba 两阶段**：①视觉-语言对齐共训（练推理）②极简策略头低成本微调。"推理够强 → 动作易得"（Liu et al., 2024）。
- **VLM2VLA / Actions-as-Language**（2025）：把动作表示成**自然语言数字串**→落入 VLM 既有词表→**仅用 LoRA** 微调即可避免遗忘，无需昂贵 VLM 级共训；并"混合冻结与微调的视觉编码器"保留表征；实测保住 VQA 与零样本泛化（Princeton IROM, 2025）。
- **共训配比**：机器人:视觉语言 **50/50** 采样平衡梯度、防表征坍塌（Preserving Pretrained Representations, 2025）。
- **SmolVLA**（450M）：动作专家只读 **中间层(第 N 层)** VLM 特征而非最后层→显著省算力；**异步推理** +30% 速度、2× 吞吐；社区数据预训练 +26.6% 成功率（HF LeRobot, 2025）。
- **TinyVLA**：小 VLM(<1B)+扩散头，**免 OpenX 级预训练**，LoRA 微调即超 OpenVLA 的速度与数据效率（Wen et al., 2024）。

### 矛盾与空白（Contradictions & Gaps）

- **精度 vs 遗忘**：连续 flow-matching 精度最高但遗忘最重；离散/语言化保先验但量化误差略高。→ 对 UAV 平滑航点，建议**离散 token 为主 + 轻量残差回归修正**折中。
- **域外推**：绝大多数证据来自桌面 manipulation（SE(3) 抓取），**无一针对 UAV 低空导航航点**；Ego3D/Adaptive-Grid 在无标定 UAV 相机、开阔场景的有效性需自证。
- **Mamba×VLA 稀缺**：RoboMamba 是唯一强证据；Mamba 在长时序动作 chunk 上的表现缺乏系统消融——这既是风险也是我们的**创新空位**。

**DA 检查点 2（通过，带保留）**：无 cherry-pick（同时给出各头短板与遗忘代价）；主要外推风险已在"空白"显式声明，进入方案时以"消融验证"而非"直接全量"落地。

---

## Phase 4 — 针对 AeroMamba 的改进方案（Recommendations）

### 1. 选型（Model Selection）
- **主干**：维持 Mamba-2（结构红利见下），保留 `[text|proprio|vision]` 因果序 + 末 token 全局（RoboMamba 约定，已实现）。
- **视觉编码器**（按序 A/B 测）：`C-RADIO-B(90M)` 单塔聚合（首选，等价 base 开销拿双塔能力）→ `DINOv3-B + SigLIP2`(Cobra 式，质量上限) → `Theia-B`（效率/动作优先）。指标：`grounding_iou / P4_shuffle / center_std / CLM`。

### 2. 充分利用 Mamba 结构优势（核心要求）
- **流式时序观测**：把"多帧 FPV + 本体历史"作为序列输入，利用 Mamba **恒定显存/恒定单步延迟**，在 10Hz 控制下延迟不随历史增长膨胀——这是相对 Transformer-VLA 最大且未兑现的红利。
- **递归推理时状态复用**：部署时用 Mamba 的 recurrent 单步更新做流式，天然契合 TemporalEnsemble/action chunking，无 KV-cache 管理。

### 3. 输出头范式切换（最高杠杆）
- **动作**：将航点离散化为 **动作 token 走 Mamba 语言头**（FAST/离散扩散思路），或 **语言即动作数字串**（VLM2VLA），替代外挂回归头 → 天然契合 Mamba 的 next-token 强项、规避分离头遗忘；如需平滑再叠加**极小残差回归**修正。
- **grounding**：**停用外挂 bbox 回归头**；改为（a）语言头输出**位置 token/指向点**（RoboPoint/PaliGemma 式），并（b）注入**位置编码**（见 4）。grounding 权重已降到 1.0，作辅助探针。

### 4. 让 AirZoo/Open3DVQA 深度真正发挥作用
- 引入 **Ego3D 式位置编码**：对有深度的样本（Open3DVQA 自带、AirZoo 度量深度、或 ZoeDepth 估计）反投影→正弦+MLP→**加到视觉特征**。这既是 SpatialVLA 验证过的空间增益来源，也给了 AirZoo（无语言标注）一个**高价值、低成本的接入点**（只做深度辅助/位置编码，不需造伪语言）。

### 5. 高效优雅的训练流程（收敛为两阶段）
- **Stage A（对齐+推理共训）**：冻结视觉编码器；**Mamba 用 LoRA**；机器人(动作/grounding):视觉语言 **50/50 共训**；动作/grounding 头可只读**中间层**特征省算力。
- **Stage B（极简头微调）**：主干基本冻结，按 RoboMamba 只低成本微调轻量头。
- **不做 OpenX 级预训练**（TinyVLA），靠 LoRA + 语言化动作实现数据高效、避免遗忘。
- **部署**：异步推理 + action chunking 达 10Hz。

### 6. 数据
- 动作**语言化表示**（数字串）统一进 CLM 词表；保持 50/50 共训配比。
- 增补**空间可供性/空间推理**类数据（利于 grounding 与泛化）。
- AirZoo：仅作**深度辅助/位置编码**素材接入（不进 grounding 监督），小规模消融先行。

---

## Phase 5 — 审阅（Review）

**主编意见**：证据充分、引用可核；建议明确"manipulation→UAV 导航"的外推为**受控消融**而非直接全量，避免过度承诺。动作 token 化对"连续平滑航点"的量化误差需在 UAV 域实测。→ **Minor Revision（已在方案中以 A/B、消融、残差修正回应）**。

**伦理**：数据许可需留意——AirZoo(mit)、SigLIP2/Theia 较宽松、C-RADIO(NVIDIA 开放许可可商用)、**DINOv3 许可偏研究向**，商用落地前须复核。无捏造、无害用途；通过。

**DA 检查点 3（通过）**："So what?"——最高杠杆是**输出头范式切换 + 位置编码注入**，而非继续调回归头；最强反论"Mamba×UAV 证据薄"已转化为创新点并以消融兜底。

## Phase 6 — 结论

保留 `Mamba-2 + DinoSigLIP/C-RADIO` 主干；**把外挂回归头换成"语言头 token 化输出 + 深度位置编码注入"**；训练收敛为"冻结编码器 + Mamba-LoRA + 50/50 共训 + 极简/中间层头 + 免大预训练"，并把 Mamba 的恒定显存流式递归用于时序观测与 10Hz 实时控制。分歧点（动作离散化精度、UAV 外推）以受控 A/B 消融落地。

---

## 局限（Limitations）
- 证据主体来自桌面 manipulation，UAV 低空导航航点域的外推未经本项目实验证实。
- 未做本地全量复现；所有增益为文献报告值，须在我们数据/评测上二次验证。
- 视觉编码器与动作头的组合空间大，报告只给出优先级与 A/B 顺序，非穷尽。

## 参考文献（一手来源，URL 可核验）
1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752. https://arxiv.org/pdf/2312.00752
2. Dao, T., & Gu, A. (2024). *Mamba-2 / State Space Duality (SSD).* HF docs. https://huggingface.co/docs/transformers/main/model_doc/mamba2
3. Liu, J. et al. (2024). *RoboMamba: Efficient VLA for Robotic Reasoning and Manipulation.* NeurIPS 2024. https://arxiv.org/abs/2406.04339
4. Zhao, H. et al. (2024). *Cobra: Extending Mamba to Multi-Modal LLM for Efficient Inference.* AAAI 2025. https://arxiv.org/abs/2403.14520
5. HF LeRobot (2025). *SmolVLA.* https://huggingface.co/blog/smolvla
6. Wen, J. et al. (2024). *TinyVLA: Fast, Data-Efficient VLA for Robotic Manipulation.* arXiv:2409.12514. https://arxiv.org/abs/2409.12514
7. (2025). *Discrete Diffusion VLA.* arXiv:2508.20072. https://arxiv.org/abs/2508.20072
8. NVIDIA (2025). *Pretrained to Imagine, Fine-Tuned to Act: World-Action Models* (catastrophic forgetting / knowledge insulation). https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
9. Qu, D. et al. (2025). *SpatialVLA: Exploring Spatial Representations for VLA.* RSS 2025. https://arxiv.org/abs/2501.15830
10. Princeton IROM (2025). *VLM2VLA / Actions as Language.* arXiv:2509.22195. https://arxiv.org/abs/2509.22195
11. (2025). *Enhancing Generalization in VLA by Preserving Pretrained Representations* (50/50 co-train). https://arxiv.org/abs/2509.11417
12. Meta AI (2025). *DINOv3.* arXiv:2508.10104. https://github.com/facebookresearch/dinov3
13. NVIDIA (2025). *RADIOv2.5 / C-RADIOv3.* CVPR 2025. https://github.com/NVlabs/RADIO
14. Google (2025). *SigLIP 2.* arXiv:2502.14786. https://huggingface.co/blog/siglip2
15. Shang, J. et al. (2024). *Theia: Distilling Diverse VFMs for Robot Learning.* CoRL 2024. https://theia.theaiinstitute.com/
