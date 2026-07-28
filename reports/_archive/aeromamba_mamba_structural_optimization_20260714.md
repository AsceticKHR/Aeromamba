# 结合 Mamba 结构特性的 AeroMamba 针对性优化（文献核验版）

- 日期：2026-07-14
- 方法：deep-research three-way-scan；关键论断均回原文核验
- 视角：不同于上一份报告（通用空间推理，见 `aeromamba_optimization_deep_research_20260714.md`），本报告专注 **Mamba/SSM 结构本身**——因果 1D 扫描、固定容量递归状态、选择性门控、无注意力回看——推导针对性设计。
- AI 披露：AI 辅助检索与撰写，引用均经独立存在性与原文核验。

---

## 一、Mamba 结构特性的文献证据（WHY / HOW / WHAT）

### E1. SSM 的"模糊记忆"：精确回忆是结构性短板 — T2（NVIDIA，8B 规模系统实验）

**An Empirical Study of Mamba-based Language Models**（Waleffe et al., NVIDIA, arXiv 2406.07887）

- WHY：验证纯 SSM 能否在 8B/3.5T token 规模替代 Transformer。
- HOW：受控对比 Mamba / Mamba-2 / Transformer / Mamba-2-Hybrid，含 Phonebook 精确回忆任务。
- WHAT（已核验原文）：
  - 纯 SSM 在多数标准任务持平或超过 Transformer，但在 **MMLU（few-shot）与 Phonebook 上系统性落后**；
  - **"fuzzy memory"现象**：SSM 无法精确复述电话号码，但预测的号码与正确答案共享多位正确数字——固定大小状态保留的是**有损压缩的近似信息**；
  - 补救：**24 层 Mamba-2 + 仅 4 层自注意力 + 28 层 MLP 的混合模型在全部 12 个短上下文基准上超过对应 Transformer**。
- **对 AeroMamba 的含义**：距离定量差（8.2%）有结构性根源——不只是数据问题，Mamba 状态天生存"模糊量"而非"精确数字"。方向/类别判断（读状态的近似模式即可）符合结构长处，这与我们 direction 48.6% >> distance 8.2% 的分桶结果吻合。

### E2. 少量注意力层即可恢复 ICL/回忆 — T2（Jamba，AI21）

**Jamba: A Hybrid Transformer-Mamba Language Model**（arXiv 2403.19887）

- WHAT（已核验原文）：纯 Mamba 无法涌现 in-context learning（缺 induction heads 的近似拷贝机制）；**混合模型只要 1/8 层是注意力，ICL 即恢复到 Transformer 水平**。
- 含义：不需要重训混合主干——在关键读出位置补一个**极小的注意力"检索补丁"**就可能弥补 Mamba 的精确回看缺陷（见 M3）。

### E3. Mamba 对 prompt 顺序异常敏感：参考信息应在问题之前 — T1（Cobra, AAAI 2025）

**Cobra: Extending Mamba to Multi-Modal LLM**（arXiv 2403.14520，AAAI 2025）

- WHAT（已核验原文）：
  - **prompt 顺序消融**："Question → Reference OCR token" 顺序使 TextVQA 大幅下降（至 47.9%）；调换为 "Reference OCR token → Question" 显著回升。作者归因于 RNN 式主干的 inductive bias：**查询 token 只能从此前累积的状态中选择性提取，无法回看**；
  - **视觉 token 顺序**：Cobra 将视觉嵌入**拼接在文本之前**（vision-first）；RoboMamba 同样 image-first（原文 Figure 2 已核验）；
  - **压缩投影消融**：轻量下采样投影器（LDPv2）"significantly harms the performance on all benchmarks"，作者明确结论——**"表征压缩显著损害 Mamba 的理解能力"**（Transformer 上同款 LDP 在 MobileVLM 中工作良好）。
- **对 AeroMamba 的含义（两条直接冲突）**：
  1. AeroMamba token 顺序是 `[text | proprio | vision]`——指令在最前，等价于 Cobra 发现的**劣序**（问题在参考之前）。设计初衷（“vision 最后一个 token 累积全部上下文供动作头读取”）只照顾了动作读出，没照顾指令条件化：729/64 个视觉 token 的门控只取决于视觉内容本身，指令信息在状态中被逐步稀释——这正是评测中"指令跟随弱"的结构性解释。
  2. AeroMamba 的 64-query PerceiverResampler 正是 Cobra 证明对 Mamba 特别有害的"压缩投影"。**Mamba 不能回看原始 token，被压缩丢掉的信息永久丢失**；Transformer 可以靠注意力反复回读弥补，Mamba 不能。

### E4. 连接器级 2D 扫描弥补 1D 因果扫描的空间偏置 — T2（VL-Mamba）

**VL-Mamba**（arXiv 2403.13600）

- WHAT（已核验原文）：在**多模态连接器内**（不动 Mamba LLM 主干）加入 Vision Selective Scan（四方向 Cross-Scan），VSS-L2 变体在多数基准上优于纯 MLP 连接器。
- 含义：1D 扫描的空间方向偏置可以在投影器层面低成本补偿，不必改造主干。

### E5. 递归状态是 Mamba 的独有正资产，但有部署纪律 — T2（μVLA）

（承接上份报告 P4，此处从 Mamba 结构角度重述）

- Mamba 推理天生 O(1) 每步状态更新——**流式闭环携带全飞行历史是 Transformer 做不到的免费能力**；
- 但 μVLA（arXiv 2606.12497，已核验）证明：递归记忆若每个 chunk 才更新一次（当前 `exec_horizon=4` 即此模式），需要记忆的任务性能崩溃；须 **receding-horizon（每步重查询、执行首步）+ TBPTT（K=2 已够）**。

### 跨文献综合

- **共同 WHY**：固定容量状态 + 因果单向扫描 → ①精确回忆弱（E1）；②信息顺序决定可用性（E3）；③压缩不可逆（E3）。
- **分歧 HOW**：加注意力层（E1/E2，动主干）vs 调 token 顺序与连接器（E3/E4，零/低成本）。对 370M 预训练主干，后者优先。
- **最强 WHAT**：Cobra 的顺序消融与压缩消融是 Mamba-VLM 上的直接受控实验，与 AeroMamba 配置逐条对应——可信度最高、迁移距离最短。
- **Gap**：以上均为 VQA/操作域，UAV 连续控制下的 token 顺序消融无人做过，需自行 A/B。

---

## 二、针对性优化设计（Mamba 结构专项，按证据强度 × 成本排序）

### M1. Token 顺序重排 + 指令三明治 【证据：Cobra 受控消融；成本：极低，改拼接逻辑 + 重训】

- 现状：`[text | proprio | vision]`（指令最先，被视觉流稀释）。
- 方案 A（vision-first，与 Cobra/RoboMamba 一致）：`[vision | proprio | text]`——指令 token 在最后从视觉状态中选择性提取，动作头读最后一个指令 token；
- 方案 B（指令三明治，保守）：`[text | vision | proprio | text]`——指令重复一次拼在末尾，前段指令仍能影响早期状态，末段指令负责精确条件化。开销仅 +10~20 token。
- 落点：`model/uav_mamba_vla.py` 的 token 拼接段；Stage2/3 同步改。**这是全部建议中证据/成本比最高的一条**，建议最先 A/B（三组：现状 / A / B，用干净 hold-out 的指令敏感题 + UAV-Flow 转向类任务判定）。

### M2. 放弃/放宽视觉压缩：Mamba 承担得起 729 token 【证据：Cobra LDP 消融 + Lost in Space；成本：中，重训 Stage2】

- Cobra 证明压缩投影对 Mamba 的伤害远大于 Transformer（不可逆丢失 + 无法回看）；而 Mamba 线性复杂度意味着 729 token 的代价可接受（Transformer 上 729² 注意力才是当初引入 resampler 的动机）。
- 分级方案：
  1. 首选：Stage2 直接旁路 resampler，全量 729 token 进 Mamba（项目文档记载的原始设计就是全量直通，resampler 是后加的），实测显存/延迟；
  2. 若延迟不可接受：queries 64 → 256，并按上份报告 P2 加位置编码。
- 判定指标：Open3D probe 的 direction/other 桶 + TextVQA 式细粒度题。

### M3. 末端"检索补丁"：单层 cross-attention 读出 【证据：Jamba 1/8 注意力恢复 ICL + NVIDIA 4/56 层混合超 Transformer；成本：中低，新增 ~5-10M 参数】

- 设计：在 Mamba 最后 hidden 与视觉 token 序列之间加**一层轻量 cross-attention**（query = 最后 hidden 或动作 query，key/value = 投影后视觉 token），输出与原 hidden 残差相加后进动作头/生成头。
- 原理：给"只能读压缩状态"的读出位置补一次对原始视觉序列的**精确回看**，正是 E1/E2 证明纯 SSM 缺失、且少量注意力即可补齐的能力。729 token 的单层 cross-attn 计算量可忽略（<1ms 级）。
- 训练：Stage2 起引入（零初始化输出投影，保证初始等价于无补丁），Stage3 继续训练。
- 注意：这是本报告中改动最大的一条，放在 M1/M2 验证之后做。

### M4. 精确数值任务改分类读出 【证据：E1 fuzzy memory；成本：低】

- 结构性结论：让 Mamba 生成精确米数等于让它做 Phonebook 任务——注定低分且不可用数据修复。
- 方案：距离/高度类答案统一对数分箱模板（承接上份报告 P5），从"生成数字"改为"选区间"，与 Mamba 状态的近似表征能力对齐。方向/类别题维持生成式（结构上不吃亏，评测已证实）。

### M5. 投影器内嵌轻量 VSS 2D 扫描 【证据：VL-Mamba VSS-L2 消融；成本：中低】

- 在 `MLPProjector` 前插入四方向 selective scan 块（不动主干），补偿 1D 因果扫描对 2D 空间关系的方向偏置。与 M2（全量 token）搭配收益最大；若保留 resampler 则在 resampler 之前插入。

### M6. 递归状态流式闭环（Mamba 独有优势，按纪律部署）【证据：μVLA；成本：改协议 + Stage3 按轨迹采样】

- Mamba O(1) 步进推理使"状态携带全飞行历史"免费——这是相对 OpenVLA 类的差异化卖点，值得做；
- 部署纪律（μVLA 强消融）：`exec_horizon` 降为 1（每步重查询执行首步）+ Stage3 轨迹连续采样 TBPTT（K=2 起步）；否则记忆通道被 chunk 执行旁路，可能不增反降。

### 不建议做的

- **重训混合（Mamba+Attention）主干**：E1/E2 证明有效但需从头预训练，370M 预算下不可行；M3 检索补丁是其低成本替代。
- **在 Mamba 主干内部改造 2D cross-scan**（VMamba 式）：破坏预训练权重结构，M5 连接器方案可达到同类目的。

---

## 三、与上份报告（P1–P6）的合并执行序

```
第 0 步   P6 干净 hold-out（一切量化的前提，后台进行）
第 1 批   M1 token 顺序 A/B（证据/成本比最高）＋ M4/P5 距离分箱
第 2 批   M2 视觉 token 放宽（与 P2 位置编码合并测）＋ P1 Stage3 空间辅助头
第 3 批   M3 检索补丁 或 M5 连接器 VSS（二选一先行，避免归因混淆）＋ P3 伪深度分支
第 4 批   M6/P4 流式闭环（receding-horizon + TBPTT）
```

每批只引入一个主变量，固定三件套量化（Open3D 干净 hold-out 分桶 / Stage2 能力面板 / UAV-Flow-Eval 闭环）。

## 四、局限性声明

1. E1/E2 结论来自 8B/52B 语言模型，外推到 370M 多模态模型的效应量未知；
2. Cobra 顺序消融是 TextVQA（OCR 参考）场景，与 UAV 指令条件化在任务形态上有距离，M1 结论需 A/B 确认；
3. μVLA 基于 Transformer 背骨（OpenVLA-OFT）的显式 memory token，Mamba 隐式状态的行为可能不同；
4. 混合层比例数字（4/56、1/8）为对应论文特定规模下的结果，不构成 M3 参数量的直接依据。

## 参考文献

1. Waleffe, R. et al. (2024). An Empirical Study of Mamba-based Language Models. arXiv:2406.07887
2. Lieber, O. et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model. arXiv:2403.19887
3. Zhao, H. et al. (2025). Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference. *AAAI 2025*. arXiv:2403.14520
4. Qiao, Y. et al. (2024). VL-Mamba: Exploring State Space Models for Multimodal Learning. arXiv:2403.13600
5. Liu, J. et al. (2024). RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation. *NeurIPS 2024*. arXiv:2406.04339
6. μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models. arXiv:2606.12497
7. Pantazopoulos, G. et al. (2024). Lost in Space: Probing Fine-grained Spatial Understanding in Vision and Language Resamplers. *NAACL 2024*. arXiv:2404.13594
