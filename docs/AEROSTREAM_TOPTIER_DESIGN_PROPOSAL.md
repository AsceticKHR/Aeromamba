# AeroStream：面向顶刊的流式状态空间 UAV VLA 框架设计书

- 日期：2026-07-15（**v1.5，两项 v1.4 根因判断经代码/数据核实后更正：proprio 失配仅存在于休眠 HF 路径、flip 镜像词表审计结论阴性，见第 15 节更正记录**；v1.4 以 stage3_v2 273 条轨迹实测诊断为锚重排优先级，见第 14 节；v1.3 融入 AnoleVLA[R26]；v1.2 单卡 4090 裁剪；v1.1 按 aeromamba_opt 现行配方更正。v1.1 更正记录见第 12 节，分部件可行性见第 11 节，AnoleVLA 专项见第 13 节）
- 性质：研究设计方案（academic-paper plan/outline 模式产物），非论文正文
- 现行基线：aeromamba_opt 配方 — SigLIP2 Base 384 单塔 / MLP 投影 / Perceiver Resampler 64q（可训）/ Mamba-2-370M LoRA(r16) / token 序 `[state|delta|vision|text]` / MLP 动作头 chunk=8 body-frame 航点 / Stage3 可训 ~5.5%（43M）
- **算力约束（v1.2）**：训练算力仅一张 RTX 4090（24GB）。全部设计按此预算裁剪：视觉 token 分档、TBPTT K≤4、消融矩阵分"CoRL 必做/TPAMI 扩展"两级、缩放律降为两点。详见 §9 单卡排程。
- 证据基础：两轮 deep-research 核验报告（`reports/aeromamba_optimization_deep_research_20260714.md`、`reports/aeromamba_mamba_structural_optimization_20260714.md`），全部关键设计决策均有已核验文献支撑
- AI 披露：本设计书由 AI 辅助调研与撰写

---

## 0. 一句话定位

**AeroStream: A Streaming State-Space Vision-Language-Action Framework for Language-Conditioned UAV Flight**
——首个把 SSM 递归状态显式用作"持久飞行记忆"的 UAV VLA：全飞行历史以 O(1) 时间/内存压缩进状态，在边缘算力上实现恒定延迟的每步闭环控制，同时通过"结构对齐的多模态接口"解决纯 SSM 的精确检索（fuzzy memory）与几何感知短板。

（命名备选：SkyStream / FlowSSM / StreamPilot。投稿前需检索确认无重名。）

## 1. 顶刊叙事：为什么这是一篇论文而不是一次调参

顶刊论文需要一个**别人没有回答过的问题**。本方案回答的是：

> **RQ：当 VLA 的主干从注意力换成递归状态空间模型时，多模态接口、记忆机制与控制协议应当如何"结构对齐"地重新设计——而不是照搬 LLaVA/OpenVLA 范式？**

这个问题成立的证据（均已核验）：

1. 现有 Mamba VLA（RoboMamba, NeurIPS 2024；Cobra, AAAI 2025）**只替换了主干，完整保留了 Transformer 时代的接口范式**，且 Cobra 自己的消融已暴露该范式与 SSM 的三处失配：prompt 顺序敏感（TextVQA 掉至 47.9%）、压缩投影"significantly harms"、无法回看；
2. 现有 UAV VLA（OpenVLA-UAV、AutoFly、UAV-Track VLA）全部基于 Transformer，**单帧输入、无跨步记忆**——UAV-Flow 原文明确指出 OpenVLA-UAV"reliance on single-frame visual input imposes limitations"；
3. 历史感知 VLA（HAMLET 等）指出利用历史"incurs substantial computational overhead"——在 Transformer 上历史 = token 堆叠 = 二次成本；**而 SSM 的历史 = 状态携带 = 零边际成本**。μVLA（2026）证明了最小递归记忆的价值，但其载体是 Transformer + 外挂 memory token，且给出了部署纪律（receding-horizon）；
4. **最接近的竞品 AnoleVLA[R26]（2026-03）恰好划清了边界**：它是与本设计几乎同构的 Mamba VLA（相同 token 序、SigLIP2、线性动作头、单 4090），但——(a) 它是**操作域**（table-top/移动操作），非 UAV 飞行；(b) 它的递归状态用在**单步序列内**、以 Δstate 作为时间信号，**并未把 Mamba 状态跨控制步携带为持久飞行记忆**（=C1 仍空白）；(c) 它明确把"显式空间推理模块 + 定位损失"列为**未完成的 future work**（=C3 仍空白）；(d) 无注意力检索补丁（=C2③ 仍空白）。因此 AnoleVLA 既验证了本设计的底座选择，又反证了 C1/C2③/C3 的新颖性未被占据。
5. **仍无人做过**：把 SSM 固有递归状态跨控制步作为飞行记忆（C1）、SSM 主干下的注意力检索补丁（C2③）、以及在 UAV 域落地上述二者。
6. **第一手实证（v1.4 新增，论文动机的最强素材）**：本项目 stage3_v2 在 UAV-Flow 273 条轨迹上的诊断给出了单帧 VLA 失败模式的一手证据——能动的任务不会停（末段 15~26cm/步直到截断，终点误差中位 13.2m）、只会飞直线（净位移/路径长度 0.9~1.0，Surround 整圆完全无法执行）。这两类失败**不是调参问题而是单帧推理的信息论盲区**（"飞了多远/绕了多少角度"在输入中不存在），正是 C1 的判决性动机。论文 Intro 可直接使用这组自家失败案例开题（附轨迹对比图），比引用他人结论更有力。

这构成三层贡献（对应论文 C1–C3）+ 一层系统贡献（C4）：

| 贡献 | 内容 | 新颖性锚点 |
|---|---|---|
| **C1 流式状态记忆** | SSM 递归状态跨控制步携带全飞行历史；轨迹级 TBPTT 训练 + receding-horizon 推理协议 | 首个 native-state 记忆 VLA；对照 μVLA（外挂 token）与 HAMLET（token 堆叠） |
| **C2 结构对齐接口** | SSM 专用多模态接口三原则：①模态排序法则（查询/本体贴近读出位，指令后置）；②无损视觉通路（全 token 直通或检索补丁旁路）；③终端检索补丁（单层 cross-attn 读 resampler 前原始视觉序列） | 每条原则各有受控消融；把 Cobra/Jamba/NVIDIA 的语言域发现首次系统化为 VLA 设计法则 |
| **C3 语言-动作绑定与几何注入（轻量版，v1.4 扩充）** | 训练期旁路辅助头升级为**指令绑定分类头**：预测运动类别 + yaw 符号 + dz 符号 + 幅度对数分箱（交叉熵，对符号错误惩罚尖锐）；辅以度量分箱读出、视觉塔尾部 LoRA、位置保持——全部零额外 token、推理零开销 | 直接针对实测失败模式（yaw 符号 14/27≈随机、dz≈0、运动原语冻结）；UAV-Track VLA 旁路范式 + AnoleVLA 自陈 future work |
| **C4 边缘效率** | 恒定延迟/内存 vs 历史长度；机载（Jetson Orin 级）10Hz+ 闭环实机部署 | Science Robotics 叙事核心 |

## 2. 与 aeromamba_opt 现行配方的继承/放弃对照（"选择性参考"）

> 对照基准为 aeromamba_opt 现行配方（2026-07-14 在训版本），不是旧版 CLAUDE.md 描述。

| aeromamba_opt 现行设计 | 处置 | 依据 |
|---|---|---|
| Token 序 `[state\|delta\|vision\|text]`（指令后置、本体前置） | **强继承——AnoleVLA[R26] 用完全相同的序并给出明确理由**："本体 token 置于序列开头，使隐藏状态在整合视觉/语言前先被 agent 状态条件化" | Cobra 顺序消融（查询在参考之后）✓ + AnoleVLA 直接同构证据 ✓✓ |
| Stage2→3 串联 + LoRA 全程可训 | 继承，加第 4 阶段流式微调 | RoboMamba"推理先行、动作轻量"路线 |
| MLP 动作头（chunk=8 body-frame）+ z-score 归一 + endpoint/direction 损失 | 继承，**新增二阶加速度损失**（两阶段：先 velocity，后加 acceleration） | AnoleVLA[R26] 消融：加速度损失 SR 63.12%→67.85%（+4.73）；对 UAV 平滑飞行/安全收益更大 |
| 转向 3× 过采样 + 翻转增强（含指令镜像） | 继承 | — |
| Stage2 四源加权采样（v2 数据） | 继承，混入度量分箱模板 + region QA | SpatialVLM/O3DVQA |
| Proprio 8 维 state+delta 编码 | 继承（流式版改逐步注入） | — |
| SigLIP2 Base 384 **单塔** | **保留单塔**；几何缺口用零 token 手段补偿（aux head + 塔尾 LoRA + 位置保持，见 §3）；深度/DINOv2 通路移入 TPAMI 扩展 | Cobra 消融显示双塔在空间基准 +5~6%，作为已知代价在 Limitations 声明 |
| Perceiver Resampler 64q（可训） | **A/B 后决定**：全 576 token 直通 vs 保留+位置保持。可训状态部分缓解 Lost in Space 问题，但 Cobra 压缩伤害仍在 | Cobra LDP 消融 + Lost in Space |
| 视觉塔全程冻结 | Stage2 尾部 2-4 block 加 LoRA（保守）；AnoleVLA 证明单 4090 上 SigLIP2 端到端全微调可行，作为激进档 A/B | SpatialVLM Table 3（解冻利好细粒度距离）+ AnoleVLA[R26]（SigLIP2 端到端微调） |
| 单帧输入、chunk 整段执行 | **放弃**，流式状态 + 每步重查询 | μVLA 强消融 |

## 3. 架构设计

```
                     ┌────────────────────────── 每个控制步 t ──────────────────────────┐
 FPV frame ─► SigLIP2-Base/16@384（尾部 2-4 block LoRA）─► [576 RGB tok]
                               ▼
                    MLP 投影（+ 24×24 位置保持）
                               ▼
        ┌──────────── token 流（结构对齐序）────────────┐
        [ vision(RGB) | state | delta | instr ]              ← 指令后置（现配方已满足）
        └──────────────────────┬────────────────────────┘     proprio 移到视觉后（A/B）
                               ▼
              Mamba-2-370M 主干（LoRA r16；缩放两点 130M/370M）
        状态 S_t ◄──── S_{t-1}（跨步携带 = 飞行记忆，O(1)） ← C1
                               ▼
             h_last ──► 终端检索补丁（1 层 cross-attn，Q=h_last，
                        KV=resampler 前的原始 576 tok，零初始化残差） ← C2③
                               ▼
             ┌─────────────┬──────────────────┐
        Action Head     Aux Spatial Head    LM Head
        (MLP, chunk=8,  (方向+log距离,       (VQA/分箱
         流式执行首步)   仅训练期)            读出)
```

设计要点与依据：

- **token 序**：现行 `[state|delta|vision|text]` 已满足 Cobra 的"查询在参考之后"原则（动作头读最后一个 text token，其状态已累积视觉与本体信息）——此项**保留**。剩余 A/B 点：proprio 前置时，2 个本体 token 的信息要穿过 64/576 个视觉 token 的门控稀释才到读出位；`[vision | state | delta | instr]` 把本体贴近读出，理论上对动作条件化更有利，成本仅为改拼接顺序。
- **视觉压缩（C2② 的现实版，24GB 三档）**：Resampler 在现配方中全程可训（Stage2 CLM + Stage3 动作损失），Lost in Space 证明联合训练能恢复部分空间信息，因此其伤害小于冻结场景，但 Cobra 的结构性论证仍在（Mamba 无法回看，压缩不可逆）。单卡 24GB 下序列长度是显存/吞吐主变量，按档推进：
  - **档 1（默认）**：保留 64q resampler + 输入 24×24 位置编码 + queries 位置嵌入（PPE），近零成本；
  - **档 2（P0 A/B）**：RGB 全 576 token 直通，batch 12→4 + 梯度累积 ×3 等效；370M LoRA + bf16 在 24GB 可行，代价为 Stage2 墙钟约 ×3~4。
  - 决策规则：档 2 相对档 1 在干净 hold-out 提升 <2% → 定档 1 + 检索补丁旁路（补丁读原始 576 token，等效补偿压缩且训练成本近零）。
- **几何注入（C3 轻量版，已取消 Siamese 深度分支）**：单塔几何缺口全部用**零额外 token、推理零开销**的手段补偿——①训练期旁路空间辅助头（UAV-Track VLA 范式，度量监督把几何先验压进表征）；②视觉塔尾部 2-4 block LoRA（SpatialVLM：解冻 ViT 对细粒度距离"considerably better"）；③投影/resampler 位置保持（PPE）；④距离分箱读出。设计取舍：伪深度分支（AutoFly 路线）与 DINOv2 双塔虽有文献支撑，但分别增加 +64~576 token 的序列成本与第二座塔的显存/墙钟成本，**整体移入 TPAMI 扩展**；单塔几何上限（Cobra 双塔 +5~6% 的差距）作为已知代价写入 Limitations。若 CoRL 版空间消融显示 aux head 补偿不足，扩展项优先级再上调。
- **终端检索补丁**：补齐 SSM 无法精确回看的结构缺陷（NVIDIA fuzzy memory + Jamba 1/8 注意力恢复 ICL）。关键实现点：**KV 必须取 resampler 之前的原始 576 token**——若保留 resampler，这条通路恰好绕过压缩损失，两个方案互补而非二选一。单层、零初始化、参数 <10M、延迟 <1ms。
- **跨步状态携带（C1 核心）**：帧间不重置 Mamba 状态；每步只增量喂入新帧 token（SSM 步进推理 O(1)）。指令 token 仅在任务开始与每 N 步刷新时注入（防状态漂移，N 为消融变量）。
- **动作读出**：读终端检索补丁输出而非裸 h_last；aux head 仅训练期（UAV-Track VLA 旁路范式，推理零开销）。现行 z-score 归一 + endpoint×0.25 + direction cosine×0.5 损失配方保留。

## 4. 训练方案（四阶段）

> 以现行 aeromamba_opt 串联训练为底座增量演进，不推倒重来。

| 阶段 | 相对现行配方的增量 | 数据 | 关键点 |
|---|---|---|---|
| S1 对齐 | 无增量（沿用现行产物） | LLaVA-Pretrain | 现行 MLP Projector 保留，S1 无需重训 |
| S2 空间 VLM | +检索补丁（零初始化）；+视觉塔尾部 2-4 block LoRA；resampler 直通/位置保持按 P0 结论定 | 现行 v2 四源加权采样基础上：O3DVQA 训练切分改**度量分箱模板** + region QA；干净 hold-out 先行剔除 | 现行 batch 12 / lr 5e-5 / 加权采样配方保留 |
| S3 动作（v1.5 按核实结论更新） | **新增监督（唯一主线）**：①指令绑定分类头（运动类别+yaw 符号+dz 符号+幅度分箱）；②二阶加速度损失（AnoleVLA 两阶段）；③运动原语类（Move/Shift/Ascend/Descend/Surround/Rotate）整类过采样 + z/yaw 通道损失加权 + 按 GT 位移幅度的样本加权（对抗均值坍缩）。**卫生修复（非阻塞，半天）**：`UAVFlowHFDataset` 路径补前帧导出或加断言——该路径 velocity/delta 全零失配真实存在但**本轮训练未使用**（实际走 `UAVFlowDataset`，时间通道合法非零，已核实，见 §15）；flip 镜像词表审计已完成、结论阴性（见 §15） | UAV-Flow（单步采样） | 现行保留：z-score + endpoint/direction 损失、LoRA 全程可训；新增标签全部从现有轨迹免费提取 |
| S4 流式（新增） | 同 S3 参数继续训 | UAV-Flow（**轨迹连续采样**） | **24GB 约束：TBPTT K=2 起步、上限 K=4**（μVLA：K=2 已获大部分收益，K=8 的边际稳健性收益让给 TPAMI 扩展）；梯度检查点 + bf16 + batch 折减（48→16 级）；训练/推理均每步更新状态 |

推理协议：receding-horizon——每步重查询，执行 chunk=8 的首步（μVLA 纪律）；温度集成保留为可选平滑层。S3 之前各阶段产物与现行 `stage3_v2_20260714_082604` 检查点兼容（新增模块均零初始化旁路，可热加载续训）。

## 5. 实验设计（顶刊审稿人视角反推）

### 5.1 主实验

| 基准 | 协议 | 对比方法 |
|---|---|---|
| UAV-Flow Colosseo（闭环仿真 + 实机） | SR + nDTW，官方协议 | OpenVLA-UAV、π0-UAV、Travel-UAV（官方结果/权重）；等参数 Transformer 对照自训移入 TPAMI 扩展（单卡预算，见 §7.6） |
| OpenFly（100K 轨迹航拍 VLN） | SR/SPL | OpenFly-Agent 及榜单方法 |
| Open3D-VQA 干净 hold-out | 分桶精度 | LLaVA-1.5/Qwen2-VL（论文数）+ Cobra |
| 记忆专项（自建，关键差异化）| "绕到建筑物后再回到起点方向"“目标短暂遮挡后继续跟随"类任务——单帧模型结构性无法完成 | 所有单帧基线 + HAMLET 式历史堆叠基线 |

记忆专项是 C1 的**判决性实验**（decisive experiment）：单帧 VLA 在此类任务的失败不是调参问题而是信息论问题，预期拉开断崖差距——这是顶刊最喜欢的"结构优势可证伪验证"。

### 5.2 效率实验（C4 / Science Robotics 核心）

- 延迟/内存 vs 飞行时长曲线：AeroStream 恒定（O(1) 状态）vs Transformer+历史 token 线性增长 vs HAMLET 式压缩历史；
- 机载部署：Jetson Orin NX 上 10Hz 闭环全链路延迟分解（编码/主干/读出）；
- 能耗与最大可持续飞行时长。

### 5.3 消融矩阵（每条贡献可归因）

单卡 24GB 下消融分两级。**统一消融协议**（控制总 GPU 时数）：除标注"全量"外，一律用短程代理——Stage2 消融 = 0.5 epoch + 干净 hold-out；Stage3/4 消融 = 25% 步数 + UAV-Flow-Sim 固定 200 episode 子集。代理有效性用 P0 一组"短程 vs 全量"相关性检查背书。

**CoRL 必做（判贡献成立与否，约 9 次短程 + 3 次全量）**

| 消融 | 验证 | 预算 |
|---|---|---|
| 反向消融：指令前置（预期显著变差，验证排序法则；现行本体前置序有 AnoleVLA 直接背书，作为默认不再当作待优化项） | C2① | 短程 ×1 |
| ± 二阶加速度损失（复现 AnoleVLA +4.73 于 UAV 域，兼看 nDTW/平滑度） | 动作平滑 | 短程 ×1 |
| ± 指令绑定分类头（诊断三指标验收，见 §14；proprio 修复消融项已随 §15 更正取消） | C3 / 修复归因 | 修复轮自带 ×1 |
| 档 1（64q+位置保持）vs 档 2（576 直通）——v1.4 起降级为普通消融（见 §14.3） | C2② | 短程 ×2 + 相关性检查 |
| ± 终端检索补丁 | C2③ | 短程 ×2 |
| ± 跨步状态携带；TBPTT K∈{1,2,4} | C1 | S4 短程 ×3 |
| exec_horizon ∈ {1,4,8}（μVLA 崩溃复现，**纯推理端，零训练成本**） | C1 纪律 | 推理 ×3 |
| ± aux spatial head（λ≈0.1 单点） | C3 | 短程 ×2 |
| ± 视觉塔尾部 LoRA | C3 | 短程 ×1（与上共用基线） |
| 最终配置全量：S2→S4 一次 + 基线（现行配方）复评 | 主结果 | 全量 ×2~3 |

**TPAMI 扩展（资源允许再做）**

| 消融 | 说明 |
|---|---|
| resampler 128q 中间档；补丁 KV 取 resampler 前 vs 后 | C2 细化 |
| 伪深度分支（Siamese/独立投影对比，AutoFly 路线）；DINOv2 双塔 | C3 几何通路扩展（本版已取消，见 §3） |
| λ 扫描全网格；指令刷新周期 N 扫描 | 超参完备性 |
| TBPTT K=8；主干缩放 130M/370M 两点（790M 放弃：24GB 全 token 训练不可行） | 缩放律降为两点声明，审稿诚实处理 |
| 等参数 Transformer 对照自训 | 见 §7 风险 4 |

### 5.4 实机实验（Science Robotics 门槛）

真实四旋翼（机载 Orin）：≥3 类场景（开阔/建筑群/低空障碍）× 指令跟随 + 记忆任务 + 长时飞行稳定性（≥5 分钟连续闭环，验证状态不漂移）。视频 + 全部轨迹日志公开。

## 6. 投稿策略（诚实评估）

| 目标 | 匹配度 | 门槛 |
|---|---|---|
| **Science Robotics** | 中 | 需要"能力跃迁"级实机演示（长时记忆飞行是候选卖点）；纯 benchmark 提升不够。建议实机结果惊艳时再冲 |
| **TPAMI** | 中高 | 方法论深度 + 消融广度是强项；需补缩放律与跨任务泛化（manipulation 域迁移一节可加分） |
| **RSS / CoRL（首发）→ TPAMI 扩展** | **高（推荐路径）** | 机器人顶会对"结构-任务对齐"叙事接受度最高，审稿周期短，社区曝光后扩展 30% 内容投 TPAMI 是成熟路线 |
| NeurIPS/ICLR | 高 | 若 C2 接口法则的语言域/操作域泛化实验充分，可走 ML 主会 |

**推荐**：CoRL 2027（约 2027-02 截稿）首发 → TPAMI 扩展版。直接投 Science Robotics 的前提是记忆专项实机 demo 达到"视频本身有传播力"的水平。

## 7. 风险与预实验（按一周内可完成排序）

1. **P0 已更换（v1.5）**：诊断表明当前瓶颈是**语言-动作绑定**，不是视觉 token 压缩——原 P0（resampler 直通 A/B）降级为 CoRL 消融队列普通一项。v1.4 拟定的两项前置审计已完成（2026-07-15）：①flip 指令镜像审计**结论阴性**（53,586 条训练指令中 0 条仅含词表未覆盖方向词，矛盾对量级 1/53,586，不构成左偏成因）；②proprio 时间通道核实**失配不存在于实际训练路径**（详见 §15）。新 P0 = 指令绑定辅助头 + 原语类过采样 + 通道/幅度加权的 Stage3 重训一轮 → 复跑 273 条诊断，验收指标：冻结率（127/273 基线）、yaw 符号准确率（14/27 基线）、dz 非零率（4/19 基线）。yaw 系统性左偏成因**仍未知**（flip 假设已排除；`--symmetrize_lateral` 使均值坍缩只能解释零 yaw 而非负 yaw），挂起至绑定头上线后复查。
2. **状态漂移风险**：长时携带状态可能积累漂移 → 预实验：仿真 5 分钟连续闭环监控动作分布散度；缓解手段=指令周期刷新 + 状态范数正则。
3. **TBPTT 显存（24GB 硬约束）**：K×序列长度×激活是显存主项 → 梯度检查点 + bf16 + K 上限 4 + batch 折减；若档 2（576 直通）与 S4 叠加仍 OOM，则 S4 阶段回退档 1 token 配置（流式贡献与 token 配置解耦，不互相绑架）。
4. **C3 轻量补偿不足的风险**：取消深度分支后，若 aux head + 塔尾 LoRA 在空间消融中补偿不足（干净 hold-out 空间桶提升 <3%），则将伪深度分支从 TPAMI 扩展提前回 CoRL 版（离线预计算深度 + 过 resampler 的 +64 token 形态，增量成本最小）。
5. **实机资源**：无机载平台则 Science Robotics 路线关闭，仿真闭环 + 延迟实测支撑 CoRL/TPAMI 路线不受影响。
6. **单卡总预算**：等参数 Transformer 对照自训 ≈ 一次全量三阶段，在单卡上会挤占 2-3 周 → CoRL 版用官方 OpenVLA-UAV/π0-UAV 结果 + 规模差异声明；缩放律降为 130M/370M 两点并在 Limitations 中说明。评测（UAV-Flow-Sim 闭环、Open3D probe）尽量放本地 4060/WSL 执行，4090 专职训练，避免排队互锁。**单卡可行性已被 AnoleVLA[R26] 证伪风险排除**：467M Mamba VLA 在单张 4090（24GB）上 20 小时完成 40 万步两阶段训练，本设计 370M + LoRA 预算更宽松。

## 8. 论文大纲（IMRaD，CoRL/TPAMI 两用骨架）

1. **Intro**：单帧 VLA 的信息论局限 × Transformer 历史成本 × SSM 状态即记忆的机会 → RQ
2. **Related Work**：VLA（OpenVLA/π0）；UAV VLA（UAV-Flow/AutoFly/CognitiveDrone）；SSM 多模态与 SSM VLA（Cobra/VL-Mamba/RoboMamba/**AnoleVLA**）；记忆 VLA（μVLA/HAMLET）——定位表格化。**AnoleVLA 作为最近邻竞品单列一段**，明确 delta：本设计=跨步持久状态记忆（非单步内递归）+ UAV 飞行域 + 检索补丁 + 空间辅助（后者是 AnoleVLA 自陈的 future work）
3. **Method**：3.1 结构失配分析（E1–E3 证据链复述为动机）；3.2 结构对齐接口（C2）；3.3 流式状态记忆与 TBPTT（C1）；3.4 几何注入（C3）
4. **Experiments**：主实验 → 记忆专项 → 效率 → 消融矩阵 → 实机
5. **Limitations**：fuzzy memory 上界（精确度量任务）、状态容量与任务复杂度关系未刻画、单指令域
6. 附录：全部超参、失败案例、状态可视化（PCA 轨迹随飞行阶段演化——审稿人喜爱的可解释性图）

## 9. 里程碑与单卡排程（1× RTX 4090 训练 + 本地 4060 评测）

排程原则：4090 全时训练不空转；评测/数据构建/写作与训练**并行在本地进行**；短程消融见缝插针排在全量训练间隙。以"1U = 现行 Stage2 全量一轮的墙钟"为单位估算，CoRL 必做合计约 **10-12U**。

```
W1     审计已于 2026-07-15 完成（flip 词表阴性 / proprio 失配不存在 / action_stats z 分布确认，见 §15）
        本地：cache_params 流式推理冒烟（纯工程验证，不重训）
        ∥ HF 路径卫生修复（半天）+ 运动原语类重加权 + 绑定头标签生成
W2-3   4090: 绑定轮 Stage3 重训（+指令绑定头 +加速度损失 +通道/幅度加权，~1.5U）
        → 复跑 273 条诊断（本地），验收三指标（冻结率/yaw 符号/dz 非零率）
W4-6   4090: S2 全量（检索补丁/塔尾 LoRA/位置保持 + 干净 hold-out 数据，~2.5U）
        ∥ 本地：记忆专项任务脚本化构建（最大自建项，提前动工）
W7-9   4090: S3→S4 全量（TBPTT K≤4 + cache_params 流式，~2U）
        ∥ 本地：UAV-Flow-Sim 闭环基线；流式版重点验收"停止判断 + 曲率"（诊断问题 4/5）
W10-12 4090: CoRL 必做短程消融队列（含降级后的 resampler A/B，~3U）
        ∥ 本地：效率曲线 + 记忆专项评测
W13-14 4090: 最终配置重训（~2U）∥ 写作
W15-16 成稿 → CoRL 投稿
```

单卡不可行而被裁掉的项（论文 Limitations 如实声明）：790M 缩放点、TBPTT K=8、DINOv2 双塔对照、CoRL 版等参数 Transformer 自训。

## 10. 参考文献（编号供 §11 对照表引用；[R1-R18] 均已回原文核验关键表述，[R19-R25] 为领域公认基础文献）

**已核验（存在性 + 原文关键表述均查证）**

- [R1] Zhao, H. et al. Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference. *AAAI 2025*. arXiv:2403.14520
- [R2] Waleffe, R. et al. (NVIDIA). An Empirical Study of Mamba-based Language Models. arXiv:2406.07887
- [R3] Lieber, O. et al. (AI21). Jamba: A Hybrid Transformer-Mamba Language Model. arXiv:2403.19887
- [R4] Liu, J. et al. RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation. *NeurIPS 2024*. arXiv:2406.04339
- [R5] μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models. arXiv:2606.12497
- [R6] HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy. *ICLR 2026*. arXiv:2510.00695
- [R7] Wang, X. et al. UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning. arXiv:2505.15725
- [R8] Gao, Y. et al. OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation. arXiv:2502.18041
- [R9] Chen, B. et al. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. *CVPR 2024*. arXiv:2401.12168
- [R10] Cheng, A.-C. et al. SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models. *NeurIPS 2024*. arXiv:2406.01584
- [R11] AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild. arXiv:2602.09657
- [R12] UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models. arXiv:2604.02241
- [R13] Pantazopoulos, G. et al. Lost in Space: Probing Fine-grained Spatial Understanding in Vision and Language Resamplers. *NAACL 2024*. arXiv:2404.13594
- [R14] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models. arXiv:2510.22936
- [R15] Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning. arXiv:2602.02951
- [R16] Qiao, Y. et al. VL-Mamba: Exploring State Space Models for Multimodal Learning. arXiv:2403.13600
- [R17] Zhang, W. et al. Open3D-VQA: A Benchmark for Comprehensive Spatial Reasoning with Multimodal Large Language Model in Open Space. arXiv:2503.11094
- [R18] CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs. arXiv:2503.01378

**领域公认基础文献（供方法节引用）**

- [R19] Gu, A. & Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752
- [R20] Dao, T. & Gu, A. Transformers are SSMs (Mamba-2). *ICML 2024*. arXiv:2405.21060
- [R21] Hu, E. et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. arXiv:2106.09685
- [R22] Zhao, T. et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT，temporal ensembling 出处). *RSS 2023*. arXiv:2304.13705
- [R23] Yang, L. et al. Depth Anything V2. *NeurIPS 2024*. arXiv:2406.09414
- [R24] Tschannen, M. et al. SigLIP 2: Multilingual Vision-Language Encoders. arXiv:2502.14786
- [R25] Kim, M. J. et al. OpenVLA: An Open-Source Vision-Language-Action Model. *CoRL 2024*. arXiv:2406.09246
- [R26] Takagi, Y., Kambara, M., Yashima, D., Seno, K., Tokura, K., Sugiura, K. AnoleVLA: Lightweight Vision-Language-Action Model with Deep State Space Models for Mobile Manipulation. arXiv:2603.15046（2026-03，Keio 大学）— 全文核验：token 序 `[state,Δstate,vision,language]`、SigLIP2 端到端微调、线性动作头、两阶段 velocity→acceleration 损失（消融 +4.73）、单 4090/24GB 20h 训练、Meta-World+实机；自陈 future work=显式空间推理+定位损失

## 11. 分部件可行性证据对照表（逐条可核验）

> 每个设计部件对应：支撑证据（含核验出处）→ 可行性如何被验证 → 证据缺口的诚实标注。证据等级：**A** = 顶会同行评审 + 原文核验；**B** = 预印本 + 原文核验；**C** = 结构性事实/官方实现；**⚠** = 无直接文献，需自证。

### C1 流式状态记忆

| 部件 | 支撑证据 | 可行性验证方式 | 等级 |
|---|---|---|---|
| SSM 状态跨步携带的 O(1) 步进推理 | Mamba/Mamba-2 递归模式是官方实现的既有推理路径（`mamba_ssm` 的 `InferenceParams` 状态缓存）[R19][R20] | 工程冒烟：370M 模型上跑通带状态续推的两帧前向，比对与全序列前向的输出一致性 | C |
| "历史信息对操作/飞行任务有大收益"的前提 | HAMLET：GR00T N1.5 加历史记忆后，历史依赖任务 SR 29.2%→76.4%（+47.2）[R6]；UAV-Flow 官方指出 OpenVLA-UAV"reliance on single-frame visual input imposes limitations"[R7] | 记忆专项任务上对比 ± 状态携带 | A（R6 为 ICLR 2026 录用） |
| Transformer 做历史的成本论证（反面动机） | HAMLET 原文："naively appending 4 frames 即产生 substantial computational overhead"，其 Table 4 给出延迟/显存随历史长度增长实测 [R6] | 复现效率曲线：AeroStream（恒定）vs 多帧拼接基线（线性增长） | A |
| TBPTT 训练递归记忆的可行性与超参 | μVLA：TBPTT K=2 已获大部分收益、K=8 更稳；MIKASA-Robo SR 0.42→0.84 [R5] | S4 冒烟：K=2 短程训练 loss 有限且下降；显存实测决定 K 上限 | B |
| receding-horizon 部署纪律 | μVLA 强消融：记忆每 chunk 更新一次则 cue-recall 任务崩溃，每步重查询是必要条件 [R5] | exec_horizon∈{1,4,8} 消融复现该现象 | B |
| 指令周期刷新 / 状态漂移控制 | **无直接文献** | 预实验：仿真 5 分钟连续闭环，监控动作分布散度与状态范数；N 扫描 | ⚠ |

### C2 结构对齐接口

| 部件 | 支撑证据 | 可行性验证方式 | 等级 |
|---|---|---|---|
| 指令后置序（现配方已实现） | Cobra 顺序消融：参考在问题后 TextVQA 掉至 47.9%，调序显著回升，归因 RNN 式 inductive bias [R1]；Cobra/RoboMamba 实现均 vision-first [R1][R4] | 反向消融（指令前置，预期显著变差）作为法则验证 | A |
| proprio 移到视觉后 | **无直接文献**（由 Cobra 排序原理外推） | A/B：现行序 vs `[vision\|state\|delta\|text]` | ⚠ |
| 全 576 token 直通（弃/减压缩） | Cobra：压缩投影"significantly harms"所有基准，Transformer 上同款无碍 [R1]；Lost in Space：resampler 联合训练可部分恢复（解释现配方可训 resampler 仍能工作）[R13]；成本可行性=Mamba 线性复杂度 [R19] | P0 预实验（§7.1）：单变量 A/B，4090 实测吞吐/显存 | A + C |
| 保留 resampler 时的位置保持 | PPE：保留压缩 token 位置 ID 免参数提升 2~5% [R14]；Nüwa：空间均匀覆盖原则 [R15] | resampler 输入加 24×24 位置编码的 A/B | B |
| 终端检索补丁（单层 cross-attn，零初始化） | 必要性：NVIDIA fuzzy memory——纯 SSM 精确回忆结构性失败 [R2]；充分性：Jamba 仅 1/8 注意力层即恢复 ICL [R3]、NVIDIA 24 Mamba-2 + 4 attn 超同规模 Transformer [R2]；零初始化旁路的安全性先例：动作头零初始化（本项目）与 LoRA 零初始化 B 矩阵 [R21] | 短程微调 ± 补丁；检查初始输出与无补丁版逐位一致（零初始化验证） | B |

### C3 几何注入（轻量版；伪深度/双塔已移入 TPAMI 扩展）

| 部件 | 支撑证据 | 可行性验证方式 | 等级 |
|---|---|---|---|
| 旁路空间辅助头（训练期 only） | UAV-Track VLA：辅助 grounding head 与动作损失联合、推理零开销 [R12]；SpatialVLM：度量监督塑形表征 [R9]；**AnoleVLA 将"定位损失/空间推理模块"列为其自陈 future work[R26]——本设计率先实现即为直接贡献** | 标签从 UAV-Flow raw_logs 免费计算；± aux head 消融 | B（+竞品未占据） |
| 二阶加速度损失（两阶段） | AnoleVLA 消融[R26]：加速度损失 SR 63.12%→67.85%（+4.73）；动作=delta pose（velocity），二阶差分=acceleration | S3 加 `λ_acc` 项，± 消融 + nDTW/平滑度对比；标签免费（现有动作序列差分） | B |
| （扩展项）伪深度分支 | AutoFly：伪深度使 UAV 导航 SR+3.9/碰撞-2.6，Siamese 共享投影 > 独立投影 [R11]；SpatialRGPT：相对深度过独立 connector 显著提升 [R10]；深度可离线预计算 [R23] | 本版不实施；若 CoRL 空间消融补偿不足（§7.4 触发条件）再回补 | B（保留证据备查） |
| 距离答案对数分箱 | SpatialVLM Table 3：自由数值最难（精确档 8.4%）[R9]；O3DVQA：微调后 size/direction +10%/+5% 而 distance marginal [R17]；fuzzy memory 结构解释 [R2] | 分箱模板重生成 O3DVQA 训练切分，干净 hold-out 量化 | A |
| 视觉塔尾部 2-4 block LoRA | SpatialVLM Table 3：解冻 ViT 使细粒度距离 5.6%→8.4%（"considerably better"）[R9]；LoRA 使解冻代价可控 [R21] | Stage2 A/B，监控通用 VQA 是否回退 | A |

### C4 效率与部署

| 部件 | 支撑证据 | 可行性验证方式 | 等级 |
|---|---|---|---|
| Mamba VLA 推理速度优势 | RoboMamba：3.2B 模型推理速度为现有 VLA 3×，370M 更快 [R4]；本项目实测：370M + fast kernels 单步 ~125ms（含编码，4090） | Jetson Orin 实测延迟分解；10Hz 达标判定 | A + 内部实测 |
| 恒定延迟/内存 vs 历史长度 | SSM 状态定长是结构事实 [R19][R20]；对照组成本曲线来自 HAMLET Table 4 范式 [R6] | 飞行时长 1/3/5 分钟三点实测曲线 | C |
| chunk 执行 + 温度集成平滑 | ACT temporal ensembling 出处 [R22]；本项目已部署验证 | 已有 UAV-Flow-Eval 闭环结果 | A |

### 评测与对比可行性

| 部件 | 支撑证据 | 可行性验证方式 | 等级 |
|---|---|---|---|
| UAV-Flow 闭环协议 + 官方基线 | 官方提供 OpenVLA-UAV 与 π0-UAV 适配方案及 SR/nDTW 协议（原文 Appendix B 有实现细节）[R7]，本项目已跑通该评测链路 | 直接沿用；等参数 Transformer 对照需自训（成本最高项，见 §7 风险） | A |
| OpenFly 基准 | 100K 轨迹 + OpenFly-Agent 基线开源 [R8] | 数据规模大，列为扩展实验（TPAMI 版）而非 CoRL 首发必需 | B |
| Open3D-VQA 干净 hold-out | 官方 80/10/10 切分协议 + LLaVA/Qwen2-VL 微调对照数 [R17]；本项目已复现其评分规则 | 训练集剔除脚本已有（`build_open3d_vqa_probe_split.py`） | A |
| 记忆专项任务（自建） | 任务设计参照 μVLA cue-recall 范式 [R5] 与 UAV-Flow 任务模板 [R7] | **需自建**：UAV-Flow-Sim 场景内脚本化生成，工作量 ~1-2 周 | ⚠（设计新颖点，亦是审稿风险点，需预留打磨时间） |

### 可行性缺口汇总（全部 ⚠ 项）

1. 指令周期刷新/状态漂移——纯工程预实验可解，1 天级；
2. 记忆专项任务构建——最大的自建工作量，但也是论文判决性实验，不可省；
3. 等参数 Transformer 对照自训——预算最高的可行性风险（约一次全量 Stage1-3 训练），CoRL 版可用"官方 OpenVLA-UAV（7B）+ 规模说明"暂替，TPAMI 版补齐。

（原"proprio 位置"⚠ 项已移除：AnoleVLA[R26] 提供了本体前置序的直接同构证据，现行序为证据支持的默认，仅保留一次确认性反向消融。）

## 12. v1.1 更正记录（2026-07-14，对齐 aeromamba_opt 现行配方）

v1.0 的部分描述基于旧版 AeroMamba（CLAUDE.md 所述 DinoSigLIP + `[text|proprio|vision]` 序），与现行 aeromamba_opt 在训配方不符，逐条更正如下：

| # | v1.0 描述 | 实际情况 | 影响 |
|---|---|---|---|
| 1 | 现有 token 序为 `[text\|proprio\|vision]`，属 Cobra 劣序，需重排 | 现行序是 `[state\|delta\|vision\|text]`，**指令已在最后，Cobra 有利序已满足** | C2① 从"纠正劣序"改为"proprio 位置微调 + 反向消融验证法则"；P0 预实验变量随之更换 |
| 2 | 视觉为 DinoSigLIP 双塔、729 token（27×27） | 现行为 SigLIP2 Base/16@384 单塔、**576 patch（24×24）**，resampler 后 64 token | 全 token 直通成本比 v1.0 估计更低；但单塔损失 DINOv2 几何特征，C3 几何通路从"锦上添花"升为"补缺口"（Cobra: 双塔在空间基准 +5~6%） |
| 3 | resampler 冻结、伤害按 Lost in Space 冻结场景估计 | 现行 resampler 在 Stage2/3 **全程可训**（Stage3 可训 43M 含 resampler） | 伤害预期下调；直通 A/B 从"必改"降为"P0 验证后决定"，检索补丁旁路成为等效替代路径 |
| 4 | Stage3 未含动作归一化/方向损失/指令镜像等改进 | 现行已实现：z-score + endpoint×0.25 + direction cosine×0.5、转向 3× 过采样、翻转+指令镜像 | 训练方案表改为"以现行配方为底座的增量"，避免重复建设 |
| 5 | 训练分 3a/3b 两段 | 现行 Stage3 已合并单段、LoRA 全程可训 | S3/S4 衔接描述更新 |

更正后仍然成立的核心主张：C1 流式状态记忆（现配方仍是单帧独立推理，此缺口未变）、C2③ 检索补丁（无法回看的结构缺陷与 resampler 无关）、C3 几何注入（单塔化后缺口反而更大）、C4 效率叙事（370M+64 token 的推理预算更充裕）。

## 13. AnoleVLA[R26] 专项：验证、借鉴与差异化（v1.3 新增）

AnoleVLA（arXiv:2603.15046，Keio 大学，2026-03；Meta-World + 实机移动操作）是目前与本设计最接近的已发表工作，全文已核验。它对本提案有三重作用。

### 13.1 验证（现行设计被独立复现为正确选择）

| 现行/本设计选择 | AnoleVLA 的独立证据 |
|---|---|
| token 序 `[state\|delta\|vision\|text]`（本体前置、指令后置） | 完全相同的序，且原文明确理由："本体 token 置于开头，使隐藏状态在整合视觉/语言前先被 agent 状态条件化"——现行序不再是"待 A/B 项"而是"有文献背书的默认" |
| SigLIP2 作视觉编码器 | 同选 SigLIP2（[R24]），且端到端微调 |
| 线性/MLP 动作头从末 token 读出 | 同款；原文论证"Mamba 递归聚合已在末 token 提供足够表达力，线性头即可" |
| Δstate 作为本体时间信号 | 同款 delta-state 输入 |
| 单张 RTX 4090（24GB）训练 | 467M 模型 20h/40 万步跑通——单卡预算的存在性证明 |
| 直接回归连续动作、不离散化 | 同款连续 chunk 单次前向 |

含义：审稿人若质疑"这些设计是否 ad-hoc"，可用 AnoleVLA 的独立到达作背书；同时也意味着这些选择**不能算作本文创新点**，创新必须押在 §13.3 的差异化上。

### 13.2 借鉴（可直接落地的改进）

**二阶加速度损失（两阶段训练）** 是 AnoleVLA 的核心贡献，直接可移植到现行 Stage3：
- 阶段一：现行 velocity 级损失（z-space L1 + endpoint + direction，已具备）；
- 阶段二：加 `λ_acc·‖ΔΔŷ − ΔΔy‖₁`（动作序列的二阶时间差分，即加速度一致性）；
- 证据：AnoleVLA 消融 SR 63.12%→67.85%（+4.73），且在实机接触密集任务上把 Open 任务成功率从 55%（SmolVLA，无平滑约束、动作抖动）拉到 75%；
- 对 UAV 的适配性判断：飞行动作的平滑性对动力学可行性与安全裕度比桌面操作更关键，预期收益方向一致（幅度需在 UAV-Flow 上自测）；实现零额外标注（现有 body-frame 航点序列差分即得），是本轮性价比最高的即时改进。

**（可选）多视角融合**：AnoleVLA 实机用 3 路相机 + SSM 线性复杂度融合，二手综述指出"多视角 + SSM 是自然配对但未被系统研究"。UAV 若配下视/前视双目，可作为 TPAMI 扩展的一个低风险增量点。

### 13.3 差异化（本设计相对 AnoleVLA 的真实新颖性）

| 维度 | AnoleVLA | 本设计（AeroStream） |
|---|---|---|
| 递归状态用法 | 单步序列内递归；跨步时间性靠 Δstate 显式输入 | **Mamba 状态跨控制步携带 = 持久飞行记忆（C1）**，TBPTT 训练 + receding-horizon 部署 |
| 精确回看短板 | 未处理（fuzzy memory 缺口敞开） | **注意力检索补丁（C2③）** |
| 空间/定位能力 | **自陈 future work**："计划引入显式空间推理模块 + 定位损失" | **本设计的 C3 正是此项**：旁路空间辅助头 + 分箱读出——率先实现即为直接贡献 |
| 领域 | 桌面/移动操作（Meta-World、HSR） | UAV 语言条件飞行（UAV-Flow/OpenFly） |
| 失败主因（其误差分析） | 位置识别错误 10/20（定位不准） | C3 直接针对该失败模式 |

一句话定位更新：AnoleVLA 证明了"Mamba 做小型高效 VLA"这条路可行且有效，但它**停在单步递归 + 无空间监督 + 操作域**；AeroStream 接着把递归状态推进到跨步飞行记忆、补上它自陈缺失的空间监督、并迁移到 UAV 域——三者叠加构成相对最接近竞品的清晰增量。

## 14. 现状诊断锚定（v1.4 新增，2026-07-15）

stage3_v2（`checkpoints/stage3_v2_20260714_082604`）在 UAV-Flow 273 条轨迹上的逐类诊断（`scripts/diagnose_trajectories.py` 等，可复跑）是本版本的优先级依据。设计从"文献驱动"转为"诊断驱动"：每个设计元素必须对应一个实测失败模式。

### 14.1 实测失败模式 → 根因 → 设计元素映射

| # | 实测失败（基线数字） | 根因判定（v1.5 按核实更新） | 对应设计元素 | 轮次 |
|---|---|---|---|---|
| 1 | 运动原语类冻结：127/273 末段静止；Move/Shift/Surround/Asc-Desc 路径中位 10~19cm vs GT 1.4~27m；有物体锚点的类正常 | 语言→动作通路未建立（动作由视觉物体触发）+ 回归坍缩到小位移均值（~~proprio OOD~~ 已排除，见 §15） | 指令绑定分类头 + 运动原语类过采样 + 幅度加权 | **绑定轮（W1-3）** |
| 2 | yaw 与指令脱钩：符号一致 14/27≈随机；25/30 输出负 yaw（系统性左偏）；幅度饱和 30~48° | ~~flip 镜像不完备~~ 审计阴性已排除（见 §15）；现存唯一嫌疑=方向词无监督通路；左偏成因未知，挂起复查 | yaw 符号分类监督 + 幅度分箱 | **绑定轮** |
| 3 | 垂直通道瘫痪：Ascend/Descend 15/19 dz≈0；Land dz 中位 -4cm vs GT -272cm | z 分量在训练分布/统计量中占比极低（已核实：z std 仅为 x 的 1/8，dx 均值 +0.41 前进先验）→ z-score 后均值坍缩；垂直任务样本少 | z 通道损失加权 + 垂直类过采样 + dz 符号分类监督 | **绑定轮** |
| 4 | 不会停/超行：末段 15~26cm/步至截断；终点误差中位 13.2m | 单帧推理无"已飞多远"累计信号（~~训练时 proprio 恒零~~ 已排除：实际训练路径速度信号合法非零，见 §15——缺的是长程历史，不是当前速度） | 流式状态 + TBPTT（根治，=C1） | **S4 轮** |
| 5 | 只会直线：净位移/路径长度 0.9~1.0；Surround 整圆/Pass 弧线无法执行 | 曲率需要"已绕角度"的积分信息，单帧结构性缺失 | 流式状态 + TBPTT（=C1，无捷径） | **S4 轮** |

### 14.2 proprio 时间通道（v1.5 更正：失配不存在于实际训练路径，降级为卫生修复）

> v1.4 曾将此判定为"修复优先级最高的 bug"，2026-07-15 经代码路径 + 远端训练日志双重核实后**推翻**，详见 §15。保留本节作为记录。

核实结论：

- **实际训练路径**：`run_stage3_v2.sh` 用 `--data_root`（无 `--hf_dataset`）→ `trainer.get_dataset` 走 **`UAVFlowDataset`** 分支。远端 `stage3_v2_20260714_082604/train.log` 中 `[UAVFlowDataset] Turn oversampling x3` 为直接证据；
- 该路径的时间通道**合法且非零**：`_convert_official_log` 逐帧从"上一帧→当前帧"计算 velocity（过去帧，无泄漏）；`__getitem__` 中 `delta_state8 = state8 - prev_state8`（step_idx>0 非零，= [velocity|accel]）；语义/单位与 server `preprocess_proprio` 逐项对齐（归一化米、yaw 弧度、`vel_per_step` 还原逐帧速度）；仅每条轨迹首窗口为零，恰与 server 每 episode 首次 `/predict` 行为一致；
- **全零失配只存在于 `UAVFlowHFDataset`（HF per-row parquet）路径**——本轮训练未使用，属休眠 bug。其 docstring 中"matching the inference server's delta_state=0 compatibility path"表述已过时（server 现发非零 delta）；
- 处置：降级为**卫生修复**（半天，非阻塞）——HF 导出附 anchor 前一帧使 velocity/delta 合法计算，或直接加断言防止误用；防止未来有人用 `--hf_dataset` 重训时踩中真实失配。AnoleVLA[R26] 中 Δstate 为承重时间信号的论据仍然成立，但它支持的是"HF 路径修复而非清零"，不改变当前优先级；
- 连锁影响：失败 4（不会停）不能再归因于"训练看不到速度"——模型看得到当前速度，缺的是**累计飞行历史**（已飞总里程/已绕角度），这反而强化了 C1 流式记忆作为唯一根治手段的地位；且 **S4 流式轮不再被任何修复阻塞**。

### 14.3 优先级重排说明（v1.5 按核实结论再更新）

1. **原 P0（resampler 直通 A/B）降级**：诊断显示瓶颈不在视觉压缩——有物体锚点的任务方向大致正确（Approach/Pass 能跟对目标），说明视觉通路基本工作；坏的是语言条件化。C2② 移入普通消融队列。
2. **C3 从"空间几何"扩为"语言-动作绑定"**：分类头以 yaw/dz 符号 + 运动类别 + 幅度分箱为主要目标（直接对应失败 1/2/3 的验收指标），度量距离目标降为次要。**v1.5 起它是唯一的第一优先**——proprio 修复与 flip 审计两条候选杠杆均已被核实排除（§15），语言-动作绑定成为失败 1/2/3 唯一未被排除的主因。
3. **C1 获得第一手动机证据**：失败 4/5 就是论文 Intro 的开题素材；S4 流式轮的验收标准直接沿用诊断脚本的停止判断/曲率指标。
4. **S4 流式轮不再被修复轮阻塞（v1.5 更正）**：v1.4 认为"流式训练依赖 proprio 语义修复先行"，该前提已随 §14.2 更正消失——实际训练路径的跨步 proprio 语义本来就是正确的。绑定轮与流式轮的先后仅剩消融叙事考虑（"绑定监督"与"流式记忆"两级增益分开呈现），工程上可并行准备。
5. **验收闭环**：每轮重训后复跑同一诊断脚本，三个量化验收指标=冻结率（基线 127/273）、yaw 符号准确率（基线 14/27）、dz 非零率（基线 4/19）；S4 轮加两个=末段速度衰减（停止判断）、Surround 角度覆盖（曲率）。新增第四个训练期探针：**指令敏感度**（同一观测配不同指令测输出散度），在每轮训练中期即可暴露"语言是否进入动作"，避免问题拖到全量评测才发现。

## 15. v1.5 更正记录（2026-07-15，两项 v1.4 根因判断经核实推翻）

两项核实均基于实际生效的代码路径、远端训练日志与全量训练数据扫描，而非文档推断。

### 15.1 更正一：proprio 时间通道"训练/推理失配"不存在于实际训练路径

| 项 | v1.4 断言 | 核实结果 |
|---|---|---|
| 训练数据路径 | HF per-row 路径，velocity4/delta_state8 恒零 | 实际走 `UAVFlowDataset`（`--data_root`），远端 train.log `[UAVFlowDataset] Turn oversampling x3` 为直接证据 |
| velocity4 | 恒零 | `_convert_official_log` 逐帧从上一帧合法计算，非零 |
| delta_state8 | 恒零 | `state8 - prev_state8`（=[velocity\|accel]），step_idx>0 非零 |
| 与 server 语义对齐 | 失配（OOD） | 逐项对齐：`state8=[pose4\|vel4]`、`delta=[vel\|accel]`、归一化米/弧度、`vel_per_step` 还原逐帧速度；轨迹首窗口零 ↔ episode 首帧零 |
| 处置 | 最高优先级修复 | 降级为休眠路径卫生修复（`UAVFlowHFDataset` 补前帧或加断言，半天，非阻塞）；同时修正其过时 docstring |

连锁修正：§14.1-#1 去除"proprio OOD"根因；§14.1-#4 去除"训练看不到速度"根因（缺的是累计历史，不是当前速度——反而强化 C1）；§14.3-4 解除 S4 对修复轮的依赖；§5.3 取消 proprio 修复消融项。

### 15.2 更正二：flip 指令镜像词表审计——结论阴性

对远端全量训练集扫描（53,586 条 instruction + instruction_unified）：

- 含方向词指令 45,465 条，其中**仅含词表未覆盖方向词的：0 条**；
- 词表外方向词全数据集仅 1 次（"rightward"，同句含已覆盖词）；
- 词表确有理论缺口（anticlockwise/leftward/leftmost 等）但在本数据分布中不出现，评测集 273 条中也仅 1 条；
- 结论：矛盾训练对量级 1/53,586，**不构成 yaw 左偏成因**。左偏成因未知（flip 50% 对称 + `--symmetrize_lateral` 强制 y/yaw 均值为零，坍缩只能解释零 yaw 而非系统性负 yaw），挂起至绑定头上线后复查；下一嫌疑为 Stage2 语义先验或 turn 过采样与视觉先验的交互。

### 15.3 附带核实：action_stats z 分布（v1.4 假设成立）

远端 `action_stats_k8.json`（归一化米/弧度，k 均值）：dx mean **+0.41** / std 0.52；dy std 0.31；**dz std 0.066（仅为 dx 的 1/8）**；dyaw std 0.148；y/yaw 均值被 `--symmetrize_lateral` 强制为零。"坍缩到均值"的预测形态（前飞、dz≈0、yaw 近零漂移）与 273 条轨迹实测行为吻合——支撑绑定轮的 z/yaw 通道加权与幅度加权设计。

### 15.4 更正后的净效果

- **绑定轮（指令绑定分类头 + 原语类过采样 + 通道/幅度加权 + 加速度损失）成为唯一 P0**；
- **S4 流式轮解除阻塞**，cache_params 推理冒烟可立即做；
- 修复类工作只剩半天级卫生项；W1 审计周的工作已全部完成并出结论；
- 其余数字勘误：§14.1-#4 终点误差中位 46m → **13.2m**；#1 GT 路径中位 1.427m → **1.4~27m**（区间跨类）。
