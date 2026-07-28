# 空中 VLA 高效推理：定位、迁移矩阵与实验迭代计划

日期：2026-07-28
状态：研究方向已定，实验待执行
定位：本文是**研究方向文档**，与 `docs/AEROV2_S3_V5_REDESIGN_20260727.md`（Stage-3 架构设计）互补，不取代它。
证据来源：deep-research 双路文献扫描（空中 VLA 域 + 高效推理机制域），所有 arXiv ID 已核验存在。

---

## 0. 一句话主张

> 机器人域的高效 VLA 机制**全部只在桌面操作基准上验证过**。把它们迁到空中低层控制上，哪些成立、哪些失效、哪些本就冗余——这个问题没人回答过，而我们的架构恰好是回答它的理想载体：**我们结构性地绕开了自回归解码与迭代去噪，因此能干净地隔离出剩余瓶颈。**

论文不是"我们把无人机 VLA 加速到了实时"——文献表明这个门槛在 4090 上对 1.5B 模型是自动达成的。论文是**一次有观点的机制审计**。

---

## 1. 赛道现状：为什么这个位置是空的

### 1.1 空中域最强的系统都不报延迟

| 系统 | UAV-Flow-Sim SR | 延迟 | 部署 |
|---|---|---|---|
| WorldVLN (2605.15964) | **79.12** | **未报告** | 服务器侧；Orin NX 仅做数据收发 |
| ImagineUAV (2606.01205) | 70.9 | 6.2 s/周期（蒸馏后） | 外挂 RTX PRO 6000 边缘盒 |
| OpenVLA-UAV (UAV-Flow 基线) | 65.61 | **0.172 s / 5.8 Hz @4090** | 地面站 |
| Pi-0-UAV (chunk=10) | — | 0.289 s @4090 | 地面站 |
| AerialVLA (2603.14363) | — | 0.38 s / ~2.6 Hz @4090, 17 GB | 非机载 |

**WorldVLN 把 "model compression and inference acceleration for fully onboard UAV deployment" 明确写进 future work。** OpenFly-Agent（ICLR 2026）摘要写着 "reduce computations" 但全文无任何延迟/FLOPs/token 数报告。CosFly-VLA 同样不报延迟。

**要打的数字：0.172 s / 5.8 Hz。空中领域至今没有任何带 VLM 骨干的工作发表过 sub-100 ms。**

桌面操作域的同规模参照系：TinyVLA-1B ≈ 40 ms (4090)、MiniVLA-300M ≈ 25 ms (3090)、π0-3B ≈ 70 ms (A100)。**这个数量级差距就是本课题的全部空间。**

### 1.2 两个自称"高效空中 VLA"的工作都不构成威胁

- **LiteVLA-H (2605.00884)**：唯一以此为主题。但作者 h-index 0、零引用；正文残留 "Submission and Formatting Instructions for ICML 2026"（未评审）；闭环表自述为 "representative results" 而非实测；256M 骨干身份从未说明；**无任何公开基准 SR、无代码、无数据集**。它只做了一个双速率调度器。
- **VLA-AN (2512.15258, 浙大 FAST Lab)**：真实且严肃，但**贡献全在 runtime 层**（AWQ + flash-attn + FFN-RMSNorm 融合 + KV 预载 + CUDA graph，4100→494 ms，8.3×），架构无创新，且在**私有 benchmark** 上评测（自建 8 场景，单任务 SR 98.1%）。Orin NX 上仍只有 2–3 Hz。

**没有任何工作做过"机器人 VLA 效率机制向空中低层控制的受控迁移研究"。**

### 1.3 评测协议的既有缺陷（我们的第二个杠杆）

UAV-Flow 原论文（2505.15725）明确写 SR 由**人工目视判定**——"determine the success rate based on manual inspection of whether the trajectory semantically satisfies the instruction"，仓库 `metric.py` 只有 NDTW。WorldVLN 另述终点距离 < ε 判据但 **ε 数值与 grader 均未公开**。

**因此 79.12 / 70.9 / 65.61 与我们的几何 SR 不可直接同列。** 正确做法见 `reports/uavflow_stopping_diagnosis_20260727.md` §1。把人工 SR 换成可复现自动判据，本身是一项方法论贡献。

---

## 2. 机制迁移矩阵（论文核心贡献）

判决依据：我们的架构是**单次前向 + 连续输出 + 无迭代去噪 + ~600–700 token + 无状态单步推理**。

| 机制 | 代表工作 | 报告收益 | 对我们的判决 | 理由 |
|---|---|---|---|---|
| Action chunking | ACT 2304.13705 | — | **已具备** | 本就输出 K×4 航点 |
| Temporal ensembling | ACT | — | **有害，已删除** | RTC 实证：真机 +100/200 ms 下震荡触发保护停机；"原地转 vs 前飞"正是多模态 |
| 并行解码 | OpenVLA-OFT 2502.19645 / PD-VLA 2503.02310 | 26× 吞吐 / 3.3× 延迟 | **已结构性满足** | K 个 action query 一次前向即等价于 OFT 的 parallel decoding + continuous + L1 三件套 |
| 一致性蒸馏+早退 | CEED-VLA 2506.13725 | 4.1× | **不适用** | 加速的是我们不执行的 Jacobi 迭代 |
| 投机解码 | Spec-VLA 2507.22424 | 1.42× | **不适用** | 无 AR 动作 token 序列可供投机；其"bin ID 距离放宽接受"依赖离散词表 |
| FAST tokenizer | 2501.09747 | 5×**训练**加速（非推理） | **不适用** | 我们不产生 action token |
| diffusion/flow action expert | π0 | — | **已避开（最大胜利）** | OFT 对照：L1 = 0.0729 s/90.7%，diffusion-50 = 1.9070 s/91.1%，**26× 延迟差而 SR 相当** |
| KV cache 压缩 | KV-Efficient VLA 2509.21354 | 1.34× | **不适用** | 无状态单步，无跨步 KV；Falcon-H1 的 Mamba-2 分支本无 KV |
| RTC inpainting | 2506.07339 | +200 ms 零退化 | **机制不适用** | 需可插入引导项的迭代去噪 τ 轴。**但问题陈述适用**，可迁移形式见 §4-B3 |
| 视觉 token 剪枝 | VLA-Pruner 2511.16449 / DivPrune 2503.02175 / FastV | 1.57–1.99× | **⚠️ 边际，做成消融** | 见下 |
| 跨帧 token 缓存 | VLA-Cache 2502.02175 / LAC 2602.00686 | 1.7× / 1.76× | **⚠️ 预期结构性失效** | 见下 |
| **异步推理** | SmolVLA 2506.01844 | 任务完成快 30% | **✅ 高价值，直接可用** | 模型无关、免训练，写进 `server_v2.py` 即可 |
| **CUDA graph / 图复用** | Jetson-PI 2607.12659 | **≈3×（1420.8→476.1 ms）** | **✅ 最高性价比** | 见下 |
| 量化 | BitVLA 2506.07530 / SQAP-VLA 2509.09090 | 1.4 GB / 4.4× | **✅ 若主张机载则必需** | 4090 上收益有限；Jetson 全面 memory-bound 时收益大 |
| 动态深度/早退 | DeeR-VLA / MoLE-VLA 2503.20384 | — | 理论可行，66 层深模型上收益需实测 | — |

### 2.1 三条需要展开的判决

**CUDA graph 是被本领域严重低估的一条。** Jetson-PI 单这一项就把 π0.5 从 1420.8 ms 降到 476.1 ms。理由很关键：**VLA 的 token 长度是确定的**（相机分辨率固定、指令长度波动极小），这是 LLM 解码不具备的性质，因此图捕获几乎无损。Falcon-H1-1.5B-**Deep** 有 66 层，内核启动次数多，收益应格外大。

**视觉 token 剪枝对我们收益有限而风险高。** VLA-Perf 测得 RTX 4090 平衡算子强度 163.7 FLOPs/Byte，π0 的 vision(321.4) 与 VLM(542.8) 均为 compute-bound——方向上剪 token 有效。但我们只有 ~700 token（π0 是 800），剪 30% 大概率只换个位数毫秒。而风险侧：VLA-Pruner 的核心发现是 prefill 语义显著性会剪掉动作关键 token；我们删除 `PerceiverResampler`（64-query 压缩"饿死了策略的视觉细节"）正是同一失效模式的亲身验证。

另有两条负面实测：PD-VLA 报告 **FastV 与 SparseVLM 部署到带 chunking 的 VLA 上都没有真正提速**（FastV 的注意力掩码开销反而拖慢，SparseVLM 速度与成功率双降）；EfficientVLA Fig.1(a) 显示**一旦系统被拖入 memory-bound 区间，剪枝收益迅速消失**。

**VLA-Cache 类跨帧缓存预期在飞行中结构性失效。** 其前提是"相邻帧背景静止"，但无人机前飞时整个视场每帧平移。这是可直接量化的假设：计算 UAV-Flow 相邻帧的 patch 级变化分布，与操作基准对比。**"某类主流加速方法在空中域结构性失效"本身就是可发表的负结果。**

---

## 3. 必须诚实处理的两条限制

### 3.1 Mamba 在 600–700 token 上没有速度优势——要主动自我否证

arXiv **2507.12442** 在 **RTX 4090** 上直接测过（BF16，官方 CUDA kernel，**含 Falcon-H1-0.5B/1.5B**）：

> "Transformer 在短序列（<8K token）下最多快 **1.9×**；SSM 在 ~57K token 才出现反转，最多快 4×。"

细节：Qwen2.5-0.5B vs Mamba2-780m 短序列 TTFT 快 1.9×；能耗在 <16K token 时 Transformer 最低；SSM 的 selective scan 因串行逐元素本质，在 Jetson 上占总运行时 55% 以上。

辅证：SpecPrune-VLA 附录 A.1.5 在 A800 上发现**输入 < 2048 token 时 eager attention 持续快于 FlashAttention**（短序列下计算单元未饱和，FlashAttention 的额外控制流与逐元素操作成为主导）。

**结论**：`CLAUDE.md` 已写的"永远不要宣称渐近速度优势"现在有一手证据背书。Falcon-H1 的正当主张只有两条：
1. **参数效率**（2507.22448：1.5B-Deep 媲美 7B–10B 级）——能力/参数比，非速度。
2. **KV 内存的条件性优势**——当前单步无状态时**未兑现**；若扩展多帧历史则兑现（可引 Hymba 11× KV 缩减）。措辞必须是条件式。

能耗优势只在长上下文成立，**不要引用**。

论文里应主动写：*"我们选择混合 backbone 是为参数效率与未来长上下文可扩展性；实测确认在 600–700 token 工作点上，混合架构相对同规模 Transformer 没有延迟优势，这与 [2507.12442] 的独立测量一致。"* 审稿人自己发现会很难堪，先讲出来则成为严谨性证据。

#### 3.1.1 为什么 RoboMamba / AnoleVLA 的加速数字不构成反例

这两篇是"Mamba VLA 更快"叙事的主要来源，但**都不是架构对照实验**，不能用来支撑主干选型：

| | 模型规模 | 对比基线 | 报告加速 | 混淆因素 |
|---|---|---|---|---|
| RoboMamba (NeurIPS'24, 2406.04339) | 3.2B（2.7B LLM） | LLaMA-AdapterV2、ManipLLM（均为 LLaMA-7B 系） | 3×／7× | 参数量差 ~2.6×；基线做自回归文本生成；**论文中无等参量 Transformer 基线** |
| AnoleVLA (2603.15046) | 深层 SSM，370M 级 | π0.5（3B，flow-matching） | ~3× | 参数量差 ~8× 却只快 3×（**次线性**）；基线慢的主因是多步去噪，非注意力 |

两条读法：

- RoboMamba 无法分离"Mamba 更快"与"2.7B 比 7B 更快"。论文自己承认 2.7B 在复杂推理上不及 7B/13B。
- AnoleVLA 参数少 8× 只快 3×，说明 SSM 主干**吃掉了**一部分本应到手的参数优势——方向上恰好与"Mamba 更快"相反。且 OFT 对照（§2 表）已表明 L1 头 vs diffusion-50 单这一项就有 26× 延迟差。

正确归因是：**小模型 + 非迭代动作头 = 快**。这两个因素与 token mixer 正交，在 Transformer 上同样可得。

#### 3.1.2 来自 SSM 阵营自身的证词（最强引用）

- **Mamba-3**（2603.15569，Albert Gu 组，2026-03）摘要原话："*their theoretically linear inference remains hardware-inefficient in practice.*" MIMO 形式正是为把 memory-bound 的外积换成矩阵乘、喂饱 tensor core 而设计。
- **state-spaces/mamba 官方 issue #657**：维护者称 Mamba-2 在短序列下仍慢于 Transformer，原因是"large constant overhead"，明确建议"*for tasks involving short sequences, Transformers remain the faster and more practical choice*"，优势需 4K token 以上才显著。
- **MambaOut**（CVPR'25，2405.07992）：SSM 仅在**长序列 ∧ 自回归**双条件下划算，阈值 L > 6D。我们 D=1536 → 阈值 9216，而工作点 600–700，差一个数量级以上。该文另证给 ViT 加因果 mask 会掉点——**因果性对视觉理解是负担而非收益**，这与我们 R1 诊断（因果 last-token readout 是瓶颈、需 C2 交叉注意力恢复全可见访问）互相印证。

#### 3.1.3 对 Orin NX 迁移的直接影响

2507.12442 的算子级归因中最关键的一条：**selective scan 在边缘平台占总延迟 55% 以上**，因其串行、逐元素、bandwidth-bound。Orin NX 显存带宽远低于 4090，因此 **SSM 分支的相对劣势在边缘端只会被放大，不会缩小**。迁移阶段必须把 attention 分支与 Mamba 分支分别计时，不能只报端到端数字。

### 3.2 停止能力是发表前提，不是可选项

当前 0/273 自主停止、SR@3m 30.1、OSR@3m 63.4。**一篇跑在"永不停止、SR 30%"策略上的效率论文会被直接拒。** 能力线必须先达到大致追平 OpenVLA-UAV。

可借鉴的唯一完整先例是 **AerialVLA (2603.14363)** 的 "intrinsic stopping policy"：终止帧标注 ⟨0,0,0⟩ 位移 + 文本 `LAND` token，执行时**双条件**（生成 LAND **或** 预测近零位移）。

**没有任何空中 VLA 有学习式 progress / value head**（VLA-AN 的 temporal comparison module 是完成度检查，不是学习头）。这是干净的空位。

---

## 4. 实验迭代计划

三条轨道并行，A 与 C 是 B 的发表前提。

### Track A —— 能力线（让 SR 值得被讨论）

| 步 | 内容 | 判据 |
|---|---|---|
| A0 | 当前 20k 步 Falcon v5 sim 全量跑完，作为所有后续实验的基座 | `pos_err_m` < 类均值 0.2271 **且** `vis_share` 落在数据要求 0.215 附近。**只用 pos_err 会选走视觉盲策略**——实测 step 1000 的 pos_err 0.2095 最低但 `vis_share` 仅 0.024，step 14000 的 pos_err 0.2389 而 `vis_share` 0.247。见 `reports/s3_v5_grounding_probe_failure_and_recalibration_20260728.md` |
| A1 | **尾部填充**：目标索引越界时钳位到末帧位姿，`terminal_pad_frac` 限制填充比例 | 停止率**显著 > 0**（当前 0/273） |
| A2 | **K 扫描** 8 → 16 → 32，配 A1 | K=8 仅覆盖 0.53 m 而任务跨度中位 7.6 m；延迟成本由 VLA-Perf Takeaway 6 保证接近零（chunk 50→250 仅 +11%） |
| A3 | **progress 头**（回归归一化剩余进度，非二分类）+ 推理端双条件终止 | 依据 DAgger-Diffusion-Navigation 消融：progress 回归 91.4 > 二分类 80 >> 加权二分类 17.5 |
| A4 | 对照 AerialVLA 的 LAND-token 形式做 A/B | SR@3m 向 OSR@3m (63.4) 收敛 |

### Track B —— 效率线（论文主体）

| 步 | 内容 | 判据 / 备注 |
|---|---|---|
| **B0** | **延迟 profile + 两个免费杠杆**。四段分解：C-RADIO / projector / Falcon-H1 prefill / xattn readout+head。同时测：(a) CUDA graph 开关，(b) **eager vs FlashAttention**（<2048 token 下可能反转） | 这是整个计划的开关。预期分布：LM prefill 70–80%、视觉 15–20%、头 <5%（基于三组独立文献外推，**必须用自测替换**） |
| B1 | **机制适用性矩阵实测**：逐条验证 §2 的判决 | 产出论文核心表 |
| B2 | **VLA-Cache 失效假设验证**：计算 UAV-Flow 相邻帧 patch 级变化分布，与操作基准对比 | 负结果亦可发表 |
| B3 | **延迟鲁棒性曲线**（UAV 特有，最高新颖性）：注入 δ ∈ {0,50,100,200,400} ms × exec_horizon h ∈ {1,2,4,8}，测 SR@3m / 停止率 / 超冲 | 假设：SR 与开环位移 $v(\delta + h\Delta t)$ 单调相关（实测 v ≈ 0.188 m/step）。**VLA-Perf 明确留下的开放问题**："increased staleness may degrade action quality, which warrants further investigation" |
| B4 | **异步推理**写进 `server_v2.py`（SmolVLA 式 client-server 解耦） | 必须**同时报告反应时间**而非只报吞吐——Jetson-PI 指出异步引入感知-执行错配，且反应时间仍被延迟下界钳制。对无人机比机械臂更严重 |
| B5 | **token 剪枝消融**：用我们独有的 `vis_share` 度量剪枝的真实代价 | 整个剪枝文献都在担心"剪掉动作关键 token"，而我们有量化的视觉依赖度指标。**基准侧要求已直接测得：UAV-Flow-Sim 验证集按指令分组，组间端点散布 1.5283 m / 组内 0.4198 m → 视觉可解释份额 0.215**（`--mode grounding_eval` 输出）。注意这与代码注释里 "instr 43% / scene 26% / phase 32%" 是不同的分解（后者三向、按方差；前者两向、按散布），不可混用。**这是别人做不出的实验** |
| B6 | 量化（W8A8 / W4A16 PTQ 或 QAT），仅在主张机载时做 | 必须用 G4/G5 门控（`responsiveness`、`vis_share`、逐通道 spread）验证动作未塌缩，不能只看 loss。BitVLA 的原生 1-bit 路线走不了（需从 BitNet 权重起） |

### Track C —— 协议线（可信度）

| 步 | 内容 |
|---|---|
| C1 | 补跑 OpenVLA-UAV 缺失的 38 集（全在 Approach/Pass/Land，即其最差三类） |
| C2 | 修 Surround 包围性判据（现判据测 OpenVLA-UAV 为 0.0% 而官方 100%，导致我们**低估**对手） |
| C3 | 所有 `pos_err_m` 必须与 `scripts/trivial_action_baselines.py` 的平凡基线表并列 |
| C4 | 若主张任何架构优势，补等参量纯 Transformer 对照（MaIL, CoRL 2024 方法论标准） |

---

## 5. 报告规范（直接采用 VLA-Perf 定义）

两篇综述（2510.17111 §7.5 与 2510.24795）都独立呼吁标准化，而空中子领域尚无此规范——**严格执行本身即可主张为贡献**。

必报项：

1. **单步延迟四段分解**，毫秒，CUDA event 计时，明确 warmup 与重复次数。
2. **端到端 system latency**：图像时间戳 → 动作可用，**含 client-server 网络往返**（RTC 先例：LAN 10–20 ms）。
3. **有效控制频率**，且**明确区分推理频率与 `exec_horizon` 摊薄后的动作执行频率**。这是最容易被审稿人抓的膨胀点——OpenVLA-OFT 的 109.7 Hz 来自 8 动作 / 0.0729 s。
4. **硬件全披露**：GPU 型号显存、精度、**attention 实现与版本**（短序列下会反转）、是否用 CUDA graph、PyTorch 版本。
5. **延迟鲁棒性曲线**（B3）。
6. **参数量与显存峰值**——承载对 Falcon-H1 的正当主张。
7. **TTFA 与 prefill 占比 ρ**（LiteVLA-H 引入，对无人机贴切）。
8. 跨论文对比必须附免责声明：硬件/任务/动作空间/本体各异，用于系统级定位而非排行榜。

---

## 6. 引用卫生（已发现的坑）

- **两篇名字极像的综述是不同团队的独立工作**：`2510.17111`（中科院自动化所，四维分类，配 github.com/guanweifan/awesome-efficient-vla）vs `2510.24795`（同济等，三支柱分类，配 evla-survey.github.io）。不要混引。
- **综述 2510.17111 参考文献 [79] 误引**：把 SuSIE（机器人 subgoal 图像方法）引成了 *"SuSIE: Search using services and information extraction", ICDE 2013*——一篇无关的数据库论文。**不要从其文献表直接转引，务必回溯原文。**
- **"AeroVLA" 不存在**，特征吻合的是 **AerialVLA (2603.14363)**。
- **未评审 preprint（引用需标注）**：WorldVLN、ImagineUAV、AerialVLA、CosFly-VLA、LiteVLA-H、VLA-AN、UAV-Flow 本身。**已评审**：OpenFly（ICLR 2026）、AIR-VLA（ICML 2026）、TravelUAV（ICLR 2025）、AerialVLN（ICCV 2023）。
- **待核实**：RTC 与 VLA-Cache 的 NeurIPS 2025 归属；AIR-VLA 的 "20 Hz testbed"（来自综述转述）。
- **常见数字误引**：OpenVLA-OFT 的 "26×" 是**吞吐**增益，延迟只降 3.3×；FAST 的 "5×" 是**训练**加速不是推理；PD-VLA 的 2.52× 中并行解码只贡献 1.28×，其余来自 chunking 本身；ImagineUAV 的 "real-time" 实为 6.2 s/周期（≈0.16 Hz），其效率论证靠参数量而非延迟。

---

## AI 披露

本文的文献检索与综合由 AI 辅助研究工具完成；所有 arXiv ID 已通过官方 API 核验存在，但正文细节多来自 HTML 版本，采用前应二次核验 PDF。
