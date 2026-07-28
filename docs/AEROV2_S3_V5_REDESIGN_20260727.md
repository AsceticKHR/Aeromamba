# AeroV2 Stage-3 重设计 (v5)：从「单帧静态回归」到「Mamba 时序策略」

> 2026-07-27 · deep-research (full) → 工程落地设计
> 目标：充分利用 Falcon-H1 的 Mamba-2 结构红利，在 UAV-Flow 上高效完成大部分飞行任务
> 约束：每处改动必须有一手证据支撑；能被根因修复取代的旧补丁一律删除，不做冗余叠加

---

## 0. 一句话主张

当前 Stage-3 的三个症状——**看不见图**（`vis_sens 0.07`）、**dz 通道塌缩**（std 比 0.124）、**闭环不停止**（GT 飞 2–4 m，模型飞 8–12 m）——不是三个独立缺陷，而是**同一个结构错误的三个投影**：

> 我们把一个**部分可观测的时序控制问题**，压缩成了**单帧输入 + 单 token 读出 + 点估计回归**的静态映射。

而 Mamba-2 的固定尺寸递归状态，恰恰是为这类问题设计的。**我们付了 Mamba 的成本，却没有兑现它的红利。**

v5 因此不是"加四个模块"，而是把这一个结构错误在三个层面同时纠正：**输入时序化**（兑现 Mamba）、**读出空间化**（修视觉通路）、**输出分布化**（修损失范式）。三者互相支撑，缺一则另两者收益打折。

---

## 1. 诊断：从症状到根因（已对代码/数据逐条证伪）

| # | 症状（实测） | 根因 | 代码证据 | 文献支撑 |
|---|---|---|---|---|
| R1 | `vis_sens=0.07` 而 `instr_sens=0.37`；Stage2 同主干 `Δshuffle=+0.918` | **单 token 读出瓶颈**：`global_token = hidden[:, -1, :]` | `model/aerov2.py:792` | π0 / GR00T N1 / Octo / UniVLA **四系统无一使用末 token**；唯一 Mamba 先例 RoboMamba 用 global pooling + 2D 接触点 |
| R2 | dz std 比 0.124（vertical 组 0.032），`dz_sign_ok=0.0` | **L1 点估计收敛到条件中位数**；数据前飞先验 → "温和前飞"就是 L1 最优解 | `action_head.py:313` 纯 L1 + `mlp[-1]` 零初始化 | OpenVLA-OFT 官方 FAQ 自述 "L1 → median mode"；Diffusion Policy 的 mode averaging |
| R3 | 闭环过冲、原地转向任务却平移 5+ m | **完全没有终止/进度监督**，且动作无界 | 全代码无 progress/stop 头 | UAV-Flow 原论文点名基线"到达后仍无限移动"；AerialVLA 零位移+LAND 双条件 SR +11.57 |
| R4 | val 在 1500 步"平台" | **假平台**：`--max_steps 1500` 且 `CosineAnnealingLR(T_max=1500)`，LR 已退火到 ~0；batch 8 ⇒ 仅见 12k 样本 | `scripts/_run_s3_full.sh` | OpenVLA-OFT 150K×64、OpenVLA 200K、SmolVLA 100–200K、SpatialVLA 120K |
| R5 | `ascend:3 / descend:1` | **覆盖缺失（extrapolation），非不均衡** | 数据统计 | Deep Imbalanced Regression (ICML'21) 自述：连续标签下"某些目标值完全无数据"时重加权无效 |
| R6 | K=8 里 1/8 损失质量浪费 | `_extract_body_frame_chunk` 从 anchor 自身起算 ⇒ `gt_action[0] ≡ 0`；`action_stats.mean[0]/std[0]` 全零，std 被 clamp 到 1e-3 | `dataset.py:682`；远端 `action_stats_k8.json` | — |

### 被证伪的两条假设（不采纳）

- **「dz_sign=0.0 是 z 轴符号约定不一致（NED vs ENU）」** — 否。更简约的解释成立：vertical 组 GT `|Δz|>0.3 m`（符号 ±1），而模型 dz 预测 std 仅 0.0156 ≈ 常数 0，落在 `SIGN_DZ_M=0.15` 死区内被判为 "hold"(0)，与 ±1 **永不相等** ⇒ 准确率恰好 0.0。这是塌缩的算术后果，不是坐标 bug。`mae_dz=0.4714 ≈ gt_std=0.4907` 进一步佐证是纯塌缩。
- **「改主干为双向/多方向扫描（BSM/CSM）」** — 否。Falcon-H1 是 **parallel hybrid**（attention head 与 Mamba-2 head 在同一 mixer block 内并行，SSM:Attn:MLP = 2:1:5），2D patch 身份并未被压掉；这与 Stage2 grounding head（KV 取 projector 2D patch）能达到 IoU 0.29、`center_std` 不塌陷完全一致。且 VL-Mamba 的消融显示 BSM vs CSM 差距 <1 点，量级远不足以解释 0.07。**读出路径才是瓶颈，不是主干。**

---

## 2. 设计：三处改动，一条主线

### C1 —— 输入时序化：兑现 Mamba 的结构红利（治 R3）

**动机（理论）**：`gt_action` 是 anchor 相对量，单帧 FPV 下"我已经飞了多远 / 该不该停"是**结构上不可观测的**。UAV-Flow 原论文对 OpenVLA-UAV 的诊断正是"单帧输入使其难以确定停止点"。要让停止可学，必须让状态可观测。

**为什么这正好是 Mamba 的活**：Transformer-VLA 加历史帧要付 KV cache 与 O(L²) 的代价，所以主流做法是砍历史（OpenFly 甚至专门做 keyframe selection 来压制冗余观测）。Mamba-2 把历史压进**固定尺寸隐状态**，单步推理开销与序列长度无关。**这是我们相对 Transformer-VLA 唯一的、尚未兑现的结构优势。**

**最小实现（避免冗余）**：
```
当前帧  : C-RADIO → projector → 729 tokens   (全分辨率，回答"往哪飞")
历史 n 帧: C-RADIO → projector → 各 pool 到 M=9 tokens (3×3)   (粗粒度，回答"飞了多远")
token 序: [hist_{t-nΔ} … hist_{t-Δ} | cur(729) | text | action_queries]
```
历史帧只保留 3×3 空间栅格：**信息论上足够支撑"进度/位移"，又不足以支撑"精细定位"**，因此不会与当前帧争夺读出注意力。LM 序列长度仅增加 `n×9`（n=3 时 +27，约 +3.5%），视觉编码器多 n 次冻结前向（`no_grad`）。

**配套的进度读出**：从 action readout 的池化输出接一个标量头，回归 `progress = step_idx / (length-1)`（Huber）。数据侧零成本——`log.json` 自带 `length`。
- 证据：DAgger-Diffusion-Navigation 在同一模型上的三选一消融——**进度回归 SR 91.4 > 二值 stop 分类 80 >> 加权二值分类 17.5**（加权二值导致过早停止崩溃）。因此**明确不采用加权二值 stop 头**。
- 佐证：Self-Monitoring Agent (ICLR'19) progress monitor 在 R2R unseen 上 SR +8pp；ProgressVLA 真机开启 progress guidance 后平均步数 100.8 → 53.3。

**终止判据（推理期，双条件）**：`progress > τ` **或** 预测 chunk 位移近零。照搬 AerialVLA 的双条件（TravelUAV SR 36.39 → 47.96）。训练侧对应加终帧零位移样本。

---

### C2 —— 读出空间化：K 个 action query cross-attend 2D patch（治 R1）

**动机**：末 token 是一个被因果扫描压过的**全局摘要**；动作需要的是"目标在画面哪个 patch"。AVA-VLA 的剪枝实验显示，剪掉 70% 视觉 token 后成功率仍有 97.3%——**动作只依赖少数几个 patch**，而全局摘要恰好把这个稀疏信号平均掉了。

**为什么这条在我们模型上一定通**：Stage2 的 grounding head 用的就是**完全相同的通路**——query 来自 LM hidden，KV 取 **projector 输出的 2D patch**（`model/aerov2.py:516`），实测 DIOR IoU 0.194 / Open3D 0.290、`center_std 0.06–0.09` 不塌陷。**这证明该通路在本模型上有梯度、有空间信号**。把它复制到动作侧是性价比最高的一步，而不是引入未经本模型验证的新结构。

**实现**：
```python
ActionReadout:  queries [K, D] + ctx_proj(LM terminal hidden)
                → 2 层 (MHA over LN(vis_patch_tokens) + FFN)
                → [B, K, D]，每个 waypoint 一个 query
```

**由此删除的冗余**（关键：这是"取代"而非"叠加"）：
- `action_query`（`--no_proprio` 时的单个学习 token）→ 被 K 个 readout query 取代，删除。
- `use_grounding_target`（把 6 维框向量加到 global_token）→ **RoboGround (CVPR'25) 的干净对照**：同一个框、同一策略，低维向量注入 vs 渲染回 patch 空间，Appearance 成功率 14 → 30。低维向量注入是已知的弱形式；cross-attn 直接读 patch 从机理上包含了它。删除。

---

### C3 —— 输出分布化：HL-Gauss 分箱交叉熵替代 L1 点估计（治 R2 + R3 的幅度侧）

**动机**：R2 的本质是"L1 的最优解就是条件中位数"，这是**损失范式问题，调权重治不好**。当前代码里的 `channel_weight_z=2.5`、`lambda_var` 方差地板都是在对抗这个数学事实的补丁。

**方案**：per-(k, dim) 在归一化空间上分箱，用 **HL-Gauss**（交叉熵 + 高斯软标签）训练，解码取 softmax 期望。
- 证据：HL-Gauss 相对 MSE 在 Q-transformer 大规模机器人操作上 **+67%**（Farebrother et al., ICML'24 Oral），且论文观察到"MSE 随训练变长会退化，交叉熵不会"——这与我们 R4 要延长训练的计划直接相关。
- 证据：BridgeVLA (NeurIPS'25) 把分布式 heatmap 换成同参数量 MSE 回归，RLBench **88.2% → 31.4%**。
- 证据：RT-1 把 256-bin 分箱换成连续高斯回归，总体 **−24 分**。

**三个免费的副产品**（说明这不是冗余，而是一石多鸟）：
1. **天然有界**：期望解码落在 `[bin_min, bin_max]` 内 ⇒ 不需要再单独加 `tanh × q99`（该形式在 VLA 基准上**没有**直接对比证据，属自研；分箱是已验证的等价手段）。
2. **等概率分箱天然给小通道平等分辨率** ⇒ 不需要 `channel_weight_z/yaw=2.5`，删除。
3. **分布式输出不会塌缩成常数** ⇒ 不需要 `lambda_var` 方差地板 hack，删除。

**归一化同步改为分位数**：q01/q99 → [−1,1]，替代 z-score。这是 π0 / FAST / LeRobot-π0.5 的默认做法（openpi `NormStats{mean,std,q01,q99}`），对重尾更稳，且与等宽分箱组合后每个 bin 的样本量更均衡。

**诚实的分歧披露**：OpenVLA-OFT 在 LIBERO 上测得连续 L1 **优于** 256-bin 离散（90.7 vs 86.5），RoboVLMs 在 CALVIN 上也是连续更好。证据是分裂的。我的归纳是：**真正的分界线是"分布式输出 vs 点估计"，而不是"离散 vs 连续"**——BridgeVLA 和 RT-1 的对照组都是**点估计回归**，而 OFT 的连续 L1 之所以能赢，是因为它配了 parallel decoding + action chunking 把多模态压力转移掉了。HL-Gauss 站在"分布式"这一侧且输出仍是连续期望，兼取两者。**但这条归纳是我的推断，不是任何单篇论文的结论**，因此 C3 在冒烟阶段以 `--action_out {l1,hlgauss}` 做 A/B，由 G5 门控裁决，不预设胜负。

---

### C4 —— 训练预算与目标修正（治 R4 + R6，非架构改动）

- `--sched_total_steps` 必须等于真实全量步数；全量预算 ≥ 20k steps（当前 1500 是欠训练平台，不是收敛平台）。
- chunk 从 `start+1` 起算，去掉恒零的 k=0，回收 1/8 的损失质量；同步重算 `action_stats`。
- 保留：`--aug_flip`（水平镜像合法）、turn/class 过采样（Re-Mix 证据支持，但**不对 ascend/descend 加权**——R5 是覆盖缺失，加权只会过拟合那 4 条轨迹）。

---

## 3. 净删除清单（"设计优美"的可验证含义）

修根因之后，以下**全部是对症状的补丁**，予以删除。这一节是本设计不引入冗余的证据：

| 删除项 | 原本在补什么 | 被谁取代 |
|---|---|---|
| `lambda_var` 方差地板 | 防 L1 塌缩成常数 | C3 分布式输出 |
| `channel_weight_z/yaw = 2.5` | 抬 dz/dyaw 梯度 | C3 等概率分箱 |
| `action_query`（单 token） | `--no_proprio` 下的读出锚点 | C2 的 K 个 readout query |
| `use_grounding_target` + `target_encoder` | 注入目标框（低维向量形式） | C2 cross-attn 直读 patch |
| `proprio_residual` / `proprio_dropout` | 抑制 pose 捷径 | 直接 proprio-free（已验证 pose 对 anchor 相对动作是纯捷径） |
| `TemporalEnsemble` | 平滑执行 | **删除**：RTC (NeurIPS'25) A 级反证——TE 在多模态任务上全面变差，即使推理延迟为 0；我们的"原地转 vs 前飞"正是多模态 |
| `lambda_smooth` / `lambda_acc` | 平滑正则 | chunk 已是 anchor 累积量，本身平滑；且平滑正则会强化"模板化前飞" |

---

## 4. 冒烟门控（G1–G7）

改动只有通过全部门控才允许进入全量训练。粗体为 v5 新增/收紧。

| 门 | 检查 | 通过判据 |
|---|---|---|
| G1 | 单 batch 过拟合 | `pos_err` 降至 <40% 初值 |
| G2 | 梯度流 | 仅 readout+head(+LoRA) 有梯度；vision/projector/base-LM 严格为 0 |
| G3 | 短真实数据收敛 | val `pos_err` 下降且 `act_std > 1e-3` |
| **G1v** | **视觉能否驱动动作**（架构证伪） | 指令固定为常量时，预测端点离散度 / GT 离散度 **> 0.5** |
| **G4** | 接地（批内真实反事实） | `responsiveness > 0.3` 且 `vis_share > 0.35`（数据校准）且 `instr ≥ prop` |
| **G5** | 通道不塌缩 | 短训后 dz / dyaw 端点 pred-std 比 **> 0.25** |
| **G6** | 进度/停止 | progress 头在轨迹上单调递增；终帧触发停止条件 |
| **G7** | 时延 | 单步推理 < 100 ms（10 Hz 实时；Mamba 递归下应与历史长度无关） |

G1v 是本轮加的最有价值的一道门：它把"指标不达标"分解成"架构不行"与"还没训够"两种可区分的情形，而 G1–G5 中原本没有任何一道能做这个区分。

G4 从软诊断升为硬门是本次的核心教训：v4 通过了 G1–G4 的旧版本却在闭环崩溃，正是因为 `vis_sens` 当时只报告不门控。

---

## 5. 落地顺序（迭代，不一次性上全部）

| 迭代 | 内容 | 为什么这样切 |
|---|---|---|
| **A** | C2 读出空间化 + C3 分布式输出 + C4 预算/chunk 修正 | 不改输入接口与推理协议，可独立归因；直击 R1/R2/R4/R6 |
| **B** | C1 输入时序化 + progress/stop 头 + 推理端双条件终止 | 改数据与推理协议，须在 A 的干净基线上做 A/B；直击 R3 |
| **C** | 闭环端：执行 horizon 8 → 2–4；关闭 TemporalEnsemble | 纯推理侧，最后做，避免与训练侧改动混淆归因 |

每一迭代结束跑一次 G1–G7 + `eval_s3_systematic.py`，未过则回到设计而非加补丁。

---

---

## 6. 迭代记录（实测驱动的设计修正）

### 迭代 A-1（2026-07-27）：C3 见效，C2 失效 —— 且失效原因是我自己引入的

跑法：`--readout xattn --n_bins 128 --norm_mode quantile --chunk_offset 1`，G1 150 步 + G3 600 步。

| 门 | 结果 | 对比 v4 |
|---|---|---|
| G1 过拟合 | PASS，`pos_err` 0.2949 → 0.0330 | — |
| G2 梯度流 | PASS，`action_head` 2.34 / LoRA 0.046，`projector`/`vision`/`lm_base` **精确为 0** | 交叉注意力读取冻结 projector 输出未泄漏梯度 |
| G3 收敛 | PASS | step150 `pos_err` 0.135 vs v4 step250 的 0.178 |
| **G5 通道塌缩** | FAIL 但显著改善：dz **0.199**、dyaw **0.604** | v4 dz 0.124（且 v4 用了 1500 步，这里只有 600 步） |
| **G4 视觉接地** | FAIL：vis/instr = 0.177 | v4 ≈ 0.20，**毫无改善** |

**C3 判定：有效。** dz 在 1/3 训练预算下从 0.124 提升到 0.199，dyaw 达 0.604。配合离线微实验（`scripts/test_v5_head_contract.py::T4`，目标分布取 dz 同形的"多数为 0 + 稀有 ±0.8"）：L1 最优解收敛到 0.0031（塌缩），HL-Gauss 保留 14.3%/14.9% 尾部质量（真值 15%/15%）。这条不再是文献外推，是本项目的可复现结果。

**C2 判定：前提成立，实现有缺陷。** G4 自动打印的读出内部：

```
|queries|=0.824   |ctx_term|=2954.997
L0: attn/res=10.35%  entropy=2.728/6.356 (42.92% of uniform)  max_w=0.2584
```

- 注意力**确实学会了选择性观看**：熵只有均匀分布的 42.9%，峰值权重 0.2584 是均匀值 (1/576) 的 150 倍。C2 的机理前提被证实。
- 但 `ctx_proj(ctx)` 的范数是学习查询的 **3600 倍**，注意力分支只占残差流的 10.35%。**我把未归一化的 LLM 终端隐状态直接加进了通往输出的残差流，等于在新模块内部重建了 v4 的旁路。**

**修正（A-2）**：不给 ctx 加缩放系数打补丁，而是**取消 ctx 路径，把文本 token 直接并入 KV**：

```
kv = [projector 2D patches | trunk 文本隐状态]     queries 纯学习，无加性上下文
```

理由：
1. 语言与视觉都**只能经由注意力**到达动作，由 softmax（而非未归一化残差）仲裁二者竞争——而 softmax 之争正是 `vis_sens` 所度量的量。
2. 这是 π0 / GR00T N1 的 action expert 结构，不是自研。
3. 代码是**减少**的：删掉 `ctx_proj`，不新增 FiLM 等机构。
4. 回归测试固化：`T1::1000x text scale does not drown vision`——把文本 key 放大 1000 倍（复现 2955 vs 0.82 的真实情形），patch 敏感度 0.128 vs 单位尺度下的 0.118，不被淹没。

**同时修正的度量缺陷**：`vis_sens` 原本只用 **1 个样本 + 1 条指令**估计，同一次训练的不同 checkpoint 上在 0.0065 / 0.1312 / 0.0248 之间摆动。这个噪声量级不足以支撑架构判定。改为对 8 个场景 × 4 条探针指令取平均。KV 中的 pad 位置也加了 mask，否则文本预算大半花在 padding 上。

**未采纳的备选**（记录以备回溯）：
- 给 ctx 残差加 ReZero 式可学习标量 α（init 0）。否决理由：若旁路仍是更省力的优化路径，α 会自行增长，问题复现；这是抑制症状而非消除通路。
- L0 注意力改为非残差 + FiLM 注入 ctx。否决理由：会切断"turn left"这类**答案不在图像里**的纯指令决策的直接通路；把文本放进 KV 同时解决了这个问题且更简单。

### 迭代 A-2：旁路消除，但暴露出**度量本身**不可信

改动生效：读出内部从 `attn/res=10.35%` 变为 L0 `27907%`（残差只剩学习查询，注意力完全主导），旁路彻底消失。但 `vis_mass` 仅 57%（L0）/ 41%（L1），而均匀先验是 90%——64 个文本 key 在 softmax 里压过 576 个 patch。

更关键的是发现 **G4 的尺子在失效**：`instr_sens` 在同一次运行的 step300 是 1.03，step600 是 0.024，摆动 40 倍。原因是探针用的是四条合成句（"Turn left."）和全零图像，**全部是 OOD 输入**；模型越拟合真实指令分布，对合成探针越无反应。用一个逐渐失效的尺子做架构判定是无效的。

**度量修正**：改为**批内真实反事实**——换成同批另一个样本的真实指令 / 真实图像，并以两者 GT 端点距离 `gt_spread` 作为物理标尺：

```
responsiveness = (instr_sens + vis_sens) / gt_spread     模型响应得够不够
vis_share      = vis_sens / (instr_sens + vis_sens)      视觉拿到应有的份额没有
```

修正后 responsiveness 在 step300 / step600 / G4 三点上读数为 0.604 / 0.615 / 0.585，**稳定**。

### 迭代 A-3：门槛校准 —— 用数据而不是直觉定阈值

`--vis_share_min` 该定多少？`scripts/probe_action_information.py` 对 GT 端点方差做嵌套三分解，不需要训练任何模型：

| K | 通道 | 指令 | 场景（当前帧可解释） | 相位（只有历史能解释） |
|---|---|---|---|---|
| 8 | dx | 50% | 29% | 21% |
| 8 | dy | 33% | 17% | **50%** |
| 8 | dz | 18% | **44%** | 39% |
| 8 | dyaw | **69%** | 14% | 16% |
| 64 | dz | 29% | 60% | 11% |

四条可直接指导设计的读数：
1. **dz 的场景占比最高（44%）** —— dz 塌缩不只是损失范式问题，视觉盲的策略结构上拿不到这 44%。C2 与 C3 是互补的，缺一不可。
2. **dy 有 50% 靠相位** —— 单帧结构上给不了，这是 C1 的定量依据（不再是"文献建议"）。
3. **dyaw 69% 靠指令** —— 读出偏向文本，对 yaw 而言部分是**正确**的，不该一味压制。
4. 均值 scene/(scene+instr) = 26/(26+43) = **0.38**，因此 `vis_share_min = 0.35` 是数据推出来的，不是拍的。

### 迭代 A-4：G1v —— 对 C2 的证伪实验（决定性）

`vis_share` 低有两种互斥解释，G4 的读数区分不了：**(A) 结构上视觉到不了输出**，**(B) 600 步（一个 epoch 的 0.3%）还没学会**。设计了可证伪的实验：

> 把一个 batch 内所有样本的指令**替换成同一条**，保留各自的图像与各自的目标。此时指令的信息量为零，**只有图像能区分样本**。若模型仍无法拉开预测，视觉就是结构上到不了输出。

结果：

```
[G1v] constant instruction, 200 steps:
      pred_spread=0.9177  gt_spread=0.9492  ratio=0.967 (>0.5)
      => PASS - vision CAN drive the action
```

**仅凭图像复现了 96.7% 的真实端点离散度。** 结构假设 (A) 被证伪，C2 成立。此后不再改读出架构。

佐证：一次意外跑长的运行（2150 步）显示 `vis_share` 单调上行 0.010 → 0.081 → 0.141 → 0.201，`end_err` 0.449 → 0.364 持续下降，`act_std` 在 0.02–0.13 波动无单调衰减。**瓶颈是预算（R4），不是架构。**

### 迭代 A-5：全量预算

吞吐实测（同一配置，仅改 batch）：

| batch | s/it | 显存 | 样本/秒 |
|---|---|---|---|
| 8 | 1.89 | 8.2 GB | 4.23 |
| **16** | **2.53** | **8.7 GB** | **6.32** |
| 24 | 3.98 | (梯度检查点下不增) | 6.03 |

取 batch 16。全量：**20000 步 × 16 = 320k 样本（约 20% 个 epoch），`sched_total_steps = max_steps = 20000`**，约 14 小时。对比 v4 的 1500 步 × 8 = 12k 样本，预算提升 **27 倍**。

### 本轮净删除（相对 v4）

`lambda_var` 方差地板、`channel_weight_z/yaw=2.5`、`action_query` 单 token、`use_grounding_target` 低维框注入、`proprio_residual`/`proprio_dropout`、独立诊断脚本 `diag_v5_readout.py`（诊断已内置 G4）。旧的 `--vis_ratio_min` 与合成探针一并移除。

## 7. 已声明的局限

1. 绝大多数一手证据来自桌面 manipulation，**UAV 低空导航航点域的外推未经本项目实验证实**；C3 的"分布式 vs 点估计"归纳是我的综合，非单篇结论。
2. `ascend/descend` 的覆盖缺失（3/1 条）**任何架构改动都无法解决**，vertical 类指标的上限受此约束；需另行采集（RoboAgent 给出的可操作量级是 ~50 条真实 + 4× 增强）。
3. C1 的历史帧池化粒度（M=9）与帧间隔 Δ 未经消融，为设计假设。
4. UAV-Flow-Eval 的 episode 终止规则需读源码确认——若为固定步数预算，停止头的收益将主要体现在 **nDTW** 而非 SR。

## AI 披露
本设计的文献检索与综合由 AI 辅助研究工具完成；所有一手来源 URL 见对应调研输出，采用前应二次核验。
