# AeroV3 数据构建指引(HUGE-Bench)

日期:2026-07-28
配套:`docs/AEROV3_ENV_SETUP.md`、`docs/HUGEBENCH_STATEFUL_SMALL_VLA_DESIGN_20260728.md`

本文所有数字与字段均**亲自核验过**,包括与论文和 `info.json` 不一致之处。凡与官方说法冲突的,以本文标注的实测值为准。

---

## 1. 下载什么、不下载什么

```bash
export HF_ENDPOINT=https://hf-mirror.com      # huggingface.co 直连超时
export HUGE_ROOT=/root/autodl-tmp/huge
```

| 仓库 / 子目录 | 大小 | 要不要 |
|---|---|---|
| `yu781986168/HUGE_Dataset_v0` → `train` | 145.07 GB(5179 文件) | 训练要 |
| 　同上 → `test_seen` | 16.41 GB(580 文件) | 要 |
| 　同上 → `test_unseen` | 11.71 GB(421 文件) | 要 |
| 　同上 → `point_cloud_utm50.ply` | 6.88 GB | 不要(与环境仓库重复) |
| `yu781986168/3DGS_Mesh_Envs` | 29.44 GB(7 个 tar) | 闭环评测要;先拉 `1_office`(6.89 GB) |
| `yu781986168/HUGE_PI0` → `params` | 11.19 GB | **不要**(我们不跑 π0) |
| 　同上 → `train_state` | 28.93 GB | **不要**(优化器状态) |

7 个环境 tar:`1_office` 6.89 / `2_city` 3.28 / `3_road` 6.53 / `4_lake` 8.15 / `no1_building` 2.8 / `no3_door` 0.89 / `overhead_bridge` 0.91 GB。

> **`real_road` 在 `scene_annotations` 里有标注但没有对应的 3DGS tar。** 仓库引用 8 个环境,实际释出 7 个。任何遍历 `scene_annotations/data_3d/*` 的脚本都要跳过它。

下载脚本用 `snapshot_download(allow_patterns=...)`,失败重试,`max_workers=16`。全部环境资产于 **2026-06-24** 释出;此前只有 1 个场景,网上流传的"nDTW 全 0 / 渲染黑屏"复现失败报告都出自那个时期,与方法无关。

---

## 2. 数据格式(实测 schema)

`{split}/data/chunk-{NNN}/episode_{NNNNNN}.parquet`,每 1000 条 episode 一个 chunk 目录。

```
image         struct<bytes: binary, path: string>   # 当前帧,编码字节内嵌
first_image   struct<bytes: binary, path: string>   # episode 首帧,逐行重复存储
state         fixed_size_list<float>[4]             # 世界系 [x, y, z, angle]
actions       fixed_size_list<float>[4]             # 世界系位姿增量
env_id        string
timestamp     float
frame_index   int64
episode_index int64
index         int64
task_index    int64
```

三个必须知道的点:

1. **图像内嵌在 parquet(LeRobot 图像模式,不是视频模式)。** 训练不需要渲染器。
2. **`first_image` 逐帧重复存储,但不占磁盘。** 一个 267 帧的 episode 把同一张图存了 267 遍,而 parquet 的字典编码把重复字节压掉了 —— 实测单文件构成:`image.bytes` **98.8%**,`first_image.bytes` 仅 **1.2%**。
   > 早先"`first_image` 是 28 MB/episode 的主要来源、IO 白白翻倍"的说法**是错的**,实测推翻。真正的体积在当前帧图像上。
   >
   > **缓存仍然必要,但理由不同**:省的是逐帧重复 JPEG 解码的 CPU,不是 IO。这一点同时是架构上 `z_ep` 可缓存的数据侧佐证。
3. **指令不在 parquet 里。** 通过 `task_index` 查 `meta/tasks.jsonl`。`action_infer.py` 依次尝试 `prompt` → `task` → `instruction`,由 LeRobot 的 dataset 对象注入。

### 只读 state 不读图像

审计与信息论分析可以完全跳过图像字节,用 HTTP range read 只取需要的列:

```python
pq.read_table(f, columns=["actions", "state", "task_index", "frame_index", "env_id"])
```

参考实现:`scripts/hugebench_fetch_states.py`。

---

## 3. 单位与坐标约定

**动作是精确的世界系位姿增量,单位米。** 实测 `state[0] + cumsum(actions)` 与 `state[1:]` 的最大偏差 **0.0000 m**。

> 不存在 UAV-Flow 那种 cm/m 歧义。**不要**把 `--pos_unit auto` 那套逻辑搬过来,这里没有需要它解决的问题。

### 相机约定(作者在 issue #5 中确认)

| 任务类型 | omega | phi | kappa |
|---|---|---|---|
| 非障碍 | 固定 **-180** | 固定 **0** | `state[3]`(偏航) |
| 障碍 | 固定 | **`state[3]`** | 固定 |

两个推论:

- 相机位姿是 `state` 的确定性函数,不含额外自由度。**这是"图像是位姿的确定性函数"这一分析前提的依据。** 场景静态,视角在部分环境下接近俯视,且作者明确要求复现时保持默认值、不按任务或环境更改。
- **障碍任务用另一套约定,`state[3]` 的语义不同。** 任何跨任务统一处理 `state[3]` 的实现都会在障碍任务上静默出错。相关代码分支见 `convert_state_for_render`。

---

## 4. 阶段标注

在 HUGE-Bench 仓库内(小文件,无需从 HF 下载):

```
trajectory_generation/stage_annotations/
├── stage_segments/{train,test_seen,test_unseen}.jsonl   # frame_start, frame_end, subtask_id
├── episode_mapping/{train,test_seen,test_unseen}.jsonl  # episode_index → env/instruction
└── raw_subtasks/task_{0,building,farm,hl,orbit,orbit_multi,road}/<env_id>/subtask.txt
```

**test 两个 split 的阶段标注都已释出**,因此阶段预测准确率可以作为中间量报告,用于支撑"中间量对了终端指标才对"的因果链(见架构验证指引 L2)。

对齐方式:按 `episode_index` 取 segment 列表,`frame_start ≤ t ≤ frame_end` 落到 `subtask_id`;区间外的帧标 `-1` 并在阶段损失中屏蔽。

---

## 5. 划分、分布与已知不一致

| 项 | 实测 | 论文/元数据声称 |
|---|---|---|
| train / test_seen / test_unseen | **5,175 / 576 / 417** | 5,330 / 593 / 294 |
| train 帧数 | 1.72 M | — |
| 不同指令数 | 1,102 | `info.json`: `total_tasks: 109` |
| fps / 分辨率 | 5 / 256×256 | 一致 |
| episode 长度 | p50 **267**,max **2,340** | — |

> **`info.json` 内部自相矛盾**:`total_tasks: 109`、`splits: "0:109"`,而实际是 5,175 条 episode、1,102 条不同指令。**任何信任 `info.json` 的工具都会索引错位。** 以实际文件为准。
>
> **命名不一致**:README 写 `NSP`,论文写 `NTP`,复现脚本输出 `NSP` 字段。同一个指标。

### 指令泄漏(设计如此,不是缺陷)

- `test_seen` 与 train 的指令重叠 **96.7%** —— 考的是新初始条件下的泛化
- `test_unseen` 重叠 **9.6%** —— 考的是新指令 + 新环境

> 这两个数是按完整标注集、指令原文精确匹配复算的(`scripts/hugebench_data_qc.py`)。早先记的 95.7% / 5.8% 略低,差别在 `test_unseen` 上更明显 —— 它并非完全不重叠,约一成指令在 train 里出现过,写作时不能说成"零重叠"。

报告结果时两个 split 必须分开,合并平均会掩盖泛化差距。

### 任务族(实测,全部 5,175 条 train 标注)

**按 `task_id` 分族,不要按指令文本猜。** 关键字分类器会把 `building` 判成 orbit —— 它的指令确实写着 "Orbit the building in the upper left of the view once",但论文那个"Orbit 族 34%"只数 `hl` / `orbit` / `orbit_multi` 三个 id,不含 `building`。两种口径都对,混用就错。

| `task_id` | episode% | frame% | 阶段数 | 动作 | 我们的族 |
|---|---|---|---|---|---|
| `0` | 33.5 | 10.0 | 3 | 飞到指定目标上方 N 米 | **inspect** |
| `building` | 10.1 | 17.6 | 6 | 绕指代建筑一圈(不给半径) | orbit |
| `hl` | 13.4 | 11.1 | 6 | 在指定高度绕一圈 | orbit |
| `orbit` | 13.4 | 11.9 | 6 | 在指定高度与半径绕一圈 | orbit |
| `orbit_multi` | 7.1 | 11.3 | 6 | 螺旋下降环绕 | orbit |
| `road` | 11.7 | 17.8 | 5 | 沿道路朝指定方向巡检 | road |
| `farm` | 2.2 | 10.1 | 4 | 弓字形测绘扫描 | survey |
| `obstacle` | 8.5 | 10.3 | 1–2 | 绕障到达指定位姿 | obstacle |

三个可直接用的推论:

- **阶段数由 `task_id` 唯一决定**(orbit 系一律 6,`0` 一律 3),所以 orbit 族内的阶段预测是固定的 6 分类,log₂6 ≈ 2.58 bit —— 与我们估的约 1 bit 条件熵口径一致。
- **episode 占比与 frame 占比严重不成比例。** `0` 占 33.5% 的 episode 却只占 10.0% 的帧(p50 长度 99);`farm` 占 2.2% 的 episode 却占 10.1% 的帧(p50 长度 1,389)。**按 episode 采样和按帧采样得到的是两个不同的数据集**,报告时必须说清用的哪个。
- **episode 编号与 `task_id` 相关。** 下载到一半时统计分布会得到严重偏斜的结果(实测下到 3,137/5,175 时,orbit 帧占比读作 57.1%)。分布类统计一律走标注文件,不要走磁盘上有什么。

`obstacle` 的 `state[3]` 实测范围 [+1.915, +3.135],其余 task_id 都是 [-3.14, +3.14] —— **两套相机约定在数据里是可见的**,这为 §3 的警告提供了直接证据。

---

## 6. 官方评测协议常量

取自 `openpi/scripts/action_infer.py`,自写 rollout 时必须逐条对齐,否则数字不可比:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--exec_steps` | 10 | 每次推理执行 10 步(horizon 为 20) |
| `--smooth_overlap` | **True** | 重叠 chunk 对同一步的预测在世界系取算术平均 |
| `--truncate_to_gt` | True | 预测截断到 GT 长度 |
| `--max_steps` | 0 | ≤0 时按每条 GT 轨迹长度自动推断 |
| `--ndtw_eta` | 3.0 | `nDTW = exp(-DTW / (eta · len(gt)))` |
| `--ndtw_yaw_weight` | 0.1 | 偏航在 DTW 代价中的权重 |
| DTW `softmin gamma` | 0.5 | **soft-DTW,不是经典 DTW**,不能用标准库替换 |
| `--obstacle_angle_mode` | `yaw_legacy` | 脚本注释称 `phi` 才 "new/correct",但默认走旧约定 |

策略每步收到的观测恰好是四项:

```python
{"observation/state", "observation/image", "observation/first_image", "prompt"}
```

**渲染在每个执行步都发生,不是每次推理一次。** 一个 267 步 episode 要向 render server 发 267 次请求,GT 侧再一遍。评测的瓶颈是渲染而非策略 —— 这既是"单卡 7 天评测"的来源,也意味着**GT 侧渲染在所有消融变体间完全相同,必须缓存复用**。

---

## 7. Dataloader 要点

1. **按 episode 缓存 `first_image`**,不要逐帧解码(§2)
2. **训练样本 = (episode, 起始帧 t)**,取 `actions[t : t+20]` 作为 chunk 标签;帧步长做下采样,1.72 M 帧全量遍历在单卡上不现实
3. **阶段标签按 §4 对齐**,区间外屏蔽
4. **归一化**:动作按分位数(q01/q99)归一,统计量随下采样步长与 chunk 长度变化而重算
5. **`env_id` 参与分组**:§1 的信息论分析在同 `(env_id, instruction)` 组内进行,训练侧同样应保证 batch 内有跨组多样性,否则相位监督会退化
6. **按 episode 划验证集**,不能按帧划 —— 同 episode 的帧高度相关,按帧划会严重高估

---

## 8. 数据侧自检清单

上训练前逐条过:

- [ ] `state[0] + cumsum(actions)` 对 `state[1:]` 的最大偏差 < 1e-3 m
- [ ] `first_image` 在 episode 内所有行字节一致(确认可缓存)
- [ ] 阶段标注覆盖率:被标注的帧占比,以及 `subtask_id` 的取值域
- [ ] 每族 episode 数与帧数,确认 Orbit 族占比 ≈ 34%
- [ ] `test_seen` / `test_unseen` 的指令重叠率复现 95.7% / 5.8%
- [ ] 障碍任务与非障碍任务分别统计 `state[3]` 的取值范围,确认两套语义确实不同
- [ ] episode 长度分布,确认 p50 ≈ 267

现成脚本:`scripts/hugebench_aliasing_audit.py`(观测混叠)、`scripts/hugebench_partial_observability.py`(部分可观测两半测试)、`scripts/hugebench_fetch_states.py`(仅列读取)。
