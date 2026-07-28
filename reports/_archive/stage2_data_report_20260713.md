# Stage 2 数据优化报告（v2 混合数据集）

日期：2026-07-13 | 数据文件：`data/stage2_mixed_data_v2.json`（180MB，服务器 `/root/autodl-tmp/Aeromamba/data/`）

## 1. 背景与动机

Stage 2 是视觉塔/投影器在冻结前**唯一一次**学习"俯视角空间 grounding"的机会（Stage 3 中两者均冻结）。
旧数据 `stage2_mixed_data.json` 存在四个问题：

1. **混合比例被数据量绑架**：COCO/LLaVA 157,712 条（68%）vs Open3D-VQA 73,324 条（32%），无 source 字段，训练时无法调节；
2. **大量与 UAV-FPV 无关的样本**：室内猫狗、食物特写等占了主要梯度预算；
3. **缺"指令→运动语义"数据**：Open3D-VQA 只教静态空间关系，与 Stage 3 的"指令→动作"之间存在空白层；
4. **无认知推理数据**：UAV-Flow 指令集含 object-interactive 任务（绕行/接近特定物体），纯几何 QA 覆盖不了。

## 2. v2 数据集构成（共 204,556 条，验证全部通过）

| source | 条数 | 占比 | 来源与构建方式 |
|---|---|---|---|
| `general` | 80,621 | 39.4% | 原 COCO/LLaVA 经关键词过滤（强词命中或 ≥2 弱词），保留率 51.1%，丢弃 77,091 条室内/特写样本 |
| `aerial_spatial` | 73,166 | 35.8% | Open3D-VQA 全量保留（去 158 条内容重复） |
| `uav_motion` | 47,103 | 23.0% | **新增**：UAV-Flow 训练集 26,848 个 episode 生成（方向 QA 26,794 + 指令改写 QA ~20,309） |
| `cognitive` | 3,666 | 1.8% | **新增**：CognitiveDrone 社区复现版 RLDS 提取 |

> 占比是"原始条数占比"。实际训练配比由 `--source_weights` 显式控制（见 §6），不再被数据量绑架。

### 2.1 `uav_motion`：语言-动作绑定数据（建议 2 落地）

- 来源：`/root/autodl-tmp/datasets/uav-flow` 26,851 个 episode，**逐一核对本地 273 个评测 episode，重叠为 0**（评测 episode 本就不在训练目录），排除逻辑仍写入脚本作双保险（`data/eval_episodes.txt`）。
- 每个 episode 取首帧 + 两类 QA：
  - **方向 QA**："给定指令，无人机整体应往哪飞？" 答案由轨迹终点相对首帧机体系位移导出（前/后、左/右、升/降组合）；
  - **改写 QA**：多样化指令 `instruction` → 规范指令 `instruction_unified`（80% episode 采样）。
- **符号约定实证校验**（防止标签反向，3,000 episode 统计）：
  - 指令含 "right" 的轨迹 77% 机体系 dy>0，含 "left" 的 79% dy<0 → **dy>0 = 右**，可靠；
  - "ascend"/"descend" 与 dz 符号一致率 97–100% → **dz>0 = 上升**，可靠；
  - 净偏航与 turn left/right 措辞一致率仅 39–55%（很多"turn"指令实际靠平移完成）→ **不生成旋转类标签**，避免注入噪声。
- 成品抽查：指令中的 left/right 与答案方向一致率 **96–98%**（剩余 2–4% 属于"从左侧绕过"实际主要直行等合理情况）。

### 2.2 `cognitive`：认知推理数据（建议 3 落地）

- 官方 `ArtemLykov/CognitiveDrone_dataset` RLDS 上传不完整（缺 features.json，社区 issue 已确认），改用社区复现版 `sonnt-vinu-cair/CognitiveDrone_rlds_dataset_REPRODUCTION`（2.1GB，16 分片，含完整元数据）。
- 用纯 Python TFRecord 解析（无需 TensorFlow，适配 2GB 内存无卡实例）提取每 episode 首帧 + QA：推理式指令（"飞过写着正确答案的门"）→ 解析后的明确指令（"飞过绿色圆形小门"），恰好监督"认知解析"这一步。
- 共 3,666 条（Human Recognition/Symbol/Reasoning 各类，含 Math_desk 166 条）；抽样人工核验图文一致（三门场景与答案门颜色/形状吻合）。
- 原始 tfrecord 已删除释放 1.9GB；仅保留提取后的帧（40MB）。
- 注：复现版数据量少于论文宣称的 8k+（部分类别每类 500 条封顶），按小剂量混入（默认 7.5% 配比）符合预期用途。

### 2.3 `general` 过滤规则

保留条件：命中任一强关键词（street/building/traffic/aerial 等 29 个城市/交通词），或命中 ≥2 个弱关键词（car/tree/sky/park 等 38 个户外词）。单一弱词不保留（"tree"、"sky" 在泛化 caption 中过于常见）。被丢弃样本抽查确认为室内猫、甜点特写等无关场景。

## 3. 质量验证（`data/validate_stage2_v2.py`，全部硬检查通过）

- 总数 204,556；**id 重复 0**（v1 因 COCO 同图多对话导致 67,043 个重复 id，已统一重编号并保留 `orig_id`）；内容级重复 158 条已去除；
- **图像 100% 全量存在性检查通过**（不抽样）；每 source 随机 30 张 PIL 解码通过；
- 对话 schema：human/gpt 交替、首轮含 `<image>`、无空文本，全部通过；
- **评测泄漏 0**：uav_motion 图像路径均不属于 273 个评测 episode。

## 4. Open3D-VQA 场景与评测环境重叠核实（建议 5）

- Open3D-VQA 论文（arXiv 2503.11094）确认场景为：UrbanScene3D（深圳重建，UE4）、EmbodiedCity（北京/武汉数字孪生）、WildUAV（罗马尼亚实拍）、自采深圳实拍；
- UAV-Flow-Sim 评测环境是 UE **校园场景**（`UnrealTrack-DowntownWest` 资产包）；
- **结论：无场景重叠**，不存在"评测同场景数据"可以特殊加权；UrbanScene3D-Campus 子集（26,496 条）在语义上最接近评测环境（低空校园/建筑），如需可通过后续 A/B 单独提高其权重（当前实现按 source 级加权，子集级可再细分）。

## 5. 代码改动（建议 1 落地）

| 文件 | 改动 |
|---|---|
| `data/llava_dataset.py` | 读取每条样本的 `source` 字段（缺省 `general`），暴露 `self.sources`，打印分布 |
| `training/trainer.py` | `get_train_sampler()` 钩子 + DataLoader 接入 sampler（sampler 与 shuffle 互斥） |
| `training/stage2_vlm.py` | `--source_weights "general=1,aerial_spatial=1.5,..."` → WeightedRandomSampler；权重语义 = 各源每 epoch 概率质量比，与原始条数解耦；空参数则保持原行为 |
| `data/build_stage2_v2.py` | v2 混合数据构建（流式，2GB 内存安全；内容去重 + 重编 id） |
| `data/extract_cognitive_drone.py` | RLDS tfrecord → LLaVA 格式（纯 Python 解析） |
| `data/validate_stage2_v2.py` | 严格质量验证（schema/图像存在/泄漏/重复/解码） |
| `scripts/run_stage2_v2.sh` | v2 训练启动脚本 |

单元验证：目标配比 50/25/25 时实测采样 47.6/27.8/24.7（1 epoch 内随机波动范围内）；`--source_weights` 为空时返回 None、行为与旧版完全一致。

## 6. 推荐训练配置与 A/B 方向

默认（`scripts/run_stage2_v2.sh`）：

```
general=1.0 (25%)  aerial_spatial=1.5 (37.5%)  uav_motion=1.2 (30%)  cognitive=0.3 (7.5%)
```

建议 A/B：
1. `aerial_spatial` 1.0 vs 1.5 vs 2.0（验证俯视 grounding 增益）；
2. `uav_motion` 0（对照）vs 1.2（验证语言-动作绑定对 Stage 3 的传递效应）；
3. general 比例低于 ~20% 时关注 val PPL 是否恶化（小 Mamba 语言底子薄，防灾难性遗忘）。

## 7. 磁盘与资源

- 处理全程在 2GB 内存无卡实例完成（ijson 流式 + 纯 Python tfrecord 解析）；
- 清理：CognitiveDrone 原始 tfrecord 1.9GB 已删；数据盘余量 79GB；
- 新增占用：v2 JSON 180MB + cognitive 帧 40MB + `uav_flow_frames` 符号链接（0 字节，复用 Stage 3 数据）。

## 8. 遗留与后续

- Stage 2 重训需 GPU（当前无卡模式），启动命令：`bash scripts/run_stage2_v2.sh`；
- OpenFly 指令-轨迹对暂未引入（UAV-Flow 自生成的 47k uav_motion 已够第一轮验证，OpenFly 需另行下载数百 GB，性价比待 A/B 结果决定）；
- 若后续需要子集级加权（如单独提升 UrbanScene-Campus），只需在 build 脚本中把 source 细分为 `aerial_spatial/campus` 等再配权重。
