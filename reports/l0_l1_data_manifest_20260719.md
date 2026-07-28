# AeroMamba v2 数据处理清单（L0 CPT + L1 动作索引）

- 日期：2026-07-19
- 机器：AutoDL/seetacloud 无卡实例（128 核 CPU，2GB cgroup 内存上限），`connect.westb.seetacloud.com`
- 对应设计：`docs/MAMBA_UAV_VLA_REDESIGN_20260719.md` §2 数据选型

---

## 1. L0 空中域 CPT 数据（`/root/autodl-tmp/datasets/l0_cpt/`）

统一 schema（LLaVA 会话格式 + 来源/任务标签，逐行 JSONL 可流式加载）：

```json
{"id": "...", "source": "...", "task": "...",
 "image": "相对路径", "image_root": "stage2|l0",
 "conversations": [{"from": "human", "value": "<image>\n..."}, {"from": "gpt", "value": "..."}]}
```

`image_root` 解析：`stage2` → `/root/autodl-tmp/Aeromamba/data/`，`l0` → `/root/autodl-tmp/datasets/l0_cpt/`（见 `build_report.json`）。

### 构成（共 215,552 条）

| source | task | 条数 | 说明 |
|---|---|---|---|
| general | vqa | 80,636 | COCO/LLaVA-Instruct 户外子集（防遗忘回放） |
| aerial_spatial | spatial_vqa | 73,166 | Open3DVQA 具身 3D 空间概念 |
| uav_motion | motion | 47,103 | UAV-Flow 轨迹转运动语义 QA |
| hrvqa | aerial_vqa | 9,999 | 高分辨率航拍 VQA，每图聚合 ≤8 轮（蓄水池采样自 100 万 QA） |
| cognitive | cognitive | 3,651+15 | CognitiveDrone 认知推理 |
| airspatial | grounding | 495 | 指代表达→bbox（**归一化 0-1000**，Qwen 风格） |
| airspatial | metric_vqa | 502 | **米制距离/深度 QA**（数字 grounding 关键监督） |

### 图像

- `images/hrvqa/`：9,999 张，1024px PNG → **512px JPEG q90**（26GB 原始 → 331MB）
- `images/airspatial/`：502 张 DJI 航拍，缩至最长边 1024（bbox 已按原始尺寸归一化，缩放无关）
- 总占用：JSONL 182MB + 新增图像 696MB；stage2 图像复用原路径（coco/open3d_vqa 等，已在盘）

### 原始包保留在 `l0_cpt/raw/`（airspatial 1GB / hrvqa 3.9GB / open3dvqa_v2 2GB）

Open3DVQA-v2 新版 zip（含 EmbodiedCity/WildUAV pkl chunks）已下载未解析——现用 stage2 已处理版 73k；若需 v2 增量再补转换器。

---

## 2. L1 动作数据索引（`/root/autodl-tmp/datasets/uav-flow/metadata/`）

数据本体核实：26,795 episodes / 1,785,284 帧（与 54 个 parquet 分片行数**精确一致**），帧为 256×256 JPEG（~40KB），共 78GB；`manifest.jsonl` 带 `instruction` / `instruction_unified` 双字段。

### `l1_episode_index.jsonl`（26,795 行，每 episode 一行）

- 指令解析标签：`motion_hint`（8 类正则）、`direction_word`、**`magnitude_m` / `magnitude_deg`（从指令抽取的数字幅度）**
- 轨迹实测：`path_len_m`、`net_dx/dy/dz_m`、`yaw_delta_deg`
- 用途：绑定监督（方向/幅度/类别）、按类均衡采样、幅度-数字相关性探针

### `l1_windows_T48_S24.jsonl`（42,932 个窗口）

T=48 / stride=24 滑动窗口索引，供 S4 流式 TBPTT 训练；短 episode 保留截断窗口（不丢难样本，CosFly 教训）。

---

## 3. 磁盘现状与待决事项

- 磁盘：删除 54 个 parquet 分片（241GB）后，已用 **204GB / 530GB，空闲 327GB**（2026-07-19 执行，删除前已核对 26,795 episode 目录与 L1 索引完好；原始数据可随时从 HF `wangxiangyu0814/UAV-Flow` 重下）。
- 已知损耗：首次构建被 2GB cgroup OOM 杀死损失的 15 行已补回（215,552 = 204,556 + 9,999 + 495 + 502）。

## 4. 复现命令

```bash
# L1 索引
python data/build_l1_index.py --root /root/autodl-tmp/datasets/uav-flow --window 48 --stride 24
# L0 构建（低内存模式，2GB cgroup 安全）
python data/build_l0_cpt.py --workers 8            # 全量
python data/build_l0_cpt.py --groups hrvqa --append  # 断点续跑单组
```
