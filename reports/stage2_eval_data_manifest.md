# Stage 2 本地评测数据清单

- 日期：2026-07-14
- Stage2 checkpoint：`checkpoints/stage2_v2/best_slim.pth`（由远端 `.../stage2_v2/best.pth` mmap 瘦身）
- 污染声明：**Stage2 v2 训练已含全部 Open3D-VQA（`aerial_spatial`）→ Open3D-VQA 分数为 in-distribution probe，非干净 zero-shot 官方榜**

## 分层 A — Smoke（冒烟，不对外报官方分）

| 项 | 值 |
|---|---|
| 来源 | `stage2_mixed_data_v2.json` 训练同款 val（`val_frac=0.1`, `split_seed=42`） |
| 导出脚本 | `data/export_eval_subset.py` |
| 本地路径 | `data/stage2_eval_smoke/`（`eval_subset.json` + `images/`） |
| 规模 | 评测时每 source 取前 **20** 条（包内可含更多以便诊断） |
| 四源 | `general` / `aerial_spatial` / `uav_motion` / `cognitive` |
| 评测脚本 | `scripts/eval_stage2_capabilities.py --subset_json ... --loss_per_source 20 --gen_per_source 20` |

## 分层 B — Diagnostic（可选加厚，仍非官方）

| 项 | 值 |
|---|---|
| 同源 | 同上 val split，每源 shuffle seed=`0` |
| 规模 | loss **150** / gen **40** per source |
| 用途 | 内部 PPL + 方向词 / gate / aerial 字符串匹配 |
| 不进入官方主表 | `uav_motion`、`cognitive`、`general` |

## 分层 C — Official probe（Open3D-VQA，论文风格）

| 项 | 值 |
|---|---|
| 数据根（本地） | OneDrive `...\dataset\aeromamba\open3d_vqa\O3DVQA` |
| Sim 场景 | `EmbodiedCity/Wuhan`, `UrbanScene/Campus`, `UrbanScene/Residence` |
| Real 场景 | `RealworldUAV/*`, `WildUAV/*` |
| Split 产物 | `data/open3d_vqa_probe/`（`split_seed=42`） |
| 实测计数 | merged QA=73324；`test_real`=3266；`test_sim_10pct`=7014；`val_sim_10pct`=6999；`train_sim_80pct`=56045 |
| Test_official_probe | **10280** = Real 全量 + Sim 10% → `test_official_probe.json` |
| Val_sim_probe | **6999** → `val_sim_probe.json` |
| 模态 | 仅 RGB（无点云） |
| 打分 | `scripts/eval_open3d_vqa_probe.py`：TF/MCQ exact；SAQ 定性关键词；SAQ 定量相对误差 ∈ **[0.75, 1.25]** |
| 冒烟先行 | `reports/open3d_vqa_probe_smoke100.*`（100 条，exact≈36%） |
| 全量产出 | `reports/open3d_vqa_probe_test.{json,md,log}` |

## 分层 A 冒烟实测（2026-07-14，每源 20 条）

| source | CLM loss | PPL | exact | partial |
|---|---|---|---|---|
| aerial_spatial | 0.283 | 1.33 | 20.0% | 20.0% |
| cognitive | 0.191 | 1.21 | 30.0% | 75.0% |
| general | 1.702 | 5.48 | — | — |
| uav_motion | 0.065 | 1.07 | 90.0% | 85.0% |

详见 `reports/stage2_smoke_capability_eval.md`。

## 明确不用作官方 Stage2 主评测

- CognitiveDrone、UAV-Flow `uav_motion`、COCO/LLaVA general、UAV-Flow-Eval 闭环（留给 Stage3）
