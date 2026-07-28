# Stage2 评测详细统计

> **污染声明**：Stage2 v2 已训练全部 Open3D-VQA；Open3D 分数为 **seen / in-distribution probe**。

- Stage2 slim: `checkpoints/stage2_v2/best_slim.pth`
- 全量 probe n=10280，耗时 4.39 h
- split_seed=42

## 1. 四源冒烟（v2 val，每源 20）

| source | n | CLM loss | PPL | exact | partial | 解读 |
|---|---:|---:|---:|---:|---:|---|
| aerial_spatial | 20 | 0.283 | 1.33 | 20.0% | 20.0% | 字面匹配严、空间定位一般 |
| cognitive | 20 | 0.191 | 1.21 | 30.0% | 75.0% | 模板对、细粒度弱 |
| general | 20 | 1.702 | 5.48 | — | — | 仅看语言遗忘（无准确率） |
| uav_motion | 20 | 0.065 | 1.07 | 90.0% | 85.0% | 指令→运动绑定强 |

## 2. Open3D-VQA 全量 probe 准确率

### 2.1 总体 / 域

| 集合 | n | exact | partial |
|---|---:|---:|---:|
| ALL | 10280 | 33.7% | 38.2% |
| real | 3266 | 35.1% | 39.7% |
| sim | 7014 | 33.0% | 37.5% |

### 2.2 任务桶

| bucket | n | exact | partial | 相对总准确率 |
|---|---:|---:|---:|---|
| other_qual | 3904 | 33.7% | 35.8% | -0.0 pp |
| tf_qual | 2343 | 38.8% | 38.8% | +5.1 pp |
| direction | 2251 | 48.6% | 57.0% | +14.9 pp |
| distance_quant | 1782 | 8.2% | 18.7% | -25.4 pp |

### 2.3 前 100 条 Real 探针（开跑冒烟）

| bucket | n | exact | partial |
|---|---:|---:|---:|
| ALL | 100 | 36.0% | 40.0% |
| other_qual | 36 | 38.9% | 38.9% |
| direction | 24 | 45.8% | 54.2% |
| tf_qual | 24 | 37.5% | 37.5% |
| distance_quant | 16 | 12.5% | 25.0% |

说明：前 100 条按 JSON 顺序，几乎全是 RealworldUAV/Lab，故 domain=real only。

## 3. Test_official_probe 构成（n=10280）

### 3.1 Split / 场景

| split | n | % |
|---|---:|---:|
| test_sim_10pct | 7014 | 68.2% |
| test_real | 3266 | 31.8% |

| scene | n | % |
|---|---:|---:|
| EmbodiedCity/Wuhan | 3012 | 29.3% |
| UrbanScene/Campus | 2581 | 25.1% |
| RealworldUAV/Lab | 1748 | 17.0% |
| UrbanScene/Residence | 1421 | 13.8% |
| RealworldUAV/Park | 736 | 7.2% |
| RealworldUAV/Residence | 460 | 4.5% |
| WildUAV/Wild | 322 | 3.1% |

### 3.2 QA type / question_name

| qa_type | n | % |
|---|---:|---:|
| qualitative | 6751 | 65.7% |
| quantitative | 2184 | 21.2% |
| NONE | 1345 | 13.1% |

| question_name | n | % |
|---|---:|---:|
| `NONE` | 1345 | 13.1% |
| `left_choice` | 262 | 2.5% |
| `distance_data` | 259 | 2.5% |
| `left_relationship2agent` | 253 | 2.5% |
| `right_choice` | 250 | 2.4% |
| `above_predicate` | 246 | 2.4% |
| `wide_predicate` | 244 | 2.4% |
| `vertical_distance_data` | 242 | 2.4% |
| `horizontal_distance2agent` | 242 | 2.4% |
| `above_relationship2agent` | 241 | 2.3% |
| `behind_multichoice` | 240 | 2.3% |
| `front_predicate` | 239 | 2.3% |
| `left_predicate` | 237 | 2.3% |
| `below_predicate` | 233 | 2.3% |
| `thin_predicate` | 233 | 2.3% |
| `above_choice` | 230 | 2.2% |
| `direction_data` | 230 | 2.2% |
| `behind_predicate` | 227 | 2.2% |
| `wide_choice` | 227 | 2.2% |
| `right_relationship2agent` | 225 | 2.2% |
| `left_multichoice` | 224 | 2.2% |
| `below_choice` | 223 | 2.2% |
| `thin_choice` | 221 | 2.1% |
| `above_multichoice` | 221 | 2.1% |
| `below_multichoice` | 220 | 2.1% |
| `tall_predicate` | 216 | 2.1% |
| `front_multichoice` | 216 | 2.1% |
| `vertical_distance2agent` | 215 | 2.1% |
| `short_choice` | 211 | 2.1% |
| `height_data` | 211 | 2.1% |
| `below_relationship2agent` | 208 | 2.0% |
| `distance2agent` | 208 | 2.0% |
| `front_choice` | 205 | 2.0% |
| `right_predicate` | 204 | 2.0% |
| `horizontal_distance_data` | 204 | 2.0% |
| `behind_choice` | 201 | 2.0% |
| `short_predicate` | 201 | 2.0% |
| `tall_choice` | 201 | 2.0% |
| `width_data` | 199 | 1.9% |
| `right_multichoice` | 192 | 1.9% |
| `direction2agent` | 174 | 1.7% |

### 3.3 场景 × 任务桶（样本数）

| scene | direction | distance_quant | other_qual | tf_qual | total |
|---|---:|---:|---:|---:|---:|
| EmbodiedCity/Wuhan | 695 | 521 | 1108 | 688 | 3012 |
| RealworldUAV/Lab | 380 | 304 | 675 | 389 | 1748 |
| RealworldUAV/Park | 160 | 128 | 294 | 154 | 736 |
| RealworldUAV/Residence | 100 | 80 | 185 | 95 | 460 |
| UrbanScene/Campus | 555 | 465 | 983 | 578 | 2581 |
| UrbanScene/Residence | 291 | 228 | 532 | 370 | 1421 |
| WildUAV/Wild | 70 | 56 | 127 | 69 | 322 |

### 3.4 域 × 任务桶（样本数）

| domain | direction | distance_quant | other_qual | tf_qual |
|---|---:|---:|---:|---:|
| real | 710 | 568 | 1281 | 707 |
| sim | 1541 | 1214 | 2623 | 1636 |

### 3.5 答案平均长度（字符）

| bucket | mean_ans_len |
|---|---:|
| direction | 74.3 |
| distance_quant | 61.4 |
| other_qual | 65.9 |
| tf_qual | 32.9 |

## 4. Val_sim_probe 构成（未跑准确率，仅清单）

- n=6999
| scene | n |
|---|---:|
| EmbodiedCity/Wuhan | 2999 |
| UrbanScene/Campus | 2552 |
| UrbanScene/Residence | 1448 |

## 5. 读数要点

- 全量 exact **33.7%** 落在 seen 数据上，说明空间 QA 未被背熟。
- **direction 48.6%** 最好；**distance_quant 8.2%** 最差，拖低总分。
- Real(35.1%) 与 Sim(33.0%) 接近 → 未见差距小；但因污染，不能解读为 sim→real 泛化成功。
- Test 中 Sim 占 68.2% / Real 占 31.8%，总分更接近 Sim。
- 冒烟 `uav_motion` 90% 与 Open3D 空间弱形成对比：Stage2 更擅长指令运动语义，而非精细测距。
- 当前全量结果**没有按 scene / question_name 存预测**，故准确率只能到 domain×bucket；构成表见上文。
