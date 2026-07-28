# AeroV3 环境配置指引

日期:2026-07-28
适用:HUGE-Bench 上的 AeroV3 训练与评测。架构见 `docs/HUGEBENCH_STATEFUL_SMALL_VLA_DESIGN_20260728.md`。

本文记录的是**已在目标机器上核实过的**版本与路径,不是通用安装说明。凡标注"实测"的行均可复现。

---

## 1. 目标机器基线(实测)

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D,24564 MiB,**sm_89** |
| 驱动 | 580.142(支持 CUDA 12.x) |
| OS / 编译器 | Ubuntu 22.04,gcc 11.3.0,cmake 已装 |
| Python | 3.10.12 |
| 训练环境 | `/root/autodl-tmp/envs/aerov2`,**torch 2.4.0+cu121** |
| 系统 CUDA toolkit | `/usr/local/cuda` → **11.8**(`cuda_11.8.r11.8`) |
| CPU / 内存 | 128 核 / 503 GB |
| 数据盘 | `/root/autodl-tmp`,530 GB |
| `uv` | **未安装**(不需要,见 §4) |

---

## 2. 训练环境

训练不需要 CUDA toolkit —— PyTorch 自带运行时,只有编译自定义 CUDA 扩展时才需要 nvcc。AeroV3 不含自定义 CUDA 算子,因此训练侧直接用现有 `aerov2` 环境即可。

```bash
source /root/autodl-tmp/envs/aerov2/bin/activate   # 或 conda activate aerov2
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 期望: 2.4.0+cu121 True (8, 9)
```

### 国内网络

`huggingface.co` 在该机器上**直连超时**(实测 `ConnectTimeoutError`)。所有 HF 访问必须走镜像:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

注意 `huggingface_hub` 的部分 API 对象会在构造时捕获 endpoint,**先 export 再启动 Python**,不要在进程内 `os.environ.setdefault` 之后再实例化 `HfApi()`。

完全离线跑请设 `AEROMAMBA_OFFLINE=1`。

---

## 3. 渲染器环境(仅闭环评测需要,可推迟)

> **训练完全不需要渲染器。** 图像已内嵌在 HUGE-Bench 的 parquet 里(见数据指引 §2)。3DGS 渲染只在闭环 rollout 时用到,因此本节可以推迟到训练跑通之后再做,不阻塞主线。

### 3.1 版本冲突:必须单开一个环境

系统 nvcc 是 **11.8**,而 `aerov2` 的 torch 是 **cu121**。直接在 `aerov2` 里编译 `diff-gaussian-rasterization` 会因 CUDA 主版本不一致失败。

有两条路,推荐第一条:

**方案 A(推荐,零额外下载):给渲染器单开一个 cu118 环境。** 渲染器是独立的 socket 服务进程,与训练进程不共享 Python 环境,互不影响。系统已有 11.8 toolkit,且 **11.8 是第一个支持 Ada(sm_89)的版本**,可以编 4090。

```bash
conda create -n gsrender python=3.10 -y && conda activate gsrender
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"          # 只编 Ada,省一半编译时间

cd /root/autodl-tmp/huge/gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install plyfile tqdm
```

**方案 B:装 CUDA 12.1 toolkit 配现有 cu121 torch。** 需额外下载数 GB:

```bash
conda create -n gsrender python=3.10 -y && conda activate gsrender
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 适配文件覆盖

HUGE-Bench 只发布了 patch 文件,需要覆盖进上游 checkout:

```bash
cd /root/autodl-tmp/huge/gaussian-splatting
cp ../HUGE-Bench/gaussian_splatting/3dgs_renderer.py .
cp ../HUGE-Bench/gaussian_splatting/my_render_traj.py .
cp ../HUGE-Bench/gaussian_splatting/utils/graphics_utils.py utils/graphics_utils.py
```

`utils/graphics_utils.py` **必须覆盖** —— 上游没有 `getProjectionMatrix_with_principal`,这是 HUGE-Bench issue #1 的成因。

### 3.3 单卡共存:必须关掉显存预分配

官方命令假定渲染在 GPU 0、推理在 GPU 1。单张 4090 上两者要共存:

- `1_office` 的 PLY 6.89 GB,gaussian-splatting 会把全部高斯载入显存,预计占 5–7 GB
- 我们的 0.7B 策略 bf16 约占 2–3 GB

PyTorch 侧不预分配,所以方案 A 下问题不大。但若日后引入任何 JAX 组件,**必须先设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`**,JAX 默认吃掉 75% 显存,会让渲染服务必然 OOM。

启动:

```bash
CUDA_VISIBLE_DEVICES=0 python 3dgs_renderer.py \
  --host 127.0.0.1 --port 5550 \
  --ply_template "$HUGE_DATA_3D_ROOT/{env_id}/3dgs_ply/point_cloud_utm50.ply"
```

---

## 4. 明确**不安装**的东西

| 组件 | 为什么不装 |
|---|---|
| **openpi** | 它存在的唯一理由是运行 π0。AeroV3 是 PyTorch 的,rollout 循环按 `action_infer.py` 的协议自己写即可。跳过它就跳过了 JAX、`uv` 和一整套与 PyTorch 冲突的 CUDA 栈。π0 的基线数字直接引论文。 |
| **`uv`** | 只有 openpi 用它 |
| **Isaac Sim** | HUGE-Bench 的闭环评测**不需要** Isaac Sim,只需要 3DGS 渲染服务 + mesh 碰撞查询。Isaac Sim 仅用于原作者的数据采集。 |
| **`HUGE_PI0/train_state`** | 28.93 GB 的优化器状态,只有续训 π0 才需要 |

---

## 5. 本地(Windows)开发注意

- dataloader 用 `--workers 0`,Windows 下多进程 worker 会因 spawn 语义报错
- 远程操作统一走 `plink` / `pscp`(见 `.cursor/skills` 与 `AGENTS.md`)
- **`pkill -f <pattern>` 会误杀自己的会话**:plink 用 `bash -c '<脚本全文>'` 执行,脚本里的模式串会匹配到承载它的 bash 进程。务必排除 `$$` 与 `$PPID`,否则连接会被自己杀掉(表现为 `Software caused connection abort`)。

---

## 6. 环境自检

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m py_compile model/aerov3.py data/hugebench_dataset.py training/v3_train.py
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(0))"
python scripts/test_aerov3_wiring.py          # 形状/梯度流/token 顺序契约
python scripts/hugebench_qc.py --split test_seen --n 8   # 数据侧自检
```

渲染器就绪后追加:

```bash
python scripts/hugebench_render_fidelity.py --env_id 1_office --n 16
# 渲染 GT 位姿并与 parquet 存图逐像素比对,详见架构验证指引 L0.5
```
