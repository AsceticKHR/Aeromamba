# AeroMamba Stage 1/2 环境与数据迁移手册

本文记录在单张 RTX 4090 服务器上部署 AeroMamba、准备 Stage 1/2 数据并启动训练的完整流程。目标是换服务器后可以按顺序复现，而不是靠回忆修补环境。

## 1. 本次验证基线

| 项目 | 已验证配置 |
|---|---|
| 服务器仓库 | `/root/Aeromamba` |
| Python | `/opt/conda/bin/python`（Conda base） |
| PyTorch | `2.1.2`，PyTorch CUDA `11.8` |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| Mamba | `mamba_ssm` 与 `causal_conv1d` 可导入 |
| 视觉编码器 | `dinosiglip_so_384` |
| Stage 1 数据 | LLaVA-Pretrain，558,128 条 |
| Stage 2 数据 | AeroMamba `stage2_mixed_data.json`，231,036 条 |
| Stage 2 图像 | 82,309 张唯一图像，全部位于数据根目录可解析路径下 |

`dinosiglip_so_384` 不是单独的 SigLIP。它同时加载：

- DINOv2-L with registers：`vit_large_patch14_reg4_dinov2.lvd142m`
- SigLIP-SO400M：`vit_so400m_patch14_siglip_384`
- 两路 384×384、27×27 patch 特征按通道拼接，输出维度为 2176。

首次启动时 GPU 可能长时间保持 0%，因为 `timm` 会依次下载约 GB 级的 DINO 和 SigLIP 权重。这通常不是卡死，应先检查 Hugging Face 缓存文件是否持续增长。

## 2. 资源和磁盘预留

建议至少准备：

- NVIDIA GPU 24 GB；显存更小时从 `--batch 1` 开始。
- 80 GB 以上空闲磁盘，覆盖数据压缩包、解压内容、模型缓存和检查点。
- 稳定的长连接；训练必须用 `nohup`、`tmux` 或 `screen`，不能依赖 Web Terminal 页面保持打开。
- 系统工具：`git`、`wget`、`unzip`、`nvidia-smi`，可选 `aria2c`。

检查服务器：

```bash
nvidia-smi
df -h /root
which python
python --version
```

## 3. Windows SSH 连接

连接需要 OpenSSH、`ncat` 和私钥。`ncat` 来自 Nmap，必须在 `PATH` 中，或在 `ProxyCommand` 中写完整路径。

```powershell
ssh -i "C:\Users\user\学习\UAV source code\Aeromamba\.codex_aicloud_id_rsa" `
  -o "ProxyCommand=ncat --proxy-type socks5 --proxy member.aicloud.szu.edu.cn:30027 %h %p" `
  root@a20203730917650432464717
```

### 私钥权限错误

Windows OpenSSH 若提示 `UNPROTECTED PRIVATE KEY FILE`，不要反复重试。复制一份专用私钥并移除继承权限：

```powershell
Copy-Item .\id_rsa .\.codex_aicloud_id_rsa
icacls .\.codex_aicloud_id_rsa /inheritance:r
icacls .\.codex_aicloud_id_rsa /grant:r "$($env:USERNAME):(R)"
```

不要把私钥提交到 Git，也不要上传到服务器。

## 4. 上传代码

新服务器建议先创建仓库目录，再用 `scp` 或 `rsync` 上传代码。排除本地数据、检查点、缓存和私钥。

```bash
mkdir -p /root/Aeromamba
```

上传后确认关键文件存在：

```bash
cd /root/Aeromamba
test -f run_stage1_train.py
test -f training/stage2_vlm.py
test -f scripts/remote_stage12_pipeline.sh
```

## 5. 环境配置

### 方案 A：复用服务器已有 PyTorch（本次采用）

如果 `/opt/conda/bin/python` 已能识别 GPU，优先复用，避免重装 PyTorch 后产生 CUDA ABI 冲突。

```bash
cd /root/Aeromamba
bash scripts/setup_remote_env_base.sh
```

### 方案 B：创建独立 Conda 环境

服务器没有可用环境时执行：

```bash
cd /root/Aeromamba
bash scripts/setup_remote_env.sh
conda activate mamba2
```

不要混用两个环境的 `python` 和 `pip`。始终检查：

```bash
which python
python -m pip --version
```

### 必做验证

```bash
python - <<'PY'
import torch
import timm
import transformers
import peft
import causal_conv1d
import mamba_ssm

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("timm:", timm.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
PY
```

常见兼容问题：

- PyTorch 2.1 系列配 NumPy 2.x 容易出现扩展 ABI 问题，保持 `numpy<2`。
- 某些 Mamba 构建依赖旧版 `pkg_resources`，保持 `setuptools<70`。
- 编译 `mamba-ssm`/`causal-conv1d` 时使用 `--no-build-isolation`。
- 驱动显示的 CUDA 版本可以高于 `torch.version.cuda`；关键是驱动能兼容 PyTorch wheel 自带的 CUDA runtime。
- 不要设置 `AEROMAMBA_OFFLINE=1` 做正式训练，否则单视觉编码器会使用随机权重；双编码器仍需要预训练权重缓存。

## 6. 数据目录标准

服务器最终必须是以下结构：

```text
/root/Aeromamba/data/
├── llava_pretrain/
│   ├── blip_laion_cc_sbu_558k.json
│   ├── 00453/004539375.jpg
│   └── ...                         # 共 558,128 张图像
├── stage2_mixed_data.json          # AeroMamba 生成的混合标注
├── llava_instruct_150k.json        # 可保留，不能替代上一文件
└── coco/
    └── train2017/
        ├── 000000000009.jpg
        └── ...                     # COCO train2017 共 118,287 张
```

本地可信基准：

```text
C:\Users\user\OneDrive - The University of Hong Kong - Connect\dataset\
├── llava_pretrain\llava_pretrain\   # Stage 1
└── aeromamba\                        # Stage 2
```

## 7. Stage 1 数据：LLaVA-Pretrain

官方来源：Hugging Face 数据集 `liuhaotian/LLaVA-Pretrain`。

```bash
mkdir -p /root/Aeromamba/data/llava_pretrain
cd /root/Aeromamba/data/llava_pretrain

wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
```

中国大陆网络访问官方端点不稳定时，可使用同一 Hugging Face 仓库对象的镜像端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
wget -c "$HF_ENDPOINT/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip"
```

本次确认 `images.zip` 完整文件大小为 `27,356,108,382` 字节。下载中断时文件可能已经存在，但不能仅凭“存在”判断完整。

```bash
stat -c '%s %n' images.zip
unzip -t images.zip
unzip -q images.zip -d /root/Aeromamba/data/llava_pretrain
```

`unzip` 报 `End-of-central-directory signature not found` 基本意味着压缩包未下载完整，使用 `wget -c` 或 `aria2c --continue=true` 续传，不要直接解压或重命名掩盖问题。

## 8. Stage 2 数据：AeroMamba + COCO

### 关键区别

`stage2_mixed_data.json` 是项目侧整理后的训练标注，不是官方 `llava_instruct_150k.json` 的同名副本。两者不能直接互换：

- `llava_instruct_150k.json` 可从官方 `liuhaotian/LLaVA-Instruct-150K` 下载。
- `stage2_mixed_data.json` 必须从本地可信备份 `dataset\aeromamba` 复制到服务器。
- 如果只有官方原始 JSON，需要重新执行当初生成 mixed JSON 的项目数据处理流程；当前流水线不会猜测或静默生成它。

COCO train2017 官方下载：

```bash
mkdir -p /root/Aeromamba/data/coco
cd /root/Aeromamba/data/coco
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -t train2017.zip
unzip -q train2017.zip
```

若平台已提供 `/autodl-pub/data/COCO2017/train2017.zip`，可以直接解压该公共只读副本，避免重复下载。

## 9. 数据一致性验证

### 文件数量

```bash
find /root/Aeromamba/data/llava_pretrain -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
find /root/Aeromamba/data/coco/train2017 -type f -name '*.jpg' | wc -l
```

预期分别为：

- Stage 1 图像：`558128`
- COCO train2017：`118287`

### 标注与路径全量检查

```bash
cd /root/Aeromamba
python - <<'PY'
import json
from pathlib import Path

root = Path('/root/Aeromamba/data')
pre_root = root / 'llava_pretrain'

with open(pre_root / 'blip_laion_cc_sbu_558k.json', encoding='utf-8') as f:
    pre = json.load(f)
with open(root / 'stage2_mixed_data.json', encoding='utf-8') as f:
    stage2 = json.load(f)

pre_missing = [x['image'] for x in pre if not (pre_root / x['image']).is_file()]
stage2_images = sorted({x['image'] for x in stage2})
stage2_missing = [x for x in stage2_images if not (root / x).is_file()]

print('stage1 samples:', len(pre))
print('stage1 missing:', len(pre_missing))
print('stage2 samples:', len(stage2))
print('stage2 unique images:', len(stage2_images))
print('stage2 missing:', len(stage2_missing))

assert len(pre) == 558_128
assert not pre_missing
assert len(stage2) == 231_036
assert len(stage2_images) == 82_309
assert not stage2_missing
PY
```

换服务器前后若要做字节级校验，对三个关键 JSON 生成 SHA-256 清单：

```bash
sha256sum \
  data/llava_pretrain/blip_laion_cc_sbu_558k.json \
  data/stage2_mixed_data.json \
  data/llava_instruct_150k.json | tee data/json.sha256

sha256sum -c data/json.sha256
```

## 10. 先做最小冒烟测试

不要第一次就直接跑几天。先限制每个 epoch 的训练步数，验证模型下载、数据加载、前向、反向和保存都正常。

```bash
cd /root/Aeromamba
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

/opt/conda/bin/python run_stage1_train.py \
  --data_root /root/Aeromamba/data/llava_pretrain \
  --json_name blip_laion_cc_sbu_558k.json \
  --mamba_type mamba-130m \
  --vision_type dinosiglip_so_384 \
  --use_token_pooling --pool_size 8 \
  --batch 1 --epochs 1 --max_steps 2 --max_val_steps 2 \
  --workers 0 --save_dir checkpoints/stage1_smoke
```

冒烟测试成功后再提高 batch。24 GB 显存也建议从 1 或 2 开始，观察 `nvidia-smi` 后逐步增加；出现 OOM 时优先减小 batch，其次缩短 `max_text_len`。

## 11. 正式启动 Stage 1 + Stage 2

流水线会按顺序完成数据检查、Stage 1、Stage 2：

```bash
cd /root/Aeromamba
chmod +x scripts/remote_stage12_pipeline.sh

nohup env \
  STAGE1_BATCH=16 STAGE1_EPOCHS=3 STAGE1_LR=1e-4 \
  STAGE2_BATCH=16 STAGE2_EPOCHS=5 STAGE2_LR=2e-4 \
  bash scripts/remote_stage12_pipeline.sh \
  > checkpoints/stage12_pipeline.log 2>&1 &

echo $! > checkpoints/stage12_pipeline.pid
```

如果冒烟测试没有证明 batch 16 可用，将两个 batch 改为 1 或 2。流水线固定使用：

- `mamba-130m`
- `dinosiglip_so_384`
- token pooling，`pool_size=8`
- Stage 1 的 `best.pth` 自动传给 Stage 2
- 若 `best.pth` 不存在，才回退到已有 `projector_only.pth`

## 12. 监控与判断是否卡住

```bash
tail -f /root/Aeromamba/checkpoints/stage12_pipeline.log
ps -ef | grep -E 'remote_stage12_pipeline|run_stage1_train|stage2_vlm' | grep -v grep
watch -n 2 nvidia-smi
```

模型下载期间检查缓存：

```bash
du -sh ~/.cache/huggingface/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m
du -sh ~/.cache/huggingface/hub/models--timm--vit_so400m_patch14_siglip_384.webli
find ~/.cache/huggingface/hub -name '*.incomplete' -printf '%s %p\n'
```

判断原则：

- `.incomplete` 文件持续增大且存在 HTTPS 连接：正在下载，继续等待。
- 权重下载结束后 GPU 才开始明显占用。
- 日志长时间不刷新但进程 CPU/网络仍变化：可能是 Python 输出缓冲；本流水线已设置 `PYTHONUNBUFFERED=1`。
- 进程消失：立即查看日志尾部，不要直接重新启动并覆盖上下文。

## 13. 检查点与恢复

正常输出：

```text
/root/Aeromamba/checkpoints/stage1/best.pth
/root/Aeromamba/checkpoints/stage2/best.pth
```

Stage 1 可通过 `--resume CHECKPOINT` 恢复。Stage 2 当前 CLI 没有单独的 `--resume` 参数；它的 `--stage1_ckpt` 只负责加载 Stage 1 projector，不等于恢复 Stage 2 optimizer/epoch。重新跑 Stage 2 前，应先保留旧目录：

```bash
mv checkpoints/stage2 checkpoints/stage2_backup_$(date +%Y%m%d_%H%M%S)
```

检查 checkpoint 是否可读：

```bash
python - <<'PY'
import torch
for path in ['checkpoints/stage1/best.pth', 'checkpoints/stage2/best.pth']:
    state = torch.load(path, map_location='cpu')
    print(path, state.keys() if isinstance(state, dict) else type(state))
PY
```

## 14. 最常见的坑

1. **私钥权限过宽**：Windows OpenSSH 会拒绝使用，按第 3 节收紧 ACL。
2. **`ncat` 不在 PATH**：SSH 的 ProxyCommand 会直接失败。
3. **压缩包只下载了一部分**：先 `stat` 和 `unzip -t`，再解压。
4. **Stage 1 多套一层目录**：`data_root` 必须直接包含 JSON，图像路径以该目录为基准。
5. **用官方 instruct JSON 冒充 mixed JSON**：样本数和路径语义不同，流水线会明确终止。
6. **只检查 COCO 总数，不检查 JSON 引用**：必须执行第 9 节的唯一图像路径全量检查。
7. **首次下载视觉权重时误判卡死**：DINO 与 SigLIP 是两个独立大文件，查看 `.incomplete` 增长。
8. **混用 Conda 环境**：安装时和训练时始终使用同一个 `python -m pip`。
9. **一上来 batch 16**：先用 batch 1 的两步冒烟测试，再逐步放大。
10. **关闭终端导致训练停止**：必须用 `nohup`/`tmux`，并记录 PID 和日志路径。
11. **PyTorch 2.1 使用 `expandable_segments` 崩溃**：该版本可能触发 CUDA allocator internal assert，本项目只保留 `max_split_size_mb:128`。

完成以上检查后，换环境时真正需要保存的是：代码版本、三个关键 JSON 的哈希、项目生成的 `stage2_mixed_data.json`、训练参数、日志和检查点。COCO 与 LLaVA-Pretrain 图像可以从官方渠道重新下载。
