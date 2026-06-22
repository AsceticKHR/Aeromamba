# AeroMamba

AeroMamba is a staged Mamba-based vision-language-action pipeline for UAV training and evaluation. This repository keeps the **core code only**:

- model definitions
- training stages
- inference helpers
- reproducible environment and data guides

Large artifacts such as checkpoints, datasets, archives, and local keys are excluded from version control.

## Repository Layout

```text
Aeromamba/
├── configs/                  # experiment and model configs
├── data/                     # dataset loaders and local data notes
├── docs/                     # reproducible environment + dataset guide
├── inference/                # evaluation / serving code
├── model/                    # core model modules
├── scripts/                  # training launchers and utility scripts
├── training/                 # stage1 / stage2 / stage3 training logic
├── run_stage1_train.py       # stage 1 entry point
├── requirements.txt          # Python dependencies
└── Dockerfile                # optional container entry
```

## What Is Tracked

Keep in Git:

- source code
- configs
- documentation
- lightweight utility scripts
- dependency manifests

Do not commit:

- `checkpoints/`
- `data/`
- `*.pth`, `*.ckpt`, `*.pt`
- `*.zip`, `*.tar.gz`, `*.part`
- SSH keys and local credentials

The ignore rules are in [`.gitignore`](./.gitignore).

## Environment Setup

The full setup and data-recovery workflow is documented in
[`docs/STAGE1_STAGE2_ENV_AND_DATA_GUIDE.md`](./docs/STAGE1_STAGE2_ENV_AND_DATA_GUIDE.md).

Typical flow:

1. Verify Python, CUDA, and GPU availability.
2. Install dependencies from `requirements.txt`.
3. Prepare the dataset directory structure.
4. Restore or download Stage 1 and Stage 2 datasets.
5. Launch Stage 1 and Stage 2 training.
6. Validate checkpoints and run inference smoke tests.

## Dataset Paths

The project expects a stable data root on the server:

```text
/root/Aeromamba/data/
├── llava_pretrain/
├── stage2_mixed_data.json
├── llava_instruct_150k.json
└── coco/
```

Recommended Windows-side mirror:

```text
C:\Users\user\OneDrive - The University of Hong Kong - Connect\dataset\
├── llava_pretrain\
└── aeromamba\
```

If the current environment changes, copy the exact folder layout rather than rewriting dataset references inside the code.

## Training Stages

### Stage 1

Stage 1 aligns the visual projector to the language model space using the LLaVA-Pretrain data.

Recommended launcher:

```bash
python run_stage1_train.py
```

If you want the staged scripts directly:

```bash
python training/stage1_align.py --help
python scripts/train_mamba.py --stage 1
```

### Stage 2

Stage 2 fine-tunes the VLM stack with AeroMamba mixed data and COCO image paths.

Recommended launcher:

```bash
python training/stage2_vlm.py --help
python scripts/train_mamba.py --stage 2
```

### Stage 3

Stage 3 trains the action head and proprioception branch for downstream UAV control.

```bash
python training/stage3_action.py --help
python scripts/train_mamba.py --stage 3
```

## Reproduce From Scratch

1. Clone the repository and create a clean Python environment.
2. Install dependencies from `requirements.txt`.
3. Put the datasets in the exact folder structure documented above.
4. Confirm the Stage 1 LLaVA-Pretrain JSON and image counts.
5. Confirm the Stage 2 mixed JSON and COCO image paths.
6. Start Stage 1 training and wait for a valid checkpoint.
7. Start Stage 2 training from the Stage 1 checkpoint.
8. Run the inference smoke test before any long experiment.

## Useful Checks

```bash
python --version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import json; print('ok')"
```

For dataset integrity, use the checks described in the docs guide instead of relying on file presence alone.

## Notes

- Keep training runs on `nohup`, `tmux`, or `screen` if the session is remote.
- Do not commit private keys, backup archives, or model weights.
- If you move to a new machine, treat the docs guide as the source of truth for data restore and launch order.
