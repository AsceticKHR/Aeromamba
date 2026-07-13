# Repository Guidelines

## Project Structure & Module Organization

This repository implements the AeroMamba UAV vision-language-action training stack.

- `model/`: core model modules, including vision encoders, Mamba VLA wrapper, projector, resampler, proprio encoder, and action heads.
- `training/`: Stage 1/2/3 training entry points, trainer utilities, and architecture presets.
- `data/`: dataset loaders plus UAV-Flow preparation and validation scripts.
- `inference/`: HTTP inference server and evaluation-facing runtime code.
- `scripts/`: smoke tests, launch scripts, benchmarks, and remote training helpers.
- `configs/`: YAML model and training configuration examples.
- `docs/` and `reports/`: design notes and analysis artifacts. Keep large generated reports out of commits unless they are intentionally curated.

Do not commit datasets, checkpoints, private keys, logs, cache folders, or local evaluation outputs.

## Build, Test, and Development Commands

Use a Python environment with dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
python -m py_compile data/dataset.py model/uav_mamba_vla.py training/stage3_action.py
python scripts/test_aeromamba_opt_contract.py
python data/validate_uavflow_stage3.py --data_root /path/to/uav-flow --chunk_size 8
```

- `py_compile` catches syntax/import-shape mistakes quickly.
- `test_aeromamba_opt_contract.py` checks the AeroMamba-Opt token/state/loss contract.
- `validate_uavflow_stage3.py` checks prepared UAV-Flow folders before full training.

## Coding Style & Naming Conventions

- Use Python 3 style with 4-space indentation and type hints where practical.
- Prefer explicit names such as `state8`, `delta_state8`, `vis_tokens`, and `gt_action`.
- Keep training changes stage-specific: Stage 1/2 CLM logic in `training/stage*_*.py`, shared behavior in `training/trainer.py`.
- Avoid hidden fallbacks that silently change training semantics, especially tokenizer, instruction, and dataset paths.

## Testing Guidelines

- Add lightweight tests under `scripts/test_*.py` for model contracts and smoke checks.
- For architecture changes, test shape compatibility, finite loss, checkpoint loading, and token ordering.
- Before remote training, run data validation and a small-step smoke run before full Stage 1→3 execution.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, e.g. `Add full-stage UAV-Flow training pipeline` or `Clean obsolete reports`.

- Keep commits focused and mention the affected subsystem.
- PRs should describe the training/inference impact, list validation commands, and note checkpoint compatibility.
- Include logs or metrics for training changes; include API examples for inference changes.

## Security & Configuration Tips

- Never commit SSH keys, Hugging Face tokens, Bita credentials, checkpoints, or dataset archives.
- Keep machine-specific paths in scripts configurable through environment variables.
- Use `.gitignore` for generated artifacts such as `checkpoints/`, caches, and local reports.
