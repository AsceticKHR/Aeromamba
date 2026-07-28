# Repository Guidelines

## Project Structure & Module Organization

This repository implements the AeroMamba UAV vision-language-action training stack.
Architecture specifics and the v1/v2 split live in `CLAUDE.md`; read it first.

- `model/`: vision encoders (`vision.py`), the active `AeroV2` model (`aerov2.py`),
  projector, proprio encoder, and action heads. `uav_mamba_vla.py` is legacy v1.
- `training/`: `v2_stage*.py` are the active entry points. `stage1_align.py`,
  `stage2_vlm.py`, `stage3_action.py`, and `trainer.py` belong to legacy v1.
- `data/`: dataset loaders plus UAV-Flow preparation, stats, and validation scripts.
- `inference/`: `server_v2.py` is active; `server.py` is the legacy v1 server.
- `scripts/`: gates, evaluation, probes, benchmarks, and remote helpers.
- `configs/`: YAML examples. These are documentation only — nothing parses them.
- `docs/`: two design docs only, see `CLAUDE.md`. `reports/`: current evidence;
  `reports/_archive/` is superseded v1-era material and should not be cited.

Do not commit datasets, checkpoints, private keys, logs, cache folders, or local
evaluation outputs.

## Build, Test, and Development Commands

Use a Python environment with dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
python -m py_compile data/dataset.py model/aerov2.py training/v2_stage3_action.py
AEROMAMBA_OFFLINE=1 python scripts/test_aerov2_wiring.py
python scripts/test_v5_head_contract.py
python scripts/qc_s3_units.py
python data/validate_uavflow_stage3.py --data_root /path/to/uav-flow --chunk_size 8
```

- `py_compile` catches syntax/import-shape mistakes quickly.
- `test_aerov2_wiring.py` checks the v2 token/shape/grad contract without weights.
- `test_v5_head_contract.py` covers the readout and output-head contracts,
  including the regression test that text keys cannot drown out vision patches.
- `qc_s3_units.py` catches the metre/centimetre mismatch between real and sim.
- `validate_uavflow_stage3.py` checks prepared UAV-Flow folders before training.

## Coding Style & Naming Conventions

- Use Python 3 style with 4-space indentation and type hints where practical.
- Prefer explicit names such as `vis_tokens`, `gt_action`, `pos_err_m`, `vis_share`.
- Keep training changes stage-specific inside the relevant `training/v2_stage*.py`.
- Avoid hidden fallbacks that silently change training semantics, especially
  tokenizer, instruction, position-unit, and dataset paths.
- When a root cause is fixed, delete the patch it replaces rather than stacking
  both. `CLAUDE.md` lists what has already been removed and why.

## Testing Guidelines

- Add lightweight tests under `scripts/test_*.py` for model contracts and smoke checks.
- For architecture changes, test shape compatibility, finite loss, checkpoint
  loading, and token ordering.
- No full training run starts before its stage gates pass. Stage 3 gates are
  G1/G1v/G2/G3/G4/G5 in `--mode smoke`.
- Report offline action error only alongside the trivial-baseline table from
  `scripts/trivial_action_baselines.py`; the raw number is not interpretable alone.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, e.g. `Add full-stage UAV-Flow
training pipeline` or `Clean obsolete reports`.

- Keep commits focused and mention the affected subsystem.
- PRs should describe the training/inference impact, list validation commands,
  and note checkpoint compatibility.
- Include logs or metrics for training changes; include API examples for
  inference changes.

## Security & Configuration Tips

- Never commit SSH keys, Hugging Face tokens, Bita credentials, checkpoints, or
  dataset archives.
- Keep machine-specific paths in scripts configurable through environment variables.
- Use `.gitignore` for generated artifacts such as `checkpoints/`, caches, and
  local reports.
