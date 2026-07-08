# AeroMamba UAV-Flow Evaluation Framework

This document defines a standard evaluation plan for AeroMamba on UAV-Flow-Eval,
using the existing Windows UnrealCV simulator and a WSL-hosted AeroMamba inference
server.

## 1. Existing Evaluation Contract

`UAV-Flow-Eval/batch_run_act_all.py` drives the simulator and calls an HTTP
inference server:

- Endpoint: `POST http://127.0.0.1:<port>/predict`
- Request JSON:
  - `image`: base64 PNG, resized by the evaluator to `224x224`
  - `proprio`: current relative UAV state, `[x, y, z, yaw_deg]`
  - `instr`: natural-language navigation instruction
- Expected response JSON:
  - `status`: `"success"`
  - `action`: list of waypoints, each `[x, y, z, yaw_rad]`
  - `action_ori`: optional raw model output for debugging
  - `done`: optional boolean

The evaluator interprets `action` as local-frame waypoints relative to the
initial task pose. For each returned waypoint it:

1. Reads `relative_x, relative_y, relative_z, relative_yaw_rad`.
2. Converts yaw from radians to degrees.
3. Rotates local `x/y` by the initial yaw.
4. Adds the initial world position.
5. Moves the UnrealCV drone/object to that absolute pose.
6. Logs `{'state': [[x, y, z], [0, yaw_deg, 0]]}`.

`UAV-Flow-Eval/metric.py` evaluates generated JSON logs against `test_jsons`
using class-grouped nDTW.

## 2. OpenVLA Reference Behavior

`UAV-Openvla/vla-scripts/openvla_act.py` implements the reference server:

- Loads model once at startup.
- Exposes `/predict` and `/reset`.
- Builds prompt:
  `In: Current State: {x,y,z,yaw}, What action should the uav take to {instruction}?\nOut:`
- Calls `model.predict_action(...)`.
- Returns one action waypoint after rotating/offsetting the raw model output.

AeroMamba should keep the same server contract so that `batch_run_act_all.py`
does not need simulator-side changes.

## 3. AeroMamba Adapter Design

Create a server script such as `scripts/aeromamba_act_server.py`.

### Runtime Split

- Windows:
  - Runs `UAV-Flow-Eval/batch_run_act_all.py`.
  - Runs UnrealCV packaged environment.
  - Sends HTTP requests to `127.0.0.1:<port>`.
- WSL `Ubuntu-20.04`:
  - Runs AeroMamba inference server.
  - Recommended env:
    `/home/khr/miniconda3/envs/aeromamba_wsl3`
  - This env has CUDA Torch, `mamba_ssm`, and `causal_conv1d`.

### Model Loading

Use the exact training architecture:

- `vision_type="siglip2_base_384"`
- `mamba_type="mamba-2-370m"`
- `token_resampler="perceiver"`
- `num_visual_queries=32`
- `resampler_layers=2`
- `resampler_heads=8`
- `action_head_type="dynamics"`
- `chunk_size=5`
- `lora_r=16`
- `lora_alpha=32`

Load:

- Stage 3 checkpoint:
  `checkpoints/remote_backup/stage3_b24_w24/best.pth`
- Or WSL path mapped from Windows:
  `/mnt/c/Users/user/.../Aeromamba/checkpoints/remote_backup/stage3_b24_w24/best.pth`

The checkpoint should be loaded with `strict=False` only if the checkpoint stores
trainer metadata or has harmless missing optimizer keys. Model weights themselves
should match the architecture.

### Input Preprocessing

For each `/predict` request:

1. Decode `image` with PIL.
2. Convert to RGB.
3. Use `build_vision_transform("siglip2_base_384")` or the model's training
   transform.
4. Build the same prompt family used by OpenVLA:
   `In: Current State: {state}, What action should the uav take to {instruction}?\nOut:`
5. Tokenize with the same tokenizer path used during training. If the Mamba2
   tokenizer is unavailable and training used GPT-2 fallback, use the same GPT-2
   fallback to avoid token mismatch.
6. Convert `proprio` to tensor `[1, 4]`, dtype `float32`.

### Model Call

Call:

```python
with torch.inference_mode():
    out = model.predict_step(pixel_values, input_ids, proprio=proprio)
    action_chunk = out["action"][0]  # [K, 4]
```

Then:

- Detach to CPU.
- Convert to `float32` NumPy/list.
- Return all `K` waypoints as `action`.
- Also return `action_ori` for debugging.

Unlike OpenVLA, do not double-apply current-position transforms unless an
empirical sanity check proves the checkpoint predicts one-step deltas rather than
relative task-frame poses. The current UAV-Flow training target appears to be
relative trajectory waypoints, matching the evaluator's expected local-frame log.

## 4. Standard Evaluation Levels

### Level 0: Offline Sanity Check

Goal: verify weights, shapes, and no OOM before simulator.

Inputs:

- One UAV-Flow image.
- One instruction.
- One proprio `[0, 0, 0, 0]`.

Checks:

- Model loads on WSL GPU.
- `predict_step` returns `[1, 5, 4]`.
- Values are finite.
- Inference latency is recorded.
- If CUDA OOM occurs, terminate immediately and report.

### Level 1: HTTP Contract Test

Goal: ensure Windows evaluator can talk to WSL server.

Procedure:

1. Start AeroMamba server in WSL on port `5007`.
2. From Windows, send one synthetic `/predict` request.
3. Validate JSON schema:
   - `status == "success"`
   - `action` is `K x 4`
   - all values finite
4. Call `/reset` and confirm success.

### Level 2: Small Simulator Smoke Test

Goal: confirm closed-loop simulator execution.

Command pattern:

```powershell
cd "C:\Users\user\学习\UAV source code\UAV-Flow-Eval"
python batch_run_act_all.py `
  --server_port 5007 `
  --json_folder .\test_jsons_smoke `
  --images_dir .\results\UnrealTrack-DowntownWest-ContinuousColor-v0\aeromamba_smoke `
  --max_steps 20
```

Use 3-5 task JSONs covering:

- Move
- Turn/Rotate
- Object-conditioned instruction
- Long forward navigation

Checks:

- No HTTP timeout.
- No simulator crash.
- Logs and 2D/3D plots are produced.
- Trajectory does not instantly collapse to near-zero movement unless instruction
  requires stop.

### Level 3: Full UAV-Flow-Eval Run

Goal: produce comparable benchmark results.

Command pattern:

```powershell
cd "C:\Users\user\学习\UAV source code\UAV-Flow-Eval"
python batch_run_act_all.py `
  --server_port 5007 `
  --images_dir .\results\UnrealTrack-DowntownWest-ContinuousColor-v0\aeromamba `
  --max_steps 100 `
  --instruction_type instruction
```

Then modify or wrap `metric.py` so `model_list = ["aeromamba"]`, or add a CLI
argument for `--model aeromamba`, and run:

```powershell
python metric.py
```

Primary metric:

- Overall Mean nDTW

Secondary metrics to add:

- Per-class nDTW
- Success-like threshold counts, e.g. nDTW >= `0.5`, `0.7`, `0.9`
- Mean inference latency
- HTTP failure count
- OOM/crash count
- Mean trajectory length and early-stop count

## 5. Safety Rules

- Start with `max_steps=20`.
- Keep WSL server batch size at `1`.
- Use `torch.inference_mode()`.
- Prefer `float32` for first run. Try autocast only after correctness is proven.
- If OOM occurs:
  - Stop the server immediately.
  - Report peak memory and failing step.
  - Do not retry with larger settings.
- Keep simulator results separate:
  - `results/.../aeromamba_smoke`
  - `results/.../aeromamba`

## 6. Recommended Implementation Order

1. Add `scripts/aeromamba_act_server.py`.
2. Add `scripts/test_aeromamba_http_client.py` for one request.
3. Add a small `test_jsons_smoke` subset or a CLI option to limit task count.
4. Patch `metric.py` to accept `--model` and `--result_dir`.
5. Run Level 0 and Level 1.
6. Run Level 2 smoke test.
7. Run Level 3 full evaluation.

