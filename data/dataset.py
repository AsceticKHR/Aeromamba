"""
UAV-Flow Dataset Loader for AeroMamba-VLA.

Handles:
  - UAV-Flow JSON trajectory files (real-world and simulation splits)
  - Image loading and transforms
  - Action chunk extraction (K consecutive steps per sample)
  - Proprioceptive state parsing
  - Instruction tokenisation

UAV-Flow trajectory JSON format (one file per episode):
    [
        {
            "image_path":  "path/to/frame.jpg",
            "instruction": "fly forward to the red building",
            "state":  [[rel_x, rel_y, rel_z], [roll, yaw, pitch]],
            "action": [abs_x, abs_y, abs_z, abs_yaw]
        },
        ...
    ]

If trajectories are chunked (UAV-Flow style, single-instruction per traj):
    The loader uses a sliding window of size K over the steps for i.i.d. samples.
"""

from __future__ import annotations

import bisect
import glob
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def yaw_to_sincos(yaw_rad: float) -> Tuple[float, float]:
    """Convert yaw angle (radians) to (sin, cos) pair."""
    return math.sin(yaw_rad), math.cos(yaw_rad)


def normalize_action(
    action: List[float],
    pos_scale: float = 100.0,  # cm → normalised meters
) -> np.ndarray:
    """
    Normalise raw UAV action [dx, dy, dz, dyaw_deg] to model training range.
    Positions divided by pos_scale, yaw converted to radians.
    """
    dx, dy, dz = action[0] / pos_scale, action[1] / pos_scale, action[2] / pos_scale
    dyaw_rad = math.radians(action[3]) if len(action) > 3 else 0.0
    return np.array([dx, dy, dz, dyaw_rad], dtype=np.float32)


def parse_proprio(state: List[List[float]]) -> np.ndarray:
    """
    Extract [rel_x, rel_y, rel_z, rel_yaw] from UAV-Flow state format:
        state[0] = [rel_x, rel_y, rel_z]
        state[1] = [roll, yaw, pitch]
    """
    rel_xyz = state[0][:3]                            # [3]
    rel_yaw = math.radians(state[1][1])               # yaw (index 1) → radians
    return np.array(rel_xyz + [rel_yaw], dtype=np.float32)  # [4]


def parse_state8(state: List[List[float]]) -> np.ndarray:
    """
    Extract AeroMamba-Opt 8D state:
      [x, y, z, yaw_rad, vx, vy, vz, yaw_rate_rad]
    """
    pose = parse_proprio(state)
    velocity = [0.0, 0.0, 0.0, 0.0]
    if len(state) > 2 and isinstance(state[2], (list, tuple)):
        for idx, value in enumerate(state[2][:4]):
            try:
                velocity[idx] = float(value)
            except (TypeError, ValueError):
                velocity[idx] = 0.0
    return np.concatenate([pose, np.asarray(velocity, dtype=np.float32)]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class UAVFlowDataset(Dataset):
    """
    PyTorch Dataset for UAV-Flow trajectory data.

    Each sample is a dict:
        pixel_values:  [3, H, W]           float32  (model-ready, normalised)
        input_ids:     [L]                 int64    (tokenised instruction)
        proprio:       [4]                 float32  (Δx, Δy, Δz, Δyaw)
        gt_action:     [K, 4]             float32  (chunk of K waypoints)

    Args:
        data_root:      Root directory with trajectory JSON files.
        tokenizer:      HuggingFace tokenizer.
        transform:      torchvision transform pipeline (from VisionEncoder).
        chunk_size:     K future waypoints per sample.
        max_text_len:   Max instruction token length.
        split:          'train' | 'val' | 'test'
        pos_scale:      Divisor for position normalisation (cm → ~meters).
        aug_flip:       Random horizontal flip augmentation (train only).
        json_extension: File extension for trajectory files.
    """

    def __init__(
        self,
        data_root:     str,
        tokenizer:     AutoTokenizer,
        transform,
        chunk_size:    int   = 5,
        max_text_len:  int   = 64,
        split:         str   = "train",
        pos_scale:     float = 100.0,
        aug_flip:      bool  = True,
        json_extension:str   = ".json",
    ):
        super().__init__()
        self.data_root    = Path(data_root)
        self.tokenizer    = tokenizer
        self.transform    = transform
        self.chunk_size   = chunk_size
        self.max_text_len = max_text_len
        self.pos_scale    = pos_scale
        self.aug_flip     = aug_flip and (split == "train")
        self.split        = split

        # Collect trajectory files. Supports both the legacy AeroMamba list
        # format and the official UAV-Flow folder format:
        #   <trajectory_id>/000000.jpg
        #   <trajectory_id>/log.json
        if json_extension == ".json":
            official_logs = sorted(self.data_root.rglob("log.json"))
            legacy_files = sorted(
                path for path in self.data_root.rglob("*.json") if path.name != "log.json"
            )
            self.traj_files = official_logs + legacy_files
        else:
            self.traj_files = sorted(self.data_root.rglob(f"*{json_extension}"))
        if not self.traj_files:
            raise FileNotFoundError(
                f"No {json_extension} files found under {data_root}. "
                "Check data_root path."
            )

        # Build flat index: (traj_file_idx, step_idx) for each valid chunk
        self.index: List[Tuple[int, int]] = []
        self.trajectories: List[List[Dict[str, Any]]] = []

        for traj_idx, traj_path in enumerate(self.traj_files):
            traj = self._load_trajectory(traj_path)
            if not isinstance(traj, list) or len(traj) < chunk_size:
                continue
            self.trajectories.append(traj)
            # Sliding window: every step where a full K-chunk is available
            for step_idx in range(len(traj) - chunk_size + 1):
                self.index.append((len(self.trajectories) - 1, step_idx))

        if not self.index:
            raise ValueError(
                f"No valid trajectory chunks found (chunk_size={chunk_size}). "
                "Ensure trajectories have >= chunk_size steps."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, step_idx = self.index[idx]
        traj = self.trajectories[traj_idx]

        # ── Anchor step (what the model sees NOW) ────────────────────────────
        anchor = traj[step_idx]

        # ── Image ────────────────────────────────────────────────────────────
        img = self._load_image(anchor, traj_idx, step_idx)
        do_flip = self.aug_flip and random.random() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # transform may return a plain tensor (single encoder)
        # or a dict {"dino": ..., "siglip": ...} (DinoSigLIPTransform)
        pixel_values = self.transform(img)

        # ── Instruction ───────────────────────────────────────────────────────
        instruction = anchor.get("instruction", "")
        input_ids   = self._tokenize(instruction)

        # ── Proprioception ────────────────────────────────────────────────────
        state  = anchor.get("state", [[0, 0, 0], [0, 0, 0]])
        proprio = torch.tensor(parse_proprio(state), dtype=torch.float32)
        state8 = torch.tensor(parse_state8(state), dtype=torch.float32)
        if step_idx > 0:
            prev_state = traj[step_idx - 1].get("state", [[0, 0, 0], [0, 0, 0]])
            prev_state8 = torch.tensor(parse_state8(prev_state), dtype=torch.float32)
            delta_state8 = state8 - prev_state8
        else:
            delta_state8 = torch.zeros_like(state8)

        # ── Ground-truth action chunk (K steps) ───────────────────────────────
        gt_action = self._extract_chunk(traj, step_idx)  # [K, 4]
        if do_flip:
            proprio = self._flip_proprio(proprio)
            state8 = self._flip_state8(state8)
            delta_state8 = self._flip_state8(delta_state8)
            gt_action = self._flip_action(gt_action)

        return {
            "pixel_values": pixel_values,   # Tensor [3,H,W] or dict of Tensors
            "input_ids":    input_ids,
            "proprio":      proprio,
            "state8":       state8,
            "delta_state8": delta_state8,
            "gt_action":    gt_action,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_trajectory(self, traj_path: Path) -> List[Dict[str, Any]]:
        with open(traj_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "raw_logs" in payload:
            return self._convert_official_log(payload, traj_path)
        return []

    def _convert_official_log(
        self,
        payload: Dict[str, Any],
        log_path: Path,
    ) -> List[Dict[str, Any]]:
        preprocessed_logs = payload.get("preprocessed_logs") or []
        raw_logs = payload.get("raw_logs") or []
        length = int(payload.get("length") or len(preprocessed_logs) or len(raw_logs))
        if length <= 0:
            return []

        instruction = (
            payload.get("instruction_unified")
            or payload.get("instruction")
        )
        if not instruction:
            return []
        raw_origin = self._raw_xyz(raw_logs[0] if raw_logs else [])
        traj_dir = log_path.parent
        traj = []
        for idx in range(length):
            image_path = traj_dir / f"{idx:06d}.jpg"
            if not image_path.exists():
                continue
            pose = self._pose_from_preprocessed_log(
                preprocessed_logs[idx] if idx < len(preprocessed_logs) else None
            )
            if pose is None:
                pose = self._pose_from_raw_log(
                    raw_logs[idx] if idx < len(raw_logs) else [],
                    raw_origin=raw_origin,
                )
                pos_unit_scale = 1.0
            else:
                pos_unit_scale = self.pos_scale
            state_pose = [pose["x_m"], pose["y_m"], pose["z_m"]]
            state_att = [pose["roll_deg"], pose["yaw_deg"], pose["pitch_deg"]]
            if traj:
                prev_state = traj[-1]["state"]
                prev_pose = prev_state[0]
                prev_yaw = math.radians(prev_state[1][1])
                yaw = math.radians(pose["yaw_deg"])
                dyaw = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                velocity = [
                    state_pose[0] - prev_pose[0],
                    state_pose[1] - prev_pose[1],
                    state_pose[2] - prev_pose[2],
                    dyaw,
                ]
            else:
                velocity = [0.0, 0.0, 0.0, 0.0]
            traj.append(
                {
                    "image_path": str(image_path.relative_to(self.data_root)).replace("\\", "/"),
                    "instruction": instruction,
                    "state": [state_pose, state_att, velocity],
                    "pose_cm": [
                        pose["x_cm"] * pos_unit_scale,
                        pose["y_cm"] * pos_unit_scale,
                        pose["z_cm"] * pos_unit_scale,
                        pose["yaw_deg"],
                    ],
                }
            )
        return traj

    @staticmethod
    def _raw_xyz(raw: Any) -> tuple[float, float, float]:
        values = []
        if isinstance(raw, (list, tuple)):
            for value in raw[:3]:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(0.0)
        while len(values) < 3:
            values.append(0.0)
        return values[0], values[1], values[2]

    @staticmethod
    def _pose_from_raw_log(
        raw: Any,
        raw_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Dict[str, float]:
        values = []
        if isinstance(raw, (list, tuple)):
            for value in raw:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(0.0)
        while len(values) < 6:
            values.append(0.0)
        rel_x = values[0] - raw_origin[0]
        rel_y = values[1] - raw_origin[1]
        rel_z = values[2] - raw_origin[2]
        return {
            "x_m": rel_x,
            "y_m": rel_y,
            "z_m": rel_z,
            "x_cm": rel_x,
            "y_cm": rel_y,
            "z_cm": rel_z,
            "roll_deg": values[3],
            "yaw_deg": values[4],
            "pitch_deg": values[5],
        }

    @staticmethod
    def _pose_from_preprocessed_log(preprocessed: Any) -> Optional[Dict[str, float]]:
        if not isinstance(preprocessed, (list, tuple)) or len(preprocessed) < 6:
            return None
        try:
            values = [float(value) for value in preprocessed[:6]]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return {
            "x_m": values[0],
            "y_m": values[1],
            "z_m": values[2],
            "x_cm": values[0],
            "y_cm": values[1],
            "z_cm": values[2],
            "roll_deg": values[3],
            "yaw_deg": values[4],
            "pitch_deg": values[5],
        }

    def _load_image(
        self,
        step: Dict,
        traj_idx: int,
        step_idx: int,
    ) -> Image.Image:
        """
        Load PIL image from step dict.
        Supports 'image_path' (absolute or relative to data_root) key.
        Falls back to a dummy white image if path missing (for dummy datasets).
        """
        img_path = step.get("image_path", None)
        if img_path is not None:
            full_path = Path(img_path)
            if not full_path.is_absolute():
                full_path = self.data_root / img_path
            if full_path.exists():
                return Image.open(full_path).convert("RGB")
        # Dummy fallback
        return Image.new("RGB", (384, 384), color=(128, 128, 128))

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)    # [L]

    def _extract_chunk(
        self,
        traj: List[Dict],
        start: int,
    ) -> torch.Tensor:
        """Extract K consecutive normalised actions as ground truth chunk."""
        if "pose_cm" in traj[start]:
            return self._extract_body_frame_chunk(traj, start)

        chunk = []
        for k in range(self.chunk_size):
            step = traj[start + k]
            raw_action = step.get(
                "action",
                step.get("state", [[0, 0, 0], [0, 0, 0]])[0] + [0]
            )
            # Handle [x,y,z] (no yaw) vs [x,y,z,yaw]
            if len(raw_action) == 3:
                raw_action = list(raw_action) + [0.0]
            norm_action = normalize_action(raw_action, self.pos_scale)
            chunk.append(norm_action)
        return torch.tensor(np.stack(chunk), dtype=torch.float32)  # [K, 4]

    def _extract_body_frame_chunk(
        self,
        traj: List[Dict],
        start: int,
    ) -> torch.Tensor:
        anchor_pose = traj[start].get("pose_cm", [0.0, 0.0, 0.0, 0.0])
        x0, y0, z0, yaw0 = [float(value) for value in anchor_pose[:4]]
        yaw0_rad = math.radians(yaw0)
        cos_yaw = math.cos(yaw0_rad)
        sin_yaw = math.sin(yaw0_rad)

        chunk = []
        for k in range(self.chunk_size):
            step = traj[start + k]
            pose = step.get("pose_cm", anchor_pose)
            x, y, z, yaw = [float(value) for value in pose[:4]]
            dx_world = x - x0
            dy_world = y - y0
            forward = cos_yaw * dx_world + sin_yaw * dy_world
            right = -sin_yaw * dx_world + cos_yaw * dy_world
            up = z - z0
            dyaw = (yaw - yaw0 + 180.0) % 360.0 - 180.0
            chunk.append(normalize_action([forward, right, up, dyaw], self.pos_scale))
        return torch.tensor(np.stack(chunk), dtype=torch.float32)

    @staticmethod
    def _flip_proprio(proprio: torch.Tensor) -> torch.Tensor:
        flipped = proprio.clone()
        if flipped.numel() >= 2:
            flipped[1] = -flipped[1]
        if flipped.numel() >= 4:
            flipped[3] = -flipped[3]
        return flipped

    @staticmethod
    def _flip_state8(state: torch.Tensor) -> torch.Tensor:
        flipped = state.clone()
        for idx in (1, 3, 5, 7):
            if flipped.numel() > idx:
                flipped[idx] = -flipped[idx]
        return flipped

    @staticmethod
    def _flip_action(action: torch.Tensor) -> torch.Tensor:
        flipped = action.clone()
        if flipped.ndim >= 2 and flipped.size(-1) >= 2:
            flipped[..., 1] = -flipped[..., 1]
        if flipped.ndim >= 2 and flipped.size(-1) >= 4:
            flipped[..., 3] = -flipped[..., 3]
        return flipped


class UAVFlowHFDataset(Dataset):
    """
    HuggingFace loader for wangxiangyu0814/UAV-Flow.

    The dataset rows contain:
        id, frame_idx, image, log

    `log` is a JSON string with `raw_logs`, a future trajectory. Each raw log
    entry is interpreted as [x, y, z, roll, yaw, pitch, timestamp]. Stage-3
    actions are built as relative future waypoints from the first raw-log pose:
        [dx, dy, dz, dyaw_deg]
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        transform,
        split: str = "train",
        data_files: Optional[str] = None,
        cache_dir: Optional[str] = None,
        chunk_size: int = 5,
        max_text_len: int = 64,
        instruction: str = "Navigate the UAV along the planned trajectory.",
        pos_scale: float = 100.0,
        aug_flip: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.chunk_size = chunk_size
        self.max_text_len = max_text_len
        self.instruction = instruction
        self.pos_scale = pos_scale
        self.aug_flip = aug_flip
        self.sequential_loading_preferred = False
        self._parquet_files: List[Path] = []
        self._logical_row_groups: List[Tuple[int, int, int]] = []
        self._logical_cumsum: List[int] = []
        self._row_group_cache_key: Optional[Tuple[int, int]] = None
        self._row_group_cache: Optional[Dict[str, Any]] = None

        if dataset_name == "parquet" and data_files:
            self.ds = None
            self.sequential_loading_preferred = True
            self._init_local_parquet(data_files)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "UAVFlowHFDataset requires the `datasets` package. "
                    "Install it with `pip install datasets pyarrow`."
                ) from exc

            load_kwargs = {"split": split, "cache_dir": cache_dir}
            if data_files:
                load_kwargs["data_files"] = data_files
            self.ds = load_dataset(dataset_name, **load_kwargs)
            if len(self.ds) == 0:
                raise ValueError(f"HuggingFace dataset {dataset_name!r} split {split!r} is empty.")

    def __len__(self) -> int:
        if self._logical_cumsum:
            return self._logical_cumsum[-1]
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._get_local_parquet_row(int(idx)) if self._logical_cumsum else self.ds[int(idx)]

        img = self._coerce_image(row["image"])
        do_flip = self.aug_flip and random.random() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.transform(img)

        log_payload = self._parse_log_payload(row.get("log", ""))
        # Use LLM-diversified instruction first (matches official UAV-Flow), fallback to unified
        instruction = (
            log_payload.get("instruction")
            or log_payload.get("instruction_unified")
        )
        if not instruction:
            raise ValueError("UAV-Flow sample is missing instruction; refusing fixed fallback.")
        input_ids = self._tokenize(instruction)

        # preprocessed_logs/raw_logs use local/world Cartesian positions and
        # [roll_deg, yaw_deg, pitch_deg] orientation ordering.
        preprocessed_logs = self._parse_preprocessed_logs(log_payload)
        raw_logs, motion_pos_scale = self._parse_motion_logs(log_payload)

        # Proprioception: use actual state from preprocessed_logs (not zeros)
        anchor_pp = preprocessed_logs[0]
        proprio = torch.tensor(
            [
                float(anchor_pp[0]),
                float(anchor_pp[1]),
                float(anchor_pp[2]),
                math.radians(float(anchor_pp[4])) if len(anchor_pp) > 4 else 0.0,
            ],
            dtype=torch.float32,
        )
        state8 = torch.tensor(
            self._state8_from_preprocessed(preprocessed_logs, 0),
            dtype=torch.float32,
        )
        delta_state8 = torch.tensor(
            self._state8_delta_from_preprocessed(preprocessed_logs, 0),
            dtype=torch.float32,
        )

        gt_action = self._extract_relative_actions(raw_logs, motion_pos_scale=motion_pos_scale)
        if do_flip:
            proprio = UAVFlowDataset._flip_proprio(proprio)
            state8 = UAVFlowDataset._flip_state8(state8)
            delta_state8 = UAVFlowDataset._flip_state8(delta_state8)
            gt_action = UAVFlowDataset._flip_action(gt_action)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "proprio": proprio,
            "state8": state8,
            "delta_state8": delta_state8,
            "gt_action": gt_action,
        }

    def _init_local_parquet(self, data_files: Any) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Local parquet training requires `pyarrow`. "
                "Install it with `pip install pyarrow`."
            ) from exc

        patterns: List[str] = []
        if isinstance(data_files, (str, os.PathLike)):
            patterns = [str(data_files)]
        elif isinstance(data_files, dict):
            for value in data_files.values():
                if isinstance(value, (list, tuple)):
                    patterns.extend(str(item) for item in value)
                else:
                    patterns.append(str(value))
        else:
            patterns = [str(item) for item in data_files]

        files: List[Path] = []
        for pattern in patterns:
            matches = glob.glob(pattern)
            files.extend(Path(match) for match in (matches or [pattern]))
        self._parquet_files = sorted(path for path in files if path.exists())
        if not self._parquet_files:
            raise FileNotFoundError(f"No parquet files matched data_files={data_files!r}")

        row_groups: List[Tuple[int, int, int]] = []
        for file_index, parquet_path in enumerate(self._parquet_files):
            parquet_file = pq.ParquetFile(str(parquet_path))
            for row_group_index in range(parquet_file.num_row_groups):
                row_count = parquet_file.metadata.row_group(row_group_index).num_rows
                if row_count > 0:
                    row_groups.append((file_index, row_group_index, row_count))

        rng = random.Random(20260629)
        rng.shuffle(row_groups)
        total_rows = 0
        self._logical_row_groups = row_groups
        self._logical_cumsum = []
        for _, _, row_count in row_groups:
            total_rows += row_count
            self._logical_cumsum.append(total_rows)
        if total_rows == 0:
            raise ValueError(f"No rows found in parquet files matched by {data_files!r}")

    def _get_local_parquet_row(self, idx: int) -> Dict[str, Any]:
        import pyarrow.parquet as pq

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        group_position = bisect.bisect_right(self._logical_cumsum, idx)
        group_start = 0 if group_position == 0 else self._logical_cumsum[group_position - 1]
        row_offset = idx - group_start
        file_index, row_group_index, _ = self._logical_row_groups[group_position]
        cache_key = (file_index, row_group_index)
        if self._row_group_cache_key != cache_key:
            parquet_path = self._parquet_files[file_index]
            row_group = pq.ParquetFile(str(parquet_path), memory_map=True).read_row_group(
                row_group_index,
                columns=["image", "log"],
            )
            self._row_group_cache = row_group.to_pydict()
            self._row_group_cache_key = cache_key

        assert self._row_group_cache is not None
        return {
            "image": self._row_group_cache["image"][row_offset],
            "log": self._row_group_cache["log"][row_offset],
        }

    @staticmethod
    def _coerce_image(image_value: Any) -> Image.Image:
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            if image_bytes:
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_path = image_value.get("path")
            if image_path:
                return Image.open(image_path).convert("RGB")
        raise ValueError("UAV-Flow row does not contain a valid image")

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            return input_ids.squeeze(0)

        input_ids = list(input_ids)[: self.max_text_len]
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        if len(input_ids) < self.max_text_len:
            input_ids.extend([pad_id] * (self.max_text_len - len(input_ids)))
        return torch.tensor(input_ids, dtype=torch.long)

    def _parse_log_payload(self, log_value: Any) -> Dict[str, Any]:
        if isinstance(log_value, dict):
            return log_value
        if isinstance(log_value, str):
            try:
                payload = json.loads(log_value)
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}
        return {}

    def _parse_raw_logs(self, payload: Dict[str, Any]) -> List[List[float]]:
        raw_logs = payload.get("raw_logs", [])

        rows = []
        for row in raw_logs:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                try:
                    values = [float(x) for x in row]
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in values):
                    rows.append(values)

        if not rows:
            rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        while len(rows) < self.chunk_size:
            rows.append(rows[-1])
        return rows

    def _parse_motion_logs(self, payload: Dict[str, Any]) -> Tuple[List[List[float]], float]:
        preprocessed_logs = payload.get("preprocessed_logs", [])
        rows = []
        for row in preprocessed_logs:
            if isinstance(row, (list, tuple)) and len(row) >= 6:
                try:
                    values = [float(x) for x in row[:6]]
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in values):
                    rows.append(values)
        if rows:
            while len(rows) < self.chunk_size:
                rows.append(rows[-1])
            return rows, self.pos_scale
        return self._parse_raw_logs(payload), 1.0

    def _parse_preprocessed_logs(self, payload: Dict[str, Any]) -> List[List[float]]:
        """
        Parse preprocessed_logs: local Cartesian [x, y, z, roll_deg, yaw_deg, pitch_deg].
        """
        pp_logs = payload.get("preprocessed_logs", [])

        rows = []
        for row in pp_logs:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                try:
                    values = [float(x) for x in row]
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in values):
                    rows.append(values)

        if not rows:
            rows = [[0.0] * 6]
        while len(rows) < self.chunk_size:
            rows.append(rows[-1])
        return rows

    @staticmethod
    def _pose4_from_preprocessed(row: List[float]) -> np.ndarray:
        yaw = math.radians(float(row[4])) if len(row) > 4 else 0.0
        return np.asarray([float(row[0]), float(row[1]), float(row[2]), yaw], dtype=np.float32)

    def _velocity4_from_preprocessed(self, rows: List[List[float]], idx: int) -> np.ndarray:
        if idx <= 0 or idx >= len(rows):
            return np.zeros(4, dtype=np.float32)
        current = self._pose4_from_preprocessed(rows[idx])
        previous = self._pose4_from_preprocessed(rows[idx - 1])
        delta = current - previous
        delta[3] = (delta[3] + math.pi) % (2 * math.pi) - math.pi
        return delta.astype(np.float32)

    def _state8_from_preprocessed(self, rows: List[List[float]], idx: int) -> np.ndarray:
        idx = min(max(idx, 0), len(rows) - 1)
        pose = self._pose4_from_preprocessed(rows[idx])
        velocity = self._velocity4_from_preprocessed(rows, idx)
        return np.concatenate([pose, velocity]).astype(np.float32)

    def _state8_delta_from_preprocessed(self, rows: List[List[float]], idx: int) -> np.ndarray:
        current = self._state8_from_preprocessed(rows, idx)
        if idx + 1 >= len(rows):
            return np.zeros_like(current)
        nxt = self._state8_from_preprocessed(rows, idx + 1)
        delta = nxt - current
        delta[3] = (delta[3] + math.pi) % (2 * math.pi) - math.pi
        delta[7] = (delta[7] + math.pi) % (2 * math.pi) - math.pi
        return delta.astype(np.float32)

    @staticmethod
    def _yaw_delta_deg(yaw: float, yaw0: float) -> float:
        return (yaw - yaw0 + 180.0) % 360.0 - 180.0

    def _extract_relative_actions(
        self,
        raw_logs: List[List[float]],
        motion_pos_scale: float = 1.0,
    ) -> torch.Tensor:
        anchor = raw_logs[0]
        x0, y0, z0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
        yaw0 = float(anchor[4]) if len(anchor) > 4 else 0.0
        yaw0_rad = math.radians(yaw0)
        cos_yaw = math.cos(yaw0_rad)
        sin_yaw = math.sin(yaw0_rad)

        actions = []
        for row in raw_logs[: self.chunk_size]:
            yaw = float(row[4]) if len(row) > 4 else yaw0
            dx_world = float(row[0]) - x0
            dy_world = float(row[1]) - y0
            raw_action = [
                (cos_yaw * dx_world + sin_yaw * dy_world) * motion_pos_scale,
                (-sin_yaw * dx_world + cos_yaw * dy_world) * motion_pos_scale,
                (float(row[2]) - z0) * motion_pos_scale,
                self._yaw_delta_deg(yaw, yaw0),
            ]
            actions.append(normalize_action(raw_action, self.pos_scale))
        return torch.tensor(np.stack(actions), dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dummy Dataset (for smoke-tests and CI without real data)
# ──────────────────────────────────────────────────────────────────────────────

class DummyUAVDataset(Dataset):
    """
    Synthetic dataset that returns random tensors with realistic shapes.
    Useful for overfitting tests and checking gradient flow without real data.

    Args:
        dual_vision : If True, pixel_values is a dict {"dino": ..., "siglip": ...}
                      matching DinoSigLIPEncoder output format.
    """

    def __init__(
        self,
        size:         int  = 200,
        chunk_size:   int  = 5,
        max_text_len: int  = 64,
        vocab_size:   int  = 50277,
        img_size:     int  = 384,
        dual_vision:  bool = True,
    ):
        self.size         = size
        self.chunk_size   = chunk_size
        self.max_text_len = max_text_len
        self.vocab_size   = vocab_size
        self.img_size     = img_size
        self.dual_vision  = dual_vision

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.dual_vision:
            # DinoSigLIP: two separate normalised tensors
            pixel_values = {
                "dino":   torch.randn(3, self.img_size, self.img_size),
                "siglip": torch.randn(3, self.img_size, self.img_size),
            }
        else:
            pixel_values = torch.randn(3, self.img_size, self.img_size)

        return {
            "pixel_values": pixel_values,
            "input_ids":    torch.randint(0, self.vocab_size, (self.max_text_len,)),
            "proprio":      torch.randn(4),
            "state8":       torch.randn(8),
            "delta_state8": torch.randn(8),
            "gt_action":    torch.randn(self.chunk_size, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Custom collate function
# ──────────────────────────────────────────────────────────────────────────────

def aero_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for AeroMamba DataLoader.

    Handles pixel_values that can be either:
      - a plain Tensor [3, H, W]  →  stacked to [B, 3, H, W]
      - a dict {"dino": [3,H,W], "siglip": [3,H,W]}
        →  {"dino": [B,3,H,W], "siglip": [B,3,H,W]}

    All other fields use the default torch.stack collation.

    Usage:
        DataLoader(dataset, collate_fn=aero_collate_fn, ...)
    """
    from torch.utils.data.dataloader import default_collate

    # Separate pixel_values from the rest
    sample_pv = batch[0]["pixel_values"]

    if isinstance(sample_pv, dict):
        # Dual-vision path: collate each stream independently
        keys = sample_pv.keys()
        collated_pv = {
            k: torch.stack([s["pixel_values"][k] for s in batch], dim=0)
            for k in keys
        }
    else:
        # Single-vision path: standard stack
        collated_pv = torch.stack([s["pixel_values"] for s in batch], dim=0)

    # Collate the remaining fields normally
    rest = default_collate(
        [{k: v for k, v in s.items() if k != "pixel_values"} for s in batch]
    )
    rest["pixel_values"] = collated_pv
    return rest
