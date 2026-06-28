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

        # Collect all trajectory files
        self.traj_files: List[Path] = sorted(
            self.data_root.rglob(f"*{json_extension}")
        )
        if not self.traj_files:
            raise FileNotFoundError(
                f"No {json_extension} files found under {data_root}. "
                "Check data_root path."
            )

        # Build flat index: (traj_file_idx, step_idx) for each valid chunk
        self.index: List[Tuple[int, int]] = []
        self.trajectories: List[List[Dict[str, Any]]] = []

        for traj_idx, traj_path in enumerate(self.traj_files):
            with open(traj_path, "r", encoding="utf-8") as f:
                traj = json.load(f)
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
        if self.aug_flip and random.random() < 0.5:
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

        # ── Ground-truth action chunk (K steps) ───────────────────────────────
        gt_action = self._extract_chunk(traj, step_idx)  # [K, 4]

        return {
            "pixel_values": pixel_values,   # Tensor [3,H,W] or dict of Tensors
            "input_ids":    input_ids,
            "proprio":      proprio,
            "gt_action":    gt_action,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

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
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "UAVFlowHFDataset requires the `datasets` package. "
                "Install it with `pip install datasets pyarrow`."
            ) from exc

        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.chunk_size = chunk_size
        self.max_text_len = max_text_len
        self.instruction = instruction
        self.pos_scale = pos_scale
        self.aug_flip = aug_flip

        load_kwargs = {"split": split, "cache_dir": cache_dir}
        if data_files:
            load_kwargs["data_files"] = data_files
        self.ds = load_dataset(dataset_name, **load_kwargs)
        if len(self.ds) == 0:
            raise ValueError(f"HuggingFace dataset {dataset_name!r} split {split!r} is empty.")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ds[int(idx)]

        img = row["image"].convert("RGB")
        if self.aug_flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.transform(img)

        input_ids = self._tokenize(self.instruction)
        raw_logs = self._parse_raw_logs(row.get("log", ""))

        anchor = raw_logs[0]
        yaw0 = float(anchor[4]) if len(anchor) > 4 else 0.0
        proprio = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                math.radians(yaw0),
            ],
            dtype=torch.float32,
        )

        gt_action = self._extract_relative_actions(raw_logs)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "proprio": proprio,
            "gt_action": gt_action,
        }

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)

    def _parse_raw_logs(self, log_text: str) -> List[List[float]]:
        try:
            payload = json.loads(log_text)
            raw_logs = payload.get("raw_logs", [])
        except Exception:
            raw_logs = []

        rows = []
        for row in raw_logs:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                rows.append([float(x) for x in row])

        if not rows:
            rows = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        while len(rows) < self.chunk_size:
            rows.append(rows[-1])
        return rows

    @staticmethod
    def _yaw_delta_deg(yaw: float, yaw0: float) -> float:
        return (yaw - yaw0 + 180.0) % 360.0 - 180.0

    def _extract_relative_actions(self, raw_logs: List[List[float]]) -> torch.Tensor:
        anchor = raw_logs[0]
        x0, y0, z0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
        yaw0 = float(anchor[4]) if len(anchor) > 4 else 0.0

        actions = []
        for row in raw_logs[: self.chunk_size]:
            yaw = float(row[4]) if len(row) > 4 else yaw0
            raw_action = [
                float(row[0]) - x0,
                float(row[1]) - y0,
                float(row[2]) - z0,
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
