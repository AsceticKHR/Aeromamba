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
import re
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


# Horizontal-flip augmentation mirrors the image and negates y/yaw, so any
# left/right (or rotation-direction) wording in the instruction must be
# swapped too — otherwise the model is trained on contradictory
# text/action pairs (e.g. "pass from the left" paired with a right-side
# trajectory), which directly undermines the turn-oversampling strategy.
_LEFT_RIGHT_PATTERN = re.compile(
    r"\b(left|right|clockwise|counter-clockwise|counterclockwise)\b",
    re.IGNORECASE,
)
_LEFT_RIGHT_SWAP = {
    "left": "right",
    "right": "left",
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
    "counter-clockwise": "clockwise",
}


def mirror_instruction_lr(text: str) -> str:
    """Swap left/right (and rotation-direction) words to stay consistent
    with a horizontally-flipped image + negated y/yaw action target."""

    def _replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        replacement = _LEFT_RIGHT_SWAP.get(word.lower(), word)
        if word.isupper():
            return replacement.upper()
        if word[:1].isupper():
            return replacement.capitalize()
        return replacement

    return _LEFT_RIGHT_PATTERN.sub(_replace, text)


# ──────────────────────────────────────────────────────────────────────────────
# Instruction binding labels (AeroStream Round A)
# ──────────────────────────────────────────────────────────────────────────────
#
# Motion class is inferred from the instruction with ordered regex templates
# (UAV-Flow instructions are highly templated). Unmatched instructions get
# ignore_index=-100 and simply do not contribute to the binding CE loss.
# Mirroring (mirror_instruction_lr) only swaps direction words and never
# changes the motion class, so inference may run on either form.
BINDING_IGNORE_INDEX = -100

# Canonical class order — MUST match model/binding_head.py MOTION_CLASSES.
MOTION_CLASSES = [
    "move", "shift", "turn", "rotate", "ascend",
    "descend", "land", "approach", "pass", "surround",
]
_MOTION_CLASS_TO_IDX = {name: i for i, name in enumerate(MOTION_CLASSES)}

# Ordered matching rules (most specific first); the rule ORDER decides which
# template wins, the returned index is always in MOTION_CLASSES order.
_MOTION_CLASS_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    ("surround", re.compile(r"\b(orbit|circle|surround|revolve|around)\b", re.I)),
    ("land",     re.compile(r"\b(land|landing|touch\s*down)\b", re.I)),
    ("ascend",   re.compile(r"\b(ascend|rise|climb|lift|upward|elevate)\b|\b(fly|move|go)\s+(up|higher)\b", re.I)),
    ("descend",  re.compile(r"\b(descend|lower|downward|drop|sink)\b|\b(fly|move|go)\s+(down)\b", re.I)),
    ("rotate",   re.compile(r"\b(rotate|spin|clockwise|counter-?clockwise|anticlockwise)\b|\bdegrees\b", re.I)),
    ("turn",     re.compile(r"\bturn\b|\bface\b|\bfacing\b", re.I)),
    ("pass",     re.compile(r"\b(pass|through|past)\b", re.I)),
    ("approach", re.compile(r"\b(approach|toward|towards|closer|close\s+to|near|get\s+to|reach)\b", re.I)),
    ("shift",    re.compile(r"\b(shift|sideways|laterally|strafe|side|translate|translating|translation)\b", re.I)),
    ("move",     re.compile(r"\b(move|moving|fly|go|proceed|advance|head|forward|backward|retreat|withdraw|navigate)\b|\b(back(ing)?|step(ping)?)\s+(up|away|out|off|back)\b", re.I)),
]

# Classes whose windows get whole-class oversampling (change plan §A2): the
# motion primitives that stage3_v2 diagnostics showed frozen / collapsed.
OVERSAMPLE_MOTION_CLASSES = {"move", "shift", "ascend", "descend", "surround", "rotate"}

# Chunk-endpoint displacement norm (metres), log-spaced → 9 bins.
MAGNITUDE_BIN_EDGES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

YAW_SIGN_THRESHOLD_RAD = math.radians(2.0)   # |Δyaw endpoint| > 2°
DZ_SIGN_THRESHOLD_M = 0.05                    # |Δz endpoint| > 0.05 m


def infer_motion_class(instruction: str) -> int:
    """Instruction → motion class index in MOTION_CLASSES order, or
    BINDING_IGNORE_INDEX when no template matches."""
    for class_name, pattern in _MOTION_CLASS_RULES:
        if pattern.search(instruction):
            return _MOTION_CLASS_TO_IDX[class_name]
    return BINDING_IGNORE_INDEX


def binding_labels_from_action(
    gt_action: torch.Tensor,
    motion_class: int,
) -> Dict[str, torch.Tensor]:
    """
    Derive binding labels from the (possibly flipped) gt_action chunk [K, 4]
    in normalised units (positions ~metres, yaw radians). gt_action rows are
    cumulative offsets from the anchor, so the endpoint row IS the chunk
    displacement. Because these are computed AFTER flip augmentation, the
    labels stay consistent with the mirrored action targets for free.
    """
    endpoint = gt_action[-1]
    dyaw = float(endpoint[3])
    dz = float(endpoint[2])
    disp_m = float(torch.linalg.vector_norm(endpoint[:3]))

    if dyaw > YAW_SIGN_THRESHOLD_RAD:
        yaw_sign = 1
    elif dyaw < -YAW_SIGN_THRESHOLD_RAD:
        yaw_sign = 2
    else:
        yaw_sign = 0

    if dz > DZ_SIGN_THRESHOLD_M:
        dz_sign = 1
    elif dz < -DZ_SIGN_THRESHOLD_M:
        dz_sign = 2
    else:
        dz_sign = 0

    magnitude_bin = bisect.bisect_right(MAGNITUDE_BIN_EDGES, disp_m)

    return {
        "motion_class": torch.tensor(motion_class, dtype=torch.long),
        "yaw_sign": torch.tensor(yaw_sign, dtype=torch.long),
        "dz_sign": torch.tensor(dz_sign, dtype=torch.long),
        "magnitude_bin": torch.tensor(magnitude_bin, dtype=torch.long),
        # endpoint norm for magnitude-aware sample weighting in the loss
        "endpoint_norm_m": torch.tensor(disp_m, dtype=torch.float32),
    }


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
        pos_unit:       Unit of `preprocessed_logs` xyz: 'm', 'cm', or 'auto'.
                        Real UAV-Flow is metres; UAV-Flow-Sim is centimetres
                        (UE). 'auto' samples trajectories and picks by magnitude.
                        Getting this wrong multiplies action targets by ~100 and
                        makes every offline metre-metric meaningless.
        aug_flip:       Random horizontal flip augmentation (train only).
        aug_vision:     Appearance augmentation (color jitter / grayscale /
                        blur) to shrink the real-photo → Unreal-render domain
                        gap. Label-free (no geometric change).
        oversample_turn_factor:
                        Replicate turn-heavy chunk samples N× in the index.
                        UAV-Flow is dominated by straight forward flight;
                        without oversampling the model regresses to "fly
                        forward" and ignores turn commands.
        oversample_turn_deg:
                        A chunk counts as turn-heavy when the max cumulative
                        |Δyaw| from the anchor exceeds this many degrees.
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
        pos_unit:      str   = "auto",
        aug_flip:      bool  = True,
        aug_vision:    bool  = False,
        oversample_turn_factor: int = 1,
        oversample_turn_deg: float = 10.0,
        oversample_class_factor: int = 1,
        emit_binding_labels: bool = False,
        json_extension:str   = ".json",
        chunk_offset:  int   = 0,
        terminal_pad_frac: float = 0.0,
    ):
        super().__init__()
        self.data_root    = Path(data_root)
        self.tokenizer    = tokenizer
        self.transform    = transform
        self.chunk_size   = chunk_size
        # Waypoints are cumulative offsets from the anchor pose, so step
        # `start + 0` is identically (0, 0, 0, 0) and its per-(k, dim) std is
        # exactly zero. With offset 0 an eighth of the K=8 loss mass is spent
        # on a target that carries no information. offset=1 starts the chunk at
        # the first *future* waypoint.
        self.chunk_offset = int(chunk_offset)
        # Recording stops when the pilot stops, so the drone is still moving at
        # ~92% of cruise speed in the final steps of 89% of trajectories, and no
        # window ever carries a "you have arrived, hold position" target. The
        # eval harness, however, ends an episode only when the policy emits
        # <3cm/step for 10 steps, so a policy trained on unpadded windows can
        # never terminate. Clamping the target index past the end repeats the
        # final pose, which turns those steps into genuine zero-displacement
        # labels. It also keeps short trajectories usable at large K.
        self.terminal_pad_frac = float(terminal_pad_frac)
        self.max_pad_steps = int(chunk_size * self.terminal_pad_frac)
        self.max_text_len = max_text_len
        self.pos_scale    = pos_scale
        unit = str(pos_unit).lower().strip()
        if unit not in ("auto", "m", "cm", "meter", "meters", "centimeter", "centimeters"):
            raise ValueError(
                f"pos_unit must be 'auto'|'m'|'cm', got {pos_unit!r}")
        if unit in ("meter", "meters"):
            unit = "m"
        elif unit in ("centimeter", "centimeters"):
            unit = "cm"
        self.pos_unit_arg = unit
        self.aug_flip     = aug_flip and (split == "train")
        self.split        = split

        self.aug_vision = aug_vision and (split == "train")
        self.vision_aug = None
        if self.aug_vision:
            from torchvision import transforms as T
            self.vision_aug = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.RandomGrayscale(p=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
            ])

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

        # Resolve preprocessed xyz units BEFORE parsing trajectories. Real
        # UAV-Flow stores metres; UAV-Flow-Sim stores centimetres. The loader
        # historically always did pose_cm = preprocessed * 100, which is correct
        # only for metres and silently 100×-inflates sim action targets.
        self.pos_unit = (
            self._detect_preprocessed_pos_unit(self.traj_files)
            if self.pos_unit_arg == "auto"
            else self.pos_unit_arg
        )
        print(
            f"[UAVFlowDataset] preprocessed pos_unit={self.pos_unit} "
            f"(arg={self.pos_unit_arg}, pos_scale={self.pos_scale})",
            flush=True,
        )

        self.emit_binding_labels = emit_binding_labels

        # Build flat index: (traj_file_idx, step_idx) for each valid chunk
        self.index: List[Tuple[int, int]] = []
        self.trajectories: List[List[Dict[str, Any]]] = []
        # traj_files entries that survived the length filter, aligned index-wise
        # with self.trajectories (traj_files is not, since short ones are skipped).
        self.kept_traj_files: List[Path] = []
        # Per-trajectory motion class (regex over the unmirrored instruction;
        # mirroring never changes the class).
        self.traj_motion_class: List[int] = []
        oversample = max(1, int(oversample_turn_factor)) if split == "train" else 1
        class_oversample = max(1, int(oversample_class_factor)) if split == "train" else 1
        n_turn_samples = 0
        n_class_samples = 0
        class_counter: Dict[int, int] = {}

        for traj_idx, traj_path in enumerate(self.traj_files):
            traj = self._load_trajectory(traj_path)
            if (not isinstance(traj, list)
                    or len(traj) < chunk_size + self.chunk_offset - self.max_pad_steps):
                continue
            self.trajectories.append(traj)
            self.kept_traj_files.append(traj_path)
            t = len(self.trajectories) - 1

            instruction = (
                traj[0].get("instruction")
                or traj[0].get("instruction_unified")
                or ""
            )
            motion_class = infer_motion_class(instruction)
            self.traj_motion_class.append(motion_class)
            class_counter[motion_class] = class_counter.get(motion_class, 0) + 1
            class_name = (
                MOTION_CLASSES[motion_class]
                if motion_class != BINDING_IGNORE_INDEX
                else None
            )
            traj_class_factor = (
                class_oversample
                if class_name in OVERSAMPLE_MOTION_CLASSES
                else 1
            )

            yaws = None
            if oversample > 1:
                yaws = [
                    float(step.get("state", [[0, 0, 0], [0, 0, 0]])[1][1])
                    for step in traj
                ]
            # Sliding window. A window may run at most max_pad_steps past the
            # end of the trajectory; those steps hold the final pose.
            last = len(traj) - 1
            n_start = len(traj) - chunk_size - self.chunk_offset + 1 + self.max_pad_steps
            for step_idx in range(max(n_start, 1)):
                self.index.append((t, step_idx))
                turn_factor = 1
                if yaws is not None:
                    yaw0 = yaws[step_idx]
                    max_dyaw = max(
                        abs((yaws[min(step_idx + self.chunk_offset + k, last)]
                             - yaw0 + 180.0) % 360.0 - 180.0)
                        for k in range(chunk_size)
                    )
                    if max_dyaw >= oversample_turn_deg:
                        turn_factor = oversample
                        n_turn_samples += 1
                # Turn- and class-oversampling combine via max, not multiply
                # (change plan §A2), so turn-heavy windows in oversampled
                # classes are not double-boosted.
                factor = max(turn_factor, traj_class_factor)
                if factor > 1:
                    self.index.extend([(t, step_idx)] * (factor - 1))
                    if factor == traj_class_factor and turn_factor <= traj_class_factor:
                        n_class_samples += 1

        if oversample > 1:
            print(
                f"[UAVFlowDataset] Turn oversampling x{oversample}: "
                f"{n_turn_samples} turn chunks (>{oversample_turn_deg}deg) "
                f"({len(self.index)} total samples)"
            )
        if class_oversample > 1:
            named = {
                (MOTION_CLASSES[k] if k != BINDING_IGNORE_INDEX else "unmatched"): v
                for k, v in sorted(class_counter.items(), key=lambda kv: -kv[1])
            }
            print(
                f"[UAVFlowDataset] Class oversampling x{class_oversample} for "
                f"{sorted(OVERSAMPLE_MOTION_CLASSES)}: {n_class_samples} class-boosted "
                f"windows; trajectory class counts: {named}"
            )

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
        if self.vision_aug is not None:
            img = self.vision_aug(img)
        do_flip = self.aug_flip and random.random() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # transform may return a plain tensor (single encoder)
        # or a dict {"dino": ..., "siglip": ...} (DinoSigLIPTransform)
        pixel_values = self.transform(img)

        # ── Instruction ───────────────────────────────────────────────────────
        instruction = anchor.get("instruction") or anchor.get("instruction_unified")
        if not instruction:
            raise ValueError(
                f"UAV-Flow sample at traj={traj_idx} step={step_idx} is missing "
                "instruction; refusing fixed fallback."
            )
        if do_flip:
            instruction = mirror_instruction_lr(instruction)
        input_ids = self._tokenize(instruction)

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

        sample = {
            "pixel_values": pixel_values,   # Tensor [3,H,W] or dict of Tensors
            "input_ids":    input_ids,
            "proprio":      proprio,
            "state8":       state8,
            "delta_state8": delta_state8,
            "gt_action":    gt_action,
        }
        if self.emit_binding_labels:
            # Labels are derived from the POST-flip gt_action, so yaw/dz signs
            # automatically match the mirrored targets; motion class comes
            # from the trajectory template match (flip-invariant).
            sample["binding_labels"] = binding_labels_from_action(
                gt_action, self.traj_motion_class[traj_idx]
            )
        return sample

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
                # raw_logs are already centimetres in both real and sim dumps.
                x_cm = float(pose["x_cm"])
                y_cm = float(pose["y_cm"])
                z_cm = float(pose["z_cm"])
            elif self.pos_unit == "cm":
                # Sim: preprocessed xyz is already cm. Do NOT × pos_scale again.
                x_cm = float(pose["x_cm"])
                y_cm = float(pose["y_cm"])
                z_cm = float(pose["z_cm"])
            else:
                # Real: preprocessed xyz is metres → convert to cm for pose_cm.
                x_cm = float(pose["x_m"]) * self.pos_scale
                y_cm = float(pose["y_m"]) * self.pos_scale
                z_cm = float(pose["z_m"]) * self.pos_scale
            state_pose = [x_cm / self.pos_scale, y_cm / self.pos_scale, z_cm / self.pos_scale]
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
                        x_cm,
                        y_cm,
                        z_cm,
                        pose["yaw_deg"],
                    ],
                }
            )
        return traj

    @staticmethod
    def _detect_preprocessed_pos_unit(
        traj_files: List[Path],
        sample_n: int = 64,
        seed: int = 0,
    ) -> str:
        """Classify preprocessed xyz as metres or centimetres by magnitude.

        Empirically (UAV-Flow vs UAV-Flow-Sim):
          real  median step ≈ 0.13, median span ≈ 7
          sim   median step ≈ 20,   median span ≈ 500
        A step median above 2.0 cannot be metres at 5 Hz cruise, so → cm.
        """
        import random as _random

        candidates = [p for p in traj_files if p.name == "log.json"]
        if not candidates:
            candidates = list(traj_files)
        if not candidates:
            return "m"
        rng = _random.Random(seed)
        sample = candidates if len(candidates) <= sample_n else rng.sample(
            candidates, sample_n)
        step_norms: List[float] = []
        spans: List[float] = []
        for path in sample:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pp = payload.get("preprocessed_logs") or []
            pts = []
            for row in pp:
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    try:
                        pts.append([float(row[0]), float(row[1]), float(row[2])])
                    except (TypeError, ValueError):
                        continue
            if len(pts) < 2:
                continue
            import math as _math
            for i in range(1, len(pts)):
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
                dz = pts[i][2] - pts[i - 1][2]
                step_norms.append(_math.sqrt(dx * dx + dy * dy + dz * dz))
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            dz = pts[-1][2] - pts[0][2]
            spans.append(_math.sqrt(dx * dx + dy * dy + dz * dz))
        if not step_norms:
            print("[UAVFlowDataset] unit detect: no usable preprocessed logs; "
                  "defaulting to metres", flush=True)
            return "m"
        step_norms.sort()
        spans.sort()
        step_med = step_norms[len(step_norms) // 2]
        span_med = spans[len(spans) // 2] if spans else 0.0
        unit = "cm" if (step_med > 2.0 or span_med > 50.0) else "m"
        print(
            f"[UAVFlowDataset] unit detect: step_med={step_med:.4f} "
            f"span_med={span_med:.4f} → {unit} "
            f"(n_files={len(sample)}, n_steps={len(step_norms)})",
            flush=True,
        )
        return unit

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
        last = len(traj) - 1
        for k in range(self.chunk_size):
            # Clamping past the end repeats the terminal pose: the label becomes
            # "hold here", which is the only stopping signal in this data.
            step = traj[min(start + self.chunk_offset + k, last)]
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
        last = len(traj) - 1
        for k in range(self.chunk_size):
            step = traj[min(start + self.chunk_offset + k, last)]
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
        # AeroStream hygiene (change plan §A7): this per-row HF format carries
        # no frame BEFORE the anchor, so delta_state8 is forced to zero and
        # velocity4 at the anchor is zero — a REAL train/infer mismatch with
        # the current server (which sends non-zero per-step velocity/delta).
        # The production Stage-3 path uses UAVFlowDataset (--data_root) whose
        # temporal channels are computed from past frames and are correct.
        print(
            "[UAVFlowHFDataset] WARNING: per-row HF format has no pre-anchor "
            "frame -> anchor velocity/delta_state8 are ZERO, which mismatches "
            "the inference server's non-zero temporal channels. Prefer "
            "UAVFlowDataset via --data_root for Stage-3 training; only use "
            "this path if you re-export rows with a preceding anchor frame."
        )
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
        if do_flip:
            instruction = mirror_instruction_lr(instruction)
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
        """
        `rows` (preprocessed_logs) holds the anchor frame at idx=0 followed
        only by *future* steps used to build the gt_action chunk — there is
        no frame preceding the anchor available in this per-row HF format.

        Computing nxt - current here would set delta_state8 to (approximately)
        the first ground-truth action step, i.e. leak the label into the
        model input. We therefore return zeros — the only leak-free option
        for this format, but note this is a KNOWN train/infer mismatch: the
        inference server sends non-zero per-step velocity/delta after the
        first frame of an episode (only UAVFlowDataset's step_idx == 0
        windows are legitimately zero). Fixing it properly requires
        re-exporting rows with one pre-anchor frame; until then this path
        should not be used for Stage-3 training (see the constructor
        warning).
        """
        current = self._state8_from_preprocessed(rows, idx)
        return np.zeros_like(current)

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
