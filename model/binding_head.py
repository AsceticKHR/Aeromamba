"""
Instruction Binding Head (AeroStream Round A, change plan §A1).

Training-only auxiliary classification head that shares the action head's
input `h_last` [B, D_m] and predicts four instruction/action-binding targets:

    motion_class  : 10 classes (move/shift/turn/rotate/ascend/descend/
                    land/approach/pass/surround), regex-derived from the
                    instruction; unmatched samples carry ignore_index=-100.
    yaw_sign      : 3 classes (0=none, 1=positive, 2=negative), from the
                    chunk-endpoint |Δyaw| > 2°.
    dz_sign       : 3 classes (0=level, 1=up, 2=down), from the chunk-endpoint
                    |Δz| > 0.05 m.
    magnitude_bin : log-binned chunk-endpoint displacement norm; edges
                    [0.1, 0.2, 0.5, 1, 2, 5, 10, 30] m → 9 bins.

Why cross-entropy instead of more regression: the diagnosed failure modes
(yaw sign ≈ random, dz ≈ 0, motion-primitive freezing) are *sign/category*
errors that L1 regression on z-scored targets penalises too softly. CE on
signs/classes gives sharp gradients exactly where the policy collapses.

Design constraints (change plan):
  - Output layers are zero-initialised → at load time the head contributes
    zero logits everywhere and does not perturb the pretrained policy.
  - Parameter count < 1M.
  - Forward is only invoked on the training path; predict_step / stream_step
    never touch it (zero deployment cost).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MOTION_CLASSES = [
    "move", "shift", "turn", "rotate", "ascend",
    "descend", "land", "approach", "pass", "surround",
]
MOTION_CLASS_TO_IDX = {name: i for i, name in enumerate(MOTION_CLASSES)}

# Chunk-endpoint displacement norm (metres), log-spaced. bisect over the 8
# edges yields 9 bins: [0,0.1) ... [30,inf).
MAGNITUDE_BIN_EDGES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
NUM_MAGNITUDE_BINS = len(MAGNITUDE_BIN_EDGES) + 1

BINDING_TASKS = ("motion_class", "yaw_sign", "dz_sign", "magnitude_bin")


class InstructionBindingHead(nn.Module):
    """Shared bottleneck MLP + four linear classification branches."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_motion_classes: int = len(MOTION_CLASSES),
        num_magnitude_bins: int = NUM_MAGNITUDE_BINS,
        bottleneck_ratio: float = 0.25,
    ):
        super().__init__()
        h = max(64, int(hidden_size * bottleneck_ratio))
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, h),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {
                "motion_class": nn.Linear(h, num_motion_classes),
                "yaw_sign": nn.Linear(h, 3),
                "dz_sign": nn.Linear(h, 3),
                "magnitude_bin": nn.Linear(h, num_magnitude_bins),
            }
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Zero-init classification outputs: uniform logits at load time, so
        # attaching the head to a pretrained policy is a no-op until trained.
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h_last: torch.Tensor) -> Dict[str, torch.Tensor]:
        """h_last [B, D_m] → dict of logits per binding task."""
        z = self.shared(h_last)
        return {name: head(z) for name, head in self.heads.items()}


def binding_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    ignore_index: int = -100,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """
    Mean of the four cross-entropies (each with ignore_index), plus per-task
    accuracy diagnostics on the valid subset.

    Returns (loss | None, detail). loss is None when no task has any valid
    label in the batch.
    """
    losses = []
    detail: Dict[str, float] = {}
    for task in BINDING_TASKS:
        task_logits = logits.get(task)
        task_labels = labels.get(task)
        if task_logits is None or task_labels is None:
            continue
        task_labels = task_labels.long().view(-1)
        valid = task_labels != ignore_index
        if not bool(valid.any()):
            continue
        ce = F.cross_entropy(
            task_logits.float(), task_labels, ignore_index=ignore_index
        )
        losses.append(ce)
        with torch.no_grad():
            pred_cls = task_logits.argmax(dim=-1)
            acc = (pred_cls[valid] == task_labels[valid]).float().mean().item()
        detail[f"{task}_acc"] = acc
    if not losses:
        return None, detail
    total = torch.stack(losses).mean()
    detail["binding_ce"] = total.item()
    return total, detail
