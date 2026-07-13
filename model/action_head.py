"""
UAV Action Head for AeroMamba-VLA.

Direct chunk regression — no intermediate sequence decoder.

Architecture (RoboMamba two_mlp dimension-cascade style):
    global_token [B, D_m]
        ↓  3-layer MLP  (D_m → D_m/2 → D_m/4 → K*4)
    [B, K*4]
        ↓  reshape
    [B, K, 4]   action chunk: (Δx, Δy, Δz, Δyaw_rad)

Why simpler than the original SSMDecoder approach:
  - K=5 is tiny; Cross-Attention decoder is overkill for such a short sequence.
  - Direct L1 regression over K*4 outputs is more stable than sin/cos split
    (no normalisation constraints, no atan2 at inference time).
  - Matches OpenVLA-OFT "parallel decode + L1 regression" that achieves 26×
    faster inference than token-by-token autoregressive decoding.
  - Zero-init on the final layer keeps initial actions near zero,
    which is safe for physical deployment.

References:
  RoboMamba (lmzpai/roboMamba)  — two_mlp policy head
  OpenVLA-OFT (Kim et al. 2025) — action chunking + L1 regression
  ACT (Zhao et al. 2023)        — temporal ensemble for smooth execution
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ---------------------------------------------------------------------------
# Action normalisation mixin
# ---------------------------------------------------------------------------

class ActionNormalizerMixin:
    """
    Per-(k, dim) z-score normalisation for action targets.

    Buffers `action_mean` / `action_std` [K, action_dim] are registered on the
    head so they persist in checkpoints. Defaults (mean=0, std=1) make the
    normalisation an identity, keeping old checkpoints fully compatible.

    Training : loss is computed in normalised (z) space
               → every DOF / horizon position contributes equally.
    Inference: `denormalize()` maps raw head output back to physical units.

    Bonus: with a zero-initialised output layer, the initial physical
    prediction equals the dataset mean action (gentle forward motion) instead
    of a frozen zero — avoiding the "stationary at start" failure mode.
    """

    def _init_action_stats(self, chunk_size: int, action_dim: int):
        self.register_buffer("action_mean", torch.zeros(chunk_size, action_dim))
        self.register_buffer("action_std", torch.ones(chunk_size, action_dim))

    @torch.no_grad()
    def set_normalization(self, mean, std, std_floor: float = 1e-3):
        mean = torch.as_tensor(mean, dtype=torch.float32).view_as(self.action_mean)
        std = torch.as_tensor(std, dtype=torch.float32).view_as(self.action_std)
        std = std.clamp_min(std_floor)
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Physical [.., K, 4] → z-space."""
        return (action - self.action_mean.to(action.dtype)) / self.action_std.to(action.dtype)

    def denormalize(self, action_norm: torch.Tensor) -> torch.Tensor:
        """z-space [.., K, 4] → physical units (m / rad)."""
        return action_norm * self.action_std.to(action_norm.dtype) + self.action_mean.to(action_norm.dtype)

    def has_normalization(self) -> bool:
        return bool(
            (self.action_std != 1.0).any().item() or (self.action_mean != 0.0).any().item()
        )


# ---------------------------------------------------------------------------
# Action Head
# ---------------------------------------------------------------------------

class UAVActionHead(ActionNormalizerMixin, nn.Module):
    """
    Lightweight direct-regression action head for UAV navigation.

    Predicts K future 4-DOF waypoints from Mamba's last hidden state in a
    single forward pass (no autoregressive decoding).

    Args:
        mamba_hidden_size : D_m from the Mamba backbone (default 1024).
        chunk_size        : K future waypoints per prediction (default 5).
        action_dim        : Per-step action dimension (default 4:
                            Δx, Δy, Δz, Δyaw_rad).
    """

    def __init__(
        self,
        mamba_hidden_size: int = 1024,
        chunk_size:        int = 5,
        action_dim:        int = 4,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        out_dim = chunk_size * action_dim
        d = mamba_hidden_size

        # RoboMamba-style dimension cascade: D → D/2 → D/4 → K*4
        self.mlp = nn.Sequential(
            nn.Linear(d,        d // 2),
            nn.SiLU(),
            nn.Linear(d // 2,  d // 4),
            nn.SiLU(),
            nn.Linear(d // 4,  out_dim),
        )
        self._init_action_stats(chunk_size, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the output layer: initial actions near zero for safe deployment
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        global_token: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            global_token : [B, D_m]  — last hidden state from Mamba backbone

        Returns:
            dict with:
                'action' : [B, K, 4]  predicted chunk (Δx, Δy, Δz, Δyaw_rad)
        """
        raw    = self.mlp(global_token)                          # [B, K*4]
        action = raw.view(-1, self.chunk_size, self.action_dim)  # [B, K, 4]
        return {"action": action}


class UAVDynamicsActionHead(ActionNormalizerMixin, nn.Module):
    """
    Dynamics-aware lightweight action chunk head for UAV-Flow.

    Instead of predicting K waypoints independently, this head rolls a compact
    recurrent dynamics state forward for K steps. The recurrent state is
    conditioned on the Mamba global token and the current UAV proprioception,
    which makes smooth velocity/yaw-rate evolution an architectural prior
    rather than relying only on loss regularisation.
    """

    def __init__(
        self,
        mamba_hidden_size: int = 1024,
        chunk_size: int = 5,
        action_dim: int = 4,
        proprio_dim: int = 4,
        hidden_ratio: float = 0.5,
        action_bound: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.action_bound = action_bound
        d = mamba_hidden_size
        h = max(128, int(d * hidden_ratio))

        self.context = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.SiLU(),
        )
        self.proprio_proj = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, h),
            nn.SiLU(),
        )
        self.step_embed = nn.Parameter(torch.randn(chunk_size, h) * 0.02)
        self.cell = nn.GRUCell(input_size=action_dim + h, hidden_size=h)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, action_dim),
        )
        self._init_action_stats(chunk_size, action_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final = self.delta_head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        global_token: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> dict:
        batch = global_token.size(0)
        state = self.context(global_token)
        if proprio is None:
            proprio = global_token.new_zeros(batch, self.proprio_dim)
        prop = self.proprio_proj(proprio.to(dtype=global_token.dtype))
        hidden = state + prop

        prev_action = global_token.new_zeros(batch, self.action_dim)
        actions = []
        for step in range(self.chunk_size):
            step_context = prop + self.step_embed[step].unsqueeze(0)
            cell_input = torch.cat([prev_action, step_context], dim=-1)
            hidden = self.cell(cell_input, hidden)
            delta = self.delta_head(hidden)
            if self.action_bound > 0:
                delta = self.action_bound * torch.tanh(delta / self.action_bound)
            actions.append(delta)
            prev_action = delta

        return {"action": torch.stack(actions, dim=1)}


# Backward compatibility alias (for any code importing the old name)
UAVActionChunkHead = UAVActionHead


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def aero_action_loss(
    pred:          dict,
    gt_action:     torch.Tensor,    # [B, K, 4]  physical units (Δm, Δm, Δm, Δyaw_rad)
    head=None,                       # ActionNormalizerMixin head (for z-score stats)
    lambda_smooth: float = 0.0,
    lambda_endpoint: float = 0.25,
    lambda_direction: float = 0.5,
    lambda_acc: float = 0.0,
) -> tuple:
    """
    Action-chunk loss in per-(k, dim) z-score space.

    Rationale (UAV-Flow specifics):
      - Raw targets are dominated by forward motion (x ≈ 0.13 m/step) while
        lateral / vertical / yaw components are 10-100× smaller. A physical-
        space loss lets the model win by predicting the forward mean and
        ignoring turns. Normalising each (horizon-position, DOF) to unit
        variance makes every degree of freedom contribute equally.
      - Pure L1 (not Smooth-L1): Smooth-L1's quadratic zone (|e|<beta)
        underweights exactly the small-magnitude errors that matter after
        normalisation; L1 keeps constant gradient toward the target.

    Components:
      main      : L1 in z-space over all K×4 outputs.
      endpoint  : extra L1 on the final waypoint (goal accuracy).
      direction : 1−cos between predicted and GT endpoint xy displacement in
                  *physical* space (heading correctness for turn commands).
      smooth/acc: optional temporal regularisers (z-space).

    `pred["action"]` is interpreted as z-space output when `head` carries
    normalisation stats; otherwise the loss operates in physical space
    (backward compatible with old checkpoints / no-stats runs).

    Returns:
        (total_loss, detail_dict)
    """
    pred_action = pred["action"]    # [B, K, 4]  (z-space if head has stats)

    if head is not None and hasattr(head, "normalize"):
        gt_norm = head.normalize(gt_action)
        pred_phys = head.denormalize(pred_action)
    else:
        gt_norm = gt_action
        pred_phys = pred_action

    # Main: pure L1 in normalised space
    loss_main = F.l1_loss(pred_action, gt_norm)

    # Endpoint: extra weight on the final waypoint
    loss_endpoint = F.l1_loss(pred_action[:, -1], gt_norm[:, -1])

    # Direction: endpoint xy heading in physical space
    pred_xy = pred_phys[:, -1, :2].float()
    gt_xy = gt_action[:, -1, :2].float()
    valid_direction = (pred_xy.norm(dim=-1) > 1e-4) & (gt_xy.norm(dim=-1) > 1e-4)
    if valid_direction.any():
        cosine = F.cosine_similarity(pred_xy[valid_direction], gt_xy[valid_direction], dim=-1)
        loss_direction = (1.0 - cosine).mean()
    else:
        loss_direction = pred_action.new_zeros(())

    # Smoothness: penalise jerk (2nd-order finite difference along K dimension)
    if lambda_smooth > 0 and pred_action.size(1) >= 3:
        d1 = pred_action[:, 1:] - pred_action[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        loss_smooth = d2.pow(2).mean()
    else:
        loss_smooth = pred_action.new_zeros(())

    if lambda_acc > 0 and pred_action.size(1) >= 2:
        pred_delta = pred_action[:, 1:] - pred_action[:, :-1]
        gt_delta = gt_norm[:, 1:] - gt_norm[:, :-1]
        loss_acc = F.l1_loss(pred_delta, gt_delta)
    else:
        loss_acc = pred_action.new_zeros(())

    total = (
        loss_main
        + lambda_smooth * loss_smooth
        + lambda_endpoint * loss_endpoint
        + lambda_direction * loss_direction
        + lambda_acc * loss_acc
    )

    # Physical-unit diagnostics (no grad)
    with torch.no_grad():
        err_phys = (pred_phys.float() - gt_action.float()).abs()
        pos_err_m = err_phys[..., :3].mean()
        yaw_err_deg = err_phys[..., 3].mean() * (180.0 / math.pi)
        end_pos_err_m = err_phys[:, -1, :3].mean()

    detail = {
        "main": loss_main.item(),
        "endpoint": loss_endpoint.item(),
        "direction": loss_direction.item(),
        "pos_err_m": pos_err_m.item(),
        "yaw_err_deg": yaw_err_deg.item(),
        "end_pos_err_m": end_pos_err_m.item(),
    }
    if lambda_smooth > 0:
        detail["smooth"] = loss_smooth.item()
    if lambda_acc > 0:
        detail["acc"] = loss_acc.item()
    return total, detail


# ---------------------------------------------------------------------------
# Temporal Ensemble (inference utility)
# ---------------------------------------------------------------------------

class TemporalEnsemble:
    """
    Aggregates action-chunk predictions over a sliding window using
    exponentially decaying weights to produce smooth executed actions.

    Reference: ACT (Zhao et al. 2023) — temporal ensemble strategy.

    Usage:
        ens = TemporalEnsemble(window=5, decay=0.7)
        for each control step:
            chunk = model(...)["action"][0]  # [K, 4]
            action = ens.update(chunk)       # [4]  ← execute this
    """

    def __init__(self, window: int = 5, decay: float = 0.7):
        self.window = window
        self.decay  = decay
        self.buffer: deque = deque(maxlen=window)

    def update(self, new_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            new_chunk : [K, 4]  model prediction for current step
        Returns:
            action    : [4]     exponentially weighted average action
        """
        self.buffer.append(new_chunk)
        weights = [self.decay ** (len(self.buffer) - 1 - i)
                   for i in range(len(self.buffer))]
        w_sum   = sum(weights)
        act_dim = new_chunk.shape[-1]
        action_sum = torch.zeros(act_dim, device=new_chunk.device,
                                  dtype=new_chunk.dtype)
        for j, (chunk, w) in enumerate(zip(self.buffer, weights)):
            # Each buffered chunk[i] predicted steps from time i forwards;
            # to get the estimate for *current* step, we index into the chunk.
            idx = len(self.buffer) - 1 - j
            if idx < chunk.shape[0]:
                action_sum += w * chunk[idx]
        return action_sum / w_sum

    def reset(self):
        self.buffer.clear()
