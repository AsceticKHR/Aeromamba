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

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ---------------------------------------------------------------------------
# Action Head
# ---------------------------------------------------------------------------

class UAVActionHead(nn.Module):
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


class UAVDynamicsActionHead(nn.Module):
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
    gt_action:     torch.Tensor,    # [B, K, 4]  (Δx, Δy, Δz, Δyaw_rad)
    lambda_smooth: float = 0.0,
    lambda_endpoint: float = 2.0,
    lambda_direction: float = 0.0,
    lambda_acc: float = 0.0,
) -> tuple:
    """
    Unified action-chunk loss.

    Components:
      - Smooth-L1 over all K steps × 4 DOF  (main regression objective)
      - Smoothness regularisation: penalises jerk (2nd-order finite difference)
        along the chunk dimension, encouraging physically plausible trajectories.

    Args:
        pred          : Output dict from UAVActionHead.forward()
        gt_action     : Ground-truth [B, K, 4]
        lambda_smooth : Weight for the smoothness term (default 0.1)

    Returns:
        (total_loss, detail_dict)
    """
    pred_action = pred["action"]    # [B, K, 4]

    # Main: Smooth-L1 over the full action chunk
    loss_main = F.smooth_l1_loss(pred_action, gt_action, beta=0.1)

    # Intuitive Metric: Exact L1 distance
    l1_err = F.l1_loss(pred_action, gt_action)

    loss_endpoint = F.smooth_l1_loss(pred_action[:, -1], gt_action[:, -1], beta=0.1)

    pred_xy = pred_action[:, -1, :2]
    gt_xy = gt_action[:, -1, :2]
    pred_norm = pred_xy.norm(dim=-1)
    gt_norm = gt_xy.norm(dim=-1)
    valid_direction = (pred_norm > 1e-4) & (gt_norm > 1e-4)
    if valid_direction.any():
        cosine = F.cosine_similarity(pred_xy[valid_direction], gt_xy[valid_direction], dim=-1)
        loss_direction = (1.0 - cosine).mean()
    else:
        loss_direction = pred_action.new_zeros(())

    # Smoothness: penalise jerk (2nd-order finite difference along K dimension)
    if pred_action.size(1) >= 3:
        d1 = pred_action[:, 1:] - pred_action[:, :-1]   # velocity  [B, K-1, 4]
        d2 = d1[:, 1:] - d1[:, :-1]                     # accel     [B, K-2, 4]
        loss_smooth = d2.pow(2).mean()
    else:
        loss_smooth = pred_action.new_zeros(()).squeeze()

    if pred_action.size(1) >= 2:
        pred_delta = pred_action[:, 1:] - pred_action[:, :-1]
        gt_delta = gt_action[:, 1:] - gt_action[:, :-1]
        loss_acc = F.l1_loss(pred_delta, gt_delta)
    else:
        loss_acc = pred_action.new_zeros(()).squeeze()

    total = (
        loss_main
        + lambda_smooth * loss_smooth
        + lambda_endpoint * loss_endpoint
        + lambda_direction * loss_direction
        + lambda_acc * loss_acc
    )
    detail = {
        "main": loss_main.item(),
        "endpoint": loss_endpoint.item(),
        "direction": loss_direction.item(),
        "smooth": loss_smooth.item(),
        "acc": loss_acc.item(),
        "l1_err": l1_err.item(),
    }
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
