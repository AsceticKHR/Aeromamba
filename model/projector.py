"""
MLP Projector for AeroMamba-VLA.

Follows LLaVA-1.5 / OpenVLA convention: a simple 2-layer GELU-MLP that maps
SigLIP patch features from D_v to Mamba's D_m dimension.

Key design decision: NO token compression.
  All N_vis patch tokens are passed to the Mamba backbone intact.
  The Mamba backbone (O(n) linear complexity) handles long sequences natively,
  so compressing 576→64 via cross-attention (costly + information-lossy) is
  unnecessary and contradicts Mamba's strength.

Reference:
  LLaVA-1.5 "Improved Baselines with Visual Instruction Tuning" (Liu et al. 2023)
  OpenVLA-OFT (Kim et al. 2025)

Input:  [B, N_vis, D_v]   (SigLIP-L-384: N_vis=576, D_v=1024)
Output: [B, N_vis, D_m]   (token count unchanged, dimension aligned to Mamba)
"""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Two-layer GELU MLP projector with LayerNorm output.

    Maps vision patch features D_v → D_m for the Mamba backbone.
    Token count (N_vis patches) is UNCHANGED — dimension alignment only.

    Architecture:
        Linear(D_v, D_m * 2) → GELU → Linear(D_m * 2, D_m) → LayerNorm(D_m)

    Args:
        vision_hidden_size : D_v, output feature dim of the vision encoder.
        mamba_hidden_size  : D_m, hidden size of the Mamba backbone.
    """

    def __init__(self, vision_hidden_size: int, mamba_hidden_size: int):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.mamba_hidden_size  = mamba_hidden_size

        self.net = nn.Sequential(
            nn.Linear(vision_hidden_size, mamba_hidden_size * 2),
            nn.GELU(),
            nn.Linear(mamba_hidden_size * 2, mamba_hidden_size),
            nn.LayerNorm(mamba_hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features : [B, N_vis, D_v]

        Returns:
            projected       : [B, N_vis, D_m]
        """
        return self.net(vision_features)
