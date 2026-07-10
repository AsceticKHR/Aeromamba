"""
Proprioception Encoder for AeroMamba-VLA.

Encodes the UAV relative state vector into a single token compatible with
the Mamba-2 hidden dimension, following the UAV-Flow state convention:

    Input state:  [Δx, Δy, Δz, Δyaw_rad]   (4-DOF relative pose)

Sinusoidal positional-style embedding (NeRF-inspired) is applied before the
MLP to improve the model's sensitivity to fine-grained magnitude and direction.

Architecture:
    [B, 4]
        ↓  SinusoidalStateEmbedding  → [B, 4 * (1 + 2*F)]
        ↓  Linear(in_features, D_m * 2) → SiLU
        ↓  Linear(D_m * 2, D_m) → LayerNorm
    [B, 1, D_m]   (single token ready for sequence concat)
"""

import math
import torch
import torch.nn as nn


class SinusoidalStateEmbedding(nn.Module):
    """
    NeRF-style sinusoidal embedding for scalar state values.

    Converts each element of a D-dim state vector into (1 + 2*F) features
    using logarithmically-spaced frequencies, improving MLP sensitivity
    to both fine-grained and large-scale displacements.

    Output dimension: D * (1 + 2 * num_freqs)
    """

    def __init__(self, state_dim: int, num_freqs: int = 8, max_freq_log2: int = 7):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, max_freq_log2, num_freqs)  # [F]
        self.register_buffer("freqs", freqs)
        self.state_dim  = state_dim
        self.num_freqs  = num_freqs
        self.out_dim    = state_dim * (1 + 2 * num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, state_dim]
        Returns:
            [B, state_dim * (1 + 2*num_freqs)]
        """
        x_exp   = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # [B, D, F]
        sin_f   = torch.sin(x_exp).reshape(x.size(0), -1)       # [B, D*F]
        cos_f   = torch.cos(x_exp).reshape(x.size(0), -1)       # [B, D*F]
        return torch.cat([x, sin_f, cos_f], dim=-1)             # [B, D*(1+2F)]


class ProprioEncoder(nn.Module):
    """
    Encodes the UAV proprioceptive state into a single Mamba-compatible token.

    Args:
        proprio_dim        : Input state dimension (default 4: Δx, Δy, Δz, Δyaw).
        mamba_hidden_size  : Target hidden dimension D_m.
        num_freqs          : Sinusoidal frequency bands (default 8).
        dropout            : Dropout probability (default 0.1).
    """

    def __init__(
        self,
        proprio_dim:       int   = 8,
        mamba_hidden_size: int   = 1024,
        num_freqs:         int   = 8,
        dropout:           float = 0.1,
    ):
        super().__init__()
        self.proprio_dim       = proprio_dim
        self.mamba_hidden_size = mamba_hidden_size

        self.sin_emb    = SinusoidalStateEmbedding(proprio_dim, num_freqs=num_freqs)
        in_features     = self.sin_emb.out_dim

        # 2-layer MLP: expanded → D_m, with SiLU activation and LayerNorm output
        self.mlp = nn.Sequential(
            nn.Linear(in_features,        mamba_hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mamba_hidden_size * 2, mamba_hidden_size),
            nn.LayerNorm(mamba_hidden_size),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio : [B, proprio_dim]  float32

        Returns:
            token   : [B, 1, mamba_hidden_size]  — single sequence token
        """
        embedded = self.sin_emb(proprio)   # [B, in_features]
        token    = self.mlp(embedded)      # [B, D_m]
        return token.unsqueeze(1)          # [B, 1, D_m]

    def forward_pair(
        self,
        state: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode current state and delta-state as two ordered tokens."""
        return torch.cat([self.forward(state), self.forward(delta_state)], dim=1)
