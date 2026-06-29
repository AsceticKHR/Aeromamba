"""
Lightweight visual token resamplers for AeroMamba.

The goal is to keep rich frozen vision features while sending far fewer tokens
into the Mamba backbone. This follows the efficient VLA trend of using learned
queries / resamplers rather than passing every ViT patch token downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverResamplerLayer(nn.Module):
    """One cross-attention + latent self-attention block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        inner = int(hidden_size * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.latent_norm(latents)
        key_value = self.context_norm(context)
        latents = latents + self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )[0]
        latents = latents + self.self_attn(
            query=self.self_norm(latents),
            key=self.self_norm(latents),
            value=self.self_norm(latents),
            need_weights=False,
        )[0]
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class PerceiverResampler(nn.Module):
    """
    Compress visual patch tokens into a small fixed set of learned queries.

    Input:
        vision_tokens: [B, N, D]
    Output:
        latents:       [B, Q, D]
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        batch = vision_tokens.size(0)
        latents = self.query.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            latents = layer(latents, vision_tokens)
        return self.out_norm(latents)
