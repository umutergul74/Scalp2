"""Temporal Attention modules for sequence fusion.

Multi-Head Self-Attention applied to the concatenated TCN+GRU temporal
sequences, followed by learned attention pooling. This replaces the naive
"take last timestep + concat" fusion with a richer, context-aware mechanism.

Key insight: TCN's last timestep and GRU's attention pool independently
lose cross-encoder temporal interactions. By keeping full sequences and
applying joint self-attention, the model can learn which timesteps from
TCN align with which timesteps from GRU, and which are most informative.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadTemporalAttention(nn.Module):
    """Multi-Head Self-Attention over temporal sequences.

    Applies standard scaled dot-product attention with multiple heads.
    Uses pre-LayerNorm for training stability.

    Args:
        d_model: Dimensionality of input features.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) — attention-enhanced sequence
        """
        B, T, D = x.shape

        # Pre-norm + residual connection
        residual = x
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, heads, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.out_proj(out)

        return residual + self.dropout(out)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int = 0, dropout: float = 0.1):
        super().__init__()
        if d_ff == 0:
            d_ff = d_model * 2  # Smaller expansion for efficiency
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TemporalTransformerBlock(nn.Module):
    """Single transformer block: Self-Attention + FFN with residual connections."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadTemporalAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class LearnedPooling(nn.Module):
    """Learned attention-based pooling: (B, T, D) → (B, D).

    Uses a small MLP to compute importance scores per timestep,
    then takes a weighted average. This is better than mean/max pooling
    because it learns which timesteps carry the most signal.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to single vector.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, d_model) — attention-weighted pooled vector
        """
        scores = self.score(x)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


class TemporalFusionAttention(nn.Module):
    """Full temporal fusion module: Self-Attention → FFN → Pooling.

    Takes concatenated TCN+GRU temporal sequences and produces a single
    context-aware representation vector.

    Args:
        d_model: Combined feature dimension (TCN_dim + GRU_dim).
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.pool = LearnedPooling(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention fusion and pool.

        Args:
            x: (batch, seq_len, d_model) — concatenated TCN+GRU sequences
        Returns:
            (batch, d_model) — single fused representation
        """
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.pool(x)
