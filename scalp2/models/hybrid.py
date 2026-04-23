"""Hybrid TCN+GRU encoder — v5 with Temporal Fusion Attention.

Major upgrade from v4: instead of independently pooling TCN and GRU outputs
then concatenating, we keep full temporal sequences from both encoders,
concatenate at each timestep, and apply Multi-Head Self-Attention to learn
cross-encoder temporal interactions before pooling.

This allows the model to learn:
- Which TCN patterns align with which GRU states
- Which specific timesteps carry the most predictive signal
- Complex temporal dependencies across both encoder streams
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scalp2.config import ModelConfig
from scalp2.models.attention import TemporalFusionAttention
from scalp2.models.gru import GRUEncoder
from scalp2.models.tcn import TCNEncoder


class CosineClassifier(nn.Module):
    """Cosine similarity classifier — forces angular separation in latent space.

    Unlike nn.Linear which learns arbitrary weight directions, this classifier
    normalizes both the weight vectors (class prototypes) and the input features.
    The output is `scale * cos(angle between input and class prototype)`.

    Args:
        in_features: Dimensionality of input features.
        num_classes: Number of classes.
        scale: Temperature scaling factor. Higher = sharper softmax.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) — L2-normalized features.
        Returns:
            (batch, num_classes) — scaled cosine similarities.
        """
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        return self.scale * F.linear(x, w)


class HybridEncoder(nn.Module):
    """Two-stage hybrid architecture with Temporal Fusion Attention.

    v5 Architecture:
        Input (B, seq_len, n_features)
            ├─→ TCNEncoder  → (B, seq_len, 128)  [full sequence]
            └─→ GRUEncoder  → (B, seq_len, 192)  [full sequence]
                        ↓
            Concatenate at each timestep → (B, seq_len, 320)
                        ↓
            TemporalFusionAttention:
                2× [MultiHeadSelfAttention(4 heads) + FFN]
                → LearnedPooling → (B, 320)
                        ↓
            Bottleneck: Linear(320, 48) → LN → GELU → Drop(0.4)
            Expand:     Linear(48, 96)  → LN → GELU → Drop(0.4)
            L2 Normalize → unit hypersphere
                        ↓ latent (B, 96)
            CosineClassifier(96, 3, scale=16) → (B, 3) logits

    Key improvements over v4:
        - Temporal Fusion Attention: joint attention over TCN+GRU sequences
          instead of naive concat of independently pooled vectors
        - Richer cross-encoder interactions at every timestep
        - Learned importance-weighted pooling replaces fixed last-timestep/attn-pool
    """

    def __init__(self, n_features: int, config: ModelConfig):
        super().__init__()

        self.tcn = TCNEncoder(
            input_channels=n_features,
            num_channels=config.tcn.num_channels,
            kernel_size=config.tcn.kernel_size,
            dropout=config.tcn.dropout,
            spatial_dropout=config.tcn.spatial_dropout,
            squeeze_excite=config.tcn.squeeze_excite,
            stochastic_depth=config.tcn.stochastic_depth,
        )

        self.gru = GRUEncoder(
            input_size=n_features,
            hidden_size=config.gru.hidden_size,
            num_layers=config.gru.num_layers,
            dropout=config.gru.dropout,
            bidirectional=config.gru.bidirectional,
            attention_pooling=config.gru.attention_pooling,
        )

        fusion_input_dim = self.tcn.output_dim + self.gru.output_dim
        bottleneck_dim = config.fusion.bottleneck_dim
        latent_dim = config.fusion.latent_dim

        # Check if attention fusion is enabled (default: True for v5)
        use_attn_fusion = getattr(config.fusion, 'use_attention', True)

        if use_attn_fusion:
            # v5: Temporal Fusion Attention
            n_heads = getattr(config.fusion, 'n_heads', 4)
            n_attn_layers = getattr(config.fusion, 'n_attn_layers', 2)
            attn_dropout = getattr(config.fusion, 'attn_dropout', 0.1)
            self.fusion_attention = TemporalFusionAttention(
                d_model=fusion_input_dim,
                n_heads=n_heads,
                n_layers=n_attn_layers,
                dropout=attn_dropout,
            )
            self._use_attn_fusion = True
        else:
            # v4 fallback: simple concat of pooled vectors
            self.fusion_attention = None
            self._use_attn_fusion = False

        # Information Bottleneck: compress → expand
        self.projection = nn.Sequential(
            nn.Linear(fusion_input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
            nn.Linear(bottleneck_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
        )

        # Cosine classifier
        self.classifier = CosineClassifier(latent_dim, 3, scale=16.0)
        self.latent_dim = latent_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and L2-normalized latent representation.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            logits: (batch, 3) — scaled cosine similarities
            latent: (batch, latent_dim) — L2-normalized, on unit hypersphere
        """
        if self._use_attn_fusion:
            # v5: get full sequences, fuse with attention
            tcn_seq = self.tcn(x, return_sequence=True)  # (B, T, tcn_dim)
            gru_seq = self.gru(x, return_sequence=True)  # (B, T, gru_dim)
            fused_seq = torch.cat([tcn_seq, gru_seq], dim=2)  # (B, T, total_dim)
            merged = self.fusion_attention(fused_seq)  # (B, total_dim)
        else:
            # v4 fallback: independent pooling + concat
            tcn_out = self.tcn(x)  # (B, tcn_dim)
            gru_out = self.gru(x)  # (B, gru_dim)
            merged = torch.cat([tcn_out, gru_out], dim=1)  # (B, total_dim)

        raw_latent = self.projection(merged)  # (B, latent_dim)
        latent = F.normalize(raw_latent, dim=1)  # (B, latent_dim) on unit sphere
        logits = self.classifier(latent)  # (B, 3)

        return logits, latent

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent vectors for the XGBoost meta-learner."""
        self.eval()
        _, latent = self.forward(x)
        return latent

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
