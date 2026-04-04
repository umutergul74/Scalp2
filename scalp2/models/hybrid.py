"""Hybrid TCN+GRU encoder — v4 with L2 normalization and cosine classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scalp2.config import ModelConfig
from scalp2.models.gru import GRUEncoder
from scalp2.models.tcn import TCNEncoder


class CosineClassifier(nn.Module):
    """Cosine similarity classifier — forces angular separation in latent space.

    Unlike nn.Linear which learns arbitrary weight directions, this classifier
    normalizes both the weight vectors (class prototypes) and the input features.
    The output is `scale * cos(angle between input and class prototype)`.

    This forces the latent space to organize ANGULARLY = directly creates
    the cluster separation we see in t-SNE.

    Used in face recognition (ArcFace, CosFace) where it's state-of-the-art.

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
        # Normalize weight vectors (class prototypes)
        w = F.normalize(self.weight, dim=1)
        # Normalize input (should already be normalized, but safety)
        x = F.normalize(x, dim=1)
        # Cosine similarity = dot product of normalized vectors
        return self.scale * F.linear(x, w)


class HybridEncoder(nn.Module):
    """Two-stage hybrid architecture: TCN + GRU → bottleneck → L2 norm → cosine classifier.

    v4 Architecture:
        Input (B, seq_len, n_features)
            ├─→ TCNEncoder + SE + SpatialDrop + StochDepth → (B, 64)
            └─→ GRUEncoder + Attention Pooling → (B, 128)
                        ↓
            Concatenate → (B, 192)
                        ↓
            Bottleneck: Linear(192, 32) → LN → GELU → Drop(0.4)  [compress]
            Expand:     Linear(32, 64)  → LN → GELU → Drop(0.4)  [expand]
            L2 Normalize → unit hypersphere                        [NEW]
                        ↓ latent (B, 64) — for XGBoost meta-learner
            CosineClassifier(64, 3, scale=16) → (B, 3) logits     [NEW]

    Key improvements over v2:
        - L2 normalization: all latent vectors on unit sphere → 
          distances become cosine similarity → t-SNE works dramatically better
        - Cosine classifier: forces angular class separation →
          classes MUST occupy different directions = cluster separation
        - Smaller latent (64 vs 128): less curse of dimensionality,
          cleaner XGBoost features, better t-SNE visualization

    Total parameters: ~290K (smaller than v2's 420K).
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

        # Information Bottleneck: compress → expand
        self.projection = nn.Sequential(
            # Compress: force noise removal (192 → 32)
            nn.Linear(fusion_input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
            # Expand: reconstruct meaningful representation (32 → 64)
            nn.Linear(bottleneck_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
        )

        # Cosine classifier instead of Linear — forces angular separation
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
        tcn_out = self.tcn(x)  # (B, 64)
        gru_out = self.gru(x)  # (B, 128)

        merged = torch.cat([tcn_out, gru_out], dim=1)  # (B, 192)
        raw_latent = self.projection(merged)  # (B, 64)

        # L2 normalize — constrains to unit hypersphere
        # This makes cosine similarity = dot product
        # Directly improves t-SNE visualization quality
        latent = F.normalize(raw_latent, dim=1)  # (B, 64) on unit sphere

        logits = self.classifier(latent)  # (B, 3)

        return logits, latent

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent vectors for the XGBoost meta-learner.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, latent_dim) — L2-normalized, detached
        """
        self.eval()
        _, latent = self.forward(x)
        return latent

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
