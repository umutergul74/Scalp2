"""Hybrid TCN+GRU encoder — parallel feature extraction with information bottleneck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scalp2.config import ModelConfig
from scalp2.models.gru import GRUEncoder
from scalp2.models.tcn import TCNEncoder


class HybridEncoder(nn.Module):
    """Two-stage hybrid architecture: parallel TCN + GRU → bottleneck → classifier.

    Architecture:
        Input (B, seq_len, n_features)
            ├─→ TCNEncoder → (B, 64)     [local patterns, chart formations]
            └─→ GRUEncoder → (B, 128)    [temporal dependencies, market memory]
                        ↓
            Concatenate → (B, 192)
                        ↓
            Information Bottleneck:
            Linear(192, 64) → LayerNorm → GELU → Dropout(0.4)     [compress]
            Linear(64, 128)  → LayerNorm → GELU → Dropout(0.4)     [expand]
                        ↓ (latent vector for XGBoost meta-learner)
            Linear(128, 3) → (B, 3) logits [short, hold, long]

    The bottleneck (192→64→128) forces the model to discard noise and
    keep only the most informative patterns. Without it, the model
    can pass noise straight through to the latent space.

    Total parameters: ~330K (deliberately small to prevent overfitting).
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
            # Compress: force noise removal
            nn.Linear(fusion_input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
            # Expand: reconstruct meaningful representation
            nn.Linear(bottleneck_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
        )

        self.classifier = nn.Linear(latent_dim, 3)
        self.latent_dim = latent_dim

        # Contrastive projection head (SupCon paper: separate MLP for contrastive loss)
        # This decouples cluster formation from classification — critical for clean t-SNE
        self.contrastive_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 32),
        )

    def contrastive_project(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent vectors into contrastive space with L2 normalization.

        Args:
            latent: (batch, latent_dim)
        Returns:
            (batch, 32) — L2-normalized projection for SupCon loss
        """
        projected = self.contrastive_head(latent)
        return F.normalize(projected, dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and latent representation.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            logits: (batch, 3) — class logits [short, hold, long]
            latent: (batch, latent_dim) — for XGBoost meta-learner
        """
        tcn_out = self.tcn(x)  # (B, 64)
        gru_out = self.gru(x)  # (B, 128)

        merged = torch.cat([tcn_out, gru_out], dim=1)  # (B, 192)
        latent = self.projection(merged)  # (B, 128)
        logits = self.classifier(latent)  # (B, 3)

        return logits, latent

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent vectors for the XGBoost meta-learner.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, latent_dim) — detached numpy-ready tensor
        """
        self.eval()
        _, latent = self.forward(x)
        return latent

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
