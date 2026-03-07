"""Hybrid TCN+GRU encoder — parallel feature extraction with fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from scalp2.config import ModelConfig
from scalp2.models.gru import GRUEncoder
from scalp2.models.tcn import TCNEncoder


class HybridEncoder(nn.Module):
    """Two-stage hybrid architecture: parallel TCN + GRU → fusion → classifier.

    Architecture:
        Input (B, seq_len, n_features)
            ├─→ TCNEncoder → (B, 64)     [local patterns, chart formations]
            └─→ GRUEncoder → (B, 128)    [temporal dependencies, market memory]
                        ↓
            Concatenate → (B, 192)
                        ↓
            Linear(192, 128) → LayerNorm → ReLU → Dropout(0.3) → (B, 128)
                        ↓ (latent vector for XGBoost meta-learner)
            Linear(128, 3) → (B, 3) logits [short, hold, long]

    Total parameters: ~315K (deliberately small to prevent overfitting).
    """

    def __init__(self, n_features: int, config: ModelConfig):
        super().__init__()

        self.tcn = TCNEncoder(
            input_channels=n_features,
            num_channels=config.tcn.num_channels,
            kernel_size=config.tcn.kernel_size,
            dropout=config.tcn.dropout,
        )

        self.gru = GRUEncoder(
            input_size=n_features,
            hidden_size=config.gru.hidden_size,
            num_layers=config.gru.num_layers,
            dropout=config.gru.dropout,
            bidirectional=config.gru.bidirectional,
        )

        fusion_input_dim = self.tcn.output_dim + self.gru.output_dim

        self.projection = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion.latent_dim),
            nn.LayerNorm(config.fusion.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion.dropout),
        )

        self.classifier = nn.Linear(config.fusion.latent_dim, 3)
        self.latent_dim = config.fusion.latent_dim

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
