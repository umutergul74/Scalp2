"""Center Loss — pull each vector toward its class centroid.

Maintains a learnable centroid per class. Each training step:
1. Compute distance between each latent vector and its class centroid.
2. Update centroids with exponential moving average.

This creates tight, circular clusters in latent space — exactly what
we need for clean t-SNE visualization.

Works alongside SupCon:
    - SupCon pushes different classes apart (inter-class separation)
    - Center Loss pulls same class together (intra-class compactness)

Reference: Wen et al., "A Discriminative Feature Learning Approach
for Deep Face Recognition" (ECCV 2016).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center Loss with learnable class centroids.

    Args:
        num_classes: Number of classes (3 for short/hold/long).
        latent_dim: Dimensionality of the latent space.
        alpha: Learning rate for centroid updates (EMA decay).
            Higher = centroids move faster. 0.5 is standard.
    """

    def __init__(
        self,
        num_classes: int = 3,
        latent_dim: int = 64,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        # Learnable centroids — initialized to zero, will adapt during training
        self.centers = nn.Parameter(
            torch.randn(num_classes, latent_dim) * 0.01,
            requires_grad=False,  # Updated via EMA, not backprop
        )

    @torch.no_grad()
    def _update_centers(
        self, latent: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Update centroids with exponential moving average.

        Args:
            latent: (batch, latent_dim) — detached latent vectors.
            labels: (batch,) — class labels.
        """
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            class_mean = latent[mask].mean(dim=0)
            self.centers[c] = (
                (1 - self.alpha) * self.centers[c] + self.alpha * class_mean
            )

    def forward(
        self, latent: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute center loss and update centroids.

        Args:
            latent: (batch, latent_dim) — L2-normalized feature vectors.
            labels: (batch,) — class labels {0, 1, 2}.

        Returns:
            Scalar center loss: mean squared distance to class centroids.
        """
        # Get centroid for each sample's class
        batch_centers = self.centers[labels]  # (B, latent_dim)

        # Squared L2 distance to centroid
        loss = ((latent - batch_centers) ** 2).sum(dim=1).mean()

        # Update centroids (EMA, no gradient)
        self._update_centers(latent.detach(), labels)

        return loss
