"""Supervised Contrastive Loss — force latent space to form meaningful clusters.

Based on: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020).

This loss directly addresses the noise memorization problem by:
    - Pulling same-class latent vectors together (Long ↔ Long)
    - Pushing different-class vectors apart (Long ↔ Short)

Without this, the latent space becomes a noise cloud (as seen in t-SNE).
With this, we enforce island-like clusters that carry real signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Given a batch of latent vectors and their labels, computes the
    InfoNCE-style contrastive loss where positives are samples
    from the same class and negatives are from different classes.

    Args:
        temperature: Scaling factor for cosine similarity. Lower values
            make the loss more sensitive to hard negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, latent: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            latent: (batch, latent_dim) — L2-normalized feature vectors.
            labels: (batch,) — class labels {0, 1, 2}.

        Returns:
            Scalar contrastive loss.
        """
        device = latent.device
        batch_size = latent.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize
        features = F.normalize(latent, dim=1)

        # Cosine similarity matrix: (B, B)
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask: same-class pairs are positives (excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-contrast (diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask = mask.masked_fill(self_mask, 0)

        # Count positives per anchor (at least 1 to avoid division by zero)
        positives_count = mask.sum(dim=1).clamp(min=1)

        # For numerical stability, subtract max
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Mask out self-similarity
        logits = logits.masked_fill(self_mask, float('-inf'))

        # Log-sum-exp of all non-self entries (denominator)
        exp_logits = torch.exp(logits)
        log_sum_exp = torch.log(exp_logits.sum(dim=1).clamp(min=1e-8))

        # Mean of log-probabilities over positive pairs
        # Use where() to avoid -inf * 0 = nan from masked positions
        safe_logits = torch.where(mask.bool(), logits, torch.zeros_like(logits))
        positive_logits = safe_logits.sum(dim=1) / positives_count
        loss = -positive_logits + log_sum_exp

        # Only compute loss for anchors that have at least one positive
        has_positives = mask.sum(dim=1) > 0
        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss[has_positives].mean()
