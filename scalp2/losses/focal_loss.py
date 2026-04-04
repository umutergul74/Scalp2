"""Focal Loss — down-weight easy examples, focus on hard ones.

Standard CE treats all examples equally, which means the dominant Hold
class gets most of the gradient budget. Focal Loss fixes this by
multiplying each example's loss by (1-p)^gamma, where p is the model's
confidence in the correct class.

Easy examples (p ≈ 1.0): weight → 0, nearly ignored.
Hard examples (p ≈ 0.3): weight → (0.7)^2 = 0.49, gets full attention.

This shifts training focus to Long/Short boundary cases = higher F1.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing and class weights.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
            gamma=0 is equivalent to standard CE.
            gamma=2.0 is the standard value from the paper.
        label_smoothing: Smoothing factor for soft targets.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (batch, num_classes) — raw model outputs.
            labels: (batch,) — class indices.
            weight: Optional (num_classes,) class weights.

        Returns:
            Scalar focal loss.
        """
        # Standard CE per sample (unreduced)
        ce_loss = F.cross_entropy(
            logits, labels, weight=weight,
            label_smoothing=self.label_smoothing, reduction="none"
        )

        # Probability of the correct class
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Focal modulation: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Weighted loss
        loss = focal_weight * ce_loss

        return loss.mean()
