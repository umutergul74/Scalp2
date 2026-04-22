"""Differentiable Rank IC Loss — directly optimize Information Coefficient.

This loss maximizes the Pearson correlation between the model's directional
score (P(Long) - P(Short)) and the actual forward returns. Since we evaluate
models using Spearman IC, and Pearson on softmax outputs approximates Spearman
well, this loss directly aligns training with our evaluation metric.

Key insight: Focal Loss optimizes classification accuracy, but accuracy != IC.
A model can have high accuracy but low IC (e.g., predicting "hold" for everything).
RankIC Loss forces the model to produce scores that RANK bars correctly by
their future return potential.

Reference:
    - "Learning to Rank for Information Retrieval" (Li, 2011)
    - "Deep Learning for Quantitative Finance" (Lim et al., 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankICLoss(nn.Module):
    """Differentiable IC loss using Pearson correlation as a smooth proxy.

    Computes: loss = -corr(score, forward_returns)

    Where score = P(Long) - P(Short) from the model's softmax output.

    The negative sign means minimizing this loss = maximizing IC.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, forward_returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative Pearson correlation between score and returns.

        Args:
            logits: (batch, 3) — raw model outputs [short, hold, long]
            forward_returns: (batch,) — actual forward returns per bar

        Returns:
            Scalar loss: -correlation (lower = better IC)
        """
        if logits.shape[0] < 4:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Directional score: P(Long) - P(Short)
        probs = F.softmax(logits, dim=1)
        score = probs[:, 2] - probs[:, 0]

        # Center both signals
        score_c = score - score.mean()
        ret_c = forward_returns - forward_returns.mean()

        # Pearson correlation
        cov = (score_c * ret_c).mean()
        std_s = score_c.std().clamp(min=self.eps)
        std_r = ret_c.std().clamp(min=self.eps)

        corr = cov / (std_s * std_r)

        # Maximize correlation = minimize -correlation
        return -corr


class PairwiseRankLoss(nn.Module):
    """Pairwise ranking loss — directly enforces correct ordering.

    For each pair of bars (i, j) where return_i > return_j,
    the model should assign score_i > score_j.

    This is more robust than Pearson correlation for non-linear
    relationships and handles outliers better.

    Uses a margin-based hinge loss:
        loss = max(0, margin - (score_i - score_j))  when ret_i > ret_j

    Args:
        margin: Minimum desired score gap between correctly ordered pairs.
        n_pairs: Number of random pairs to sample per batch (for efficiency).
    """

    def __init__(self, margin: float = 0.1, n_pairs: int = 256):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs

    def forward(
        self, logits: torch.Tensor, forward_returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Args:
            logits: (batch, 3) — raw model outputs
            forward_returns: (batch,) — actual forward returns

        Returns:
            Scalar pairwise ranking loss.
        """
        batch_size = logits.shape[0]
        if batch_size < 4:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = F.softmax(logits, dim=1)
        score = probs[:, 2] - probs[:, 0]

        # Sample random pairs
        n = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        idx_i = torch.randint(0, batch_size, (n,), device=logits.device)
        idx_j = torch.randint(0, batch_size, (n,), device=logits.device)

        # Avoid self-pairs
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Return differences and score differences
        ret_diff = forward_returns[idx_i] - forward_returns[idx_j]
        score_diff = score[idx_i] - score[idx_j]

        # Sign: +1 if ret_i > ret_j, -1 if ret_i < ret_j
        target = torch.sign(ret_diff)

        # Filter out pairs with identical returns
        nonzero = target.abs() > 0
        if nonzero.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target = target[nonzero]
        score_diff = score_diff[nonzero]

        # Margin ranking loss: max(0, margin - target * score_diff)
        loss = F.relu(self.margin - target * score_diff)

        return loss.mean()
