"""Differentiable Sharpe Ratio loss for finance-grounded optimization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharpeLoss(nn.Module):
    """Differentiable Sharpe Ratio loss.

    Converts model logits to soft position weights via softmax,
    computes portfolio returns, and returns the negative Sharpe ratio
    as the loss (minimizing loss = maximizing Sharpe).

    Position mapping:
        position = P(long) - P(short)  (P(hold) contributes 0)

    Portfolio return per bar:
        r_portfolio = position * forward_return

    Loss:
        loss = -mean(r_portfolio) / (std(r_portfolio) + eps)
    """

    def __init__(self, eps: float = 1e-8, annualization: float = 1.0):
        super().__init__()
        self.eps = eps
        self.annualization = annualization

    def forward(
        self, logits: torch.Tensor, forward_returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative Sharpe ratio loss.

        Args:
            logits: (batch, 3) — raw model outputs [short, hold, long]
            forward_returns: (batch,) — actual forward returns per bar

        Returns:
            Scalar loss (negative Sharpe ratio).
        """
        # Soft position: p(long) - p(short)
        probs = F.softmax(logits, dim=1)
        position = probs[:, 2] - probs[:, 0]  # long - short

        # Portfolio returns
        portfolio_returns = position * forward_returns

        # Sharpe ratio
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std() + self.eps

        sharpe = mean_ret / std_ret * self.annualization

        return -sharpe
