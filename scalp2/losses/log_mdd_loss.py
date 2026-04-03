"""Logarithmic Maximum Drawdown loss + combined loss with contrastive term."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMDDLoss(nn.Module):
    """Logarithmic Maximum Drawdown loss.

    Computes the cumulative PnL curve from predicted positions,
    finds the maximum drawdown, and returns log(1 + MDD).

    Combined loss strategy:
        total_loss = alpha * CE_loss + beta * SupCon_loss + (1 - alpha - beta) * Aux_loss

    Alpha is annealed from 1.0 → 0.5 over training to start with stable
    CE gradients and gradually introduce the finance-aware objective.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, forward_returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute log maximum drawdown loss.

        Args:
            logits: (batch, 3) — raw model outputs [short, hold, long]
            forward_returns: (batch,) — actual forward returns per bar

        Returns:
            Scalar loss: log(1 + max_drawdown)
        """
        # Soft position
        probs = F.softmax(logits, dim=1)
        position = probs[:, 2] - probs[:, 0]

        # Portfolio returns
        portfolio_returns = position * forward_returns

        # Cumulative PnL curve
        cum_returns = torch.cumsum(portfolio_returns, dim=0)

        # Running maximum
        running_max = torch.cummax(cum_returns, dim=0)[0]

        # Drawdown at each point
        drawdown = running_max - cum_returns

        # Maximum drawdown
        max_drawdown = drawdown.max()

        return torch.log1p(max_drawdown + self.eps)


def compute_combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    forward_returns: torch.Tensor,
    class_weights: torch.Tensor | None,
    alpha: float,
    auxiliary_loss_fn: nn.Module,
    contrastive_loss_fn: nn.Module | None = None,
    latent: torch.Tensor | None = None,
    contrastive_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined CE + contrastive + auxiliary finance loss.

    Loss formula:
        total = alpha * CE(label_smoothing) + beta * SupCon + (1-alpha-beta) * Aux

    Args:
        logits: (batch, 3)
        labels: (batch,) — class indices {0, 1, 2}
        forward_returns: (batch,) — actual forward returns
        class_weights: Optional class weights for CE loss
        alpha: CE weighting factor (1.0 = pure CE, 0.0 = pure auxiliary)
        auxiliary_loss_fn: SharpeLoss or LogMDDLoss instance
        contrastive_loss_fn: Optional SupConLoss instance
        latent: Optional (batch, latent_dim) for contrastive loss
        contrastive_weight: Beta — weight for contrastive loss
        label_smoothing: Label smoothing factor (0.0 = hard labels)

    Returns:
        total_loss: Scalar combined loss
        loss_components: Dict with individual loss values for logging
    """
    # CE loss with label smoothing
    ce_loss = F.cross_entropy(
        logits, labels, weight=class_weights, label_smoothing=label_smoothing
    )

    # Auxiliary finance loss (Sharpe or LogMDD)
    aux_loss = auxiliary_loss_fn(logits, forward_returns)

    # Contrastive loss on latent space
    con_loss = torch.tensor(0.0, device=logits.device)
    if contrastive_loss_fn is not None and latent is not None and contrastive_weight > 0:
        con_loss = contrastive_loss_fn(latent, labels)

    # Combined: alpha * CE + beta * SupCon + (1 - alpha - beta) * Aux
    beta = contrastive_weight
    aux_weight = max(0.0, 1.0 - alpha - beta)
    total = alpha * ce_loss + beta * con_loss + aux_weight * aux_loss

    components = {
        "ce_loss": ce_loss.item(),
        "aux_loss": aux_loss.item(),
        "con_loss": con_loss.item(),
        "total_loss": total.item(),
        "alpha": alpha,
        "beta": beta,
    }

    return total, components
