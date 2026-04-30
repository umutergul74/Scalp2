"""Logarithmic Maximum Drawdown loss + combined loss with Focal, SupCon, Center terms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scalp2.losses.focal_loss import FocalLoss


class LogMDDLoss(nn.Module):
    """Logarithmic Maximum Drawdown loss.

    Computes the cumulative PnL curve from predicted positions,
    finds the maximum drawdown, and returns log(1 + MDD).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, forward_returns: torch.Tensor,
        rt_cost: float = 0.0,
    ) -> torch.Tensor:
        """Compute log maximum drawdown loss.

        Args:
            logits: (batch, 3) — raw model outputs [short, hold, long]
            forward_returns: (batch,) — actual forward returns per bar
            rt_cost: Round-trip cost as decimal (e.g. 0.0008 for 8bps)

        Returns:
            Scalar loss: log(1 + max_drawdown)
        """
        probs = F.softmax(logits, dim=1)
        position = probs[:, 2] - probs[:, 0]
        # Cost-aware: penalize every unit of position taken
        cost_penalty = position.abs() * rt_cost
        portfolio_returns = position * forward_returns - cost_penalty
        cum_returns = torch.cumsum(portfolio_returns, dim=0)
        running_max = torch.cummax(cum_returns, dim=0)[0]
        drawdown = running_max - cum_returns
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
    center_loss_fn: nn.Module | None = None,
    rank_ic_loss_fn: nn.Module | None = None,
    latent: torch.Tensor | None = None,
    contrastive_weight: float = 0.0,
    center_loss_weight: float = 0.0,
    rank_ic_weight: float = 0.0,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    rt_cost: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined Focal/CE + SupCon + RankIC + auxiliary finance loss.

    v5 Loss formula:
        total = α × Focal(label_smoothing)     # Classification (hard example focus)
              + β × SupCon(temp=0.10)           # Push classes apart
              + γ × CenterLoss                  # Pull each class to its centroid
              + δ × RankIC                      # Directly optimize IC (NEW)
              + (1-α-β-γ-δ) × SharpeLoss       # Financial awareness

    Args:
        logits: (batch, 3)
        labels: (batch,) — class indices {0, 1, 2}
        forward_returns: (batch,) — actual forward returns
        class_weights: Optional class weights
        alpha: Classification loss weight (annealed)
        auxiliary_loss_fn: SharpeLoss or LogMDDLoss instance
        contrastive_loss_fn: Optional SupConLoss instance
        center_loss_fn: Optional CenterLoss instance
        rank_ic_loss_fn: Optional RankICLoss instance
        latent: Optional (batch, latent_dim) for contrastive/center loss
        contrastive_weight: β — SupCon weight
        center_loss_weight: γ — Center Loss weight
        rank_ic_weight: δ — RankIC weight (directly optimizes IC)
        label_smoothing: Label smoothing factor
        focal_gamma: Focal Loss gamma (0.0 = standard CE)

    Returns:
        total_loss: Scalar combined loss
        loss_components: Dict with individual loss values for logging
    """
    # Classification loss: Focal Loss (if gamma > 0) or CE
    if focal_gamma > 0:
        focal_fn = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        cls_loss = focal_fn(logits, labels, weight=class_weights)
    else:
        cls_loss = F.cross_entropy(
            logits, labels, weight=class_weights,
            label_smoothing=label_smoothing,
        )

    # Auxiliary finance loss (Sharpe or LogMDD) — cost-aware
    aux_loss = auxiliary_loss_fn(logits, forward_returns, rt_cost=rt_cost)

    # Contrastive loss on latent space
    con_loss = torch.tensor(0.0, device=logits.device)
    if contrastive_loss_fn is not None and latent is not None and contrastive_weight > 0:
        con_loss = contrastive_loss_fn(latent, labels)

    # Center loss on latent space
    cen_loss = torch.tensor(0.0, device=logits.device)
    if center_loss_fn is not None and latent is not None and center_loss_weight > 0:
        cen_loss = center_loss_fn(latent, labels)

    # RankIC loss — directly maximize IC
    ric_loss = torch.tensor(0.0, device=logits.device)
    if rank_ic_loss_fn is not None and rank_ic_weight > 0:
        ric_loss = rank_ic_loss_fn(logits, forward_returns)

    # Combined: α*Focal + β*SupCon + γ*Center + δ*RankIC + (1-α-β-γ-δ)*Aux
    beta = contrastive_weight
    gamma = center_loss_weight
    delta = rank_ic_weight
    aux_weight = max(0.0, 1.0 - alpha - beta - gamma - delta)

    total = (
        alpha * cls_loss
        + beta * con_loss
        + gamma * cen_loss
        + delta * ric_loss
        + aux_weight * aux_loss
    )

    components = {
        "cls_loss": cls_loss.item(),
        "aux_loss": aux_loss.item(),
        "con_loss": con_loss.item(),
        "cen_loss": cen_loss.item(),
        "ric_loss": ric_loss.item(),
        "total_loss": total.item(),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }

    return total, components
