"""Performance metrics for backtesting and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 35040) -> float:
    """Annualized Sharpe ratio (assumes 15m bars: 96*365 = 35040/year)."""
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 35040) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 35040) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd < 1e-10:
        return 0.0
    annual_return = np.mean(returns) * periods_per_year
    return float(annual_return / mdd)


def win_rate(returns: np.ndarray) -> float:
    """Fraction of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def expectancy(returns: np.ndarray) -> float:
    """Average R-multiple per trade."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))


def evaluate_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    confidence_threshold: float = 0.70,
) -> dict:
    """Evaluate model predictions with comprehensive metrics.

    Args:
        predictions: (n, 3) probability matrix.
        labels: (n,) true labels {0, 1, 2}.
        returns: (n,) actual forward returns.
        confidence_threshold: Minimum confidence to count as trade.

    Returns:
        Dict of performance metrics.
    """
    # Filter by confidence
    max_probs = predictions.max(axis=1)
    confident_mask = max_probs >= confidence_threshold
    pred_classes = predictions.argmax(axis=1)

    # Only evaluate on actionable signals (not hold=1)
    trade_mask = confident_mask & (pred_classes != 1)
    n_trades = trade_mask.sum()

    if n_trades == 0:
        return {"n_trades": 0, "message": "No trades generated above threshold"}

    # Trade returns: long → positive return, short → negative return
    trade_returns = np.zeros(n_trades)
    trade_labels = labels[trade_mask]
    trade_preds = pred_classes[trade_mask]
    trade_actual_returns = returns[trade_mask]

    for i in range(n_trades):
        if trade_preds[i] == 2:  # Long
            trade_returns[i] = trade_actual_returns[i]
        elif trade_preds[i] == 0:  # Short
            trade_returns[i] = -trade_actual_returns[i]

    # Accuracy on traded signals
    accuracy = float(np.mean(trade_preds == trade_labels))

    return {
        "n_trades": int(n_trades),
        "accuracy": accuracy,
        "sharpe": sharpe_ratio(trade_returns),
        "sortino": sortino_ratio(trade_returns),
        "max_drawdown": max_drawdown(trade_returns),
        "calmar": calmar_ratio(trade_returns),
        "win_rate": win_rate(trade_returns),
        "profit_factor": profit_factor(trade_returns),
        "expectancy": expectancy(trade_returns),
        "avg_confidence": float(max_probs[trade_mask].mean()),
        "trades_per_day": float(n_trades / max(len(returns) / 96, 1)),
    }
