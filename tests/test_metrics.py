from __future__ import annotations

import numpy as np

from scalp2.utils.metrics import (
    drawdown_series_from_equity,
    max_drawdown,
    max_drawdown_from_equity,
)


def test_max_drawdown_uses_compounded_equity():
    returns = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)

    assert np.isclose(max_drawdown(returns), 0.625)


def test_max_drawdown_from_equity_is_peak_relative():
    equity = np.array([150.0, 75.0, 112.5, 56.25], dtype=np.float64)

    assert np.isclose(
        max_drawdown_from_equity(equity, initial_equity=100.0),
        0.625,
    )


def test_drawdown_series_can_seed_initial_equity():
    equity = np.array([90.0, 95.0, 80.0], dtype=np.float64)

    drawdown = drawdown_series_from_equity(equity, initial_equity=100.0)

    assert np.allclose(drawdown, np.array([0.10, 0.05, 0.20], dtype=np.float64))
