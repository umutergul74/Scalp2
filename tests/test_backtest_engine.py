"""Tests for notebook-aligned backtest helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.analysis.backtest_engine import (
    build_signal_audit_table,
    simulate_forward_backtest,
    simulate_walk_forward_backtest,
)
from scalp2.config import load_config
from scalp2.execution.strategy_logic import plan_trade_from_probabilities


def _base_config():
    config = load_config("config.yaml")
    config.execution.time_of_day_filter.enabled = False
    config.execution.regime_direction_filter = False
    config.execution.min_adx = 0.0
    config.execution.min_atr_percentile = 0.0
    config.regime.choppy_threshold = 0.95
    return config


def _market_frame(length: int) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=length, freq="15min")
    base = np.full(length, 100.0, dtype=np.float32)
    return pd.DataFrame(
        {
            "open": base,
            "high": base,
            "low": base,
            "close": base,
            "atr_14": np.full(length, 1.0, dtype=np.float32),
            "adx": np.full(length, 40.0, dtype=np.float32),
        },
        index=index,
    )


def test_plan_trade_skips_hold_and_zero_atr():
    config = _base_config()
    now = pd.Timestamp("2026-01-01 00:00:00")

    hold_trade = plan_trade_from_probabilities(
        config=config,
        probs=np.array([0.30, 0.55, 0.15], dtype=np.float32),
        current_regime="bull",
        choppy_prob=0.0,
        current_atr=1.0,
        current_price=100.0,
        current_time=now,
    )
    assert hold_trade.direction is None
    assert hold_trade.reason == "hold"

    no_atr_trade = plan_trade_from_probabilities(
        config=config,
        probs=np.array([0.05, 0.05, 0.90], dtype=np.float32),
        current_regime="bull",
        choppy_prob=0.0,
        current_atr=0.0,
        current_price=100.0,
        current_time=now,
    )
    assert no_atr_trade.direction is None
    assert no_atr_trade.reason == "no_atr"


def test_forward_backtest_small_position_skip_does_not_consume_daily_limit():
    config = _base_config()
    config.execution.max_trades_per_day = 1
    df = _market_frame(3)
    probs = np.array(
        [
            [0.01, 0.00, 0.99],
            [0.01, 0.00, 0.99],
            [0.05, 0.90, 0.05],
        ],
        dtype=np.float32,
    )
    regime_probs = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))

    result = simulate_forward_backtest(
        df=df,
        probs=probs,
        regime_probs=regime_probs,
        config=config,
        initial_balance=1.0,
    )

    assert result["skip_reasons"].get("position_too_small") == 2
    assert result["skip_reasons"].get("daily_limit", 0) == 0


def test_walk_forward_backtest_can_finish_trade_after_last_prediction_bar():
    config = _base_config()
    config.model.seq_len = 1
    config.labeling.max_holding_bars = 2
    df = _market_frame(6)
    wf_predictions = [
        {
            "fold_idx": 0,
            "test_start": 0,
            "test_probabilities": np.array([[0.01, 0.00, 0.99]], dtype=np.float32),
            "regime_probs": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        }
    ]

    result = simulate_walk_forward_backtest(
        df=df,
        wf_predictions=wf_predictions,
        config=config,
        initial_balance=1000.0,
    )

    assert len(result["trades_df"]) == 1
    trade = result["trades_df"].iloc[0]
    assert int(trade["entry_bar"]) == 2
    assert int(trade["exit_bar"]) >= 3
    assert trade["status"] == "TIME"


def test_forward_backtest_closes_last_open_trade_at_end_of_data():
    config = _base_config()
    config.labeling.max_holding_bars = 10
    df = _market_frame(3)
    probs = np.array(
        [
            [0.01, 0.00, 0.99],
            [0.10, 0.80, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=np.float32,
    )
    regime_probs = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))

    result = simulate_forward_backtest(
        df=df,
        probs=probs,
        regime_probs=regime_probs,
        config=config,
        initial_balance=1000.0,
    )

    assert len(result["trades_df"]) == 1
    trade = result["trades_df"].iloc[0]
    assert trade["status"] == "EOD"
    assert int(trade["entry_bar"]) == 1
    assert int(trade["exit_bar"]) == 2
    assert trade["balance_after"] < 1000.0


def test_signal_audit_table_matches_sequential_trade_acceptance():
    config = _base_config()
    config.labeling.max_holding_bars = 2
    df = _market_frame(6)
    probs = np.array(
        [
            [0.01, 0.00, 0.99],
            [0.01, 0.00, 0.99],
            [0.01, 0.00, 0.99],
            [0.10, 0.80, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=np.float32,
    )
    regime_probs = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(df), 1))

    audit_df = build_signal_audit_table(
        df=df,
        probs=probs,
        regime_probs=regime_probs,
        config=config,
    )
    backtest_result = simulate_forward_backtest(
        df=df,
        probs=probs,
        regime_probs=regime_probs,
        config=config,
    )
    backtest_statuses = list(backtest_result["trades_df"]["status"])
    audit_statuses = list(audit_df["Sonuc"])

    assert len(audit_df) == len(backtest_statuses)
    assert audit_statuses == backtest_statuses
