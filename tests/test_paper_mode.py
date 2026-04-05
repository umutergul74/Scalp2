"""Tests for paper-trading accounting and state synchronization."""

import pytest

from scalp2.config import TradeManagementConfig
from scalp2.execution.trade_manager import TradeManager
from scalp2.live.paper import (
    active_trade_to_trade_state,
    backfill_legacy_partial_tp_state,
    effective_paper_balance,
    equity_impact_pct,
    marked_to_market_pnl_frac,
    normalize_close_reason,
    pnl_usd_from_return_frac,
    protection_status_for_close,
    sync_active_trade_from_trade_state,
)
from scalp2.live.state import ActiveTrade, BotState


def _make_trade() -> ActiveTrade:
    return ActiveTrade(
        direction="LONG",
        entry_price=50_000.0,
        stop_loss=49_920.0,
        take_profit=50_200.0,
        atr_at_entry=100.0,
        position_size_usd=4_000.0,
        position_size_frac=0.20,
        confidence=0.75,
        entry_time="2026-04-04T00:00:00+00:00",
        entry_equity=1_000.0,
    )


def test_effective_paper_balance_uses_persisted_pnl():
    state = BotState(start_balance=1_000.0, total_pnl_usd=125.0)
    assert state.current_balance() == pytest.approx(1_125.0)
    assert effective_paper_balance(1_000.0, 125.0) == pytest.approx(1_125.0)


def test_zero_pnl_trade_is_recorded_as_breakeven():
    state = BotState()
    state.record_trade(0.0)
    assert state.daily_stats.trades == 1
    assert state.daily_stats.wins == 0
    assert state.daily_stats.losses == 0
    assert state.daily_stats.breakevens == 1


def test_partial_tp_then_breakeven_keeps_realized_profit():
    cfg = TradeManagementConfig(
        partial_tp_1_atr=0.5,
        partial_tp_1_pct=0.5,
        full_tp_atr=2.0,
        trailing_activation_atr=99.0,
    )
    manager = TradeManager(cfg, max_holding_bars=10)
    trade = _make_trade()

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_060.0,
        current_low=49_990.0,
        current_close=50_040.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    assert trade.partial_tp_done is True
    assert trade.remaining_size_frac == pytest.approx(0.5)
    assert trade.realized_pnl_frac > 0

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_010.0,
        current_low=49_990.0,
        current_close=50_000.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    pnl_usd = pnl_usd_from_return_frac(trade.realized_pnl_frac, trade.position_size_usd)
    assert state.pnl == pytest.approx(trade.realized_pnl_frac)
    assert pnl_usd == pytest.approx(2.0, abs=1e-6)
    assert equity_impact_pct(pnl_usd, trade.entry_equity) == pytest.approx(0.2)


def test_partial_then_full_tp_accumulates_both_legs():
    cfg = TradeManagementConfig(
        partial_tp_1_atr=0.5,
        partial_tp_1_pct=0.5,
        full_tp_atr=2.0,
        trailing_activation_atr=99.0,
    )
    manager = TradeManager(cfg, max_holding_bars=10)
    trade = _make_trade()

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_060.0,
        current_low=50_000.0,
        current_close=50_050.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_220.0,
        current_low=50_100.0,
        current_close=50_200.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    total_pnl_usd = pnl_usd_from_return_frac(state.pnl, trade.position_size_usd)
    assert state.remaining_size == pytest.approx(0.0)
    assert state.pnl == pytest.approx(0.0025, abs=1e-8)
    assert total_pnl_usd == pytest.approx(10.0, abs=1e-6)


def test_mark_to_market_includes_realized_and_remaining_legs():
    trade = _make_trade()
    trade.partial_tp_done = True
    trade.remaining_size_frac = 0.5
    trade.realized_pnl_frac = 0.0005

    total_frac = marked_to_market_pnl_frac(trade, current_price=50_100.0)
    total_pnl_usd = pnl_usd_from_return_frac(total_frac, trade.position_size_usd)

    assert total_frac == pytest.approx(0.0015, abs=1e-8)
    assert total_pnl_usd == pytest.approx(6.0, abs=1e-6)


def test_positive_stop_close_is_not_treated_as_loss_protection():
    assert normalize_close_reason("SL", 0.001) == "TRAIL"
    assert normalize_close_reason("SL", 0.0) == "BE"
    assert protection_status_for_close("SL", 0.001).value == "CLOSED_TP"
    assert protection_status_for_close("SL", 0.0).value == "PARTIAL_TP"


def test_direct_stop_does_not_fake_partial_tp():
    cfg = TradeManagementConfig(
        partial_tp_1_atr=0.5,
        partial_tp_1_pct=0.5,
        full_tp_atr=2.0,
        trailing_activation_atr=99.0,
    )
    manager = TradeManager(cfg, max_holding_bars=10)
    trade = _make_trade()

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_010.0,
        current_low=49_900.0,
        current_close=49_950.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    assert state.status.value == "CLOSED_SL"
    assert trade.partial_tp_done is False
    assert trade.realized_pnl_frac < 0


def test_profitable_trailing_stop_without_tp1_stays_non_partial():
    cfg = TradeManagementConfig(
        partial_tp_1_atr=0.6,
        partial_tp_1_pct=0.5,
        full_tp_atr=2.0,
        trailing_activation_atr=0.2,
        trailing_distance_atr=0.1,
    )
    manager = TradeManager(cfg, max_holding_bars=10)
    trade = _make_trade()

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_030.0,
        current_low=50_000.0,
        current_close=50_045.0,
    )
    sync_active_trade_from_trade_state(trade, state)
    assert trade.partial_tp_done is False
    assert trade.stop_loss == pytest.approx(50_035.0)

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_040.0,
        current_low=50_030.0,
        current_close=50_022.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    assert state.status.value == "CLOSED_SL"
    assert trade.partial_tp_done is False
    assert trade.realized_pnl_frac > 0


def test_partial_tp_then_time_exit_keeps_realized_profit():
    cfg = TradeManagementConfig(
        partial_tp_1_atr=0.5,
        partial_tp_1_pct=0.5,
        full_tp_atr=9.0,
        trailing_activation_atr=99.0,
    )
    manager = TradeManager(cfg, max_holding_bars=2)
    trade = _make_trade()

    state = active_trade_to_trade_state(trade)
    state = manager.update(
        state,
        current_high=50_060.0,
        current_low=50_000.0,
        current_close=50_050.0,
    )
    sync_active_trade_from_trade_state(trade, state)

    state = active_trade_to_trade_state(trade)
    state.bars_held = 1
    state = manager.update(
        state,
        current_high=50_025.0,
        current_low=50_001.0,
        current_close=50_025.0,
    )

    assert state.status.value == "CLOSED_TIME"
    assert state.pnl == pytest.approx(0.00075, abs=1e-8)


def test_backfill_legacy_partial_tp_state_restores_missing_accounting():
    trade = _make_trade()
    trade.partial_tp_done = True

    repaired = backfill_legacy_partial_tp_state(
        trade,
        partial_tp_atr=0.5,
        partial_tp_pct=0.5,
    )

    assert repaired is True
    assert trade.remaining_size_frac == pytest.approx(0.5)
    assert trade.realized_pnl_frac == pytest.approx(0.0005, abs=1e-8)
