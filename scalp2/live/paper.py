"""Paper-trading helpers for balance, accounting, and state conversion."""

from __future__ import annotations

from scalp2.execution.trade_manager import TradeState, TradeStatus
from scalp2.live.state import ActiveTrade


def effective_paper_balance(
    start_balance: float,
    total_pnl_usd: float,
    fallback_balance: float = 1000.0,
) -> float:
    """Return current paper equity using persisted PnL when available."""
    base = start_balance if start_balance > 0 else fallback_balance
    return max(base + total_pnl_usd, 0.0)


def active_trade_to_trade_state(trade: ActiveTrade) -> TradeState:
    """Convert persisted bot state into the runtime TradeManager format."""
    status = TradeStatus.PARTIAL_TP if trade.partial_tp_done else TradeStatus.OPEN
    return TradeState(
        direction=trade.direction,
        entry_price=trade.entry_price,
        current_stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        atr_at_entry=trade.atr_at_entry,
        remaining_size=trade.remaining_size_frac,
        status=status,
        bars_held=trade.bars_held,
        pnl=trade.realized_pnl_frac,
        adaptive_partial_tp_atr=trade.adaptive_partial_tp_atr,
        adaptive_full_tp_atr=trade.adaptive_full_tp_atr,
        adaptive_trailing_act_atr=trade.adaptive_trailing_act_atr,
        adaptive_trailing_dist_atr=trade.adaptive_trailing_dist_atr,
    )


def sync_active_trade_from_trade_state(
    trade: ActiveTrade,
    runtime_state: TradeState,
) -> None:
    """Persist TradeManager updates back into the bot state."""
    trade.stop_loss = runtime_state.current_stop_loss
    trade.take_profit = runtime_state.take_profit
    trade.bars_held = runtime_state.bars_held
    trade.remaining_size_frac = runtime_state.remaining_size
    trade.realized_pnl_frac = runtime_state.pnl
    trade.partial_tp_done = trade.partial_tp_done or bool(runtime_state.partial_fills)


def directional_return_frac(
    entry_price: float,
    exit_price: float,
    direction: str,
) -> float:
    """Return the signed price move fraction for a long or short trade."""
    if direction == "LONG":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def marked_to_market_pnl_frac(trade: ActiveTrade, current_price: float) -> float:
    """Return total PnL as a weighted fraction of original notional."""
    live_return = directional_return_frac(
        trade.entry_price, current_price, trade.direction
    )
    return trade.realized_pnl_frac + live_return * trade.remaining_size_frac


def pnl_usd_from_return_frac(
    return_frac: float,
    position_notional_usd: float,
) -> float:
    """Convert a weighted return fraction into USD PnL."""
    return return_frac * position_notional_usd


def equity_impact_pct(pnl_usd: float, entry_equity: float) -> float:
    """Convert USD PnL into percentage impact on account equity."""
    if entry_equity <= 0:
        return 0.0
    return pnl_usd / entry_equity * 100.0


def normalize_close_reason(
    reason: str,
    pnl_frac: float,
    tolerance: float = 1e-9,
) -> str:
    """Refine stop-based close reasons using the realized total PnL."""
    if reason != "SL":
        return reason
    if pnl_frac > tolerance:
        return "TRAIL"
    if abs(pnl_frac) <= tolerance:
        return "BE"
    return "SL"


def protection_status_for_close(reason: str, pnl_frac: float) -> TradeStatus:
    """Map a close reason to protection bookkeeping without false SL penalties."""
    normalized = normalize_close_reason(reason, pnl_frac)
    if normalized == "SL":
        return TradeStatus.CLOSED_SL
    if normalized == "TIME":
        return TradeStatus.CLOSED_TIME
    if normalized == "REGIME":
        return TradeStatus.CLOSED_REGIME
    if normalized in {"TP", "TRAIL"}:
        return TradeStatus.CLOSED_TP
    if normalized == "exchange_close":
        return TradeStatus.CLOSED_SL
    return TradeStatus.PARTIAL_TP


def backfill_legacy_partial_tp_state(
    trade: ActiveTrade,
    partial_tp_atr: float,
    partial_tp_pct: float,
    tolerance: float = 1e-12,
) -> bool:
    """Infer missing partial-TP accounting for trades restored from older state files."""
    if not trade.partial_tp_done:
        return False
    if trade.remaining_size_frac < 0.999 or abs(trade.realized_pnl_frac) > tolerance:
        return False
    if partial_tp_atr <= 0 or partial_tp_pct <= 0 or trade.entry_price <= 0:
        return False

    trade.remaining_size_frac = max(0.0, 1.0 - partial_tp_pct)
    realized_move = (partial_tp_atr * trade.atr_at_entry) / trade.entry_price
    trade.realized_pnl_frac = realized_move * partial_tp_pct
    return True
