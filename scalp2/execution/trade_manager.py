"""Dynamic trade management — partial TPs, breakeven, trailing stops."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from scalp2.config import TradeManagementConfig

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    OPEN = "OPEN"
    PARTIAL_TP = "PARTIAL_TP"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TIME = "CLOSED_TIME"
    CLOSED_REGIME = "CLOSED_REGIME"


@dataclass
class TradeState:
    """Mutable state of an active trade."""

    direction: str  # "LONG" or "SHORT"
    entry_price: float
    current_stop_loss: float
    take_profit: float
    atr_at_entry: float
    remaining_size: float = 1.0  # Fraction of position still open
    status: TradeStatus = TradeStatus.OPEN
    bars_held: int = 0
    max_favorable_excursion: float = 0.0
    pnl: float = 0.0
    partial_fills: list = field(default_factory=list)


class TradeManager:
    """Manage open trades with dynamic stop/target adjustment.

    Rules:
        - Partial TP 1: Close 50% at 0.6 ATR profit, move SL to breakeven
        - Full TP: Close remaining at 1.2 ATR profit
        - Breakeven: After partial TP, SL moves to entry + spread
        - Trailing: After 0.8 ATR profit, trail 0.5 ATR behind price
        - Time stop: Close at max_holding bars
        - Regime stop: Close if regime flips to choppy
    """

    def __init__(self, config: TradeManagementConfig, max_holding_bars: int = 10):
        self.config = config
        self.max_holding_bars = max_holding_bars

    def update(
        self,
        trade: TradeState,
        current_high: float,
        current_low: float,
        current_close: float,
        is_choppy: bool = False,
    ) -> TradeState:
        """Update trade state based on latest bar.

        Args:
            trade: Current trade state.
            current_high: Current bar high.
            current_low: Current bar low.
            current_close: Current bar close.
            is_choppy: Whether current regime is choppy.

        Returns:
            Updated trade state.
        """
        if trade.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):
            return trade

        trade.bars_held += 1
        is_long = trade.direction == "LONG"

        # Calculate unrealized PnL
        if is_long:
            unrealized = (current_close - trade.entry_price) / trade.entry_price
            favorable = (current_high - trade.entry_price) / trade.entry_price
        else:
            unrealized = (trade.entry_price - current_close) / trade.entry_price
            favorable = (trade.entry_price - current_low) / trade.entry_price

        trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable)

        # Check stops first (worst case)
        if self._check_stop_loss(trade, current_high, current_low, is_long):
            return trade

        # Check regime change
        if is_choppy:
            trade.status = TradeStatus.CLOSED_REGIME
            trade.pnl = unrealized * trade.remaining_size
            logger.info("Trade closed: regime change to choppy (PnL=%.4f)", trade.pnl)
            return trade

        # Check time barrier
        if trade.bars_held >= self.max_holding_bars:
            trade.status = TradeStatus.CLOSED_TIME
            trade.pnl = unrealized * trade.remaining_size
            logger.info("Trade closed: time barrier (PnL=%.4f)", trade.pnl)
            return trade

        # Check partial TP
        atr_move = favorable * trade.entry_price / (trade.atr_at_entry + 1e-10)

        if trade.status == TradeStatus.OPEN and atr_move >= self.config.partial_tp_1_atr:
            self._execute_partial_tp(trade, current_close, is_long)

        # Check full TP
        if atr_move >= self.config.full_tp_atr:
            trade.pnl += unrealized * trade.remaining_size
            trade.remaining_size = 0.0
            trade.status = TradeStatus.CLOSED_TP
            logger.info("Trade closed: full TP (PnL=%.4f)", trade.pnl)
            return trade

        # Trailing stop adjustment
        if atr_move >= self.config.trailing_activation_atr:
            self._update_trailing_stop(trade, current_close, is_long)

        return trade

    def _check_stop_loss(
        self, trade: TradeState, high: float, low: float, is_long: bool
    ) -> bool:
        """Check if stop loss was hit."""
        if is_long and low <= trade.current_stop_loss:
            trade.pnl += (
                (trade.current_stop_loss - trade.entry_price)
                / trade.entry_price
                * trade.remaining_size
            )
            trade.remaining_size = 0.0
            trade.status = TradeStatus.CLOSED_SL
            logger.info("Trade closed: stop loss (PnL=%.4f)", trade.pnl)
            return True

        if not is_long and high >= trade.current_stop_loss:
            trade.pnl += (
                (trade.entry_price - trade.current_stop_loss)
                / trade.entry_price
                * trade.remaining_size
            )
            trade.remaining_size = 0.0
            trade.status = TradeStatus.CLOSED_SL
            logger.info("Trade closed: stop loss (PnL=%.4f)", trade.pnl)
            return True

        return False

    def _execute_partial_tp(
        self, trade: TradeState, price: float, is_long: bool
    ) -> None:
        """Execute partial take profit and move to breakeven."""
        partial_size = trade.remaining_size * self.config.partial_tp_1_pct

        if is_long:
            partial_pnl = (price - trade.entry_price) / trade.entry_price * partial_size
        else:
            partial_pnl = (trade.entry_price - price) / trade.entry_price * partial_size

        trade.pnl += partial_pnl
        trade.remaining_size -= partial_size
        trade.status = TradeStatus.PARTIAL_TP
        trade.partial_fills.append({"price": price, "size": partial_size, "pnl": partial_pnl})

        # Move stop to breakeven
        trade.current_stop_loss = trade.entry_price

        logger.info(
            "Partial TP: closed %.0f%% @ %.2f (PnL=%.4f), SL → breakeven",
            self.config.partial_tp_1_pct * 100, price, partial_pnl,
        )

    def _update_trailing_stop(
        self, trade: TradeState, price: float, is_long: bool
    ) -> None:
        """Update trailing stop behind price."""
        trail_dist = self.config.trailing_distance_atr * trade.atr_at_entry

        if is_long:
            new_sl = price - trail_dist
            if new_sl > trade.current_stop_loss:
                trade.current_stop_loss = new_sl
        else:
            new_sl = price + trail_dist
            if new_sl < trade.current_stop_loss:
                trade.current_stop_loss = new_sl
