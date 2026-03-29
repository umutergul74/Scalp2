"""Dynamic trade management — partial TPs, breakeven, trailing stops, SL protection."""

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
    # Adaptive TP/SL overrides (None = use config defaults)
    adaptive_partial_tp_atr: float | None = None
    adaptive_full_tp_atr: float | None = None
    adaptive_trailing_act_atr: float | None = None
    adaptive_trailing_dist_atr: float | None = None


class TradeManager:
    """Manage open trades with dynamic stop/target adjustment.

    Rules:
        - Partial TP 1: Close 50% at 0.6 ATR profit, move SL to breakeven
        - Full TP: Close remaining at 1.2 ATR profit
        - Breakeven: After partial TP, SL moves to entry + spread
        - Trailing: After 0.8 ATR profit, trail 0.5 ATR behind price
        - Time stop: Close at max_holding bars
        - Regime stop: Close if regime flips to choppy
        - Cooldown: Pause N bars after SL
        - Price distance: Block re-entry near SL price
        - Consecutive SL cap: Halt after N consecutive SLs
    """

    def __init__(self, config: TradeManagementConfig, max_holding_bars: int = 10):
        self.config = config
        self.max_holding_bars = max_holding_bars

        # ── Enhancement 1: Consecutive SL Protection State ──
        self._cooldown_until_bar: int = -1  # Global bar index until which entry blocked
        self._last_sl_price: float = 0.0
        self._last_sl_direction: str = ""
        self._consecutive_sl_count: int = 0
        self._current_bar: int = 0  # Monotonically increasing bar counter

    # ── Entry Gate ────────────────────────────────────────────────────────

    def can_enter_trade(
        self,
        direction: str,
        entry_price: float,
        current_atr: float,
    ) -> tuple[bool, str]:
        """Check if a new trade is permitted by SL protection rules.

        Args:
            direction: "LONG" or "SHORT".
            entry_price: Proposed entry price.
            current_atr: Current ATR value.

        Returns:
            (allowed, skip_reason) tuple.
        """
        cfg = self.config

        # 1. Cooldown after SL
        if cfg.cooldown.enabled and self._current_bar < self._cooldown_until_bar:
            remaining = self._cooldown_until_bar - self._current_bar
            logger.info("Cooldown active: %d bars remaining", remaining)
            return False, "cooldown"

        # 2. Price distance block (same direction only)
        if (
            cfg.price_distance_block.enabled
            and self._last_sl_price > 0
            and direction == self._last_sl_direction
            and current_atr > 0
        ):
            distance_atr = abs(entry_price - self._last_sl_price) / current_atr
            if distance_atr < cfg.price_distance_block.min_atr_distance:
                logger.info(
                    "Price too close to last SL: %.2f ATR (min: %.2f)",
                    distance_atr,
                    cfg.price_distance_block.min_atr_distance,
                )
                return False, "price_too_close"

        # 3. Consecutive SL cap
        if (
            cfg.consecutive_sl_cap.enabled
            and self._consecutive_sl_count >= cfg.consecutive_sl_cap.max_consecutive
        ):
            logger.info(
                "Consecutive SL cap reached: %d/%d",
                self._consecutive_sl_count,
                cfg.consecutive_sl_cap.max_consecutive,
            )
            return False, "consecutive_sl_cap"

        return True, "approved"

    def record_trade_result(self, status: TradeStatus, exit_price: float, direction: str, atr: float) -> None:
        """Record the result of a closed trade for protection state tracking.

        Args:
            status: How the trade was closed.
            exit_price: Price at which the trade was closed.
            direction: Trade direction ("LONG" or "SHORT").
            atr: ATR at entry for distance calculations.
        """
        cfg = self.config

        if status == TradeStatus.CLOSED_SL:
            self._consecutive_sl_count += 1
            self._last_sl_price = exit_price
            self._last_sl_direction = direction

            # Activate cooldown
            if cfg.cooldown.enabled:
                self._cooldown_until_bar = self._current_bar + cfg.cooldown.bars_after_sl
                logger.info(
                    "SL #%d — cooldown activated for %d bars (until bar %d)",
                    self._consecutive_sl_count,
                    cfg.cooldown.bars_after_sl,
                    self._cooldown_until_bar,
                )

        elif status == TradeStatus.CLOSED_REGIME:
            # Shorter cooldown for regime closes
            if cfg.cooldown.enabled:
                self._cooldown_until_bar = self._current_bar + cfg.cooldown.bars_after_regime_close

        elif status in (TradeStatus.CLOSED_TP, TradeStatus.PARTIAL_TP):
            # Win resets consecutive SL counter
            if cfg.consecutive_sl_cap.enabled and cfg.consecutive_sl_cap.reset_after_win:
                if self._consecutive_sl_count > 0:
                    logger.info(
                        "Win resets consecutive SL counter (%d → 0)",
                        self._consecutive_sl_count,
                    )
                self._consecutive_sl_count = 0
            # Clear SL price tracking on win
            self._last_sl_price = 0.0
            self._last_sl_direction = ""

    def advance_bar(self) -> None:
        """Advance the internal bar counter. Call once per 15m cycle."""
        self._current_bar += 1

    def get_protection_state(self) -> dict:
        """Export protection state for persistence / logging."""
        return {
            "cooldown_until_bar": self._cooldown_until_bar,
            "last_sl_price": self._last_sl_price,
            "last_sl_direction": self._last_sl_direction,
            "consecutive_sl_count": self._consecutive_sl_count,
            "current_bar": self._current_bar,
        }

    def set_protection_state(self, state: dict) -> None:
        """Restore protection state from persistence."""
        self._cooldown_until_bar = state.get("cooldown_until_bar", -1)
        self._last_sl_price = state.get("last_sl_price", 0.0)
        self._last_sl_direction = state.get("last_sl_direction", "")
        self._consecutive_sl_count = state.get("consecutive_sl_count", 0)
        self._current_bar = state.get("current_bar", 0)

    # ── Trade Update (existing logic) ────────────────────────────────────

    def update(
        self,
        trade: TradeState,
        current_high: float,
        current_low: float,
        current_close: float,
        is_choppy: bool = False,
        structural_levels: dict | None = None,
    ) -> TradeState:
        """Update trade state based on latest bar.

        Args:
            trade: Current trade state.
            current_high: Current bar high.
            current_low: Current bar low.
            current_close: Current bar close.
            is_choppy: Whether current regime is choppy.
            structural_levels: Dict with vwap, fvg_bull, fvg_bear,
                swing_high, swing_low absolute prices.

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
            # Protect profitable positions with partial TP already taken —
            # let trailing stop manage the exit instead of force-closing
            if trade.status == TradeStatus.PARTIAL_TP and unrealized > 0:
                logger.info(
                    "Choppy regime but partial TP already taken and in profit "
                    "(unrealized=%.4f) — trailing stop protects",
                    unrealized,
                )
                # Tighten trailing stop to lock in more profit during choppy
                trail_dist = (
                    trade.adaptive_trailing_dist_atr
                    or self.config.trailing_distance_atr
                ) * trade.atr_at_entry * 0.5  # 50% tighter in choppy
                if is_long:
                    tight_sl = current_close - trail_dist
                    if tight_sl > trade.current_stop_loss:
                        trade.current_stop_loss = tight_sl
                else:
                    tight_sl = current_close + trail_dist
                    if tight_sl < trade.current_stop_loss:
                        trade.current_stop_loss = tight_sl
            else:
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

        partial_tp = trade.adaptive_partial_tp_atr or self.config.partial_tp_1_atr
        if trade.status == TradeStatus.OPEN and atr_move >= partial_tp:
            # Use exact limit price instead of bar close
            tp_dist = partial_tp * trade.atr_at_entry
            tp_price = trade.entry_price + tp_dist if is_long else trade.entry_price - tp_dist
            self._execute_partial_tp(trade, tp_price, is_long)

        # Check full TP
        full_tp = trade.adaptive_full_tp_atr or self.config.full_tp_atr
        if atr_move >= full_tp:
            tp_pct = (full_tp * trade.atr_at_entry) / trade.entry_price
            trade.pnl += tp_pct * trade.remaining_size
            trade.remaining_size = 0.0
            trade.status = TradeStatus.CLOSED_TP
            logger.info("Trade closed: full TP (PnL=%.4f)", trade.pnl)
            return trade

        # Trailing stop adjustment
        trailing_act = trade.adaptive_trailing_act_atr or self.config.trailing_activation_atr
        if atr_move >= trailing_act:
            self._update_trailing_stop(trade, current_close, is_long)

        # ── Smart Exit Engine: protect trailing SL from sweep zones ──
        if structural_levels:
            self._adjust_sl_to_structure(trade, is_long, structural_levels)

        return trade

    def _adjust_sl_to_structure(
        self,
        trade: TradeState,
        is_long: bool,
        levels: dict,
    ) -> None:
        """Dynamically nudge trailing SL away from sweep zones.

        If the current trailing SL sits inside a swing high/low liquidity
        pool, move it behind the structural level so market-maker sweeps
        don't prematurely stop us out.
        """
        struct_cfg = self.config.structural_exit
        if not struct_cfg.enabled or trade.atr_at_entry <= 0:
            return

        buffer = struct_cfg.sweep_buffer_atr * trade.atr_at_entry

        if is_long:
            swing_low = levels.get("swing_low")
            if swing_low is not None and not _isnan(swing_low):
                # SL is dangerously close to the swing low
                if abs(trade.current_stop_loss - swing_low) < buffer:
                    safe_sl = swing_low - buffer
                    # Only move SL further away, never closer
                    if safe_sl < trade.current_stop_loss:
                        max_stretch = struct_cfg.max_sl_stretch_atr * trade.atr_at_entry
                        if trade.current_stop_loss - safe_sl <= max_stretch:
                            logger.debug(
                                "Trailing sweep shield: SL %.1f → %.1f (swing_low=%.1f)",
                                trade.current_stop_loss, safe_sl, swing_low,
                            )
                            trade.current_stop_loss = safe_sl
        else:
            swing_high = levels.get("swing_high")
            if swing_high is not None and not _isnan(swing_high):
                if abs(trade.current_stop_loss - swing_high) < buffer:
                    safe_sl = swing_high + buffer
                    if safe_sl > trade.current_stop_loss:
                        max_stretch = struct_cfg.max_sl_stretch_atr * trade.atr_at_entry
                        if safe_sl - trade.current_stop_loss <= max_stretch:
                            logger.debug(
                                "Trailing sweep shield: SL %.1f → %.1f (swing_high=%.1f)",
                                trade.current_stop_loss, safe_sl, swing_high,
                            )
                            trade.current_stop_loss = safe_sl

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
        trail_dist = (trade.adaptive_trailing_dist_atr or self.config.trailing_distance_atr) * trade.atr_at_entry

        if is_long:
            new_sl = price - trail_dist
            if new_sl > trade.current_stop_loss:
                trade.current_stop_loss = new_sl
        else:
            new_sl = price + trail_dist
            if new_sl < trade.current_stop_loss:
                trade.current_stop_loss = new_sl


def _isnan(value) -> bool:
    """Safe NaN check for float values."""
    try:
        import math
        return math.isnan(value)
    except (TypeError, ValueError):
        return True
