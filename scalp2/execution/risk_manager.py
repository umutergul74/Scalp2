"""Risk management — daily limits, regime gating, exposure control."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from scalp2.config import ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Aggregated daily trading statistics."""

    date: date
    trades_taken: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0


class RiskManager:
    """Gate trades based on risk rules and daily limits.

    Rules:
        - Max N trades per day (default: 2)
        - Halt during choppy regime
        - Max daily loss limit
        - Track daily P&L
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.daily_stats: dict[date, DailyStats] = {}
        self.current_date: date | None = None

    def _get_today(self, timestamp: datetime) -> DailyStats:
        """Get or create today's stats."""
        today = timestamp.date()
        if today not in self.daily_stats:
            self.daily_stats[today] = DailyStats(date=today)
        self.current_date = today
        return self.daily_stats[today]

    def can_trade(
        self,
        timestamp: datetime,
        choppy_prob: float = 0.0,
        choppy_threshold: float = 0.5,
    ) -> tuple[bool, str]:
        """Check if a new trade is permitted.

        Args:
            timestamp: Current time.
            choppy_prob: Probability of choppy regime.
            choppy_threshold: Threshold above which trading halts.

        Returns:
            (allowed, reason) tuple.
        """
        stats = self._get_today(timestamp)

        # Check daily trade limit
        if stats.trades_taken >= self.config.max_trades_per_day:
            return False, f"daily_limit ({stats.trades_taken}/{self.config.max_trades_per_day})"

        # Check regime
        if choppy_prob > choppy_threshold:
            return False, f"choppy_regime (P={choppy_prob:.3f})"

        return True, "approved"

    def record_trade(self, timestamp: datetime, pnl: float) -> None:
        """Record a completed trade."""
        stats = self._get_today(timestamp)
        stats.trades_taken += 1
        stats.total_pnl += pnl

        if pnl > 0:
            stats.wins += 1
        elif pnl < 0:
            stats.losses += 1

        stats.peak_pnl = max(stats.peak_pnl, stats.total_pnl)
        drawdown = stats.peak_pnl - stats.total_pnl
        stats.max_drawdown = max(stats.max_drawdown, drawdown)

        logger.info(
            "Trade recorded: PnL=%.4f | Daily: %d trades, total_pnl=%.4f, W/L=%d/%d",
            pnl,
            stats.trades_taken,
            stats.total_pnl,
            stats.wins,
            stats.losses,
        )

    def get_daily_summary(self, timestamp: datetime) -> dict:
        """Get summary of today's trading activity."""
        stats = self._get_today(timestamp)
        return {
            "date": str(stats.date),
            "trades": stats.trades_taken,
            "pnl": stats.total_pnl,
            "wins": stats.wins,
            "losses": stats.losses,
            "win_rate": stats.wins / max(stats.trades_taken, 1),
            "max_drawdown": stats.max_drawdown,
        }
