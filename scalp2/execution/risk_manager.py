"""Portfolio-level risk management — daily/weekly loss limits, drawdown halt, win streak control."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

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
    """Portfolio-level risk gating beyond individual trade management.

    Rules:
        - Max N trades per day
        - Halt during choppy regime
        - Daily loss limit (% of equity)
        - Weekly loss limit (% of equity)
        - Max drawdown halt (% of peak equity)
        - Win streak position size reduction (anti-overconfidence)
        - Track daily P&L
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.daily_stats: dict[date, DailyStats] = {}
        self.current_date: date | None = None

        # ── Enhancement 5: Portfolio-level tracking ──
        self._cumulative_pnl_pct: float = 0.0  # Since bot start
        self._peak_pnl_pct: float = 0.0
        self._consecutive_wins: int = 0
        self._halted: bool = False
        self._halt_reason: str = ""

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
        """Check if a new trade is permitted by all risk rules.

        Args:
            timestamp: Current time.
            choppy_prob: Probability of choppy regime.
            choppy_threshold: Threshold above which trading halts.

        Returns:
            (allowed, reason) tuple.
        """
        # 0. Check if system is halted
        if self._halted:
            return False, f"risk_halt ({self._halt_reason})"

        stats = self._get_today(timestamp)

        # 1. Check daily trade limit
        if stats.trades_taken >= self.config.max_trades_per_day:
            return False, f"daily_limit ({stats.trades_taken}/{self.config.max_trades_per_day})"

        # 2. Check regime
        if choppy_prob > choppy_threshold:
            return False, f"choppy_regime (P={choppy_prob:.3f})"

        # 3. Check daily loss limit
        risk_cfg = self.config.risk_limits
        if abs(stats.total_pnl) > 0 and stats.total_pnl < 0:
            daily_loss_pct = abs(stats.total_pnl)  # Already in pct from record_trade
            if daily_loss_pct >= risk_cfg.daily_loss_limit_pct:
                logger.warning(
                    "Daily loss limit hit: %.2f%% >= %.2f%%",
                    daily_loss_pct, risk_cfg.daily_loss_limit_pct,
                )
                return False, f"daily_loss_limit ({daily_loss_pct:.1f}%)"

        # 4. Check weekly loss limit
        weekly_pnl = self._get_weekly_pnl(timestamp)
        if weekly_pnl < -risk_cfg.weekly_loss_limit_pct:
            logger.warning(
                "Weekly loss limit hit: %.2f%% >= %.2f%%",
                abs(weekly_pnl), risk_cfg.weekly_loss_limit_pct,
            )
            return False, f"weekly_loss_limit ({weekly_pnl:.1f}%)"

        # 5. Check drawdown halt
        current_dd = self._peak_pnl_pct - self._cumulative_pnl_pct
        if current_dd >= risk_cfg.drawdown_halt_pct:
            self._halted = True
            self._halt_reason = f"drawdown_{current_dd:.1f}%"
            logger.critical(
                "DRAWDOWN HALT: %.2f%% drawdown from peak — ALL trading stopped!",
                current_dd,
            )
            return False, f"drawdown_halt ({current_dd:.1f}%)"

        return True, "approved"

    def get_position_size_modifier(self) -> float:
        """Get position size multiplier based on win/loss streaks.

        Returns:
            Multiplier in (0, 1]. 1.0 = no modification, <1.0 = reduce size.
        """
        ws_cfg = self.config.risk_limits.win_streak_reduction
        if not ws_cfg.enabled:
            return 1.0

        if self._consecutive_wins >= ws_cfg.after_wins:
            logger.info(
                "Win streak reduction: %d consecutive wins, size × %.2f",
                self._consecutive_wins, ws_cfg.size_multiplier,
            )
            return ws_cfg.size_multiplier

        return 1.0

    def record_trade(self, timestamp: datetime, pnl_pct: float) -> None:
        """Record a completed trade with percentage PnL.

        Args:
            timestamp: Trade close time.
            pnl_pct: Leveraged PnL as percentage of equity.
        """
        stats = self._get_today(timestamp)
        stats.trades_taken += 1
        stats.total_pnl += pnl_pct

        if pnl_pct > 0:
            stats.wins += 1
            self._consecutive_wins += 1
        elif pnl_pct < 0:
            stats.losses += 1
            self._consecutive_wins = 0  # Reset on loss

        stats.peak_pnl = max(stats.peak_pnl, stats.total_pnl)
        drawdown = stats.peak_pnl - stats.total_pnl
        stats.max_drawdown = max(stats.max_drawdown, drawdown)

        # Update cumulative tracking
        self._cumulative_pnl_pct += pnl_pct
        self._peak_pnl_pct = max(self._peak_pnl_pct, self._cumulative_pnl_pct)

        logger.info(
            "Trade recorded: PnL=%.2f%% | Daily: %d trades, total=%.2f%%, W/L=%d/%d | "
            "Cum: %.2f%% (peak: %.2f%%, DD: %.2f%%) | Win streak: %d",
            pnl_pct,
            stats.trades_taken,
            stats.total_pnl,
            stats.wins,
            stats.losses,
            self._cumulative_pnl_pct,
            self._peak_pnl_pct,
            self._peak_pnl_pct - self._cumulative_pnl_pct,
            self._consecutive_wins,
        )

    def _get_weekly_pnl(self, timestamp: datetime) -> float:
        """Sum PnL for the current ISO week."""
        today = timestamp.date()
        week_start = today - timedelta(days=today.weekday())
        total = 0.0
        for d, stats in self.daily_stats.items():
            if d >= week_start:
                total += stats.total_pnl
        return total

    def reset_halt(self) -> None:
        """Manually reset a drawdown halt (requires human intervention)."""
        if self._halted:
            logger.warning("Risk halt MANUALLY reset by operator. Resetting high-water mark.")
            self._halted = False
            self._halt_reason = ""
            # FIX: Reset the peak tracking so current_dd becomes 0, avoiding instant re-halt
            self._peak_pnl_pct = self._cumulative_pnl_pct

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
            "cumulative_pnl": self._cumulative_pnl_pct,
            "peak_pnl": self._peak_pnl_pct,
            "current_dd": self._peak_pnl_pct - self._cumulative_pnl_pct,
            "consecutive_wins": self._consecutive_wins,
            "halted": self._halted,
        }

    def get_state_dict(self) -> dict:
        """Export state for persistence."""
        return {
            "cumulative_pnl_pct": self._cumulative_pnl_pct,
            "peak_pnl_pct": self._peak_pnl_pct,
            "consecutive_wins": self._consecutive_wins,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }

    def set_state_dict(self, data: dict) -> None:
        """Restore state from persistence."""
        self._cumulative_pnl_pct = data.get("cumulative_pnl_pct", 0.0)
        self._peak_pnl_pct = data.get("peak_pnl_pct", 0.0)
        self._consecutive_wins = data.get("consecutive_wins", 0)
        self._halted = data.get("halted", False)
        self._halt_reason = data.get("halt_reason", "")
