"""Persistent bot state — survives crashes and restarts via JSON file."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_FILE = "bot_state.json"


@dataclass
class ActiveTrade:
    """Represents an open position on-exchange."""

    direction: str                    # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_at_entry: float
    position_size_usd: float
    position_size_frac: float        # fraction of equity
    confidence: float
    entry_time: str                  # ISO format
    order_id: str = ""               # exchange order ID
    sl_order_id: str = ""
    tp_order_id: str = ""
    partial_tp_done: bool = False    # True if TP1 (50%) already hit
    bars_held: int = 0


@dataclass
class DailyStats:
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_usd: float = 0.0


@dataclass
class BotState:
    """Full bot state that gets persisted to disk."""

    active_trade: Optional[ActiveTrade] = None
    daily_stats: DailyStats = field(default_factory=DailyStats)
    last_signal_time: str = ""       # ISO format — prevents duplicate signals
    total_pnl_usd: float = 0.0
    total_trades: int = 0
    start_balance: float = 0.0
    paper_mode: bool = True

    def save(self, state_dir: str | Path) -> None:
        """Persist state to JSON."""
        path = Path(state_dir) / _STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls, state_dir: str | Path) -> BotState:
        """Load state from JSON, or return fresh state."""
        path = Path(state_dir) / _STATE_FILE
        if not path.exists():
            logger.info("No saved state found, starting fresh")
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            state = cls._from_dict(data)
            logger.info("Loaded state: %d total trades, active=%s",
                        state.total_trades, state.active_trade is not None)
            return state
        except Exception as e:
            logger.error("Failed to load state: %s — starting fresh", e)
            return cls()

    def reset_daily_if_needed(self, now: datetime) -> None:
        """Reset daily counters at midnight UTC."""
        today = now.strftime("%Y-%m-%d")
        if self.daily_stats.date != today:
            self.daily_stats = DailyStats(date=today)

    def record_trade(self, pnl_usd: float) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        self.total_pnl_usd += pnl_usd
        self.daily_stats.trades += 1
        if pnl_usd >= 0:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1
        self.daily_stats.pnl_usd += pnl_usd

    # ── serialization helpers ─────────────────────────────────────────────

    def _to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def _from_dict(cls, data: dict) -> BotState:
        state = cls()
        # Restore active trade
        at = data.get("active_trade")
        if at and isinstance(at, dict):
            state.active_trade = ActiveTrade(**at)
        # Restore daily stats
        ds = data.get("daily_stats")
        if ds and isinstance(ds, dict):
            state.daily_stats = DailyStats(**ds)
        state.last_signal_time = data.get("last_signal_time", "")
        state.total_pnl_usd = data.get("total_pnl_usd", 0.0)
        state.total_trades = data.get("total_trades", 0)
        state.start_balance = data.get("start_balance", 0.0)
        state.paper_mode = data.get("paper_mode", True)
        return state
