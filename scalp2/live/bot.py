"""Scalp2 Live Trading Bot — async 24/7 daemon for VPS deployment.

Usage:
    python -m scalp2.live.bot [--config config.yaml] [--checkpoint-dir ./checkpoints]

Environment variables required:
    BINANCE_API_KEY, BINANCE_API_SECRET
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (optional)
    PAPER_TRADE=true/false  (default: true)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal as signal_mod
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env file if present (for local dev — on VPS, systemd EnvironmentFile handles this)
def _load_dotenv():
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

import numpy as np
import torch

from scalp2.config import load_config
from scalp2.execution.signal_generator import SignalGenerator, Direction
from scalp2.models.hybrid import HybridEncoder
from scalp2.models.meta_learner import XGBoostMetaLearner
from scalp2.utils.serialization import load_fold_artifacts

from scalp2.live.data_pipeline import DataPipeline
from scalp2.live.exchange import BinanceExecutor
from scalp2.live.notifier import TelegramNotifier
from scalp2.live.state import BotState, ActiveTrade

logger = logging.getLogger(__name__)

# How long after candle close to start (seconds) — ensures candle is finalized
_CANDLE_BUFFER_SECS = 5

# 15-minute interval in seconds
_INTERVAL_SECS = 15 * 60


class LiveBot:
    """Production-grade async 24/7 trading daemon.

    Lifecycle:
        1. Load model artifacts from checkpoint
        2. Restore state from JSON (crash recovery)
        3. Enter async main loop:
           a. Sleep until next 15m candle close (asyncio.sleep)
           b. Fetch data → build features → generate signal
           c. Execute trade if signal is LONG/SHORT
           d. Monitor open positions for TP1/SL
           e. Save state after every action
        4. On SIGTERM / SIGINT: graceful shutdown + close sessions
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        checkpoint_dir: str = "./checkpoints",
        state_dir: str = "./state",
        fold_idx: int | None = None,
    ):
        # Config
        self.config = load_config(config_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_dir = Path(state_dir)

        # Mode
        paper_env = os.environ.get("PAPER_TRADE", "true").lower()
        self.paper_mode = paper_env in ("true", "1", "yes")

        # Components
        self.notifier = TelegramNotifier()
        self.executor = BinanceExecutor(
            paper_mode=self.paper_mode,
            leverage=self.config.execution.position_sizing.leverage,
        )

        # Load model artifacts
        self._load_model(fold_idx)

        # Signal generator (reuses existing module)
        self.signal_gen = SignalGenerator(
            config=self.config,
            model=self.encoder,
            meta_learner=self.xgb,
            regime_detector=self.regime_detector,
            scaler=self.scaler,
            top_feature_indices=self.top_indices,
        )

        # Data pipeline
        self.pipeline = DataPipeline(
            config=self.config,
            executor=self.executor,
            scaler=self.scaler,
            feature_names=self.feature_names,
        )

        # State (restore from disk)
        self.state = BotState.load(self.state_dir)
        self.state.paper_mode = self.paper_mode

        # Shutdown flag
        self._running = True

        mode = "PAPER" if self.paper_mode else "🔴 LIVE"
        logger.info("=" * 60)
        logger.info("  Scalp2 Live Bot — %s MODE (async)", mode)
        logger.info("  Leverage: %dx | Daily cap: %d",
                     self.config.execution.position_sizing.leverage,
                     self.config.execution.max_trades_per_day)
        logger.info("  Features: %d | Seq len: %d",
                     len(self.feature_names), self.config.model.seq_len)
        logger.info("=" * 60)

    # ── Model Loading ─────────────────────────────────────────────────────

    def _load_model(self, fold_idx: int | None = None) -> None:
        """Load the latest (or specified) fold's model artifacts."""
        if fold_idx is None:
            # Find latest fold
            fold_dirs = sorted(
                [d.name for d in self.checkpoint_dir.iterdir()
                 if d.is_dir() and d.name.startswith("fold_")]
            )
            if not fold_dirs:
                raise FileNotFoundError(f"No fold directories in {self.checkpoint_dir}")
            fold_idx = int(fold_dirs[-1].split("_")[1])

        logger.info("Loading fold %d artifacts from %s", fold_idx, self.checkpoint_dir)
        artifacts = load_fold_artifacts(self.checkpoint_dir, fold_idx, device=torch.device("cpu"))

        self.scaler = artifacts["scaler"]
        self.top_indices = artifacts["top_feature_indices"]
        self.feature_names = artifacts["feature_names"]
        self.regime_detector = artifacts.get("regime_detector", None)

        # Build encoder on CPU (inference only, fast enough)
        self.encoder = HybridEncoder(
            n_features=len(self.feature_names),
            config=self.config.model,
        )
        self.encoder.load_state_dict(artifacts["model_state"])
        self.encoder.eval()

        # XGBoost
        self.xgb = XGBoostMetaLearner(self.config.model.xgboost)
        xgb_path = self.checkpoint_dir / f"xgb_fold_{fold_idx:03d}.json"
        self.xgb.load(str(xgb_path))

        logger.info("Model loaded: %d params, %d features, fold %d",
                     self.encoder.count_parameters(), len(self.feature_names), fold_idx)

    # ── Main Loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Async main loop — runs until SIGTERM/SIGINT."""
        # Async init (set leverage on exchange)
        await self.executor.init()
        await self.notifier.info(
            f"Bot başlatıldı — {'PAPER' if self.paper_mode else '🔴 LIVE'} modu (async)"
        )

        logger.info("Bot running. Waiting for next candle close...")

        try:
            while self._running:
                try:
                    # Wait until next 15m candle close
                    await self._wait_for_candle_close()

                    if not self._running:
                        break

                    now = datetime.now(timezone.utc)
                    self.state.reset_daily_if_needed(now)

                    # Check if we have an active trade
                    if self.state.active_trade is not None:
                        await self._manage_active_trade()
                    else:
                        # Generate signal
                        await self._signal_cycle(now)

                    # Save state after every cycle
                    self.state.save(self.state_dir)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error("Main loop error: %s", e, exc_info=True)
                    await self.notifier.error(str(e))
                    await asyncio.sleep(30)  # cooldown before retry

        finally:
            await self._graceful_shutdown()

    # ── Signal Cycle ──────────────────────────────────────────────────────

    async def _signal_cycle(self, now: datetime) -> None:
        """Fetch data, generate signal, execute if actionable."""
        # Prevent duplicate signals on same bar
        now_key = now.strftime("%Y-%m-%d %H:%M")
        if self.state.last_signal_time == now_key:
            logger.debug("Signal already processed for %s", now_key)
            return

        # Check daily limit
        if self.state.daily_stats.trades >= self.config.execution.max_trades_per_day:
            logger.info("Daily trade limit reached (%d)", self.state.daily_stats.trades)
            return

        # Fetch and prepare data (async)
        data = await self.pipeline.prepare()
        if data is None:
            logger.warning("Data pipeline returned None, skipping cycle")
            return

        # Generate signal via existing SignalGenerator (CPU-bound, runs sync)
        signal = self.signal_gen.generate(
            features_scaled=data["features_scaled"],
            regime_df=data["regime_df"],
            current_atr=data["current_atr"],
            current_price=data["current_price"],
            current_time=now,
            current_adx=data["current_adx"],
            atr_percentile=data["atr_percentile"],
        )

        self.state.last_signal_time = now_key

        # ── Cycle summary: Telegram + CSV ─────────────────────────────────
        price = data["current_price"]
        atr = data["current_atr"]
        atr_pct = data["atr_percentile"]
        adx = data["current_adx"]

        if signal.direction == Direction.NO_TRADE:
            reason = signal.regime or "unknown"
            logger.info("No trade signal (reason: %s)", reason)

            # Telegram cycle summary (async)
            await self.notifier.cycle_summary(
                time_str=now_key, price=price, atr=atr,
                atr_pct=atr_pct, adx=adx,
                signal="NO_TRADE", reason=reason,
            )
            # CSV log
            self._log_cycle_csv(now_key, price, atr, atr_pct, adx, "NO_TRADE", reason)
            return

        # We have a signal — execute!
        direction = signal.direction.value
        await self.notifier.cycle_summary(
            time_str=now_key, price=price, atr=atr,
            atr_pct=atr_pct, adx=adx,
            signal=direction, reason="SIGNAL",
            confidence=signal.confidence,
            entry=signal.entry_price,
            sl=signal.stop_loss, tp=signal.take_profit,
        )
        self._log_cycle_csv(
            now_key, price, atr, atr_pct, adx, direction, "SIGNAL",
            signal.confidence, signal.entry_price, signal.stop_loss, signal.take_profit,
        )
        await self._execute_signal(signal, data["current_atr"])

    async def _execute_signal(self, signal, current_atr: float) -> None:
        """Place orders on exchange for a trade signal."""
        direction = signal.direction.value
        leverage = self.config.execution.position_sizing.leverage

        # Calculate position size in USD
        balance = await self.executor.get_balance()
        size_usd = balance * signal.position_size * leverage

        if size_usd < 10:  # Binance min notional
            logger.info("Position too small ($%.2f), skipping", size_usd)
            return

        logger.info(
            "Executing %s: $%.1f (%.1f%% × %dx), conf=%.3f",
            direction, size_usd, signal.position_size * 100, leverage, signal.confidence,
        )

        # Place orders (SL + TP placed concurrently inside open_position)
        result = await self.executor.open_position(
            direction=direction,
            size_usd=size_usd,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        # Record active trade in state
        self.state.active_trade = ActiveTrade(
            direction=direction,
            entry_price=result["filled_price"],
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            atr_at_entry=current_atr,
            position_size_usd=size_usd,
            position_size_frac=signal.position_size,
            confidence=signal.confidence,
            entry_time=datetime.now(timezone.utc).isoformat(),
            order_id=result["order_id"],
            sl_order_id=result["sl_order_id"],
            tp_order_id=result["tp_order_id"],
        )
        self.state.save(self.state_dir)

        # Telegram alert (async)
        await self.notifier.trade_opened(
            direction=direction,
            entry=result["filled_price"],
            sl=signal.stop_loss,
            tp=signal.take_profit,
            size_usd=size_usd,
            confidence=signal.confidence,
            regime=signal.regime,
        )

    # ── Trade Management ──────────────────────────────────────────────────

    async def _manage_active_trade(self) -> None:
        """Check and manage an open trade — partial TP, breakeven, time stop."""
        trade = self.state.active_trade
        if trade is None:
            return

        trade.bars_held += 1

        # Get current price
        try:
            price = await self.executor.get_ticker_price()
        except Exception as e:
            logger.warning("Failed to get price: %s", e)
            return

        is_long = trade.direction == "LONG"
        atr = trade.atr_at_entry
        tm = self.config.execution.trade_management

        # Check if exchange already closed the position (SL/TP hit)
        if not self.paper_mode:
            pos = await self.executor.get_open_position()
            if pos is None:
                # Position closed by exchange (SL or TP hit)
                logger.info("Position closed by exchange")
                await self._finalize_trade(price, "exchange_close")
                return

        # Calculate unrealized PnL
        if is_long:
            unrealized_pct = (price - trade.entry_price) / trade.entry_price
        else:
            unrealized_pct = (trade.entry_price - price) / trade.entry_price

        atr_move = unrealized_pct * trade.entry_price / (atr + 1e-10)

        # Paper mode: simulate SL hit
        if self.paper_mode:
            if is_long and price <= trade.stop_loss:
                sl_pnl = (trade.stop_loss - trade.entry_price) / trade.entry_price
                await self._finalize_trade(trade.stop_loss, "SL", sl_pnl)
                return
            if not is_long and price >= trade.stop_loss:
                sl_pnl = (trade.entry_price - trade.stop_loss) / trade.entry_price
                await self._finalize_trade(trade.stop_loss, "SL", sl_pnl)
                return

        # Partial TP check
        if not trade.partial_tp_done and atr_move >= tm.partial_tp_1_atr:
            logger.info("TP1 hit (%.2f ATR), closing 50%%", atr_move)
            amount = trade.position_size_usd / trade.entry_price
            await self.executor.close_partial(trade.direction, tm.partial_tp_1_pct, amount)

            # Move SL to breakeven
            remaining_amount = amount * (1 - tm.partial_tp_1_pct)
            new_sl_id = await self.executor.modify_stop_loss(
                old_sl_order_id=trade.sl_order_id,
                direction=trade.direction,
                amount=remaining_amount,
                new_sl_price=trade.entry_price,
            )
            trade.partial_tp_done = True
            trade.sl_order_id = new_sl_id
            trade.stop_loss = trade.entry_price
            self.state.save(self.state_dir)
            await self.notifier.info(f"TP1 hit — 50% kapatıldı, SL → breakeven (${trade.entry_price:,.1f})")

        # Full TP check (paper mode)
        if self.paper_mode and atr_move >= tm.full_tp_atr:
            await self._finalize_trade(price, "TP", unrealized_pct)
            return

        # Trailing stop
        if atr_move >= tm.trailing_activation_atr:
            trail_dist = tm.trailing_distance_atr * atr
            if is_long:
                new_sl = price - trail_dist
                if new_sl > trade.stop_loss:
                    trade.stop_loss = new_sl
                    await self.executor.modify_stop_loss(
                        trade.sl_order_id, trade.direction,
                        trade.position_size_usd / trade.entry_price,
                        new_sl,
                    )
            else:
                new_sl = price + trail_dist
                if new_sl < trade.stop_loss:
                    trade.stop_loss = new_sl
                    await self.executor.modify_stop_loss(
                        trade.sl_order_id, trade.direction,
                        trade.position_size_usd / trade.entry_price,
                        new_sl,
                    )

        # Time stop
        if trade.bars_held >= self.config.labeling.max_holding_bars:
            logger.info("Time barrier reached (%d bars)", trade.bars_held)
            if not self.paper_mode:
                await self.executor.cancel_all_orders()
                await self.executor.close_position(trade.direction)
            await self._finalize_trade(price, "TIME", unrealized_pct)

    async def _finalize_trade(self, exit_price: float, reason: str, pnl_pct: float | None = None) -> None:
        """Record trade completion and clear state."""
        trade = self.state.active_trade
        if trade is None:
            return

        leverage = self.config.execution.position_sizing.leverage

        if pnl_pct is None:
            if trade.direction == "LONG":
                pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        pnl_usd = pnl_pct * trade.position_size_usd * leverage
        self.state.record_trade(pnl_usd)

        logger.info(
            "Trade closed: %s %s | entry=$%.1f exit=$%.1f | PnL=$%.2f (%.2f%%)",
            trade.direction, reason, trade.entry_price, exit_price, pnl_usd, pnl_pct * 100,
        )

        await self.notifier.trade_closed(
            direction=trade.direction,
            entry=trade.entry_price,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct * 100,
            reason=reason,
        )

        self.state.active_trade = None
        self.state.save(self.state_dir)

    # ── CSV Cycle Log ─────────────────────────────────────────────────────

    def _log_cycle_csv(
        self,
        time_str: str,
        price: float,
        atr: float,
        atr_pct: float,
        adx: float,
        signal: str,
        reason: str,
        confidence: float | None = None,
        entry: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
    ) -> None:
        """Append one row to cycle_history.csv."""
        import csv
        csv_path = self.state_dir / "cycle_history.csv"
        write_header = not csv_path.exists()
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "time", "price", "atr", "atr_pct", "adx",
                        "signal", "reason", "confidence", "entry", "sl", "tp",
                    ])
                writer.writerow([
                    time_str,
                    f"{price:.1f}",
                    f"{atr:.1f}",
                    f"{atr_pct:.3f}",
                    f"{adx:.1f}",
                    signal,
                    reason,
                    f"{confidence:.3f}" if confidence else "",
                    f"{entry:.1f}" if entry else "",
                    f"{sl:.1f}" if sl else "",
                    f"{tp:.1f}" if tp else "",
                ])
        except Exception as e:
            logger.warning("CSV log error: %s", e)

    # ── Scheduler ─────────────────────────────────────────────────────────

    async def _wait_for_candle_close(self) -> None:
        """Async sleep until the next 15-minute candle close + buffer."""
        now = datetime.now(timezone.utc)
        current_ts = now.timestamp()

        # Next 15m boundary
        interval = _INTERVAL_SECS
        next_boundary = ((int(current_ts) // interval) + 1) * interval
        next_close = next_boundary + _CANDLE_BUFFER_SECS

        wait_secs = next_close - current_ts
        if wait_secs < 0:
            wait_secs = 0

        next_time = datetime.fromtimestamp(next_close, tz=timezone.utc)
        logger.info("Waiting %.0fs until candle close at %s", wait_secs, next_time.strftime("%H:%M:%S"))

        # Sleep in short intervals to allow graceful shutdown
        end_time = time.time() + wait_secs
        while time.time() < end_time and self._running:
            remaining = end_time - time.time()
            await asyncio.sleep(min(5.0, remaining))

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info("Shutdown signal received (%s)", signal_mod.Signals(signum).name)
        self._running = False

    async def _graceful_shutdown(self) -> None:
        """Save state, close sessions, and notify on shutdown."""
        logger.info("Graceful shutdown...")
        self.state.save(self.state_dir)

        if self.state.active_trade is not None:
            logger.info(
                "Active trade left open with SL/TP on exchange — safe to restart"
            )
            await self.notifier.info(
                "Bot durduruluyor — açık pozisyon var, SL/TP emirleri borsada duruyor!"
            )
        else:
            await self.notifier.info("Bot durduruluyor — açık pozisyon yok.")

        # Daily summary
        ds = self.state.daily_stats
        if ds.trades > 0:
            balance = await self.executor.get_balance()
            await self.notifier.daily_summary(
                date=ds.date, trades=ds.trades,
                wins=ds.wins, losses=ds.losses,
                pnl_usd=ds.pnl_usd, balance=balance,
            )

        # Close async sessions
        await self.notifier.close()
        await self.executor.close()

        logger.info("Shutdown complete.")


# ── CLI Entry Point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scalp2 Live Trading Bot")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Model checkpoint dir")
    parser.add_argument("--state-dir", default="./state", help="Bot state persistence dir")
    parser.add_argument("--log-dir", default="./logs", help="Log file directory")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold index (default: latest)")
    args = parser.parse_args()

    # Ensure log directory exists
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scalp2_bot.log"

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )

    # Suppress noisy libraries
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("hmmlearn").setLevel(logging.WARNING)

    bot = LiveBot(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        state_dir=args.state_dir,
        fold_idx=args.fold,
    )

    # Register signal handlers
    signal_mod.signal(signal_mod.SIGINT, bot._shutdown_handler)
    signal_mod.signal(signal_mod.SIGTERM, bot._shutdown_handler)

    # Run the async event loop
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
