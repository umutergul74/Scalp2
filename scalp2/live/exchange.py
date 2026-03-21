"""Binance Futures execution — orders, positions, balance.

Supports both live and paper trading modes.
API keys loaded exclusively from environment variables.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)

# Max retries for transient exchange errors
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 15]  # exponential-ish backoff


class BinanceExecutor:
    """Thin wrapper around CCXT for Binance USDM Futures.

    Security:
        - API keys from env vars only (never hardcoded)
        - IP whitelist enforced on Binance side
        - Futures-only permission, no withdraw
    """

    def __init__(
        self,
        paper_mode: bool = True,
        leverage: int = 10,
        symbol: str = "BTC/USDT:USDT",
    ):
        self.paper_mode = paper_mode
        self.leverage = leverage
        self.symbol = symbol

        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")

        if not paper_mode and (not api_key or not api_secret):
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set for live trading"
            )

        self.exchange = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        })

        # Set leverage on init
        if not paper_mode and api_key:
            self._retry(lambda: self.exchange.set_leverage(leverage, self.symbol))
            logger.info("Leverage set to %dx for %s", leverage, symbol)

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info("BinanceExecutor initialized: %s mode, %dx leverage", mode_str, leverage)

    # ── Balance ───────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Get available USDT balance (futures wallet)."""
        if self.paper_mode:
            return 1000.0  # default paper balance
        balance = self._retry(lambda: self.exchange.fetch_balance())
        usdt = balance.get("USDT", {})
        free = usdt.get("free", 0.0)
        logger.info("USDT balance: %.2f (total: %.2f)", free, usdt.get("total", 0.0))
        return float(free)

    # ── Market Data ───────────────────────────────────────────────────────

    def fetch_ohlcv(self, timeframe: str = "15m", limit: int = 400) -> list:
        """Fetch OHLCV candles from Binance."""
        return self._retry(
            lambda: self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        )

    def get_ticker_price(self) -> float:
        """Get current market price."""
        ticker = self._retry(lambda: self.exchange.fetch_ticker(self.symbol))
        return float(ticker["last"])

    # ── Order Execution ───────────────────────────────────────────────────

    def open_position(
        self,
        direction: str,
        size_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> dict:
        """Open a new position with SL and TP orders on exchange.

        Args:
            direction: "LONG" or "SHORT"
            size_usd: Position size in USDT
            entry_price: Expected entry price (for logging)
            stop_loss: Stop-loss price
            take_profit: Take-profit price

        Returns:
            dict with order_id, sl_order_id, tp_order_id
        """
        side = "buy" if direction == "LONG" else "sell"
        amount = size_usd / entry_price  # BTC quantity

        if self.paper_mode:
            logger.info(
                "[PAPER] %s %.6f BTC @ $%.1f (SL=$%.1f, TP=$%.1f)",
                side.upper(), amount, entry_price, stop_loss, take_profit,
            )
            return {
                "order_id": f"paper_{int(time.time())}",
                "sl_order_id": f"paper_sl_{int(time.time())}",
                "tp_order_id": f"paper_tp_{int(time.time())}",
                "filled_price": entry_price,
            }

        # Market entry
        entry_order = self._retry(
            lambda: self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=amount,
            )
        )
        order_id = entry_order["id"]
        filled_price = float(entry_order.get("average", entry_price))
        logger.info("Entry filled: %s %.6f @ $%.1f (id=%s)", side, amount, filled_price, order_id)

        # SL and TP as stop-market and take-profit-market
        close_side = "sell" if direction == "LONG" else "buy"

        sl_order = self._retry(
            lambda: self.exchange.create_order(
                symbol=self.symbol,
                type="stop_market",
                side=close_side,
                amount=amount,
                params={"stopPrice": stop_loss, "closePosition": False},
            )
        )

        tp_order = self._retry(
            lambda: self.exchange.create_order(
                symbol=self.symbol,
                type="take_profit_market",
                side=close_side,
                amount=amount,
                params={"stopPrice": take_profit, "closePosition": False},
            )
        )

        logger.info("SL order placed: $%.1f (id=%s)", stop_loss, sl_order["id"])
        logger.info("TP order placed: $%.1f (id=%s)", take_profit, tp_order["id"])

        return {
            "order_id": order_id,
            "sl_order_id": sl_order["id"],
            "tp_order_id": tp_order["id"],
            "filled_price": filled_price,
        }

    def close_partial(
        self,
        direction: str,
        fraction: float,
        current_amount: float,
    ) -> dict:
        """Close a fraction of the position (for partial TP)."""
        close_side = "sell" if direction == "LONG" else "buy"
        close_amount = current_amount * fraction

        if self.paper_mode:
            price = self.get_ticker_price()
            logger.info("[PAPER] Partial close: %s %.6f BTC @ $%.1f", close_side, close_amount, price)
            return {"order_id": f"paper_partial_{int(time.time())}", "filled_price": price}

        order = self._retry(
            lambda: self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=close_side,
                amount=close_amount,
            )
        )
        filled = float(order.get("average", 0))
        logger.info("Partial close filled: %.6f @ $%.1f", close_amount, filled)
        return {"order_id": order["id"], "filled_price": filled}

    def modify_stop_loss(
        self,
        old_sl_order_id: str,
        direction: str,
        amount: float,
        new_sl_price: float,
    ) -> str:
        """Cancel old SL and place new one (for breakeven / trailing)."""
        if self.paper_mode:
            logger.info("[PAPER] SL moved to $%.1f", new_sl_price)
            return f"paper_sl_{int(time.time())}"

        # Cancel old SL
        try:
            self._retry(lambda: self.exchange.cancel_order(old_sl_order_id, self.symbol))
        except Exception as e:
            logger.warning("Failed to cancel old SL %s: %s", old_sl_order_id, e)

        # Place new SL
        close_side = "sell" if direction == "LONG" else "buy"
        new_sl = self._retry(
            lambda: self.exchange.create_order(
                symbol=self.symbol,
                type="stop_market",
                side=close_side,
                amount=amount,
                params={"stopPrice": new_sl_price, "closePosition": False},
            )
        )
        logger.info("New SL placed: $%.1f (id=%s)", new_sl_price, new_sl["id"])
        return new_sl["id"]

    def cancel_all_orders(self) -> None:
        """Cancel all open orders for the symbol."""
        if self.paper_mode:
            logger.info("[PAPER] All orders cancelled")
            return
        try:
            self._retry(lambda: self.exchange.cancel_all_orders(self.symbol))
            logger.info("All open orders cancelled for %s", self.symbol)
        except Exception as e:
            logger.warning("Failed to cancel all orders: %s", e)

    def close_position(self, direction: str) -> dict:
        """Close entire position at market."""
        if self.paper_mode:
            price = self.get_ticker_price()
            logger.info("[PAPER] Full close at $%.1f", price)
            return {"filled_price": price}

        # Get current position size
        positions = self._retry(lambda: self.exchange.fetch_positions([self.symbol]))
        for pos in positions:
            contracts = abs(float(pos.get("contracts", 0)))
            if contracts > 0:
                close_side = "sell" if direction == "LONG" else "buy"
                order = self._retry(
                    lambda: self.exchange.create_order(
                        symbol=self.symbol,
                        type="market",
                        side=close_side,
                        amount=contracts,
                    )
                )
                filled = float(order.get("average", 0))
                logger.info("Full close: %.6f @ $%.1f", contracts, filled)
                return {"filled_price": filled}

        logger.warning("No open position found to close")
        return {"filled_price": self.get_ticker_price()}

    def get_open_position(self) -> Optional[dict]:
        """Check if there's a position open on exchange.

        Returns:
            dict with side, size, entry_price, pnl or None
        """
        if self.paper_mode:
            return None  # paper mode uses bot state only

        positions = self._retry(lambda: self.exchange.fetch_positions([self.symbol]))
        for pos in positions:
            contracts = abs(float(pos.get("contracts", 0)))
            if contracts > 0:
                return {
                    "side": pos.get("side", ""),
                    "size": contracts,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "notional": float(pos.get("notional", 0)),
                }
        return None

    # ── Retry logic ───────────────────────────────────────────────────────

    def _retry(self, fn, retries: int = _MAX_RETRIES):
        """Execute with exponential backoff on transient errors."""
        for attempt in range(retries):
            try:
                return fn()
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
                ccxt.RateLimitExceeded,
            ) as e:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                logger.warning(
                    "Exchange error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, retries, e, delay,
                )
                time.sleep(delay)
            except ccxt.ExchangeError as e:
                logger.error("Exchange error (non-retryable): %s", e)
                raise
        raise RuntimeError(f"Exchange call failed after {retries} attempts")
