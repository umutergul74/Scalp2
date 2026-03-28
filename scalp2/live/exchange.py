"""Binance Futures execution — async orders, positions, balance.

Uses ccxt.async_support for non-blocking I/O. All public methods are
coroutines and must be awaited.

Supports both live and paper trading modes.
API keys loaded exclusively from environment variables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

# Max retries for transient exchange errors
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 15]  # exponential-ish backoff


class BinanceExecutor:
    """Async wrapper around CCXT for Binance USDM Futures.

    Security:
        - API keys from env vars only (never hardcoded)
        - IP whitelist enforced on Binance side
        - Futures-only permission, no withdraw

    Usage:
        executor = BinanceExecutor()
        data = await executor.fetch_ohlcv("15m", limit=800)
        ...
        await executor.close()   # must close aiohttp session
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

        # Paper mode: don't pass API keys (public endpoints only, avoids auth errors)
        exchange_config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
        if not paper_mode:
            exchange_config["apiKey"] = api_key
            exchange_config["secret"] = api_secret

        self.exchange = ccxt.binanceusdm(exchange_config)

        # Enable Binance Testnet (demo.binance.com) if specified
        is_testnet = os.environ.get("BINANCE_TESTNET", "false").lower() in ("true", "1", "yes")
        if is_testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("TESTNET MODE ENABLED — Connecting to demo.binance.com")

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info("BinanceExecutor initialized: %s mode, %dx leverage", mode_str, leverage)

    async def init(self) -> None:
        """Async initialization — call after __init__ to set leverage."""
        if not self.paper_mode and os.environ.get("BINANCE_API_KEY", ""):
            await self._retry(
                lambda: self.exchange.set_leverage(self.leverage, self.symbol)
            )
            logger.info("Leverage set to %dx for %s", self.leverage, self.symbol)

    async def close(self) -> None:
        """Close the underlying aiohttp session. Must be called on shutdown."""
        try:
            await self.exchange.close()
        except Exception as e:
            logger.warning("Exchange session close error: %s", e)

    # ── Balance ───────────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        """Get available USDT balance (futures wallet)."""
        if self.paper_mode:
            return 1000.0  # default paper balance
        balance = await self._retry(lambda: self.exchange.fetch_balance())
        usdt = balance.get("USDT", {})
        free = usdt.get("free", 0.0)
        logger.info("USDT balance: %.2f (total: %.2f)", free, usdt.get("total", 0.0))
        return float(free)

    # ── Market Data ───────────────────────────────────────────────────────

    async def fetch_ohlcv(self, timeframe: str = "15m", limit: int = 400) -> list:
        """Fetch OHLCV candles from Binance."""
        return await self._retry(
            lambda: self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        )

    async def get_ticker_price(self) -> float:
        """Get current market price."""
        ticker = await self._retry(lambda: self.exchange.fetch_ticker(self.symbol))
        return float(ticker["last"])

    async def fetch_last_candle(self, timeframe: str = "15m") -> dict:
        """Fetch the last completed candle's OHLC for realistic paper SL/TP checks."""
        ohlcv = await self._retry(
            lambda: self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=2)
        )
        if len(ohlcv) < 2:
            raise ValueError(f"Expected >= 2 candles, got {len(ohlcv)}")
        # ohlcv[-2] = last completed candle, ohlcv[-1] = current (incomplete)
        candle = ohlcv[-2]
        return {"open": candle[1], "high": candle[2], "low": candle[3], "close": candle[4]}

    # ── Order Execution ───────────────────────────────────────────────────

    async def open_position(
        self,
        direction: str,
        size_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> dict:
        """Open a new position with SL and TP orders on exchange.

        In live mode, SL and TP are placed concurrently via asyncio.gather()
        for minimal latency (~500ms saved vs sequential).

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
        entry_order = await self._retry(
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

        # SL and TP placed concurrently via asyncio.gather
        close_side = "sell" if direction == "LONG" else "buy"

        sl_order, tp_order = await asyncio.gather(
            self._retry(
                lambda: self.exchange.create_order(
                    symbol=self.symbol,
                    type="stop_market",
                    side=close_side,
                    amount=amount,
                    params={"stopPrice": stop_loss, "closePosition": False},
                )
            ),
            self._retry(
                lambda: self.exchange.create_order(
                    symbol=self.symbol,
                    type="take_profit_market",
                    side=close_side,
                    amount=amount,
                    params={"stopPrice": take_profit, "closePosition": False},
                )
            ),
        )

        logger.info("SL order placed: $%.1f (id=%s)", stop_loss, sl_order["id"])
        logger.info("TP order placed: $%.1f (id=%s)", take_profit, tp_order["id"])

        return {
            "order_id": order_id,
            "sl_order_id": sl_order["id"],
            "tp_order_id": tp_order["id"],
            "filled_price": filled_price,
        }

    async def close_partial(
        self,
        direction: str,
        fraction: float,
        current_amount: float,
    ) -> dict:
        """Close a fraction of the position (for partial TP)."""
        close_side = "sell" if direction == "LONG" else "buy"
        close_amount = current_amount * fraction

        if self.paper_mode:
            price = await self.get_ticker_price()
            logger.info("[PAPER] Partial close: %s %.6f BTC @ $%.1f", close_side, close_amount, price)
            return {"order_id": f"paper_partial_{int(time.time())}", "filled_price": price}

        order = await self._retry(
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

    async def modify_stop_loss(
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
            await self._retry(lambda: self.exchange.cancel_order(old_sl_order_id, self.symbol))
        except Exception as e:
            logger.warning("Failed to cancel old SL %s: %s", old_sl_order_id, e)

        # Place new SL
        close_side = "sell" if direction == "LONG" else "buy"
        new_sl = await self._retry(
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

    async def modify_take_profit(
        self,
        old_tp_order_id: str,
        direction: str,
        amount: float,
        new_tp_price: float,
    ) -> str:
        """Cancel old TP and place new one (for fill-price adjustment)."""
        if self.paper_mode:
            logger.info("[PAPER] TP moved to $%.1f", new_tp_price)
            return f"paper_tp_{int(time.time())}"

        # Cancel old TP
        try:
            await self._retry(lambda: self.exchange.cancel_order(old_tp_order_id, self.symbol))
        except Exception as e:
            logger.warning("Failed to cancel old TP %s: %s", old_tp_order_id, e)

        # Place new TP
        close_side = "sell" if direction == "LONG" else "buy"
        new_tp = await self._retry(
            lambda: self.exchange.create_order(
                symbol=self.symbol,
                type="take_profit_market",
                side=close_side,
                amount=amount,
                params={"stopPrice": new_tp_price, "closePosition": False},
            )
        )
        logger.info("New TP placed: $%.1f (id=%s)", new_tp_price, new_tp["id"])
        return new_tp["id"]

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders for the symbol."""
        if self.paper_mode:
            logger.info("[PAPER] All orders cancelled")
            return
        try:
            await self._retry(lambda: self.exchange.cancel_all_orders(self.symbol))
            logger.info("All open orders cancelled for %s", self.symbol)
        except Exception as e:
            logger.warning("Failed to cancel all orders: %s", e)

    async def close_position(self, direction: str) -> dict:
        """Close entire position at market."""
        if self.paper_mode:
            price = await self.get_ticker_price()
            logger.info("[PAPER] Full close at $%.1f", price)
            return {"filled_price": price}

        # Get current position size
        positions = await self._retry(lambda: self.exchange.fetch_positions([self.symbol]))
        for pos in positions:
            contracts = abs(float(pos.get("contracts", 0)))
            if contracts > 0:
                close_side = "sell" if direction == "LONG" else "buy"
                order = await self._retry(
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
        return {"filled_price": await self.get_ticker_price()}

    async def get_open_position(self) -> Optional[dict]:
        """Check if there's a position open on exchange.

        Returns:
            dict with side, size, entry_price, pnl or None
        """
        if self.paper_mode:
            return None  # paper mode uses bot state only

        positions = await self._retry(lambda: self.exchange.fetch_positions([self.symbol]))
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

    async def _retry(self, fn, retries: int = _MAX_RETRIES):
        """Execute with async exponential backoff on transient errors."""
        for attempt in range(retries):
            try:
                return await fn()
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
                await asyncio.sleep(delay)
            except ccxt.ExchangeError as e:
                logger.error("Exchange error (non-retryable): %s", e)
                raise
        raise RuntimeError(f"Exchange call failed after {retries} attempts")
