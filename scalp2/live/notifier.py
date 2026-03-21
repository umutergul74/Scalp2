"""Async Telegram notification system for trade alerts and daily summaries.

Uses aiohttp for non-blocking HTTP. All methods are coroutines.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import aiohttp

logger = logging.getLogger(__name__)

# Timeout for Telegram API calls (seconds)
_TIMEOUT = aiohttp.ClientTimeout(total=10)


class TelegramNotifier:
    """Fire-and-forget async Telegram notifications.

    All methods are non-blocking coroutines and swallow errors so
    the bot never crashes because of a notification failure.

    Usage:
        notifier = TelegramNotifier()
        await notifier.info("Bot started")
        ...
        await notifier.close()   # cleanup aiohttp session
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self._session: aiohttp.ClientSession | None = None
        if not self.enabled:
            logger.warning("Telegram notifier disabled (missing token/chat_id)")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-init reusable aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_TIMEOUT)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session. Must be called on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── public API ────────────────────────────────────────────────────────

    async def trade_opened(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        size_usd: float,
        confidence: float,
        regime: str,
    ) -> None:
        emoji = "🟢" if direction == "LONG" else "🔴"
        msg = (
            f"{emoji} <b>{direction} Açıldı</b>\n"
            f"Giriş : <code>${entry:,.1f}</code>\n"
            f"SL    : <code>${sl:,.1f}</code>\n"
            f"TP    : <code>${tp:,.1f}</code>\n"
            f"Boyut : <code>${size_usd:,.1f}</code>\n"
            f"Güven : <code>{confidence*100:.1f}%</code>\n"
            f"Rejim : {regime}"
        )
        await self._send(msg)

    async def trade_closed(
        self,
        direction: str,
        entry: float,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        emoji = "✅" if pnl_usd >= 0 else "❌"
        msg = (
            f"{emoji} <b>{direction} Kapandı — {reason}</b>\n"
            f"Giriş : <code>${entry:,.1f}</code>\n"
            f"Çıkış : <code>${exit_price:,.1f}</code>\n"
            f"PnL   : <code>{'+' if pnl_usd >= 0 else ''}{pnl_usd:,.2f}$"
            f" ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)</code>"
        )
        await self._send(msg)

    async def daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        losses: int,
        pnl_usd: float,
        balance: float,
    ) -> None:
        wr = wins / max(trades, 1) * 100
        msg = (
            f"📊 <b>Günlük Rapor — {date}</b>\n"
            f"İşlem : {trades} (W:{wins} L:{losses})\n"
            f"WR    : <code>{wr:.0f}%</code>\n"
            f"PnL   : <code>{'+' if pnl_usd >= 0 else ''}{pnl_usd:,.2f}$</code>\n"
            f"Bakiye: <code>${balance:,.2f}</code>"
        )
        await self._send(msg)

    async def cycle_summary(
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
        """Send a concise 15-minute cycle summary."""
        if signal == "NO_TRADE":
            emoji = "⏸️"
            signal_line = f"Sinyal : <b>İşlem Yok</b> ({reason})"
        else:
            emoji = "🟢" if signal == "LONG" else "🔴"
            signal_line = (
                f"Sinyal : <b>{signal}</b> (güven: {confidence*100:.1f}%)\n"
                f"Giriş  : <code>${entry:,.1f}</code>\n"
                f"SL     : <code>${sl:,.1f}</code> | TP: <code>${tp:,.1f}</code>"
            )
        msg = (
            f"{emoji} <b>15dk Rapor — {time_str}</b>\n"
            f"Fiyat  : <code>${price:,.1f}</code>\n"
            f"ATR    : <code>{atr:.1f}</code> (%{atr_pct*100:.0f})\n"
            f"ADX    : <code>{adx:.1f}</code>\n"
            f"{signal_line}"
        )
        await self._send(msg)

    async def error(self, message: str) -> None:
        await self._send(f"🚨 <b>HATA</b>\n<code>{message[:500]}</code>")

    async def info(self, message: str) -> None:
        await self._send(f"ℹ️ {message}")

    # ── internal ──────────────────────────────────────────────────────────

    async def _send(self, text: str) -> None:
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram send failed: %s", body[:200])
        except Exception as e:
            logger.warning("Telegram send error: %s", e)
