"""Telegram notification system for trade alerts and daily summaries."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# Timeout for Telegram API calls (seconds)
_TIMEOUT = 10


class TelegramNotifier:
    """Fire-and-forget Telegram notifications.

    All methods are non-blocking and swallow errors so the bot
    never crashes because of a notification failure.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram notifier disabled (missing token/chat_id)")

    # ── public API ────────────────────────────────────────────────────────

    def trade_opened(
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
        self._send(msg)

    def trade_closed(
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
        self._send(msg)

    def daily_summary(
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
        self._send(msg)

    def error(self, message: str) -> None:
        self._send(f"🚨 <b>HATA</b>\n<code>{message[:500]}</code>")

    def info(self, message: str) -> None:
        self._send(f"ℹ️ {message}")

    # ── internal ──────────────────────────────────────────────────────────

    def _send(self, text: str) -> None:
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
            resp = requests.post(url, json=payload, timeout=_TIMEOUT)
            if resp.status_code != 200:
                logger.warning("Telegram send failed: %s", resp.text[:200])
        except Exception as e:
            logger.warning("Telegram send error: %s", e)
