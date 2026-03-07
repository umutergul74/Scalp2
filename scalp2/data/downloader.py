"""CCXT-based OHLCV data downloader with pagination and caching."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from scalp2.config import DataConfig

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class OHLCVDownloader:
    """Download and cache OHLCV data from a CCXT exchange."""

    def __init__(self, config: DataConfig):
        self.config = config
        try:
            exchange_cls = getattr(ccxt, config.exchange)
        except AttributeError as e:
            raise ValueError(
                f"Unknown CCXT exchange '{config.exchange}'."
            ) from e
        self.exchange = exchange_cls({"enableRateLimit": True})
        # Respect the exchange's advertised per-request delay (ms -> s).
        self._base_pause_sec = max(
            float(getattr(self.exchange, "rateLimit", 200)) / 1000.0, 0.2
        )
        self._last_request_ts = 0.0
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self, multiplier: float = 1.0) -> None:
        """Sleep to avoid sending requests faster than exchange limits."""
        min_interval = self._base_pause_sec * multiplier
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _mark_request(self) -> None:
        self._last_request_ts = time.monotonic()

    def _cache_path(self, timeframe: str) -> Path:
        symbol_safe = re.sub(r"[^A-Za-z0-9]+", "_", self.config.symbol).strip("_")
        start = self.config.date_range.start.replace("-", "")
        end = self.config.date_range.end.replace("-", "")
        return self.cache_dir / f"{symbol_safe}_{timeframe}_{start}_{end}.parquet"

    def fetch(self, timeframe: str, use_cache: bool = True) -> pd.DataFrame:
        """Fetch OHLCV data for a given timeframe.

        Args:
            timeframe: Candle interval (e.g. '15m', '1h', '4h').
            use_cache: If True, load from parquet cache when available.

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        cache_path = self._cache_path(timeframe)
        if use_cache and cache_path.exists():
            logger.info("Loading cached %s data from %s", timeframe, cache_path)
            return pd.read_parquet(cache_path)

        logger.info(
            "Downloading %s %s data from %s to %s",
            self.config.symbol,
            timeframe,
            self.config.date_range.start,
            self.config.date_range.end,
        )

        start_ms = self.exchange.parse8601(
            f"{self.config.date_range.start}T00:00:00Z"
        )
        end_ms = self.exchange.parse8601(
            f"{self.config.date_range.end}T23:59:59Z"
        )

        all_candles: list[list] = []
        since = start_ms
        limit = 1000  # Binance max per request
        consecutive_network_errors = 0
        rate_limit_retries = 0

        while since < end_ms:
            try:
                self._throttle()
                candles = self.exchange.fetch_ohlcv(
                    self.config.symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )
                self._mark_request()
                consecutive_network_errors = 0
                rate_limit_retries = 0
            except ccxt.RateLimitExceeded:
                rate_limit_retries += 1
                sleep_s = min(
                    self._base_pause_sec * (2 ** min(rate_limit_retries, 6)),
                    60.0,
                )
                logger.warning(
                    "Rate limit hit on %s, sleeping %.1fs (%d)",
                    self.config.exchange,
                    sleep_s,
                    rate_limit_retries,
                )
                time.sleep(sleep_s)
                continue
            except ccxt.NetworkError as e:
                err_msg = str(e).lower()
                if "451" in err_msg and "restricted location" in err_msg:
                    raise RuntimeError(
                        "Exchange access blocked for this location (HTTP 451). "
                        f"Current exchange: '{self.config.exchange}'. "
                        "Select an exchange available in your region."
                    ) from e

                consecutive_network_errors += 1
                rate_limit_retries = 0
                if consecutive_network_errors >= 5:
                    raise RuntimeError(
                        f"Aborting after {consecutive_network_errors} consecutive "
                        f"network errors on '{self.config.exchange}'. "
                        "Check connectivity or switch exchange."
                    ) from e

                logger.warning(
                    "Network error: %s, retrying in 5s (%d/5)",
                    e,
                    consecutive_network_errors,
                )
                time.sleep(5)
                continue

            if not candles:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + 1  # Next ms after last candle

            # Progress logging every 10k candles
            if len(all_candles) % 10000 < limit:
                logger.info(
                    "  Downloaded %d candles so far...", len(all_candles)
                )

        df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df[df["timestamp"] <= pd.Timestamp(self.config.date_range.end, tz="UTC")]
        df = df.reset_index(drop=True)

        # Downcast to float32 for memory efficiency
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float32)

        df.to_parquet(cache_path, index=False)
        logger.info(
            "Saved %d candles to %s (%.1f MB)",
            len(df),
            cache_path,
            cache_path.stat().st_size / 1e6,
        )
        return df

    def fetch_all(self, use_cache: bool = True) -> dict[str, pd.DataFrame]:
        """Fetch primary and all MTF timeframes.

        Returns:
            Dict mapping timeframe string to DataFrame.
        """
        result = {}
        all_tf = [self.config.timeframes.primary] + self.config.timeframes.mtf
        for tf in all_tf:
            result[tf] = self.fetch(tf, use_cache=use_cache)
        return result

    def fetch_funding_rate(self, use_cache: bool = True) -> pd.DataFrame:
        """Fetch historical funding rates (Binance futures)."""
        cache_path = self.cache_dir / "funding_rate.parquet"
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        logger.info("Downloading funding rate history...")
        start_ms = self.exchange.parse8601(
            f"{self.config.date_range.start}T00:00:00Z"
        )
        end_ms = self.exchange.parse8601(
            f"{self.config.date_range.end}T23:59:59Z"
        )

        all_rates: list[dict] = []
        since = start_ms
        rate_limit_retries = 0

        while since < end_ms:
            try:
                self._throttle()
                rates = self.exchange.fetch_funding_rate_history(
                    self.config.symbol, since=since, limit=1000
                )
                self._mark_request()
                rate_limit_retries = 0
            except ccxt.RateLimitExceeded:
                rate_limit_retries += 1
                sleep_s = min(
                    self._base_pause_sec * (2 ** min(rate_limit_retries, 6)),
                    60.0,
                )
                logger.warning(
                    "Funding rate limit hit on %s, sleeping %.1fs (%d)",
                    self.config.exchange,
                    sleep_s,
                    rate_limit_retries,
                )
                time.sleep(sleep_s)
                continue
            except Exception as e:
                logger.warning("Funding rate fetch error: %s", e)
                break

            if not rates:
                break

            all_rates.extend(rates)
            since = rates[-1]["timestamp"] + 1

        if not all_rates:
            logger.warning("No funding rate data available")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame(all_rates)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["timestamp", "fundingRate"]].rename(
            columns={"fundingRate": "funding_rate"}
        )
        df["funding_rate"] = df["funding_rate"].astype(np.float32)
        df.to_parquet(cache_path, index=False)
        return df
