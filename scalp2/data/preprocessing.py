"""Data cleaning, gap filling, resampling, and dtype optimization."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Binance klines CSV column names → internal names
_BINANCE_RENAME = {
    "open_time_ms": "timestamp",
    "quote_asset_volume": "quote_volume",
    "number_of_trades": "num_trades",
    "taker_buy_base_asset_volume": "taker_buy_base_vol",
    "taker_buy_quote_asset_volume": "taker_buy_quote_vol",
}

# OHLCV + extra columns worth keeping
_KEEP_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]

# Resampling rules per column
_RESAMPLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "num_trades": "sum",
    "taker_buy_base_vol": "sum",
    "taker_buy_quote_vol": "sum",
}


def load_binance_csv(path: str | Path) -> pd.DataFrame:
    """Load a Binance klines CSV and return a DatetimeIndex DataFrame.

    Renames Binance-style columns to short internal names,
    drops the ``close_time_ms`` and ``ignore`` columns,
    and sets a UTC DatetimeIndex.

    Args:
        path: Path to CSV file with Binance klines format.

    Returns:
        DataFrame indexed by UTC timestamp with float32 dtypes.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=_BINANCE_RENAME)

    # Convert ms epoch → UTC datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Keep only useful columns
    keep = [c for c in _KEEP_COLS if c in df.columns]
    df = df[keep]

    df = optimize_dtypes(df)
    df.index.name = "timestamp"

    logger.info(
        "Loaded %d bars from %s  (%s → %s)",
        len(df), Path(path).name, df.index[0], df.index[-1],
    )
    return df


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample a 15m DataFrame to a higher timeframe (1h, 4h, etc.).

    Args:
        df: Source DataFrame (must have OHLCV columns, DatetimeIndex).
        target_tf: Target timeframe string, e.g. ``'1h'``, ``'4h'``.

    Returns:
        Resampled DataFrame with the same column set.
    """
    agg = {c: rule for c, rule in _RESAMPLE_AGG.items() if c in df.columns}
    resampled = df.resample(target_tf).agg(agg).dropna(subset=["close"])
    resampled = optimize_dtypes(resampled)
    resampled.index.name = "timestamp"

    logger.info(
        "Resampled to %s: %d bars  (%s → %s)",
        target_tf, len(resampled), resampled.index[0], resampled.index[-1],
    )
    return resampled


def clean_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Clean and validate OHLCV data.

    Steps:
        1. Ensure timestamp is datetime with UTC timezone.
        2. Set timestamp as index.
        3. Fill missing bars via forward-fill (max 3 consecutive gaps).
        4. Drop remaining NaN rows.
        5. Validate OHLCV constraints (H >= L, all > 0).
        6. Optimize dtypes to float32.

    Args:
        df: Raw OHLCV DataFrame with 'timestamp' column.
        timeframe: Candle interval string for reindex frequency.

    Returns:
        Cleaned DataFrame indexed by timestamp.
    """
    df = df.copy()

    # Ensure UTC datetime index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a 'timestamp' column or DatetimeIndex")

    df = df.sort_index()

    # Remove exact duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Build complete index and reindex to find gaps
    freq_map = {
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    freq = freq_map.get(timeframe)
    if freq:
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=freq, tz="UTC"
        )
        n_missing = len(full_idx) - len(df)
        if n_missing > 0:
            logger.info("Found %d missing bars in %s data, forward-filling", n_missing, timeframe)
            df = df.reindex(full_idx)
            # Forward-fill up to 3 consecutive missing bars
            df = df.ffill(limit=3)

    # Drop any remaining NaNs
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN after gap-filling", n_dropped)

    # Validate OHLCV constraints
    invalid_hl = df["high"] < df["low"]
    if invalid_hl.any():
        logger.warning("Found %d bars where high < low, swapping", invalid_hl.sum())
        df.loc[invalid_hl, ["high", "low"]] = df.loc[
            invalid_hl, ["low", "high"]
        ].values

    invalid_neg = (df[["open", "high", "low", "close", "volume"]] <= 0).any(axis=1)
    if invalid_neg.any():
        logger.warning("Dropping %d bars with non-positive values", invalid_neg.sum())
        df = df[~invalid_neg]

    # Optimize dtypes
    df = optimize_dtypes(df)

    df.index.name = "timestamp"
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 → float32, int64 → int32 where safe."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        if (
            df[col].min() >= np.iinfo(np.int32).min
            and df[col].max() <= np.iinfo(np.int32).max
        ):
            df[col] = df[col].astype(np.int32)
    return df
