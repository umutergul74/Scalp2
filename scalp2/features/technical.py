"""Standard technical indicators computed on OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.config import TechnicalConfig


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{period}")


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=period, min_periods=period, adjust=False).mean().rename(
        f"ema_{period}"
    )


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }
    )


def compute_bollinger(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: middle, upper, lower, bandwidth, %B."""
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()

    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / (middle + 1e-10)
    pct_b = (close - lower) / (upper - lower + 1e-10)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": pct_b,
        }
    )


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.rename(f"atr_{period}")


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()

    fast_k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    slow_k = fast_k.rolling(smooth, min_periods=smooth).mean()
    slow_d = slow_k.rolling(d_period, min_periods=d_period).mean()

    return pd.DataFrame({"stoch_k": slow_k, "stoch_d": slow_d})


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Average Directional Index with +DI and -DI."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # Zero out DM when the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def compute_all_technical(df: pd.DataFrame, config: TechnicalConfig) -> pd.DataFrame:
    """Compute all technical indicators and append to DataFrame.

    Args:
        df: OHLCV DataFrame indexed by timestamp.
        config: Technical indicator configuration.

    Returns:
        DataFrame with all technical indicator columns appended.
    """
    result = df.copy()

    # RSI
    result[f"rsi_{config.rsi_period}"] = compute_rsi(df["close"], config.rsi_period)

    # EMAs
    for period in config.ema_periods:
        result[f"ema_{period}"] = compute_ema(df["close"], period)

    # EMA slopes (rate of change)
    for period in config.ema_periods:
        ema_col = f"ema_{period}"
        result[f"{ema_col}_slope"] = result[ema_col].diff() / (result[ema_col].shift(1) + 1e-10)

    # MACD
    macd_df = compute_macd(df["close"], *config.macd)
    result = pd.concat([result, macd_df], axis=1)

    # Bollinger Bands
    bb_df = compute_bollinger(df["close"], config.bollinger.period, config.bollinger.std)
    result = pd.concat([result, bb_df], axis=1)

    # ATR
    result[f"atr_{config.atr_period}"] = compute_atr(
        df["high"], df["low"], df["close"], config.atr_period
    )

    # ATR as percentage of close
    result["atr_pct"] = result[f"atr_{config.atr_period}"] / (df["close"] + 1e-10)

    # Stochastic
    stoch_df = compute_stochastic(
        df["high"],
        df["low"],
        df["close"],
        config.stochastic.k,
        config.stochastic.d,
        config.stochastic.smooth,
    )
    result = pd.concat([result, stoch_df], axis=1)

    # ADX
    adx_df = compute_adx(df["high"], df["low"], df["close"], config.adx_period)
    result = pd.concat([result, adx_df], axis=1)

    # Log returns
    result["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volume features
    result["volume_sma_20"] = df["volume"].rolling(20, min_periods=20).mean()
    result["volume_ratio"] = df["volume"] / (result["volume_sma_20"] + 1e-10)
    result["volume_zscore"] = (
        (df["volume"] - result["volume_sma_20"])
        / (df["volume"].rolling(20, min_periods=20).std() + 1e-10)
    )

    return result
