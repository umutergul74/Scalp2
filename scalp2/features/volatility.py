"""Volatility microstructure estimators — Garman-Klass and Parkinson."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.config import VolatilityConfig


def garman_klass(
    df: pd.DataFrame, window: int = 14
) -> pd.Series:
    """Garman-Klass volatility estimator.

    GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    Approximately 7.4× more efficient than close-to-close volatility
    because it uses all four OHLC prices.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
        window: Rolling window for smoothed estimator.

    Returns:
        Rolling Garman-Klass volatility series.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

    return gk.rolling(window, min_periods=window).mean().rename(f"gk_vol_{window}")


def parkinson(
    df: pd.DataFrame, window: int = 14
) -> pd.Series:
    """Parkinson volatility estimator.

    P = (1 / (4 * n * ln(2))) * sum(ln(H/L)^2)

    Uses only high-low range. Approximately 5.2× more efficient than
    close-to-close volatility.

    Args:
        df: DataFrame with 'high' and 'low' columns.
        window: Rolling window size.

    Returns:
        Rolling Parkinson volatility series.
    """
    log_hl_sq = np.log(df["high"] / df["low"]) ** 2

    factor = 1 / (4 * np.log(2))
    park = factor * log_hl_sq.rolling(window, min_periods=window).mean()

    return np.sqrt(park).rename(f"park_vol_{window}")


def compute_all_volatility(
    df: pd.DataFrame, config: VolatilityConfig
) -> pd.DataFrame:
    """Compute all volatility microstructure features.

    Args:
        df: OHLCV DataFrame.
        config: Volatility configuration.

    Returns:
        DataFrame with volatility columns appended.
    """
    result = df.copy()
    result[f"gk_vol_{config.garman_klass_window}"] = garman_klass(
        df, config.garman_klass_window
    )
    result[f"park_vol_{config.parkinson_window}"] = parkinson(
        df, config.parkinson_window
    )

    # Volatility ratio: GK / Parkinson — divergence signals regime change
    gk_col = f"gk_vol_{config.garman_klass_window}"
    pk_col = f"park_vol_{config.parkinson_window}"
    result["vol_ratio"] = result[gk_col] / (result[pk_col] + 1e-10)

    # Volatility z-score
    vol_mean = result[gk_col].rolling(96, min_periods=96).mean()  # ~1 day
    vol_std = result[gk_col].rolling(96, min_periods=96).std()
    result["vol_zscore"] = (result[gk_col] - vol_mean) / (vol_std + 1e-10)

    return result
