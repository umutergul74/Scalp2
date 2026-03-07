"""Order flow proxy features — CVD, funding rate, open interest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.config import OrderFlowConfig


def cumulative_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CVD proxy from OHLCV data.

    Formula: delta = volume * (2*close - high - low) / (high - low)

    When close is near the high, most volume is interpreted as buying.
    When close is near the low, most volume is interpreted as selling.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with cvd_delta, cvd_cumulative, and cvd_sma columns.
    """
    hl_range = df["high"] - df["low"]
    # Avoid division by zero for doji candles
    hl_range = hl_range.replace(0, np.nan).ffill()

    delta = df["volume"] * (2 * df["close"] - df["high"] - df["low"]) / (hl_range + 1e-10)

    cvd = delta.cumsum()
    cvd_sma = cvd.rolling(20, min_periods=20).mean()
    cvd_divergence = cvd - cvd_sma

    return pd.DataFrame(
        {
            "cvd_delta": delta.astype(np.float32),
            "cvd_cumulative": cvd.astype(np.float32),
            "cvd_divergence": cvd_divergence.astype(np.float32),
            "cvd_delta_zscore": (
                (delta - delta.rolling(20, min_periods=20).mean())
                / (delta.rolling(20, min_periods=20).std() + 1e-10)
            ).astype(np.float32),
        },
        index=df.index,
    )


def align_funding_rate(
    funding_df: pd.DataFrame, df_primary: pd.DataFrame
) -> pd.DataFrame:
    """Align 8-hourly funding rates to the primary timeframe.

    Uses forward-fill (backward merge) to prevent look-ahead.

    Args:
        funding_df: DataFrame with 'timestamp' and 'funding_rate' columns.
        df_primary: Primary timeframe DataFrame indexed by timestamp.

    Returns:
        DataFrame with funding rate features appended.
    """
    result = df_primary.copy()

    if funding_df.empty:
        result["funding_rate"] = np.float32(0)
        result["funding_rate_ma"] = np.float32(0)
        result["funding_rate_zscore"] = np.float32(0)
        return result

    funding = funding_df.copy()
    if "timestamp" in funding.columns:
        funding = funding.set_index("timestamp")
    funding = funding.sort_index()

    # Reindex to primary timeframe, backward fill
    aligned = funding["funding_rate"].reindex(result.index, method="ffill")

    result["funding_rate"] = aligned.astype(np.float32)
    result["funding_rate_ma"] = (
        aligned.rolling(21, min_periods=1).mean().astype(np.float32)
    )  # ~7 days at 8h intervals
    result["funding_rate_zscore"] = (
        (aligned - result["funding_rate_ma"])
        / (aligned.rolling(21, min_periods=1).std() + 1e-10)
    ).astype(np.float32)

    return result


def compute_oi_delta(
    oi_df: pd.DataFrame, df_primary: pd.DataFrame
) -> pd.DataFrame:
    """Compute open interest delta features.

    Args:
        oi_df: DataFrame with 'timestamp' and 'open_interest' columns.
        df_primary: Primary timeframe DataFrame indexed by timestamp.

    Returns:
        DataFrame with OI features appended.
    """
    result = df_primary.copy()

    if oi_df is None or oi_df.empty:
        result["oi_delta"] = np.float32(0)
        result["oi_delta_pct"] = np.float32(0)
        result["oi_volume_ratio"] = np.float32(0)
        return result

    oi = oi_df.copy()
    if "timestamp" in oi.columns:
        oi = oi.set_index("timestamp")
    oi = oi.sort_index()

    aligned_oi = oi["open_interest"].reindex(result.index, method="ffill")

    oi_delta = aligned_oi.diff()
    oi_delta_pct = oi_delta / (aligned_oi.shift(1) + 1e-10)

    result["oi_delta"] = oi_delta.astype(np.float32)
    result["oi_delta_pct"] = oi_delta_pct.astype(np.float32)
    result["oi_volume_ratio"] = (
        oi_delta.abs() / (df_primary["volume"] + 1e-10)
    ).astype(np.float32)

    return result


def compute_all_orderflow(
    df: pd.DataFrame,
    config: OrderFlowConfig,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all order flow features.

    Args:
        df: Primary OHLCV DataFrame.
        config: Order flow configuration.
        funding_df: Optional funding rate DataFrame.
        oi_df: Optional open interest DataFrame.

    Returns:
        DataFrame with order flow features appended.
    """
    result = df.copy()

    if config.cvd_proxy:
        cvd_features = cumulative_volume_delta(df)
        result = pd.concat([result, cvd_features], axis=1)

    if config.funding_rate and funding_df is not None:
        result = align_funding_rate(funding_df, result)

    if config.open_interest and oi_df is not None:
        result = compute_oi_delta(oi_df, result)

    return result
