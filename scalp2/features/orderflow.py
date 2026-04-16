"""Market Microstructure and Order Flow features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.config import OrderFlowConfig


def true_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact CVD and volume imbalance using Binance taker data.

    Binance provides exact market buy volume (`taker_buy_base_vol`).
    Therefore:
        Market Sell Vol = Total Volume - Market Buy Vol
        True Delta = Market Buy Vol - Market Sell Vol
    """
    if "taker_buy_base_vol" not in df.columns or "volume" not in df.columns:
        return pd.DataFrame(index=df.index)

    # Calculate exact aggressive volume directions
    buy_vol = df["taker_buy_base_vol"]
    sell_vol = df["volume"] - buy_vol
    
    # Delta (Net Aggressive Flow)
    delta = buy_vol - sell_vol
    
    # Ratios
    buy_ratio = buy_vol / (df["volume"] + 1e-10)
    
    # Cumulative Volume Delta (CVD)
    cvd = delta.cumsum()
    
    # CVD Divergence (CVD trend vs short SMA)
    cvd_sma = cvd.rolling(20, min_periods=20).mean()
    cvd_divergence = cvd - cvd_sma
    
    # Price - CVD absorption flag (Price goes one way, CVD goes the other)
    # E.g. Strong buying (positive delta) but price dropped = Absorption by passive sellers
    price_delta = df["close"].diff()
    absorption_bullish = (delta < 0) & (price_delta > 0) # Selling pressure absorbed, price up
    absorption_bearish = (delta > 0) & (price_delta < 0) # Buying pressure absorbed, price down
    
    # Normalize delta
    delta_zscore = (
        (delta - delta.rolling(20, min_periods=20).mean())
        / (delta.rolling(20, min_periods=20).std() + 1e-10)
    )

    return pd.DataFrame(
        {
            "cvd_delta": delta.astype(np.float32),
            "cvd_cumulative": cvd.astype(np.float32),
            "cvd_divergence": cvd_divergence.astype(np.float32),
            "cvd_delta_zscore": delta_zscore.astype(np.float32),
            "taker_buy_ratio": buy_ratio.astype(np.float32),
            "absorb_bull": absorption_bullish.astype(np.float32),
            "absorb_bear": absorption_bearish.astype(np.float32),
        },
        index=df.index,
    ).ffill().fillna(0)


def whale_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume per trade to detect institutional activity.
    
    If volume per trade spikes, large players are active.
    """
    if "num_trades" not in df.columns or "volume" not in df.columns:
        return pd.DataFrame(index=df.index)
        
    trades = df["num_trades"].replace(0, 1) # Prevent div zero
    
    # Overall volume per trade (ticket size)
    vpt = df["volume"] / trades
    
    # Spike detection (Volume per trade z-score)
    vpt_zscore = (
        (vpt - vpt.rolling(50, min_periods=20).mean())
        / (vpt.rolling(50, min_periods=20).std() + 1e-10)
    )
    
    # Is activity high?
    high_activity = (trades > trades.rolling(50).mean() * 1.5).astype(np.float32)

    return pd.DataFrame(
        {
            "vol_per_trade": vpt.astype(np.float32),
            "vpt_zscore": vpt_zscore.astype(np.float32),
            "high_activity": high_activity,
        },
        index=df.index,
    ).ffill().fillna(0)


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

    if funding_df is None or funding_df.empty:
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

    if config.true_orderflow:
        cvd_features = true_volume_delta(df)
        if not cvd_features.empty and len(cvd_features.columns) > 0:
            result = pd.concat([result, cvd_features], axis=1)

    if getattr(config, "whale_detector", False):
        whale_features = whale_detector(df)
        if not whale_features.empty and len(whale_features.columns) > 0:
            result = pd.concat([result, whale_features], axis=1)

    if config.funding_rate and funding_df is not None:
        result = align_funding_rate(funding_df, result)

    if config.open_interest and oi_df is not None:
        result = compute_oi_delta(oi_df, result)

    return result
