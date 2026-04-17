"""Market Microstructure and Order Flow features.

Institutional-grade features used by HFT/quant firms:
- True CVD (from Binance taker data)
- Absorption Detection (price vs volume divergence)
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Kyle's Lambda (Price Impact / Market Depth)
- Amihud Illiquidity Ratio
- Whale Detector (Volume per Trade spikes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.config import OrderFlowConfig


# ═══════════════════════════════════════════════════════════════════════════
#  1. TRUE VOLUME DELTA (Real CVD from Binance Taker Data)
# ═══════════════════════════════════════════════════════════════════════════

def true_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact CVD and volume imbalance using Binance taker data.

    Binance provides exact market buy volume (`taker_buy_base_vol`).
    Therefore:
        Market Sell Vol = Total Volume - Market Buy Vol
        True Delta = Market Buy Vol - Market Sell Vol
    """
    if "taker_buy_base_vol" in df.columns and "volume" in df.columns:
        # EXACT MODE: We have Binance's true taker volume
        buy_vol = df["taker_buy_base_vol"]
        sell_vol = df["volume"] - buy_vol
        delta = buy_vol - sell_vol
        buy_ratio = buy_vol / (df["volume"] + 1e-10)
    else:
        # PROXY MODE: Graceful degradation if CSV lacks taker data
        hl_range = df["high"] - df["low"]
        hl_range = hl_range.replace(0, np.nan).ffill()
        delta = df["volume"] * (2 * df["close"] - df["high"] - df["low"]) / (hl_range + 1e-10)
        buy_ratio = (delta + df["volume"]) / (2 * df["volume"] + 1e-10)

    # Cumulative Volume Delta (CVD)
    cvd = delta.cumsum()

    # CVD Divergence (CVD trend vs short SMA)
    cvd_sma = cvd.rolling(20, min_periods=20).mean()
    cvd_divergence = cvd - cvd_sma

    # Price - CVD absorption flag
    price_delta = df["close"].diff()
    absorption_bullish = (delta < 0) & (price_delta > 0)
    absorption_bearish = (delta > 0) & (price_delta < 0)

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


# ═══════════════════════════════════════════════════════════════════════════
#  2. VPIN — Volume-Synchronized Probability of Informed Trading
#  (Easley, López de Prado & O'Hara, 2012)
#  Detects "toxic flow" — when informed traders dominate the market.
# ═══════════════════════════════════════════════════════════════════════════

def compute_vpin(df: pd.DataFrame, n_buckets: int = 50) -> pd.DataFrame:
    """Compute VPIN — a real-time toxicity indicator.

    VPIN measures the imbalance between buy and sell volume across
    equally-sized volume buckets. High VPIN = informed traders are active.

    Args:
        df: OHLCV DataFrame (must have taker_buy_base_vol or will use proxy).
        n_buckets: Number of volume buckets for the rolling window.

    Returns:
        DataFrame with vpin and vpin_zscore columns.
    """
    if "taker_buy_base_vol" in df.columns:
        buy_vol = df["taker_buy_base_vol"].values
        sell_vol = (df["volume"] - df["taker_buy_base_vol"]).values
    else:
        # Proxy: use close location value
        hl_range = (df["high"] - df["low"]).replace(0, np.nan).ffill().values
        clv = (2 * df["close"].values - df["high"].values - df["low"].values) / (hl_range + 1e-10)
        buy_pct = (clv + 1) / 2  # Normalize to [0, 1]
        buy_vol = df["volume"].values * buy_pct
        sell_vol = df["volume"].values * (1 - buy_pct)

    total_vol = df["volume"].values
    n = len(df)

    # Volume bucket size = average volume per bar * n_buckets normalizer
    avg_vol = np.nanmean(total_vol)
    bucket_size = avg_vol  # Each "bucket" = 1 bar's average volume

    # Compute per-bar order imbalance |V_buy - V_sell| / V_total
    imbalance = np.abs(buy_vol - sell_vol) / (total_vol + 1e-10)

    # VPIN = rolling average of imbalance over n_buckets bars
    vpin_series = pd.Series(imbalance, index=df.index)
    vpin = vpin_series.rolling(n_buckets, min_periods=n_buckets // 2).mean()

    # Z-score for regime detection
    vpin_mean = vpin.rolling(200, min_periods=50).mean()
    vpin_std = vpin.rolling(200, min_periods=50).std()
    vpin_zscore = (vpin - vpin_mean) / (vpin_std + 1e-10)

    return pd.DataFrame(
        {
            "vpin": vpin.astype(np.float32),
            "vpin_zscore": vpin_zscore.astype(np.float32),
        },
        index=df.index,
    ).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
#  3. KYLE'S LAMBDA — Price Impact / Market Depth Measure
#  (Kyle, 1985 — "Continuous Auctions and Insider Trading")
#  High λ = thin market, small volume moves price a lot.
#  Low λ = deep market, absorbs large volume without price change.
# ═══════════════════════════════════════════════════════════════════════════

def compute_kyle_lambda(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute Kyle's Lambda — rolling price impact coefficient.

    λ = |ΔPrice| / Volume  (rolling average)

    When λ spikes, the market is illiquid and vulnerable to large moves.

    Args:
        df: OHLCV DataFrame.
        window: Rolling window for averaging.

    Returns:
        DataFrame with kyle_lambda and kyle_lambda_zscore columns.
    """
    abs_return = df["close"].pct_change().abs()
    volume = df["volume"]

    # Raw lambda: how much price moves per unit of volume
    raw_lambda = abs_return / (volume + 1e-10)

    # Rolling smoothed
    kyle = raw_lambda.rolling(window, min_periods=window // 2).mean()

    # Z-score
    kyle_mean = kyle.rolling(100, min_periods=20).mean()
    kyle_std = kyle.rolling(100, min_periods=20).std()
    kyle_zscore = (kyle - kyle_mean) / (kyle_std + 1e-10)

    return pd.DataFrame(
        {
            "kyle_lambda": kyle.astype(np.float32),
            "kyle_lambda_zscore": kyle_zscore.astype(np.float32),
        },
        index=df.index,
    ).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
#  4. AMIHUD ILLIQUIDITY RATIO
#  (Amihud, 2002 — "Illiquidity and Stock Returns")
#  High Amihud = illiquid market (price moves easily).
#  Used by every major quant fund for liquidity risk assessment.
# ═══════════════════════════════════════════════════════════════════════════

def compute_amihud(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute Amihud Illiquidity Ratio.

    Amihud = |r_t| / (Volume_t * Price_t)  (dollar volume adjusted)

    Args:
        df: OHLCV DataFrame.
        window: Rolling window for averaging.

    Returns:
        DataFrame with amihud and amihud_zscore columns.
    """
    abs_return = df["close"].pct_change().abs()
    dollar_volume = df["volume"] * df["close"]  # Notional volume

    raw_amihud = abs_return / (dollar_volume + 1e-10)

    # Rolling smoothed
    amihud = raw_amihud.rolling(window, min_periods=window // 2).mean()

    # Z-score
    amihud_mean = amihud.rolling(100, min_periods=20).mean()
    amihud_std = amihud.rolling(100, min_periods=20).std()
    amihud_zscore = (amihud - amihud_mean) / (amihud_std + 1e-10)

    return pd.DataFrame(
        {
            "amihud": amihud.astype(np.float32),
            "amihud_zscore": amihud_zscore.astype(np.float32),
        },
        index=df.index,
    ).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. WHALE DETECTOR — Volume per Trade Spike Detection
# ═══════════════════════════════════════════════════════════════════════════

def whale_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume per trade to detect institutional activity.

    If volume per trade spikes, large players are active.
    """
    if "num_trades" not in df.columns or "volume" not in df.columns:
        # Fallback if num_trades is missing
        vpt = df["volume"]
        trades = pd.Series(1, index=df.index)
        high_activity = pd.Series(0, index=df.index, dtype=np.float32)
    else:
        trades = df["num_trades"].replace(0, 1)
        vpt = df["volume"] / trades
        high_activity = (trades > trades.rolling(50).mean() * 1.5).astype(np.float32)

    # Spike detection (Volume per trade z-score)
    vpt_zscore = (
        (vpt - vpt.rolling(50, min_periods=20).mean())
        / (vpt.rolling(50, min_periods=20).std() + 1e-10)
    )

    return pd.DataFrame(
        {
            "vol_per_trade": vpt.astype(np.float32),
            "vpt_zscore": vpt_zscore.astype(np.float32),
            "high_activity": high_activity,
        },
        index=df.index,
    ).ffill().fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
#  6. FUNDING RATE & OPEN INTEREST (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def align_funding_rate(
    funding_df: pd.DataFrame, df_primary: pd.DataFrame
) -> pd.DataFrame:
    """Align 8-hourly funding rates to the primary timeframe."""
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

    aligned = funding["funding_rate"].reindex(result.index, method="ffill")

    result["funding_rate"] = aligned.astype(np.float32)
    result["funding_rate_ma"] = (
        aligned.rolling(21, min_periods=1).mean().astype(np.float32)
    )
    result["funding_rate_zscore"] = (
        (aligned - result["funding_rate_ma"])
        / (aligned.rolling(21, min_periods=1).std() + 1e-10)
    ).astype(np.float32)

    return result


def compute_oi_delta(
    oi_df: pd.DataFrame, df_primary: pd.DataFrame
) -> pd.DataFrame:
    """Compute open interest delta features."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — Build all order flow features
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_orderflow(
    df: pd.DataFrame,
    config: OrderFlowConfig,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all order flow and microstructure features.

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
        if not cvd_features.empty:
            result = pd.concat([result, cvd_features], axis=1)

        # VPIN — Toxic flow detection
        vpin_features = compute_vpin(df)
        if not vpin_features.empty:
            result = pd.concat([result, vpin_features], axis=1)

        # Kyle's Lambda — Price impact
        kyle_features = compute_kyle_lambda(df)
        if not kyle_features.empty:
            result = pd.concat([result, kyle_features], axis=1)

        # Amihud — Illiquidity
        amihud_features = compute_amihud(df)
        if not amihud_features.empty:
            result = pd.concat([result, amihud_features], axis=1)

    if getattr(config, "whale_detector", False):
        whale_features = whale_detector(df)
        if not whale_features.empty:
            result = pd.concat([result, whale_features], axis=1)

    if config.funding_rate and funding_df is not None:
        result = align_funding_rate(funding_df, result)

    if config.open_interest and oi_df is not None:
        result = compute_oi_delta(oi_df, result)

    return result
