"""Smart Money Concepts (SMC/ICT) features — FVG, liquidity sweeps, VWAP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scalp2.config import SmartMoneyConfig


def fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.001) -> pd.DataFrame:
    """Detect Fair Value Gaps (FVGs) and compute distance features.

    Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
    Bearish FVG: candle[i-2].low > candle[i].high  (gap down)

    Args:
        df: OHLCV DataFrame.
        min_gap_pct: Minimum gap size as fraction of price.

    Returns:
        DataFrame with FVG indicator and distance columns.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    fvg_bullish = np.zeros(n, dtype=np.float32)
    fvg_bearish = np.zeros(n, dtype=np.float32)
    fvg_bull_dist = np.full(n, np.nan, dtype=np.float32)
    fvg_bear_dist = np.full(n, np.nan, dtype=np.float32)

    # Track active (unfilled) FVGs
    active_bull_fvgs: list[tuple[float, float]] = []  # (gap_low, gap_high)
    active_bear_fvgs: list[tuple[float, float]] = []

    for i in range(2, n):
        # Detect new bullish FVG
        gap = low[i] - high[i - 2]
        if gap > min_gap_pct * close[i]:
            fvg_bullish[i] = 1.0
            active_bull_fvgs.append((high[i - 2], low[i]))

        # Detect new bearish FVG
        gap = low[i - 2] - high[i]
        if gap > min_gap_pct * close[i]:
            fvg_bearish[i] = 1.0
            active_bear_fvgs.append((high[i], low[i - 2]))

        # Fill (remove) FVGs that price has revisited
        active_bull_fvgs = [
            (gl, gh) for gl, gh in active_bull_fvgs if low[i] > gl
        ]
        active_bear_fvgs = [
            (gl, gh) for gl, gh in active_bear_fvgs if high[i] < gh
        ]

        # Distance to nearest unfilled FVG
        if active_bull_fvgs:
            nearest = min(active_bull_fvgs, key=lambda x: abs(close[i] - x[1]))
            fvg_bull_dist[i] = (close[i] - nearest[1]) / (close[i] + 1e-10)
        if active_bear_fvgs:
            nearest = min(active_bear_fvgs, key=lambda x: abs(close[i] - x[0]))
            fvg_bear_dist[i] = (nearest[0] - close[i]) / (close[i] + 1e-10)

    return pd.DataFrame(
        {
            "fvg_bullish": fvg_bullish,
            "fvg_bearish": fvg_bearish,
            "fvg_bull_dist": fvg_bull_dist,
            "fvg_bear_dist": fvg_bear_dist,
        },
        index=df.index,
    )


def liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect liquidity sweeps of prior swing highs/lows.

    A sweep occurs when price breaks a prior swing level and then reverses,
    suggesting stop-hunting by institutional players.

    Args:
        df: OHLCV DataFrame.
        lookback: Window for identifying swing points.

    Returns:
        DataFrame with sweep indicators and timing features.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    sweep_high = np.zeros(n, dtype=np.float32)
    sweep_low = np.zeros(n, dtype=np.float32)
    bars_since_sweep = np.full(n, np.nan, dtype=np.float32)

    last_sweep_bar = -999

    for i in range(lookback, n):
        window_high = np.max(high[i - lookback : i])
        window_low = np.min(low[i - lookback : i])

        # Sweep high: wick above prior high but close below
        if high[i] > window_high and close[i] < window_high:
            sweep_high[i] = 1.0
            last_sweep_bar = i

        # Sweep low: wick below prior low but close above
        if low[i] < window_low and close[i] > window_low:
            sweep_low[i] = 1.0
            last_sweep_bar = i

        if last_sweep_bar >= 0:
            bars_since_sweep[i] = np.float32(i - last_sweep_bar)

    return pd.DataFrame(
        {
            "sweep_high": sweep_high,
            "sweep_low": sweep_low,
            "bars_since_sweep": bars_since_sweep,
        },
        index=df.index,
    )


def vwap_distance(
    df: pd.DataFrame, session_hours: int = 24
) -> pd.DataFrame:
    """Rolling VWAP with distance-to-VWAP features.

    Args:
        df: OHLCV DataFrame with ATR column.
        session_hours: Rolling window in hours for VWAP calculation.

    Returns:
        DataFrame with VWAP and distance columns.
    """
    # Determine bars per session based on index frequency
    if hasattr(df.index, "freq") and df.index.freq is not None:
        freq_minutes = df.index.freq.delta.total_seconds() / 60
    else:
        # Estimate from first two timestamps
        if len(df) > 1:
            delta = (df.index[1] - df.index[0]).total_seconds() / 60
            freq_minutes = max(delta, 1)
        else:
            freq_minutes = 15

    bars_per_session = int(session_hours * 60 / freq_minutes)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    vwap = tp_vol.rolling(bars_per_session, min_periods=1).sum() / (
        df["volume"].rolling(bars_per_session, min_periods=1).sum() + 1e-10
    )

    # Distance normalized by ATR if available, else by price
    atr_col = [c for c in df.columns if c.startswith("atr_")]
    if atr_col:
        atr = df[atr_col[0]]
        vwap_dist_atr = (df["close"] - vwap) / (atr + 1e-10)
    else:
        vwap_dist_atr = (df["close"] - vwap) / (df["close"] + 1e-10)

    vwap_dist_pct = (df["close"] - vwap) / (vwap + 1e-10)

    return pd.DataFrame(
        {
            "vwap": vwap.astype(np.float32),
            "vwap_dist_pct": vwap_dist_pct.astype(np.float32),
            "vwap_dist_atr": vwap_dist_atr.astype(np.float32),
        },
        index=df.index,
    )


def compute_all_smart_money(
    df: pd.DataFrame, config: SmartMoneyConfig
) -> pd.DataFrame:
    """Compute all Smart Money Concept features.

    Args:
        df: OHLCV DataFrame (should include ATR for VWAP normalization).
        config: Smart money configuration.

    Returns:
        DataFrame with all SMC features appended.
    """
    result = df.copy()

    fvg = fair_value_gaps(df, config.fvg_min_gap_pct)
    result = pd.concat([result, fvg], axis=1)

    sweeps = liquidity_sweeps(df, config.liquidity_sweep_lookback)
    result = pd.concat([result, sweeps], axis=1)

    vwap = vwap_distance(df, config.vwap_session_hours)
    result = pd.concat([result, vwap], axis=1)

    return result
