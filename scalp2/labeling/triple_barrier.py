"""Dynamic Triple Barrier Method with ATR-scaled barriers — numba-accelerated."""

from __future__ import annotations

import logging

import numba
import numpy as np
import pandas as pd

from scalp2.config import LabelConfig

logger = logging.getLogger(__name__)

# Sentinel for unlabelable bars (insufficient forward data)
LABEL_SENTINEL = -999


@numba.njit(cache=True)
def _triple_barrier_long(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_hold: int,
    min_ret: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated triple barrier labeling for long entries.

    Returns:
        labels: 1 (TP hit), -1 (SL hit), 0 (time barrier)
        touch_times: bar offset where barrier was hit
        returns: realized return at touch
    """
    n = len(close)
    labels = np.full(n, LABEL_SENTINEL, dtype=np.int64)
    touch_times = np.zeros(n, dtype=np.int64)
    returns = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i + max_hold >= n:
            break
        if atr[i] < 1e-10:
            continue

        entry = close[i]
        tp = entry + tp_mult * atr[i]
        sl = entry - sl_mult * atr[i]

        label = 0  # default: time barrier
        touch = max_hold
        ret = (close[i + max_hold] - entry) / entry

        for j in range(1, max_hold + 1):
            # Check TP: did the high reach TP?
            if high[i + j] >= tp:
                label = 1
                touch = j
                ret = (tp - entry) / entry
                break
            # Check SL: did the low reach SL?
            if low[i + j] <= sl:
                label = -1
                touch = j
                ret = (sl - entry) / entry
                break

        # Filter micro-moves
        if abs(ret) < min_ret and label != 0:
            label = 0

        labels[i] = label
        touch_times[i] = touch
        returns[i] = ret

    return labels, touch_times, returns


@numba.njit(cache=True)
def _triple_barrier_short(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_hold: int,
    min_ret: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated triple barrier labeling for short entries."""
    n = len(close)
    labels = np.full(n, LABEL_SENTINEL, dtype=np.int64)
    touch_times = np.zeros(n, dtype=np.int64)
    returns = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i + max_hold >= n:
            break
        if atr[i] < 1e-10:
            continue

        entry = close[i]
        tp = entry - tp_mult * atr[i]  # TP below for shorts
        sl = entry + sl_mult * atr[i]  # SL above for shorts

        label = 0
        touch = max_hold
        ret = (entry - close[i + max_hold]) / entry

        for j in range(1, max_hold + 1):
            if low[i + j] <= tp:
                label = 1  # TP hit (profitable short)
                touch = j
                ret = (entry - tp) / entry
                break
            if high[i + j] >= sl:
                label = -1  # SL hit
                touch = j
                ret = (entry - sl) / entry
                break

        if abs(ret) < min_ret and label != 0:
            label = 0

        labels[i] = label
        touch_times[i] = touch
        returns[i] = ret

    return labels, touch_times, returns


def triple_barrier_labels(
    df: pd.DataFrame,
    config: LabelConfig,
) -> pd.DataFrame:
    """Apply Dynamic Triple Barrier Method to label each bar.

    Combines long and short perspectives:
        - Label 1: Long TP hit (bullish setup)
        - Label -1: Short TP hit (bearish setup)
        - Label 0: No clear directional edge (time barrier or SL)

    For the XGBoost meta-learner, labels are remapped to {0, 1, 2}:
        0 = short, 1 = hold, 2 = long

    Args:
        df: Feature DataFrame with ATR column and OHLCV.
        config: Labeling configuration.

    Returns:
        DataFrame with label columns appended.
    """
    atr_col = f"atr_{config.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"ATR column '{atr_col}' not found. Compute features first.")

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    atr = df[atr_col].values.astype(np.float64)

    # Long perspective
    long_labels, long_touch, long_ret = _triple_barrier_long(
        close, high, low, atr,
        config.tp_multiplier,
        config.sl_multiplier,
        config.max_holding_bars,
        config.min_return_threshold,
    )

    # Short perspective
    short_labels, short_touch, short_ret = _triple_barrier_short(
        close, high, low, atr,
        config.tp_multiplier,
        config.sl_multiplier,
        config.max_holding_bars,
        config.min_return_threshold,
    )

    # Combine: if long TP hit → label 1 (long), if short TP hit → label -1 (short)
    # If both hit TP, prefer the one that hit faster
    n = len(df)
    combined_labels = np.zeros(n, dtype=np.int64)
    combined_returns = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if long_labels[i] == LABEL_SENTINEL:
            combined_labels[i] = LABEL_SENTINEL
            continue

        long_tp = long_labels[i] == 1
        short_tp = short_labels[i] == 1

        if long_tp and short_tp:
            # Both hit TP — choose faster
            if long_touch[i] <= short_touch[i]:
                combined_labels[i] = 1
                combined_returns[i] = long_ret[i]
            else:
                combined_labels[i] = -1
                combined_returns[i] = -short_ret[i]
        elif long_tp:
            combined_labels[i] = 1
            combined_returns[i] = long_ret[i]
        elif short_tp:
            combined_labels[i] = -1
            combined_returns[i] = -short_ret[i]
        else:
            combined_labels[i] = 0
            combined_returns[i] = long_ret[i]  # Time barrier return

    # Build result
    result = df.copy()
    result["tb_label"] = combined_labels
    result["tb_return"] = combined_returns.astype(np.float32)

    # Remap for classifier: -1 → 0 (short), 0 → 1 (hold), 1 → 2 (long)
    label_map = {LABEL_SENTINEL: LABEL_SENTINEL, -1: 0, 0: 1, 1: 2}
    result["tb_label_cls"] = result["tb_label"].map(label_map).astype(np.int64)

    # Drop unlabelable rows
    valid_mask = result["tb_label"] != LABEL_SENTINEL
    n_dropped = (~valid_mask).sum()
    result = result[valid_mask]

    # Log label distribution
    label_counts = result["tb_label"].value_counts().sort_index()
    logger.info(
        "Triple barrier labels — Short: %d (%.1f%%), Hold: %d (%.1f%%), Long: %d (%.1f%%), "
        "Dropped: %d unlabelable bars",
        label_counts.get(-1, 0),
        100 * label_counts.get(-1, 0) / len(result),
        label_counts.get(0, 0),
        100 * label_counts.get(0, 0) / len(result),
        label_counts.get(1, 0),
        100 * label_counts.get(1, 0) / len(result),
        n_dropped,
    )

    return result
