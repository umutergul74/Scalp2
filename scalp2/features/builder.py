"""Feature engineering orchestrator — builds the complete feature matrix."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scalp2.config import FeatureConfig
from scalp2.features.orderflow import compute_all_orderflow
from scalp2.features.smart_money import compute_all_smart_money
from scalp2.features.technical import compute_all_technical
from scalp2.features.volatility import compute_all_volatility
from scalp2.features.wavelet import wavelet_denoise

logger = logging.getLogger(__name__)

# Columns that are raw market data and should not be used as model features
RAW_DATA_COLS = {
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
}


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all features for a single timeframe.

    Pipeline order:
        1. Wavelet denoising (close, volume)
        2. Technical indicators
        3. Volatility microstructure
        4. Order flow proxies
        5. Smart money concepts

    Args:
        df: Clean OHLCV DataFrame indexed by timestamp.
        config: Feature engineering configuration.
        funding_df: Optional funding rate data.
        oi_df: Optional open interest data.

    Returns:
        DataFrame with all feature columns appended (~80-120 columns).
    """
    result = df.copy()

    # 1. Wavelet denoising
    for col in config.wavelet.apply_to:
        if col in result.columns:
            logger.info("Applying wavelet denoising to %s", col)
            denoised = wavelet_denoise(
                result[col],
                wavelet=config.wavelet.wavelet,
                level=config.wavelet.level,
                threshold_mode=config.wavelet.threshold_mode,
                window=config.wavelet.window,
            )
            result[f"{col}_denoised"] = denoised.astype(np.float32)

            # Denoised vs raw divergence — useful feature
            result[f"{col}_noise"] = (
                (result[col] - result[f"{col}_denoised"])
                / (result[col].abs() + 1e-10)
            ).astype(np.float32)

    # 2. Technical indicators
    logger.info("Computing technical indicators")
    result = compute_all_technical(result, config.technical)

    # 3. Volatility microstructure
    logger.info("Computing volatility features")
    result = compute_all_volatility(result, config.volatility)

    # 4. Order flow proxies
    logger.info("Computing order flow features")
    result = compute_all_orderflow(result, config.orderflow, funding_df, oi_df)

    # 5. Smart money concepts
    logger.info("Computing smart money features")
    result = compute_all_smart_money(result, config.smart_money)

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Extract model feature column names (excludes raw OHLCV)."""
    return [c for c in df.columns if c not in RAW_DATA_COLS]


def drop_warmup_nans(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Drop initial rows where too many features are NaN (warmup period).

    Args:
        df: Feature DataFrame.
        threshold: Maximum fraction of NaN columns to tolerate per row.

    Returns:
        DataFrame with warmup rows removed.
    """
    feature_cols = get_feature_columns(df)
    nan_frac = df[feature_cols].isna().mean(axis=1)

    # Find first row where NaN fraction drops below threshold
    valid_mask = nan_frac <= threshold
    if not valid_mask.any():
        logger.warning("All rows exceed NaN threshold!")
        return df

    first_valid = valid_mask.idxmax()
    n_dropped = df.index.get_loc(first_valid)
    logger.info(
        "Dropping %d warmup rows (%.1f%% of data)",
        n_dropped,
        100 * n_dropped / len(df),
    )

    result = df.loc[first_valid:].copy()

    # Fill any remaining sporadic NaNs via forward-fill then zero
    remaining_nans = result[feature_cols].isna().sum().sum()
    if remaining_nans > 0:
        logger.info("Forward-filling %d remaining NaN values", remaining_nans)
        result[feature_cols] = result[feature_cols].ffill().fillna(0)

    return result
