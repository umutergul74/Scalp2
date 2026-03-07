"""Multi-timeframe feature alignment — no look-ahead."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def align_mtf_features(
    df_primary: pd.DataFrame,
    df_htf: pd.DataFrame,
    htf_label: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Align higher-timeframe features onto the primary timeframe index.

    Uses merge_asof with direction='backward' to ensure only completed
    HTF bars are used — this prevents look-ahead bias.

    Args:
        df_primary: Primary timeframe DataFrame (e.g. 15m), indexed by timestamp.
        df_htf: Higher timeframe DataFrame (e.g. 1h, 4h), indexed by timestamp.
        htf_label: Prefix for HTF columns (e.g. '1h', '4h').
        feature_cols: Which columns from df_htf to include. If None, all columns
            except OHLCV are included.

    Returns:
        df_primary with HTF features appended as new columns.
    """
    if feature_cols is None:
        ohlcv = {
            "open", "high", "low", "close", "volume",
            "quote_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol",
        }
        feature_cols = [c for c in df_htf.columns if c not in ohlcv]

    if not feature_cols:
        logger.warning("No HTF features to align for %s", htf_label)
        return df_primary

    # Prepare HTF dataframe for merge
    htf = df_htf[feature_cols].copy()

    # Rename columns with prefix
    htf.columns = [f"{htf_label}_{col}" for col in htf.columns]

    # Reset index for merge_asof (requires sorted datetime columns)
    primary_reset = df_primary.reset_index()
    htf_reset = htf.reset_index()

    # Ensure both have the same column name for the timestamp
    ts_col = primary_reset.columns[0]  # 'timestamp'
    htf_ts_col = htf_reset.columns[0]
    htf_reset = htf_reset.rename(columns={htf_ts_col: ts_col})

    merged = pd.merge_asof(
        primary_reset.sort_values(ts_col),
        htf_reset.sort_values(ts_col),
        on=ts_col,
        direction="backward",
    )
    merged = merged.set_index(ts_col)

    logger.info(
        "Aligned %d %s features onto primary timeframe (%d rows)",
        len(feature_cols),
        htf_label,
        len(merged),
    )
    return merged


def build_mtf_dataset(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> pd.DataFrame:
    """Build a unified dataset with 15m base + 1H/4H context features.

    All HTF indicator columns in df_1h and df_4h (everything except OHLCV)
    are aligned to the 15m index using backward merge to avoid look-ahead.

    Args:
        df_15m: 15-minute DataFrame with features already computed.
        df_1h: 1-hour DataFrame with features already computed.
        df_4h: 4-hour DataFrame with features already computed.

    Returns:
        Unified DataFrame on 15m index with all features.
    """
    result = df_15m.copy()
    result = align_mtf_features(result, df_1h, htf_label="1h")
    result = align_mtf_features(result, df_4h, htf_label="4h")

    # Forward-fill any NaNs introduced at the start (HTF warmup)
    n_nan_before = result.isna().sum().sum()
    result = result.ffill()
    n_filled = n_nan_before - result.isna().sum().sum()
    if n_filled > 0:
        logger.info("Forward-filled %d NaN values from HTF alignment warmup", n_filled)

    return result
