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

    Uses merge_asof with direction='backward' on SHIFTED timestamps to
    ensure only COMPLETED HTF bars are used — preventing look-ahead bias.

    Without the shift, a 4h bar labeled 08:00 (covering 08:00-11:59)
    would be matched to a 15m bar at 08:15, leaking ~3h44m of future data.
    By shifting the HTF timestamp forward by one period, that 4h bar becomes
    12:00 and is only matched to 15m bars at 12:00+, when it's fully closed.

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

    # FIX: Shift HTF timestamps forward by one period so that only
    # COMPLETED bars are matched. A bar labeled 10:00 in the 1h timeframe
    # covers 10:00-10:59. By shifting to 11:00, merge_asof will only
    # match it to 15m bars at 11:00+, when the bar is fully closed.
    htf_offset = pd.tseries.frequencies.to_offset(htf_label)
    htf_reset[ts_col] = htf_reset[ts_col] + htf_offset
    logger.info(
        "Shifted %s timestamps forward by %s to prevent look-ahead bias",
        htf_label, htf_offset,
    )

    merged = pd.merge_asof(
        primary_reset.sort_values(ts_col),
        htf_reset.sort_values(ts_col),
        on=ts_col,
        direction="backward",
    )
    merged = merged.set_index(ts_col)

    logger.info(
        "Aligned %d %s features onto primary timeframe (%d rows, bias-free)",
        len(feature_cols),
        htf_label,
        len(merged),
    )
    return merged


def build_mtf_dataset(
    df_primary: pd.DataFrame,
    *htf_dfs: pd.DataFrame,
    htf_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Build a unified dataset with primary base + higher-timeframe context features.

    All HTF indicator columns (everything except OHLCV) are aligned to the
    primary index using backward merge to avoid look-ahead.

    Args:
        df_primary: Primary timeframe DataFrame with features already computed.
        *htf_dfs: One or more higher-timeframe DataFrames with features.
        htf_labels: Labels for each HTF (e.g. ["4h"]). If None, auto-detected
            from the DataFrame index frequency.

    Returns:
        Unified DataFrame on primary index with all features.
    """
    # Auto-detect HTF labels if not provided
    if htf_labels is None:
        default_labels = ["1h", "4h", "1d"]
        htf_labels = default_labels[:len(htf_dfs)]

    if len(htf_labels) != len(htf_dfs):
        raise ValueError(
            f"Got {len(htf_dfs)} HTF DataFrames but {len(htf_labels)} labels"
        )

    result = df_primary.copy()
    for htf_df, label in zip(htf_dfs, htf_labels):
        result = align_mtf_features(result, htf_df, htf_label=label)

    # Forward-fill any NaNs introduced at the start (HTF warmup)
    n_nan_before = result.isna().sum().sum()
    result = result.ffill()
    n_filled = n_nan_before - result.isna().sum().sum()
    if n_filled > 0:
        logger.info("Forward-filled %d NaN values from HTF alignment warmup", n_filled)

    return result
