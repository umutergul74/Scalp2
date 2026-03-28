"""Async live data pipeline — fetch candles, build features, prepare model input."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from scalp2.config import Config
from scalp2.data.preprocessing import clean_ohlcv, resample_ohlcv
from scalp2.data.mtf_builder import build_mtf_dataset
from scalp2.features.builder import build_features, drop_warmup_nans, get_feature_columns
from scalp2.live.exchange import BinanceExecutor

logger = logging.getLogger(__name__)

# Extra bars to fetch for feature warmup (wavelet=256 + MTF NaN + seq_len=64)
# 400 is NOT enough — warmup drops ~340 rows, leaving only ~60.
# 800 bars → ~460 usable rows after warmup → plenty for seq_len=64.
_WARMUP_BARS = 800


class DataPipeline:
    """Async data pipeline: fetch live candles, compute features, scale, return model-ready window.

    Reuses the exact same feature engineering as training to avoid
    train/live skew.
    """

    def __init__(
        self,
        config: Config,
        executor: BinanceExecutor,
        scaler,
        feature_names: list[str],
    ):
        self.config = config
        self.executor = executor
        self.scaler = scaler
        self.feature_names = feature_names
        self.seq_len = config.model.seq_len

    async def prepare(self) -> Optional[dict]:
        """Fetch data, build features, scale, and return model input.

        Returns:
            dict with keys:
                - features_scaled: (seq_len, n_features) array
                - regime_df: DataFrame for regime detector
                - current_atr: float
                - current_adx: float
                - current_price: float
                - atr_percentile: float
                - df_full: full DataFrame (for trade management context)
            or None if data fetch fails.
        """
        try:
            # Fetch 15m candles (async)
            raw_15m = await self.executor.fetch_ohlcv("15m", limit=_WARMUP_BARS)
            if not raw_15m or len(raw_15m) < self.seq_len + 100:
                logger.error("Insufficient 15m data: %d bars", len(raw_15m) if raw_15m else 0)
                return None

            df_15m = self._candles_to_df(raw_15m)
            df_15m = clean_ohlcv(df_15m, "15m")

            # Resample to 1H and 4H (CPU-bound, runs sync)
            df_1h = resample_ohlcv(df_15m, "1h")
            df_4h = resample_ohlcv(df_15m, "4h")

            # Build features (same pipeline as training)
            df_15m_feat = build_features(df_15m, self.config.features)
            df_1h_feat = build_features(df_1h, self.config.features)
            df_4h_feat = build_features(df_4h, self.config.features)

            # Multi-timeframe merge
            df_full = build_mtf_dataset(df_15m_feat, df_1h_feat, df_4h_feat)
            df_full = drop_warmup_nans(df_full)

            if len(df_full) < self.seq_len + 10:
                logger.error("Too few rows after feature engineering: %d", len(df_full))
                return None

            # Align features with training feature set
            missing = [c for c in self.feature_names if c not in df_full.columns]
            if missing:
                logger.warning("%d features missing, zero-filling: %s", len(missing), missing[:5])
                for col in missing:
                    df_full[col] = 0.0

            # Extract and scale
            raw_features = df_full[self.feature_names].values.astype(np.float32)
            scaled = self.scaler.transform(raw_features).astype(np.float32)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Get the last seq_len bars as model window
            window = scaled[-self.seq_len:]

            # Current bar values for filtering
            last_row = df_full.iloc[-1]
            atr_col = f"atr_{self.config.labeling.atr_period}"
            current_atr = float(last_row.get(atr_col, 0.0)) if atr_col in df_full.columns else 0.0
            current_adx = float(last_row.get("adx", 999.0)) if "adx" in df_full.columns else 999.0
            current_price = float(last_row["close"])

            # ATR percentile (rolling rank over 96 bars ≈ 24h — matches backtest)
            if atr_col in df_full.columns:
                atr_series = df_full[atr_col]
                atr_pctile = float(
                    atr_series.rolling(96, min_periods=10).rank(pct=True).iloc[-1]
                )
                if np.isnan(atr_pctile):
                    atr_pctile = 1.0
            else:
                atr_pctile = 1.0

            # Regime DataFrame (last seq_len bars for forward-only HMM)
            regime_df = df_full.iloc[-self.seq_len:]

            logger.info(
                "Pipeline ready: %d features, price=$%.1f, ATR=%.1f, ADX=%.1f",
                len(self.feature_names), current_price, current_atr, current_adx,
            )

            # Extra metrics for Telegram display
            rsi = float(last_row.get("rsi_14", 0.0)) if "rsi_14" in df_full.columns else 0.0
            ema_9 = float(last_row.get("ema_9", 0.0)) if "ema_9" in df_full.columns else 0.0
            ema_21 = float(last_row.get("ema_21", 0.0)) if "ema_21" in df_full.columns else 0.0
            bb_pct_b = float(last_row.get("bb_pct_b", 0.5)) if "bb_pct_b" in df_full.columns else 0.5
            stoch_k = float(last_row.get("stoch_k", 50.0)) if "stoch_k" in df_full.columns else 50.0
            vol_ratio = float(last_row.get("volume_ratio", 1.0)) if "volume_ratio" in df_full.columns else 1.0
            plus_di = float(last_row.get("plus_di", 0.0)) if "plus_di" in df_full.columns else 0.0
            minus_di = float(last_row.get("minus_di", 0.0)) if "minus_di" in df_full.columns else 0.0
            macd_hist = float(last_row.get("macd_hist", 0.0)) if "macd_hist" in df_full.columns else 0.0

            return {
                "features_scaled": window,
                "regime_df": regime_df,
                "current_atr": current_atr,
                "current_adx": current_adx,
                "current_price": current_price,
                "atr_percentile": atr_pctile,
                "df_full": df_full,
                "indicators": {
                    "rsi": rsi,
                    "ema_9": ema_9,
                    "ema_21": ema_21,
                    "bb_pct_b": bb_pct_b,
                    "stoch_k": stoch_k,
                    "vol_ratio": vol_ratio,
                    "plus_di": plus_di,
                    "minus_di": minus_di,
                    "macd_hist": macd_hist,
                },
            }

        except Exception as e:
            logger.error("Data pipeline error: %s", e, exc_info=True)
            return None

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        """Convert CCXT candle list to DataFrame."""
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        df = df.set_index("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float32)
        return df
