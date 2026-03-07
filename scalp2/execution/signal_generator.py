"""Full inference pipeline — from raw data to trade signal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import torch

from scalp2.config import Config
from scalp2.models.hybrid import HybridEncoder
from scalp2.models.meta_learner import XGBoostMetaLearner
from scalp2.regime.hmm import RegimeDetector

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


@dataclass
class TradeSignal:
    direction: Direction
    confidence: float
    entry_price: float
    take_profit: float
    stop_loss: float
    position_size: float
    regime: str
    timestamp: datetime
    probabilities: dict  # {short, hold, long}


class SignalGenerator:
    """Generate trade signals from the full model pipeline.

    Pipeline:
        1. Check daily trade limit
        2. Check regime — halt if choppy
        3. Check ADX — skip if no trend
        4. Check ATR percentile — skip low volatility
        5. Extract latent from HybridEncoder
        6. Build meta-features
        7. Predict via XGBoost meta-learner
        8. Apply confidence threshold
        9. Determine direction & TP/SL
       10. Compute position size (fractional Kelly)
    """

    def __init__(
        self,
        config: Config,
        model: HybridEncoder,
        meta_learner: XGBoostMetaLearner,
        regime_detector: RegimeDetector,
        scaler,
        top_feature_indices: np.ndarray,
        device: torch.device | None = None,
    ):
        self.config = config
        self.model = model
        self.meta_learner = meta_learner
        self.regime_detector = regime_detector
        self.scaler = scaler
        self.top_feature_indices = top_feature_indices
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.daily_trade_count = 0
        self.last_trade_date = None

    def _reset_daily_counter(self, current_time: datetime) -> None:
        """Reset daily trade counter at midnight UTC."""
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date

    def generate(
        self,
        features_scaled: np.ndarray,
        regime_df,
        current_atr: float,
        current_price: float,
        current_time: datetime,
        current_adx: float = 999.0,
        atr_percentile: float = 1.0,
    ) -> TradeSignal:
        """Generate a trade signal from prepared features.

        Args:
            features_scaled: (seq_len, n_features) scaled feature window.
            regime_df: DataFrame with regime feature columns for latest bars.
            current_atr: Current ATR value.
            current_price: Current close price.
            current_time: Current timestamp.
            current_adx: Current ADX value (skip if below min_adx).
            atr_percentile: Rolling ATR percentile 0-1 (skip if below min).

        Returns:
            TradeSignal with direction and parameters.
        """
        self._reset_daily_counter(current_time)
        exec_cfg = self.config.execution

        # 1. Check daily limit
        if self.daily_trade_count >= exec_cfg.max_trades_per_day:
            logger.info("Daily trade limit reached (%d)", exec_cfg.max_trades_per_day)
            return self._no_trade(current_price, current_time, "daily_limit")

        # 2. Check regime
        regime_probs = self.regime_detector.predict_proba(regime_df)
        current_regime = self.regime_detector.current_regime(regime_df)

        if regime_probs[-1, RegimeDetector.CHOPPY] > self.config.regime.choppy_threshold:
            logger.info("Choppy regime detected (P=%.3f), skipping", regime_probs[-1, 2])
            return self._no_trade(current_price, current_time, f"choppy_{current_regime}")

        # 3. Check ADX — no trend below threshold
        if current_adx < exec_cfg.min_adx:
            logger.info("ADX too low (%.1f < %.1f), skipping", current_adx, exec_cfg.min_adx)
            return self._no_trade(current_price, current_time, "low_adx")

        # 4. Check ATR percentile — no edge in ultra-low volatility
        if atr_percentile < exec_cfg.min_atr_percentile:
            logger.info("ATR percentile too low (%.2f < %.2f), skipping",
                        atr_percentile, exec_cfg.min_atr_percentile)
            return self._no_trade(current_price, current_time, "low_volatility")

        # 5. Extract latent
        x = torch.from_numpy(features_scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.model.extract_latent(x).cpu().numpy()

        # 6. Build meta-features
        handcrafted = features_scaled[-1:, self.top_feature_indices]
        regime_input = regime_probs[-1:].astype(np.float32)
        meta_features = XGBoostMetaLearner.build_meta_features(
            latent, handcrafted, regime_input
        )

        # 7. XGBoost prediction
        probs = self.meta_learner.predict_proba(meta_features)[0]
        prob_dict = {"short": float(probs[0]), "hold": float(probs[1]), "long": float(probs[2])}

        # 8. Confidence check
        max_prob = max(probs[0], probs[2])
        if max_prob < exec_cfg.confidence_threshold:
            logger.info(
                "Low confidence (max=%.3f < %.3f), skipping",
                max_prob, exec_cfg.confidence_threshold,
            )
            return self._no_trade(current_price, current_time, "low_confidence")

        # 9. Determine direction
        if probs[2] > probs[0]:
            direction = Direction.LONG
            confidence = float(probs[2])
            tp = current_price + exec_cfg.trade_management.full_tp_atr * current_atr
            sl = current_price - self.config.labeling.sl_multiplier * current_atr
        else:
            direction = Direction.SHORT
            confidence = float(probs[0])
            tp = current_price - exec_cfg.trade_management.full_tp_atr * current_atr
            sl = current_price + self.config.labeling.sl_multiplier * current_atr

        # 10. Position sizing (fractional Kelly)
        position_size = self._kelly_size(confidence, exec_cfg)

        self.daily_trade_count += 1

        signal = TradeSignal(
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            take_profit=tp,
            stop_loss=sl,
            position_size=position_size,
            regime=current_regime,
            timestamp=current_time,
            probabilities=prob_dict,
        )

        logger.info(
            "SIGNAL: %s @ %.2f | conf=%.3f | TP=%.2f SL=%.2f | size=%.4f | regime=%s",
            direction.value, current_price, confidence, tp, sl,
            position_size, current_regime,
        )

        return signal

    def _kelly_size(self, confidence: float, exec_cfg) -> float:
        """Fractional Kelly criterion position sizing.

        f = (p*b - q) / b, capped at max_fraction.
        b = TP/SL ratio, p = confidence, q = 1-p.
        """
        b = self.config.labeling.tp_multiplier / self.config.labeling.sl_multiplier
        p = confidence
        q = 1 - p

        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(kelly, 0)  # No negative sizing

        # Apply fractional Kelly and cap
        size = kelly * exec_cfg.position_sizing.kelly_fraction
        return min(size, exec_cfg.position_sizing.max_fraction)

    def _no_trade(
        self, price: float, time: datetime, reason: str
    ) -> TradeSignal:
        """Generate a NO_TRADE signal."""
        return TradeSignal(
            direction=Direction.NO_TRADE,
            confidence=0.0,
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            position_size=0.0,
            regime=reason,
            timestamp=time,
            probabilities={"short": 0.0, "hold": 1.0, "long": 0.0},
        )
