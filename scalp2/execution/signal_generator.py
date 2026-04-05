"""Full inference pipeline from prepared features to a trade signal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import torch

from scalp2.config import Config
from scalp2.execution.risk_manager import RiskManager
from scalp2.execution.strategy_logic import (
    apply_structural_adjustments,
    compute_adaptive_tp_sl,
    compute_kelly_size,
    plan_trade_from_probabilities,
)
from scalp2.execution.trade_manager import TradeManager
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
    probabilities: dict
    market_regime: str = ""
    adaptive_tp_sl: dict | None = None


class SignalGenerator:
    """Generate trade signals from the full live inference pipeline."""

    def __init__(
        self,
        config: Config,
        model: HybridEncoder,
        meta_learner: XGBoostMetaLearner,
        regime_detector: RegimeDetector,
        scaler,
        top_feature_indices: np.ndarray,
        device: torch.device | None = None,
        trade_manager: TradeManager | None = None,
        risk_manager: RiskManager | None = None,
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
        self.trade_manager = trade_manager
        self.risk_manager = risk_manager

    def _reset_daily_counter(self, current_time: datetime) -> None:
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
        structural_levels: dict | None = None,
    ) -> TradeSignal:
        """Generate a trade signal from prepared features."""
        self._reset_daily_counter(current_time)

        if self.daily_trade_count >= self.config.execution.max_trades_per_day:
            logger.info(
                "Daily trade limit reached (%d)",
                self.config.execution.max_trades_per_day,
            )
            return self._no_trade(current_price, current_time, "daily_limit")

        regime_probs = self.regime_detector.predict_proba_online(regime_df)
        current_regime = self.regime_detector.current_regime_online(regime_df)

        x = torch.from_numpy(features_scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.model.extract_latent(x).cpu().numpy()

        handcrafted = features_scaled[-1:, self.top_feature_indices]
        regime_input = regime_probs[-1:].astype(np.float32)
        meta_features = XGBoostMetaLearner.build_meta_features(
            latent,
            handcrafted,
            regime_input,
        )

        probs = self.meta_learner.predict_proba(meta_features)[0]
        prob_dict = {
            "short": float(probs[0]),
            "hold": float(probs[1]),
            "long": float(probs[2]),
        }

        planned = plan_trade_from_probabilities(
            config=self.config,
            probs=probs,
            current_regime=current_regime,
            choppy_prob=float(regime_probs[-1, RegimeDetector.CHOPPY]),
            current_atr=current_atr,
            current_price=current_price,
            current_time=current_time,
            current_adx=current_adx,
            atr_percentile=atr_percentile,
            structural_levels=structural_levels or {},
            daily_trade_count=self.daily_trade_count,
            trade_manager=self.trade_manager,
            risk_manager=self.risk_manager,
        )
        if planned.direction is None:
            return self._no_trade(
                current_price,
                current_time,
                planned.reason,
                market_regime=current_regime,
                probs=prob_dict,
            )

        self.daily_trade_count += 1

        signal = TradeSignal(
            direction=Direction(planned.direction),
            confidence=planned.confidence,
            entry_price=current_price,
            take_profit=planned.take_profit,
            stop_loss=planned.stop_loss,
            position_size=planned.position_size,
            regime=current_regime,
            timestamp=current_time,
            probabilities=prob_dict,
            market_regime=current_regime,
            adaptive_tp_sl=planned.adaptive_tp_sl,
        )

        logger.info(
            "SIGNAL: %s @ %.2f | conf=%.3f | TP=%.2f SL=%.2f | size=%.4f | regime=%s%s",
            signal.direction.value,
            current_price,
            signal.confidence,
            signal.take_profit,
            signal.stop_loss,
            signal.position_size,
            current_regime,
            (
                f" | adaptive(tp={signal.adaptive_tp_sl['adaptive_full_tp_atr']:.2f})"
                if signal.adaptive_tp_sl and "adaptive_full_tp_atr" in signal.adaptive_tp_sl
                else ""
            ),
        )

        return signal

    def _compute_adaptive_tp_sl(self, atr_percentile: float, exec_cfg) -> dict:
        """Compatibility wrapper for older callers/tests."""
        return compute_adaptive_tp_sl(self.config, atr_percentile)

    def _kelly_size(
        self,
        confidence: float,
        exec_cfg,
        adaptive: dict | None = None,
    ) -> float:
        """Compatibility wrapper for older callers/tests."""
        return compute_kelly_size(self.config, confidence, adaptive=adaptive)

    def _apply_structural_adjustments(
        self,
        direction: Direction,
        entry: float,
        tp: float,
        sl: float,
        atr: float,
        levels: dict,
    ) -> tuple[float, float]:
        """Compatibility wrapper for older callers/tests."""
        return apply_structural_adjustments(
            self.config,
            direction=direction.value,
            entry=entry,
            tp=tp,
            sl=sl,
            atr=atr,
            levels=levels,
        )

    def _no_trade(
        self,
        price: float,
        time: datetime,
        reason: str,
        market_regime: str = "",
        probs: dict | None = None,
    ) -> TradeSignal:
        return TradeSignal(
            direction=Direction.NO_TRADE,
            confidence=0.0,
            entry_price=price,
            take_profit=price,
            stop_loss=price,
            position_size=0.0,
            regime=reason,
            timestamp=time,
            probabilities=probs or {"short": 0.0, "hold": 1.0, "long": 0.0},
            market_regime=market_regime,
        )
