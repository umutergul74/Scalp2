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
from scalp2.execution.trade_manager import TradeManager
from scalp2.execution.risk_manager import RiskManager

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
    market_regime: str = ""  # actual market regime: bull/bear/choppy
    adaptive_tp_sl: dict | None = None  # per-trade adaptive TP/SL overrides


class SignalGenerator:
    """Generate trade signals from the full model pipeline.

    Pipeline:
        1. Check daily trade limit
        2. Check regime — halt if choppy (ADX override)
        3. Check time-of-day filter
        4. Check ADX — skip if no trend
        5. Check ATR percentile — skip low volatility
        6. Extract latent + predict via XGBoost
        7. Apply confidence threshold
        8. Regime-direction filter (no SHORT in bull, no LONG in bear)
        9. Compute adaptive TP/SL
       10. Determine direction & TP/SL
       11. Compute position size (fractional Kelly)
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
        structural_levels: dict | None = None,
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

        # 2. Check regime (forward-only to avoid look-ahead bias)
        regime_probs = self.regime_detector.predict_proba_online(regime_df)
        current_regime = self.regime_detector.current_regime_online(regime_df)

        # 3. Run model inference (always — so we always have probabilities)
        x = torch.from_numpy(features_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if getattr(self.config.model, "bypass_xgboost", False):
                logits, _ = self.model(x)
                # Raw TCN+GRU probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            else:
                latent = self.model.extract_latent(x).cpu().numpy()
                handcrafted = features_scaled[-1:, self.top_feature_indices]
                regime_input = regime_probs[-1:].astype(np.float32)
                meta_features = XGBoostMetaLearner.build_meta_features(
                    latent, handcrafted, regime_input
                )
                probs = self.meta_learner.predict_proba(meta_features)[0]

        prob_dict = {"short": float(probs[0]), "hold": float(probs[1]), "long": float(probs[2])}

        # 4. Check choppy regime (with ADX override)
        is_choppy = regime_probs[-1, RegimeDetector.CHOPPY] > self.config.regime.choppy_threshold
        if is_choppy and current_adx < exec_cfg.choppy_adx_override:
            logger.info("Choppy regime detected (P=%.3f), skipping", regime_probs[-1, 2])
            return self._no_trade(current_price, current_time, "choppy", market_regime=current_regime, probs=prob_dict)
        if is_choppy:
            logger.info(
                "Choppy override: ADX=%.1f >= %.1f, proceeding despite P(choppy)=%.3f",
                current_adx, exec_cfg.choppy_adx_override, regime_probs[-1, 2],
            )

        # 5. Check Time-of-Day Filter
        if exec_cfg.time_of_day_filter.enabled:
            hour = current_time.hour if hasattr(current_time, "hour") else pd.to_datetime(current_time).hour
            if hour in exec_cfg.time_of_day_filter.blocked_hours_utc:
                logger.debug("Blocked time of day (Hour %d UTC), skipping", hour)
                return self._no_trade(current_price, current_time, f"blocked_time_{hour}", market_regime=current_regime, probs=prob_dict)

        # 6. Check ADX — no trend below threshold
        if current_adx < exec_cfg.min_adx:
            logger.info("ADX too low (%.1f < %.1f), skipping", current_adx, exec_cfg.min_adx)
            return self._no_trade(current_price, current_time, "low_adx", market_regime=current_regime, probs=prob_dict)

        # 7. Check ATR percentile — no edge in ultra-low volatility
        if atr_percentile < exec_cfg.min_atr_percentile:
            logger.info("ATR percentile too low (%.2f < %.2f), skipping",
                        atr_percentile, exec_cfg.min_atr_percentile)
            return self._no_trade(current_price, current_time, "low_volatility", market_regime=current_regime, probs=prob_dict)

        # 8. Confidence check
        max_prob = max(probs[0], probs[2])
        if max_prob < exec_cfg.confidence_threshold:
            logger.info(
                "Low confidence (max=%.3f < %.3f), skipping",
                max_prob, exec_cfg.confidence_threshold,
            )
            return self._no_trade(current_price, current_time, "low_confidence", market_regime=current_regime, probs=prob_dict)

        # 9. Regime-direction filter: block SHORT in bull, LONG in bear
        if exec_cfg.regime_direction_filter:
            intended_dir = "LONG" if probs[2] > probs[0] else "SHORT"
            if current_regime == "bull" and intended_dir == "SHORT":
                logger.info("Regime-direction mismatch: SHORT blocked in bull regime")
                return self._no_trade(current_price, current_time, "regime_direction",
                                      market_regime=current_regime, probs=prob_dict)
            if current_regime == "bear" and intended_dir == "LONG":
                logger.info("Regime-direction mismatch: LONG blocked in bear regime")
                return self._no_trade(current_price, current_time, "regime_direction",
                                      market_regime=current_regime, probs=prob_dict)

        # 10. Compute adaptive TP/SL multipliers
        adaptive = self._compute_adaptive_tp_sl(atr_percentile, exec_cfg)
        full_tp_atr = adaptive.get("adaptive_full_tp_atr", exec_cfg.trade_management.full_tp_atr)

        # 11. Determine direction
        if probs[2] > probs[0]:
            direction = Direction.LONG
            confidence = float(probs[2])
            tp = current_price + full_tp_atr * current_atr
            sl = current_price - self.config.labeling.sl_multiplier * current_atr
        else:
            direction = Direction.SHORT
            confidence = float(probs[0])
            tp = current_price - full_tp_atr * current_atr
            sl = current_price + self.config.labeling.sl_multiplier * current_atr

        # 11b. Smart Exit Engine — structural TP/SL adjustments
        original_sl = sl
        tp, sl = self._apply_structural_adjustments(
            direction, current_price, tp, sl, current_atr,
            structural_levels or {},
        )

        # 12. Consecutive SL protection check (Enhancement 1)
        if self.trade_manager is not None:
            can_enter, skip_reason = self.trade_manager.can_enter_trade(
                direction=direction.value,
                entry_price=current_price,
                current_atr=current_atr,
            )
            if not can_enter:
                logger.info("Trade blocked by SL protection: %s", skip_reason)
                return self._no_trade(
                    current_price, current_time, skip_reason,
                    market_regime=current_regime, probs=prob_dict,
                )

        # 13. Portfolio risk check (Enhancement 5)
        if self.risk_manager is not None:
            can_trade, risk_reason = self.risk_manager.can_trade(
                timestamp=current_time,
            )
            if not can_trade:
                logger.info("Trade blocked by risk manager: %s", risk_reason)
                return self._no_trade(
                    current_price, current_time, risk_reason,
                    market_regime=current_regime, probs=prob_dict,
                )

        # 14. Position sizing (fractional Kelly) with risk modifier
        position_size = self._kelly_size(confidence, exec_cfg, adaptive)
        if self.risk_manager is not None:
            size_modifier = self.risk_manager.get_position_size_modifier()
            if size_modifier < 1.0:
                position_size *= size_modifier
                logger.info("Position size reduced by risk: %.2f × %.3f = %.3f",
                            size_modifier, position_size / size_modifier, position_size)

        # 14b. Option A — Risk Normalization: if SL was widened by structural
        #      adjustment, shrink position proportionally to keep $ risk constant.
        struct_cfg = exec_cfg.trade_management.structural_exit
        if struct_cfg.enabled and struct_cfg.normalize_risk and sl != original_sl:
            original_risk = abs(current_price - original_sl)
            new_risk = abs(current_price - sl)
            if new_risk > original_risk and original_risk > 0:
                risk_ratio = original_risk / new_risk
                logger.info(
                    "Risk normalization (Option A): SL widened %.1f → %.1f, "
                    "size %.4f × %.3f = %.4f",
                    original_sl, sl, position_size, risk_ratio,
                    position_size * risk_ratio,
                )
                position_size *= risk_ratio

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
            market_regime=current_regime,
            adaptive_tp_sl=adaptive if adaptive else None,
        )

        logger.info(
            "SIGNAL: %s @ %.2f | conf=%.3f | TP=%.2f SL=%.2f | size=%.4f | regime=%s%s",
            direction.value, current_price, confidence, tp, sl,
            position_size, current_regime,
            f" | adaptive(tp={full_tp_atr:.2f})" if adaptive else "",
        )

        return signal

    def _compute_adaptive_tp_sl(self, atr_percentile: float, exec_cfg) -> dict:
        """Compute adaptive TP/SL multipliers based on ATR percentile."""
        cfg = exec_cfg.trade_management.adaptive_tp_sl
        if not cfg.enabled:
            return {}

        tm = exec_cfg.trade_management
        tp_scale = sl_scale = 1.0

        if atr_percentile > cfg.high_vol_pctile:
            tp_scale = cfg.high_vol_tp_scale
            sl_scale = cfg.high_vol_sl_scale
        elif atr_percentile < cfg.low_vol_pctile:
            tp_scale = cfg.low_vol_tp_scale
            sl_scale = cfg.low_vol_sl_scale

        return {
            "adaptive_partial_tp_atr": tm.partial_tp_1_atr * tp_scale,
            "adaptive_full_tp_atr": tm.full_tp_atr * tp_scale,
            "adaptive_trailing_act_atr": tm.trailing_activation_atr * tp_scale,
            "adaptive_trailing_dist_atr": tm.trailing_distance_atr * sl_scale,
        }

    def _kelly_size(self, confidence: float, exec_cfg, adaptive: dict | None = None) -> float:
        """Fractional Kelly criterion position sizing.

        f = (p*b - q) / b, capped at max_fraction.
        b = effective TP/SL ratio accounting for partial TP exits.
        """
        tm = exec_cfg.trade_management
        partial_pct = tm.partial_tp_1_pct
        partial_atr = (adaptive or {}).get("adaptive_partial_tp_atr", tm.partial_tp_1_atr)
        full_atr = (adaptive or {}).get("adaptive_full_tp_atr", tm.full_tp_atr)
        effective_tp = partial_pct * partial_atr + (1 - partial_pct) * full_atr
        b = effective_tp / self.config.labeling.sl_multiplier
        p = confidence
        q = 1 - p

        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(kelly, 0)  # No negative sizing

        # Apply fractional Kelly and cap
        size = kelly * exec_cfg.position_sizing.kelly_fraction
        return min(size, exec_cfg.position_sizing.max_fraction)

    def _apply_structural_adjustments(
        self,
        direction: Direction,
        entry: float,
        tp: float,
        sl: float,
        atr: float,
        levels: dict,
    ) -> tuple[float, float]:
        """Adjust TP/SL based on structural market levels.

        A. FVG-Aware TP: if TP is within `fvg_proximity_atr` of an unfilled
           FVG, extend TP into/beyond the gap.
        B. Sweep-Resistant SL: if SL coincides with a swing high/low,
           nudge SL behind the structural level.
        C. VWAP-Anchored TP: if TP is heading towards VWAP and within
           margin, extend TP to VWAP.

        Returns:
            (adjusted_tp, adjusted_sl)
        """
        cfg = self.config.execution.trade_management.structural_exit
        if not cfg.enabled or atr <= 0:
            return tp, sl

        is_long = direction == Direction.LONG
        adjusted_tp = tp
        adjusted_sl = sl

        # ── A. FVG-Aware TP ───────────────────────────────────────────────
        fvg_target = levels.get("fvg_bear" if is_long else "fvg_bull")
        if fvg_target is not None and not np.isnan(fvg_target):
            # For LONG: bearish FVG above is a magnet (price gets pulled up)
            # For SHORT: bullish FVG below is a magnet (price gets pulled down)
            if is_long and fvg_target > entry:
                dist_to_fvg = abs(tp - fvg_target)
                if dist_to_fvg < cfg.fvg_proximity_atr * atr:
                    # TP is close to FVG — extend into the gap
                    adjusted_tp = fvg_target
                    logger.info(
                        "FVG stretch (LONG): TP %.1f → %.1f (FVG magnetic pull)",
                        tp, adjusted_tp,
                    )
            elif not is_long and fvg_target < entry:
                dist_to_fvg = abs(tp - fvg_target)
                if dist_to_fvg < cfg.fvg_proximity_atr * atr:
                    adjusted_tp = fvg_target
                    logger.info(
                        "FVG stretch (SHORT): TP %.1f → %.1f (FVG magnetic pull)",
                        tp, adjusted_tp,
                    )

        # ── B. Sweep-Resistant SL ─────────────────────────────────────────
        swing_level = levels.get("swing_low" if is_long else "swing_high")
        if swing_level is not None and not np.isnan(swing_level):
            buffer = cfg.sweep_buffer_atr * atr
            max_stretch = cfg.max_sl_stretch_atr * atr

            if is_long:
                # LONG SL is below entry. Danger: SL sits right on swing low pool.
                if abs(adjusted_sl - swing_level) < buffer:
                    # Nudge SL below the swing low + buffer
                    new_sl = swing_level - buffer
                    # Cap the stretch so we don't balloon risk infinitely
                    if abs(entry - new_sl) - abs(entry - sl) <= max_stretch:
                        logger.info(
                            "Sweep shield (LONG): SL %.1f → %.1f (behind swing low %.1f)",
                            adjusted_sl, new_sl, swing_level,
                        )
                        adjusted_sl = new_sl
            else:
                # SHORT SL is above entry. Danger: SL sits on swing high pool.
                if abs(adjusted_sl - swing_level) < buffer:
                    new_sl = swing_level + buffer
                    if abs(new_sl - entry) - abs(sl - entry) <= max_stretch:
                        logger.info(
                            "Sweep shield (SHORT): SL %.1f → %.1f (behind swing high %.1f)",
                            adjusted_sl, new_sl, swing_level,
                        )
                        adjusted_sl = new_sl

        # ── C. VWAP-Anchored TP ───────────────────────────────────────────
        vwap = levels.get("vwap")
        if vwap is not None and not np.isnan(vwap):
            if is_long and vwap > entry:
                dist_to_vwap = abs(adjusted_tp - vwap)
                if dist_to_vwap < cfg.vwap_margin_atr * atr and vwap > adjusted_tp:
                    logger.info(
                        "VWAP stretch (LONG): TP %.1f → %.1f (VWAP anchor)",
                        adjusted_tp, vwap,
                    )
                    adjusted_tp = vwap
            elif not is_long and vwap < entry:
                dist_to_vwap = abs(adjusted_tp - vwap)
                if dist_to_vwap < cfg.vwap_margin_atr * atr and vwap < adjusted_tp:
                    logger.info(
                        "VWAP stretch (SHORT): TP %.1f → %.1f (VWAP anchor)",
                        adjusted_tp, vwap,
                    )
                    adjusted_tp = vwap

        return adjusted_tp, adjusted_sl

    def _no_trade(
        self, price: float, time: datetime, reason: str,
        market_regime: str = "", probs: dict | None = None,
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
            probabilities=probs or {"short": 0.0, "hold": 1.0, "long": 0.0},
            market_regime=market_regime,
        )
