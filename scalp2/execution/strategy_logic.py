"""Shared strategy-planning helpers for live signals and backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from scalp2.config import Config
from scalp2.execution.risk_manager import RiskManager
from scalp2.execution.trade_manager import TradeManager


@dataclass
class PlannedTrade:
    """Result of planning a trade from probabilities and current context."""

    direction: str | None
    confidence: float
    take_profit: float
    stop_loss: float
    position_size: float
    reason: str = ""
    adaptive_tp_sl: dict | None = None


def regime_name_from_probs(regime_probs: np.ndarray) -> str:
    """Map reordered regime probabilities [bull, bear, choppy] to a label."""
    regime_idx = int(np.argmax(regime_probs))
    return ("bull", "bear", "choppy")[regime_idx]


def compute_adaptive_tp_sl(config: Config, atr_percentile: float) -> dict:
    """Compute adaptive TP/SL multipliers from the current ATR percentile."""
    cfg = config.execution.trade_management.adaptive_tp_sl
    if not cfg.enabled:
        return {}

    tm = config.execution.trade_management
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


def apply_structural_adjustments(
    config: Config,
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    atr: float,
    levels: dict | None,
) -> tuple[float, float]:
    """Apply the same structural TP/SL adjustments used by live signals."""
    cfg = config.execution.trade_management.structural_exit
    if not cfg.enabled or atr <= 0:
        return tp, sl

    levels = levels or {}
    is_long = direction == "LONG"
    adjusted_tp = tp
    adjusted_sl = sl

    fvg_target = levels.get("fvg_bear" if is_long else "fvg_bull")
    if fvg_target is not None and not np.isnan(fvg_target):
        if is_long and fvg_target > entry:
            if abs(tp - fvg_target) < cfg.fvg_proximity_atr * atr:
                adjusted_tp = fvg_target
        elif not is_long and fvg_target < entry:
            if abs(tp - fvg_target) < cfg.fvg_proximity_atr * atr:
                adjusted_tp = fvg_target

    swing_level = levels.get("swing_low" if is_long else "swing_high")
    if swing_level is not None and not np.isnan(swing_level):
        buffer = cfg.sweep_buffer_atr * atr
        max_stretch = cfg.max_sl_stretch_atr * atr

        if is_long and abs(adjusted_sl - swing_level) < buffer:
            new_sl = swing_level - buffer
            if abs(entry - new_sl) - abs(entry - sl) <= max_stretch:
                adjusted_sl = new_sl
        elif not is_long and abs(adjusted_sl - swing_level) < buffer:
            new_sl = swing_level + buffer
            if abs(new_sl - entry) - abs(sl - entry) <= max_stretch:
                adjusted_sl = new_sl

    vwap = levels.get("vwap")
    if vwap is not None and not np.isnan(vwap):
        if is_long and vwap > entry:
            if abs(adjusted_tp - vwap) < cfg.vwap_margin_atr * atr and vwap > adjusted_tp:
                adjusted_tp = vwap
        elif not is_long and vwap < entry:
            if abs(adjusted_tp - vwap) < cfg.vwap_margin_atr * atr and vwap < adjusted_tp:
                adjusted_tp = vwap

    return adjusted_tp, adjusted_sl


def compute_kelly_size(
    config: Config,
    confidence: float,
    adaptive: dict | None = None,
    size_modifier: float = 1.0,
) -> float:
    """Compute fractional Kelly size with the live strategy's effective RR math."""
    exec_cfg = config.execution
    tm = exec_cfg.trade_management
    partial_pct = tm.partial_tp_1_pct
    partial_atr = (adaptive or {}).get("adaptive_partial_tp_atr", tm.partial_tp_1_atr)
    full_atr = (adaptive or {}).get("adaptive_full_tp_atr", tm.full_tp_atr)
    effective_tp = partial_pct * partial_atr + (1.0 - partial_pct) * full_atr
    b_ratio = effective_tp / config.labeling.sl_multiplier if config.labeling.sl_multiplier > 0 else 0.0
    p = confidence
    q = 1.0 - p

    kelly = (p * b_ratio - q) / b_ratio if b_ratio > 0 else 0.0
    kelly = max(kelly, 0.0)
    size = kelly * exec_cfg.position_sizing.kelly_fraction
    size = min(size, exec_cfg.position_sizing.max_fraction)
    return size * size_modifier


def plan_trade_from_probabilities(
    config: Config,
    probs: np.ndarray,
    current_regime: str,
    choppy_prob: float,
    current_atr: float,
    current_price: float,
    current_time: datetime,
    current_adx: float = 999.0,
    atr_percentile: float = 1.0,
    structural_levels: dict | None = None,
    daily_trade_count: int = 0,
    trade_manager: TradeManager | None = None,
    risk_manager: RiskManager | None = None,
    confidence_threshold: float | None = None,
) -> PlannedTrade:
    """Plan a trade from already-computed class probabilities."""
    exec_cfg = config.execution
    probs = np.asarray(probs, dtype=np.float32)

    if not np.isfinite(current_price) or current_price <= 0:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "invalid_price")

    if not np.isfinite(current_atr) or current_atr <= 0:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "no_atr")

    if daily_trade_count >= exec_cfg.max_trades_per_day:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "daily_limit")

    is_choppy = choppy_prob > config.regime.choppy_threshold
    if is_choppy and current_adx < exec_cfg.choppy_adx_override:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "choppy")

    if exec_cfg.time_of_day_filter.enabled:
        hour = getattr(current_time, "hour", None)
        if hour is None:
            hour = datetime.fromisoformat(str(current_time)).hour
        if hour in exec_cfg.time_of_day_filter.blocked_hours_utc:
            return PlannedTrade(
                None, 0.0, current_price, current_price, 0.0, f"blocked_time_{hour}"
            )

    if current_adx < exec_cfg.min_adx:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "low_adx")

    if atr_percentile < exec_cfg.min_atr_percentile:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "low_volatility")

    predicted_class = int(np.argmax(probs))
    if predicted_class == 1:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "hold")

    max_prob = max(float(probs[0]), float(probs[2]))
    threshold = exec_cfg.confidence_threshold if confidence_threshold is None else confidence_threshold
    if max_prob < threshold:
        return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "low_confidence")

    direction = "LONG" if probs[2] > probs[0] else "SHORT"
    if exec_cfg.regime_direction_filter:
        if current_regime == "bull" and direction == "SHORT":
            return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "regime_direction")
        if current_regime == "bear" and direction == "LONG":
            return PlannedTrade(None, 0.0, current_price, current_price, 0.0, "regime_direction")

    adaptive = compute_adaptive_tp_sl(config, atr_percentile)
    full_tp_atr = adaptive.get("adaptive_full_tp_atr", exec_cfg.trade_management.full_tp_atr)

    if direction == "LONG":
        confidence = float(probs[2])
        tp = current_price + full_tp_atr * current_atr
        sl = current_price - config.labeling.sl_multiplier * current_atr
    else:
        confidence = float(probs[0])
        tp = current_price - full_tp_atr * current_atr
        sl = current_price + config.labeling.sl_multiplier * current_atr

    original_sl = sl
    tp, sl = apply_structural_adjustments(
        config,
        direction=direction,
        entry=current_price,
        tp=tp,
        sl=sl,
        atr=current_atr,
        levels=structural_levels,
    )

    if trade_manager is not None:
        can_enter, skip_reason = trade_manager.can_enter_trade(
            direction=direction,
            entry_price=current_price,
            current_atr=current_atr,
        )
        if not can_enter:
            return PlannedTrade(None, 0.0, current_price, current_price, 0.0, skip_reason)

    size_modifier = 1.0
    if risk_manager is not None:
        can_trade, risk_reason = risk_manager.can_trade(timestamp=current_time)
        if not can_trade:
            return PlannedTrade(None, 0.0, current_price, current_price, 0.0, risk_reason)
        size_modifier = risk_manager.get_position_size_modifier()

    position_size = compute_kelly_size(
        config,
        confidence=confidence,
        adaptive=adaptive,
        size_modifier=size_modifier,
    )

    struct_cfg = exec_cfg.trade_management.structural_exit
    if struct_cfg.enabled and struct_cfg.normalize_risk and sl != original_sl:
        original_risk = abs(current_price - original_sl)
        new_risk = abs(current_price - sl)
        if new_risk > original_risk and original_risk > 0:
            position_size *= original_risk / new_risk

    return PlannedTrade(
        direction=direction,
        confidence=confidence,
        take_profit=tp,
        stop_loss=sl,
        position_size=position_size,
        adaptive_tp_sl=adaptive if adaptive else None,
    )
