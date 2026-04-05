"""Shared walk-forward backtest engine aligned with current execution logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from scalp2.config import Config
from scalp2.execution.risk_manager import RiskManager
from scalp2.execution.strategy_logic import (
    plan_trade_from_probabilities,
    regime_name_from_probs,
)
from scalp2.execution.trade_manager import TradeManager, TradeState, TradeStatus
from scalp2.live.paper import normalize_close_reason, protection_status_for_close

DEFAULT_INITIAL_BALANCE = 1000.0


@dataclass
class PredictionPoint:
    """Prediction payload aligned to a single signal bar."""

    bar_index: int
    probs: np.ndarray
    regime_probs: np.ndarray
    fold_idx: int | None = None


def _atr_column(config: Config, df: pd.DataFrame) -> str:
    configured = f"atr_{config.labeling.atr_period}"
    if configured in df.columns:
        return configured
    if "atr_14" in df.columns:
        return "atr_14"
    return configured


def _adx_value(row: pd.Series) -> float:
    if "adx" in row.index:
        return float(row["adx"])
    if "adx_14" in row.index:
        return float(row["adx_14"])
    return 999.0


def _ensure_backtest_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Ensure the dataframe has the columns the backtest logic expects."""
    df_bt = df.copy()
    atr_col = _atr_column(config, df_bt)
    if atr_col in df_bt.columns and "atr_pctile" not in df_bt.columns:
        df_bt["atr_pctile"] = (
            df_bt[atr_col].rolling(96, min_periods=10).rank(pct=True).fillna(1.0)
        )
    elif "atr_pctile" not in df_bt.columns:
        df_bt["atr_pctile"] = 1.0
    return df_bt


def _structural_levels(row: pd.Series) -> dict:
    """Extract structural levels in the same shape used by live execution."""
    return {
        "vwap": float(row.get("vwap", np.nan)),
        "fvg_bull": float(row.get("fvg_bull_price", np.nan)),
        "fvg_bear": float(row.get("fvg_bear_price", np.nan)),
        "swing_high": float(row.get("swing_high_price", np.nan)),
        "swing_low": float(row.get("swing_low_price", np.nan)),
    }


def _short_reason(reason: str) -> str:
    """Compress verbose skip reasons into stable counter keys."""
    if not reason:
        return "unknown"
    short = reason.split(" ", 1)[0]
    short = short.split("(", 1)[0]
    return short.strip() or "unknown"


def _funding_intervals(entry_time, exit_time) -> int:
    """Count 8h funding timestamps crossed while the trade was open."""
    if not isinstance(entry_time, pd.Timestamp):
        entry_time = pd.Timestamp(entry_time)
    if not isinstance(exit_time, pd.Timestamp):
        exit_time = pd.Timestamp(exit_time)
    funding_times = pd.date_range(
        entry_time.normalize(),
        exit_time.normalize() + pd.Timedelta(days=1),
        freq="8h",
    )
    return int(((funding_times > entry_time) & (funding_times <= exit_time)).sum())


def flatten_walk_forward_predictions(
    wf_predictions: list[dict],
    seq_len: int,
    n_rows: int,
) -> dict[int, PredictionPoint]:
    """Flatten fold predictions into a chronological per-bar lookup."""
    prediction_map: dict[int, PredictionPoint] = {}
    for fold_data in wf_predictions:
        preds = fold_data["test_probabilities"]
        regime = fold_data.get(
            "regime_probs",
            np.full((len(preds), 3), 1.0 / 3.0, dtype=np.float32),
        )
        offset = fold_data["test_start"] + seq_len
        valid_len = min(len(preds), len(regime), max(n_rows - offset, 0))
        for local_idx in range(valid_len):
            bar_index = offset + local_idx
            prediction_map[bar_index] = PredictionPoint(
                bar_index=bar_index,
                probs=np.asarray(preds[local_idx], dtype=np.float32),
                regime_probs=np.asarray(regime[local_idx], dtype=np.float32),
                fold_idx=fold_data.get("fold_idx"),
            )
    return prediction_map


def _entry_sl_tp_from_plan(
    direction: str,
    planned_entry: float,
    actual_entry: float,
    planned_sl: float,
    planned_tp: float,
) -> tuple[float, float]:
    """Shift planned SL/TP distances to the actual next-bar entry price."""
    sl_dist = abs(planned_entry - planned_sl)
    tp_dist = abs(planned_entry - planned_tp)
    if direction == "LONG":
        return actual_entry - sl_dist, actual_entry + tp_dist
    return actual_entry + sl_dist, actual_entry - tp_dist


def _slippage_bps(config: Config, atr_value: float, median_atr: float) -> float:
    order_cfg = config.execution.order_execution
    slip_cfg = getattr(config.execution, "slippage_model", None)
    use_variable = slip_cfg is not None and slip_cfg.enabled
    if not use_variable:
        return float(order_cfg.slippage_bps)
    ratio = atr_value / (median_atr + 1e-10)
    return float(slip_cfg.base_bps + slip_cfg.volatility_scale * ratio)


def _impact_frac(config: Config, position_notional_usd: float) -> float:
    impact_cfg = getattr(config.execution, "market_impact", None)
    if impact_cfg is None or not impact_cfg.enabled or position_notional_usd <= 0:
        return 0.0
    return (
        impact_cfg.base_impact_bps
        * (position_notional_usd / impact_cfg.reference_notional_usd)
        / 10_000.0
    )


def _funding_cost_usd(
    config: Config,
    position_notional_usd: float,
    entry_time,
    exit_time,
) -> float:
    funding_cfg = getattr(config.execution, "funding_rate", None)
    if funding_cfg is None or not funding_cfg.enabled or position_notional_usd <= 0:
        return 0.0
    return (
        _funding_intervals(entry_time, exit_time)
        * (funding_cfg.fixed_rate_pct / 100.0)
        * position_notional_usd
    )


def _execution_cost_usd(
    config: Config,
    slip_bps: float,
    position_notional_usd: float,
    partial_fills: list[dict],
) -> float:
    """All live orders are modeled as market-like fills with taker fees."""
    if position_notional_usd <= 0:
        return 0.0
    taker_bps = config.execution.order_execution.taker_fee_bps
    fill_cost_frac = (taker_bps + slip_bps) / 10_000.0
    partial_fraction = float(sum(fill["size"] for fill in partial_fills))
    remaining_fraction = max(0.0, 1.0 - partial_fraction)
    total_fraction = 1.0 + partial_fraction + remaining_fraction
    return position_notional_usd * fill_cost_frac * total_fraction


def _m2m_trade_return_frac(trade: TradeState, price: float) -> float:
    if trade.direction == "LONG":
        live_return = (price - trade.entry_price) / trade.entry_price
    else:
        live_return = (trade.entry_price - price) / trade.entry_price
    return trade.pnl + live_return * trade.remaining_size


def _close_trade_balance_impact(
    config: Config,
    *,
    trade: TradeState,
    trade_meta: dict,
    current_balance: float,
    initial_balance: float,
    pnl_frac: float,
    exit_time,
    median_atr: float,
) -> dict:
    """Apply execution costs/funding and convert a realized trade into equity impact."""
    position_notional_usd = trade_meta["position_notional_usd"]
    slip_bps = _slippage_bps(config, trade.atr_at_entry, median_atr)
    gross_pnl_usd = pnl_frac * position_notional_usd
    cost_usd = _execution_cost_usd(
        config,
        slip_bps=slip_bps,
        position_notional_usd=position_notional_usd,
        partial_fills=trade.partial_fills,
    )
    impact_usd = _impact_frac(config, position_notional_usd) * position_notional_usd
    cost_usd += impact_usd
    funding_usd = _funding_cost_usd(
        config,
        position_notional_usd,
        entry_time=trade_meta["entry_time"],
        exit_time=exit_time,
    )
    net_pnl_usd = gross_pnl_usd - cost_usd - funding_usd
    next_balance = max(current_balance + net_pnl_usd, 0.0)
    equity_pnl_pct = (
        net_pnl_usd / trade_meta["entry_equity"] * 100.0
        if trade_meta["entry_equity"] > 0
        else 0.0
    )
    return {
        "slip_bps": slip_bps,
        "gross_pnl_usd": gross_pnl_usd,
        "cost_usd": cost_usd,
        "funding_usd": funding_usd,
        "net_pnl_usd": net_pnl_usd,
        "next_balance": next_balance,
        "equity_pnl_pct": equity_pnl_pct,
        "impact_bps": impact_usd / position_notional_usd * 10_000.0
        if position_notional_usd > 0
        else 0.0,
    }


def _simulate_prediction_stream(
    df: pd.DataFrame,
    prediction_map: dict[int, PredictionPoint],
    config: Config,
    *,
    signal_start_bar: int = 0,
    signal_end_bar: int | None = None,
    confidence_threshold: float | None = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    label: str = "",
    reset_guards_on_fold_change: bool = False,
) -> dict:
    """Replay the strategy over a chronological stream of class probabilities."""
    df_bt = _ensure_backtest_columns(df, config)
    atr_col = _atr_column(config, df_bt)
    median_atr = (
        float(df_bt[atr_col].median())
        if atr_col in df_bt.columns and not df_bt.empty
        else 1.0
    )
    leverage = config.execution.position_sizing.leverage

    trade_mgr = TradeManager(
        config.execution.trade_management,
        config.labeling.max_holding_bars,
    )
    risk_mgr = RiskManager(config=config.execution)

    skip_reasons: dict[str, int] = defaultdict(int)
    trades: list[dict] = []
    equity_curve = [0.0]
    bar_equity_curve = [0.0]

    current_balance = float(initial_balance)
    active: TradeState | None = None
    active_meta: dict | None = None
    pending: dict | None = None
    current_fold_idx: int | None = None
    pending_fold_reset: int | None = None
    daily_trade_count = 0
    prev_date = None
    liquidated = False

    start_bar = max(0, signal_start_bar)
    end_bar = len(df_bt) - 1 if signal_end_bar is None else min(signal_end_bar, len(df_bt) - 1)

    for bar in range(start_bar, end_bar + 1):
        row = df_bt.iloc[bar]
        current_time = row.name
        current_date = current_time.date() if hasattr(current_time, "date") else None
        if current_date != prev_date:
            daily_trade_count = 0
            prev_date = current_date

        prediction = prediction_map.get(bar)
        incoming_fold_idx = prediction.fold_idx if prediction is not None else None
        if reset_guards_on_fold_change and incoming_fold_idx is not None:
            if current_fold_idx is None:
                current_fold_idx = incoming_fold_idx
            elif incoming_fold_idx != current_fold_idx:
                pending_fold_reset = incoming_fold_idx

        trade_mgr.advance_bar()

        if pending is not None and pending["entry_bar"] == bar:
            actual_entry = float(row["open"])
            actual_sl, actual_tp = _entry_sl_tp_from_plan(
                direction=pending["direction"],
                planned_entry=pending["planned_entry"],
                actual_entry=actual_entry,
                planned_sl=pending["planned_sl"],
                planned_tp=pending["planned_tp"],
            )
            entry_equity = current_balance
            position_size_frac = pending["position_size"]
            position_notional_usd = entry_equity * position_size_frac * leverage
            active = TradeState(
                direction=pending["direction"],
                entry_price=actual_entry,
                current_stop_loss=actual_sl,
                take_profit=actual_tp,
                atr_at_entry=pending["atr"],
                adaptive_partial_tp_atr=pending["adaptive"].get("adaptive_partial_tp_atr"),
                adaptive_full_tp_atr=pending["adaptive"].get("adaptive_full_tp_atr"),
                adaptive_trailing_act_atr=pending["adaptive"].get("adaptive_trailing_act_atr"),
                adaptive_trailing_dist_atr=pending["adaptive"].get("adaptive_trailing_dist_atr"),
            )
            active_meta = {
                "entry_equity": entry_equity,
                "position_size_frac": position_size_frac,
                "position_notional_usd": position_notional_usd,
                "entry_bar": bar,
                "entry_time": current_time,
                "fold_idx": pending.get("fold_idx"),
            }
            pending = None

        if active is not None and active_meta is not None:
            prediction = prediction_map.get(bar)
            choppy_prob = float(prediction.regime_probs[2]) if prediction is not None else 0.0
            current_adx = _adx_value(row)
            is_choppy = (
                choppy_prob > config.regime.choppy_threshold
                and current_adx < config.execution.choppy_adx_override
            )

            active = trade_mgr.update(
                active,
                current_high=float(row["high"]),
                current_low=float(row["low"]),
                current_close=float(row["close"]),
                is_choppy=is_choppy,
                structural_levels=_structural_levels(row),
            )

            if active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):
                raw_reason = {
                    TradeStatus.CLOSED_TP: "TP",
                    TradeStatus.CLOSED_SL: "SL",
                    TradeStatus.CLOSED_TIME: "TIME",
                    TradeStatus.CLOSED_REGIME: "REGIME",
                }.get(active.status, "REGIME")
                close_reason = normalize_close_reason(raw_reason, active.pnl)
                if close_reason == "TP":
                    exit_price = active.take_profit
                elif close_reason in {"SL", "TRAIL", "BE"}:
                    exit_price = active.current_stop_loss
                else:
                    exit_price = float(row["close"])

                accounting = _close_trade_balance_impact(
                    config,
                    trade=active,
                    trade_meta=active_meta,
                    current_balance=current_balance,
                    initial_balance=initial_balance,
                    pnl_frac=active.pnl,
                    exit_time=current_time,
                    median_atr=median_atr,
                )
                current_balance = accounting["next_balance"]

                trade_mgr.record_trade_result(
                    status=protection_status_for_close(close_reason, active.pnl),
                    exit_price=exit_price,
                    direction=active.direction,
                    atr=active.atr_at_entry,
                )
                equity_pnl_pct = accounting["equity_pnl_pct"]
                risk_mgr.record_trade(timestamp=pd.Timestamp(current_time), pnl_pct=equity_pnl_pct)

                net_frac = accounting["net_pnl_usd"] / initial_balance
                trades.append(
                    {
                        "fold": active_meta.get("fold_idx"),
                        "direction": active.direction,
                        "entry_price": active.entry_price,
                        "bars_held": active.bars_held,
                        "status": close_reason,
                        "gross_pnl": accounting["gross_pnl_usd"] / initial_balance,
                        "unit_pnl": (accounting["net_pnl_usd"] / leverage) / initial_balance,
                        "net_pnl": net_frac,
                        "cost": accounting["cost_usd"] / initial_balance,
                        "funding_cost": accounting["funding_usd"] / initial_balance,
                        "position_size": active_meta["position_size_frac"],
                        "margin_utilization": active_meta["position_size_frac"] * leverage,
                        "slippage_bps": accounting["slip_bps"],
                        "impact_bps": accounting["impact_bps"],
                        "n_exits": 1 + len(active.partial_fills),
                        "entry_bar": active_meta["entry_bar"],
                        "exit_bar": bar,
                        "timestamp": current_time,
                        "entry_equity": active_meta["entry_equity"],
                        "balance_after": current_balance,
                    }
                )
                equity_curve.append((current_balance - initial_balance) / initial_balance)
                bar_equity_curve.append((current_balance - initial_balance) / initial_balance)
                active = None
                active_meta = None

                if current_balance <= 0:
                    liquidated = True
                    break
            else:
                mtm_usd = _m2m_trade_return_frac(active, float(row["close"])) * active_meta["position_notional_usd"]
                mtm_cost_usd = _execution_cost_usd(
                    config,
                    slip_bps=_slippage_bps(config, active.atr_at_entry, median_atr),
                    position_notional_usd=active_meta["position_notional_usd"],
                    partial_fills=active.partial_fills,
                )
                mtm_cost_usd += _impact_frac(config, active_meta["position_notional_usd"]) * active_meta["position_notional_usd"]
                mtm_equity = current_balance - mtm_cost_usd + mtm_usd
                bar_equity_curve.append((mtm_equity - initial_balance) / initial_balance)

        elif active is None:
            bar_equity_curve.append((current_balance - initial_balance) / initial_balance)

        if (
            reset_guards_on_fold_change
            and pending_fold_reset is not None
            and active is None
            and pending is None
        ):
            trade_mgr = TradeManager(
                config.execution.trade_management,
                config.labeling.max_holding_bars,
            )
            risk_mgr = RiskManager(config=config.execution)
            current_fold_idx = pending_fold_reset
            pending_fold_reset = None
            daily_trade_count = 0

        if active is not None:
            continue

        if prediction is None:
            continue

        current_atr = float(row.get(atr_col, 0.0))
        planned = plan_trade_from_probabilities(
            config=config,
            probs=prediction.probs,
            current_regime=regime_name_from_probs(prediction.regime_probs),
            choppy_prob=float(prediction.regime_probs[2]),
            current_atr=current_atr,
            current_price=float(row["close"]),
            current_time=pd.Timestamp(current_time),
            current_adx=_adx_value(row),
            atr_percentile=float(row.get("atr_pctile", 1.0)),
            structural_levels=_structural_levels(row),
            daily_trade_count=daily_trade_count,
            trade_manager=trade_mgr,
            risk_manager=risk_mgr,
            confidence_threshold=confidence_threshold,
        )
        if planned.direction is None:
            skip_reasons[_short_reason(planned.reason)] += 1
            continue

        position_notional_usd = current_balance * planned.position_size * leverage
        if position_notional_usd < 10.0:
            skip_reasons["position_too_small"] += 1
            continue

        if bar + 1 > end_bar:
            skip_reasons["no_next_bar"] += 1
            continue

        daily_trade_count += 1
        pending = {
            "direction": planned.direction,
            "planned_entry": float(row["close"]),
            "planned_sl": planned.stop_loss,
            "planned_tp": planned.take_profit,
            "atr": current_atr,
            "adaptive": planned.adaptive_tp_sl or {},
            "position_size": planned.position_size,
            "entry_bar": bar + 1,
            "signal_bar": bar,
            "fold_idx": prediction.fold_idx,
        }

    if active is not None and active_meta is not None and not liquidated:
        final_row = df_bt.iloc[end_bar]
        final_time = final_row.name
        exit_price = float(final_row["close"])
        close_reason = "EOD"
        pnl_frac = _m2m_trade_return_frac(active, exit_price)
        accounting = _close_trade_balance_impact(
            config,
            trade=active,
            trade_meta=active_meta,
            current_balance=current_balance,
            initial_balance=initial_balance,
            pnl_frac=pnl_frac,
            exit_time=final_time,
            median_atr=median_atr,
        )
        current_balance = accounting["next_balance"]
        trade_mgr.record_trade_result(
            status=TradeStatus.CLOSED_TIME,
            exit_price=exit_price,
            direction=active.direction,
            atr=active.atr_at_entry,
        )
        equity_pnl_pct = accounting["equity_pnl_pct"]
        risk_mgr.record_trade(timestamp=pd.Timestamp(final_time), pnl_pct=equity_pnl_pct)

        trades.append(
            {
                "fold": active_meta.get("fold_idx"),
                "direction": active.direction,
                "entry_price": active.entry_price,
                "bars_held": active.bars_held,
                "status": close_reason,
                "gross_pnl": accounting["gross_pnl_usd"] / initial_balance,
                "unit_pnl": (accounting["net_pnl_usd"] / leverage) / initial_balance,
                "net_pnl": accounting["net_pnl_usd"] / initial_balance,
                "cost": accounting["cost_usd"] / initial_balance,
                "funding_cost": accounting["funding_usd"] / initial_balance,
                "position_size": active_meta["position_size_frac"],
                "margin_utilization": active_meta["position_size_frac"] * leverage,
                "slippage_bps": accounting["slip_bps"],
                "impact_bps": accounting["impact_bps"],
                "n_exits": 1 + len(active.partial_fills),
                "entry_bar": active_meta["entry_bar"],
                "exit_bar": end_bar,
                "timestamp": final_time,
                "entry_equity": active_meta["entry_equity"],
                "balance_after": current_balance,
            }
        )
        equity_curve.append((current_balance - initial_balance) / initial_balance)
        bar_equity_curve.append((current_balance - initial_balance) / initial_balance)

    trades_df = pd.DataFrame(trades)
    cumulative_pnl = (current_balance - initial_balance) / initial_balance
    return {
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "bar_equity_curve": bar_equity_curve,
        "cumulative_pnl": cumulative_pnl,
        "skip_reasons": dict(skip_reasons),
        "liquidated": liquidated,
        "current_balance": current_balance,
        "initial_balance": initial_balance,
        "leverage": leverage,
        "label": label,
    }


def simulate_walk_forward_backtest(
    df: pd.DataFrame,
    wf_predictions: list[dict],
    config: Config,
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
) -> dict:
    """Backtest sequential walk-forward predictions without fold-boundary resets."""
    prediction_map = flatten_walk_forward_predictions(
        wf_predictions,
        seq_len=config.model.seq_len,
        n_rows=len(df),
    )
    if not prediction_map:
        return {
            "trades_df": pd.DataFrame(),
            "equity_curve": [0.0],
            "bar_equity_curve": [0.0],
            "cumulative_pnl": 0.0,
            "skip_reasons": {},
            "liquidated": False,
            "current_balance": initial_balance,
            "initial_balance": initial_balance,
            "leverage": config.execution.position_sizing.leverage,
            "label": "walk_forward",
        }
    start_bar = min(prediction_map)
    return _simulate_prediction_stream(
        df,
        prediction_map,
        config,
        signal_start_bar=start_bar,
        initial_balance=initial_balance,
        label="walk_forward",
        reset_guards_on_fold_change=True,
    )


def simulate_forward_backtest(
    df: pd.DataFrame,
    probs: np.ndarray,
    regime_probs: np.ndarray,
    config: Config,
    *,
    signal_start_bar: int = 0,
    confidence_threshold: float | None = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
) -> dict:
    """Backtest a single contiguous prediction stream, e.g. forward test bars."""
    prediction_map = {
        idx: PredictionPoint(
            bar_index=idx,
            probs=np.asarray(probs[idx], dtype=np.float32),
            regime_probs=np.asarray(regime_probs[idx], dtype=np.float32),
            fold_idx=None,
        )
        for idx in range(min(len(probs), len(regime_probs), len(df)))
    }
    return _simulate_prediction_stream(
        df,
        prediction_map,
        config,
        signal_start_bar=signal_start_bar,
        confidence_threshold=confidence_threshold,
        initial_balance=initial_balance,
        label="forward_test",
    )


def build_signal_audit_table(
    df: pd.DataFrame,
    probs: np.ndarray,
    regime_probs: np.ndarray,
    config: Config,
    *,
    signal_start_bar: int = 0,
    confidence_threshold: float | None = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
) -> pd.DataFrame:
    """Create an audit table aligned with the sequential backtest execution path."""
    df_bt = _ensure_backtest_columns(df, config)
    atr_col = _atr_column(config, df_bt)
    median_atr = (
        float(df_bt[atr_col].median())
        if atr_col in df_bt.columns and not df_bt.empty
        else 1.0
    )
    leverage = config.execution.position_sizing.leverage
    rows: list[dict] = []

    trade_mgr = TradeManager(
        config.execution.trade_management,
        config.labeling.max_holding_bars,
    )
    risk_mgr = RiskManager(config=config.execution)

    current_balance = float(initial_balance)
    active: TradeState | None = None
    active_meta: dict | None = None
    pending: dict | None = None
    daily_trade_count = 0
    prev_date = None

    n_bars = min(len(df_bt), len(probs), len(regime_probs))
    if n_bars == 0:
        return pd.DataFrame(rows)

    for bar in range(signal_start_bar, len(df_bt)):
        row = df_bt.iloc[bar]
        current_date = row.name.date() if hasattr(row.name, "date") else None
        if current_date != prev_date:
            daily_trade_count = 0
            prev_date = current_date

        trade_mgr.advance_bar()

        if pending is not None and pending["entry_bar"] == bar:
            actual_entry = float(row["open"])
            actual_sl, actual_tp = _entry_sl_tp_from_plan(
                direction=pending["direction"],
                planned_entry=pending["planned_entry"],
                actual_entry=actual_entry,
                planned_sl=pending["planned_sl"],
                planned_tp=pending["planned_tp"],
            )
            entry_equity = current_balance
            position_size_frac = pending["position_size"]
            position_notional_usd = entry_equity * position_size_frac * leverage
            active = TradeState(
                direction=pending["direction"],
                entry_price=actual_entry,
                current_stop_loss=actual_sl,
                take_profit=actual_tp,
                atr_at_entry=pending["atr"],
                adaptive_partial_tp_atr=pending["adaptive"].get("adaptive_partial_tp_atr"),
                adaptive_full_tp_atr=pending["adaptive"].get("adaptive_full_tp_atr"),
                adaptive_trailing_act_atr=pending["adaptive"].get("adaptive_trailing_act_atr"),
                adaptive_trailing_dist_atr=pending["adaptive"].get("adaptive_trailing_dist_atr"),
            )
            active_meta = {
                "signal_time": pending["signal_time"],
                "entry_time": pd.Timestamp(row.name),
                "confidence": pending["confidence"],
                "entry_equity": entry_equity,
                "position_size_frac": position_size_frac,
                "position_notional_usd": position_notional_usd,
                "initial_sl": actual_sl,
                "initial_tp": actual_tp,
            }
            pending = None

        if active is not None and active_meta is not None:
            is_choppy = (
                bar < n_bars
                and float(regime_probs[bar][2]) > config.regime.choppy_threshold
                and _adx_value(row) < config.execution.choppy_adx_override
            )
            active = trade_mgr.update(
                active,
                current_high=float(row["high"]),
                current_low=float(row["low"]),
                current_close=float(row["close"]),
                is_choppy=is_choppy,
                structural_levels=_structural_levels(row),
            )

            if active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):
                raw_reason = {
                    TradeStatus.CLOSED_TP: "TP",
                    TradeStatus.CLOSED_SL: "SL",
                    TradeStatus.CLOSED_TIME: "TIME",
                    TradeStatus.CLOSED_REGIME: "REGIME",
                }.get(active.status, "REGIME")
                outcome = normalize_close_reason(raw_reason, active.pnl)
                if outcome == "TP":
                    exit_price = active.take_profit
                elif outcome in {"SL", "TRAIL", "BE"}:
                    exit_price = active.current_stop_loss
                else:
                    exit_price = float(row["close"])

                accounting = _close_trade_balance_impact(
                    config,
                    trade=active,
                    trade_meta=active_meta,
                    current_balance=current_balance,
                    initial_balance=initial_balance,
                    pnl_frac=active.pnl,
                    exit_time=pd.Timestamp(row.name),
                    median_atr=median_atr,
                )
                current_balance = accounting["next_balance"]

                trade_mgr.record_trade_result(
                    status=protection_status_for_close(outcome, active.pnl),
                    exit_price=exit_price,
                    direction=active.direction,
                    atr=active.atr_at_entry,
                )
                equity_pnl_pct = accounting["equity_pnl_pct"]
                risk_mgr.record_trade(timestamp=pd.Timestamp(row.name), pnl_pct=equity_pnl_pct)

                partial_tp_atr = (
                    active.adaptive_partial_tp_atr
                    or config.execution.trade_management.partial_tp_1_atr
                )
                tp1_price = (
                    active.entry_price + partial_tp_atr * active.atr_at_entry
                    if active.direction == "LONG"
                    else active.entry_price - partial_tp_atr * active.atr_at_entry
                )
                rows.append(
                    {
                        "Tarih": active_meta["signal_time"].strftime("%Y-%m-%d %H:%M"),
                        "Yon": active.direction,
                        "Guven": f"{active_meta['confidence']:.3f}",
                        "Entry": f"{active.entry_price:.1f}",
                        "SL": f"{active_meta['initial_sl']:.1f}",
                        "TP1 (50%)": f"{tp1_price:.1f}",
                        "TP2 (Full)": f"{active_meta['initial_tp']:.1f}",
                        "Pozisyon": f"{active_meta['position_size_frac']*100:.2f}%",
                        "Sonuc": outcome,
                        "PnL%": f"{equity_pnl_pct:+.3f}%",
                        "Exit": f"{exit_price:.1f}",
                    }
                )
                active = None
                active_meta = None

                if current_balance <= 0:
                    break
            else:
                continue

        if bar >= n_bars:
            continue

        planned = plan_trade_from_probabilities(
            config=config,
            probs=np.asarray(probs[bar], dtype=np.float32),
            current_regime=regime_name_from_probs(np.asarray(regime_probs[bar], dtype=np.float32)),
            choppy_prob=float(regime_probs[bar][2]),
            current_atr=float(row.get(atr_col, 0.0)),
            current_price=float(row["close"]),
            current_time=pd.Timestamp(row.name),
            current_adx=_adx_value(row),
            atr_percentile=float(row.get("atr_pctile", 1.0)),
            structural_levels=_structural_levels(row),
            daily_trade_count=daily_trade_count,
            trade_manager=trade_mgr,
            risk_manager=risk_mgr,
            confidence_threshold=confidence_threshold,
        )
        if planned.direction is None:
            continue

        position_notional_usd = current_balance * planned.position_size * leverage
        if position_notional_usd < 10.0:
            continue

        entry_bar = bar + 1
        if entry_bar >= len(df_bt):
            continue

        daily_trade_count += 1
        pending = {
            "direction": planned.direction,
            "planned_entry": float(row["close"]),
            "planned_sl": planned.stop_loss,
            "planned_tp": planned.take_profit,
            "atr": float(row.get(atr_col, 0.0)),
            "adaptive": planned.adaptive_tp_sl or {},
            "position_size": planned.position_size,
            "entry_bar": entry_bar,
            "signal_time": pd.Timestamp(row.name),
            "confidence": planned.confidence,
        }

    if active is not None and active_meta is not None:
        final_row = df_bt.iloc[-1]
        exit_price = float(final_row["close"])
        pnl_frac = _m2m_trade_return_frac(active, exit_price)
        accounting = _close_trade_balance_impact(
            config,
            trade=active,
            trade_meta=active_meta,
            current_balance=current_balance,
            initial_balance=initial_balance,
            pnl_frac=pnl_frac,
            exit_time=pd.Timestamp(final_row.name),
            median_atr=median_atr,
        )
        current_balance = accounting["next_balance"]
        trade_mgr.record_trade_result(
            status=TradeStatus.CLOSED_TIME,
            exit_price=exit_price,
            direction=active.direction,
            atr=active.atr_at_entry,
        )
        equity_pnl_pct = accounting["equity_pnl_pct"]
        risk_mgr.record_trade(timestamp=pd.Timestamp(final_row.name), pnl_pct=equity_pnl_pct)

        partial_tp_atr = (
            active.adaptive_partial_tp_atr
            or config.execution.trade_management.partial_tp_1_atr
        )
        tp1_price = (
            active.entry_price + partial_tp_atr * active.atr_at_entry
            if active.direction == "LONG"
            else active.entry_price - partial_tp_atr * active.atr_at_entry
        )
        rows.append(
            {
                "Tarih": active_meta["signal_time"].strftime("%Y-%m-%d %H:%M"),
                "Yon": active.direction,
                "Guven": f"{active_meta['confidence']:.3f}",
                "Entry": f"{active.entry_price:.1f}",
                "SL": f"{active_meta['initial_sl']:.1f}",
                "TP1 (50%)": f"{tp1_price:.1f}",
                "TP2 (Full)": f"{active_meta['initial_tp']:.1f}",
                "Pozisyon": f"{active_meta['position_size_frac']*100:.2f}%",
                "Sonuc": "EOD",
                "PnL%": f"{equity_pnl_pct:+.3f}%",
                "Exit": f"{exit_price:.1f}",
            }
        )

    return pd.DataFrame(rows)
