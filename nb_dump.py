# Install dependencies
# torch, numpy, pandas are pre-installed by Colab â€” do NOT reinstall them.
# Pinning versions fights Colab's environment and causes resolver conflicts.
!pip install -q xgboost ccxt PyWavelets hmmlearn numba scikit-learn pyyaml \
    tqdm pyarrow matplotlib seaborn

from google.colab import drive
drive.mount('/content/drive')

import os, sys, json

# Clone/update repo from GitHub (local Colab filesystem â€” fast)
REPO_DIR = '/content/scalp2_repo'
if os.path.exists(os.path.join(REPO_DIR, '.git')):
    !git -C {REPO_DIR} pull --ff-only
else:
    !git clone https://github.com/sergul74/Scalp2.git {REPO_DIR}

if not os.path.exists(os.path.join(REPO_DIR, 'scalp2', '__init__.py')):
    raise FileNotFoundError('scalp2 package not found after clone!')

sys.path.insert(0, REPO_DIR)

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scalp2.config import load_config
from scalp2.execution.trade_manager import TradeManager, TradeState, TradeStatus
from scalp2.utils.metrics import sharpe_ratio, sortino_ratio, max_drawdown, win_rate, profit_factor

config = load_config(f'{REPO_DIR}/config.yaml')
DATA_DIR = '/content/drive/MyDrive/scalp2/data/processed'

# Load labeled dataset and walk-forward predictions
import pickle

df = pd.read_parquet(f'{DATA_DIR}/BTC_USDT_labeled.parquet')

with open(f'{DATA_DIR}/wf_predictions.pkl', 'rb') as f:
    wf_predictions = pickle.load(f)

print(f'Loaded {len(df)} bars, {len(wf_predictions)} walk-forward folds')

# Walk-forward backtest with trade management + realistic costs + regime filters
# Fixes applied:
#   1.2  Entry at NEXT bar open (not signal bar close)
#   1.3  Choppy regime filter using forward-only HMM probs
#   1.4  Fractional Kelly position sizing (matches live)
#   1.6  Multi-leg slippage model (per-exit cost)
#   1.7  Leverage, variable slippage, funding rate, market impact
from tqdm import tqdm

# Suppress per-trade log spam from TradeManager
logging.getLogger('scalp2.execution.trade_manager').setLevel(logging.WARNING)

seq_len = config.model.seq_len
exec_cfg = config.execution
trade_mgmt_cfg = config.execution.trade_management
label_cfg = config.labeling
order_cfg = config.execution.order_execution

trade_mgr = TradeManager(trade_mgmt_cfg, label_cfg.max_holding_bars)

# --- Transaction costs from config (Binance USDM Futures) ---
MAKER_FEE_BPS = order_cfg.maker_fee_bps
TAKER_FEE_BPS = order_cfg.taker_fee_bps
SLIPPAGE_BPS = order_cfg.slippage_bps

# --- Leverage & margin ---
LEVERAGE = exec_cfg.position_sizing.leverage
MAINT_MARGIN = exec_cfg.position_sizing.maintenance_margin

# --- Variable slippage model ---
slip_cfg = getattr(exec_cfg, 'slippage_model', None)
USE_VAR_SLIPPAGE = slip_cfg is not None and slip_cfg.enabled
if USE_VAR_SLIPPAGE and 'atr_14' in df.columns:
    MEDIAN_ATR = df['atr_14'].median()
else:
    MEDIAN_ATR = 1.0

def get_slippage_bps(atr_val):
    """Variable slippage: base + volatility component."""
    if not USE_VAR_SLIPPAGE:
        return SLIPPAGE_BPS
    ratio = atr_val / (MEDIAN_ATR + 1e-10)
    return slip_cfg.base_bps + slip_cfg.volatility_scale * ratio

# --- Funding rate model ---
funding_cfg = getattr(exec_cfg, 'funding_rate', None)
USE_FUNDING = funding_cfg is not None and funding_cfg.enabled

def count_funding_intervals(entry_time, exit_time):
    """Count 8h funding intervals between entry and exit."""
    if not USE_FUNDING:
        return 0
    funding_times = pd.date_range(
        entry_time.normalize(),
        exit_time.normalize() + pd.Timedelta(days=1),
        freq='8h'
    )
    return int(((funding_times > entry_time) & (funding_times <= exit_time)).sum())

# --- Market impact model ---
impact_cfg = getattr(exec_cfg, 'market_impact', None)
USE_MARKET_IMPACT = impact_cfg is not None and impact_cfg.enabled

def market_impact_frac(position_size, price):
    """Proportional market impact for large positions (returned as fraction)."""
    if not USE_MARKET_IMPACT:
        return 0.0
    notional = position_size * price * LEVERAGE
    return impact_cfg.base_impact_bps * (notional / impact_cfg.reference_notional_usd) / 10_000

# Per-leg cost functions (in fraction, not bps)
def entry_cost_frac(slip_bps):
    """Entry is a limit order: maker fee + slippage."""
    return (MAKER_FEE_BPS + slip_bps) / 10_000

def exit_cost_frac(status, slip_bps):
    """Exit cost depends on how the trade closed.
    SL/time/regime exits are market orders (taker fee).
    TP exits are limit orders (maker fee).
    """
    if status in (TradeStatus.CLOSED_SL, TradeStatus.CLOSED_TIME,
                  TradeStatus.CLOSED_REGIME, 'CLOSED_FOLD_END'):
        return (TAKER_FEE_BPS + slip_bps) / 10_000
    else:  # CLOSED_TP
        return (MAKER_FEE_BPS + slip_bps) / 10_000

# Legacy flat cost for comparison
FLAT_RT_COST = 2 * (SLIPPAGE_BPS + MAKER_FEE_BPS) / 10_000
print(f'TX cost model: multi-leg | maker={MAKER_FEE_BPS}bps taker={TAKER_FEE_BPS}bps '
      f'slip={SLIPPAGE_BPS}bps | flat RT={FLAT_RT_COST*10000:.0f}bps (for reference)')
print(f'Leverage: {LEVERAGE}x | Variable slippage: {USE_VAR_SLIPPAGE} | '
      f'Funding rate: {USE_FUNDING} | Market impact: {USE_MARKET_IMPACT}')

# --- Kelly sizing params ---
# Effective b-ratio accounts for partial TP
partial_pct = trade_mgmt_cfg.partial_tp_1_pct   # 0.5
partial_atr = trade_mgmt_cfg.partial_tp_1_atr   # 0.6
full_tp_atr = trade_mgmt_cfg.full_tp_atr        # 1.2
effective_tp = partial_pct * partial_atr + (1 - partial_pct) * full_tp_atr
kelly_b = effective_tp / label_cfg.sl_multiplier
kelly_fraction = exec_cfg.position_sizing.kelly_fraction   # 0.25
kelly_max = exec_cfg.position_sizing.max_fraction          # 0.02
print(f'Kelly sizing: effective_b={kelly_b:.2f} (was {label_cfg.tp_multiplier/label_cfg.sl_multiplier:.2f}), '
      f'fraction={kelly_fraction}, max={kelly_max}')

# --- Precompute ADX and ATR percentile for filtering ---
MIN_ADX = exec_cfg.min_adx
MIN_ATR_PCTILE = exec_cfg.min_atr_percentile
CHOPPY_THRESHOLD = config.regime.choppy_threshold

# Rolling ATR percentile (96-bar window = ~24h on 15m)
if 'atr_14' in df.columns:
    df['atr_pctile'] = df['atr_14'].rolling(96, min_periods=10).rank(pct=True)
    df['atr_pctile'] = df['atr_pctile'].fillna(1.0)
else:
    df['atr_pctile'] = 1.0

print(f'Filters: min_adx={MIN_ADX}, min_atr_pctile={MIN_ATR_PCTILE}, '
      f'choppy_threshold={CHOPPY_THRESHOLD}')

all_trades = []
equity_curve = [0.0]
cumulative_pnl = 0.0
skip_reasons = {'low_adx': 0, 'low_volatility': 0, 'low_conf': 0,
                'hold': 0, 'daily_cap': 0, 'no_atr': 0, 'choppy': 0,
                'no_next_bar': 0}
liquidated = False

def _close_trade(active, position_size, entry_bar, bar, row, fold_idx,
                 status_override=None, gross_override=None):
    """Record a closed trade with all cost layers and leverage."""
    global cumulative_pnl

    gross = gross_override if gross_override is not None else active.pnl
    status_val = status_override if status_override else active.status.value

    # Variable slippage based on ATR at entry
    slip_bps = get_slippage_bps(active.atr_at_entry)

    # Multi-leg cost
    cost = entry_cost_frac(slip_bps)
    if active.partial_fills:
        cost += exit_cost_frac(TradeStatus.CLOSED_TP, slip_bps) * partial_pct
        exit_status = status_override or active.status
        cost += exit_cost_frac(exit_status, slip_bps) * (1 - partial_pct)
    else:
        exit_status = status_override or active.status
        cost += exit_cost_frac(exit_status, slip_bps)

    # Market impact
    impact = market_impact_frac(position_size, active.entry_price)
    cost += impact

    # Funding rate
    entry_ts = df.index[entry_bar]
    exit_ts = row.name
    n_funding = count_funding_intervals(entry_ts, exit_ts)
    funding = n_funding * (funding_cfg.fixed_rate_pct / 100.0) if USE_FUNDING else 0.0

    # Per-unit net (unleveraged)
    unit_net = (gross - cost - funding) * position_size

    # Leveraged portfolio PnL
    leveraged_net = unit_net * LEVERAGE

    cumulative_pnl += leveraged_net

    all_trades.append(dict(
        fold=fold_idx, direction=active.direction,
        entry_price=active.entry_price,
        bars_held=active.bars_held,
        status=status_val,
        gross_pnl=gross * position_size * LEVERAGE,
        unit_pnl=unit_net,
        net_pnl=leveraged_net,
        cost=cost * position_size * LEVERAGE,
        funding_cost=funding * position_size * LEVERAGE,
        position_size=position_size,
        margin_utilization=position_size * LEVERAGE,
        slippage_bps=slip_bps,
        impact_bps=impact * 10_000,
        n_exits=1 + len(active.partial_fills),
        entry_bar=entry_bar, exit_bar=bar,
        timestamp=row.name,
    ))
    equity_curve.append(cumulative_pnl)

for fold_data in tqdm(wf_predictions, desc='Backtesting folds'):
    if liquidated:
        break

    fold_idx = fold_data['fold_idx']
    test_start = fold_data['test_start']
    test_end = fold_data['test_end']
    preds = fold_data['test_probabilities']   # (n_preds, 3)
    n_preds = len(preds)
    pred_offset = test_start + seq_len        # first bar with a prediction

    # Regime probs (forward-only, from stage2_trainer)
    # Fallback: if not present (old wf_predictions), skip regime filter
    regime_probs = fold_data.get('regime_probs', None)
    has_regime = regime_probs is not None

    active = None
    entry_bar = None
    pending_signal = None   # (direction, atr, sl_raw, confidence, pred_idx)
    daily_count = 0
    prev_date = None

    for i in range(n_preds):
        bar = pred_offset + i
        if bar >= len(df):
            break

        row = df.iloc[bar]
        cur_date = row.name.date() if hasattr(row.name, 'date') else None
        if cur_date != prev_date:
            daily_count = 0
            prev_date = cur_date

        # ---- manage active trade ----
        if active is not None:
            # Check choppy regime for active trade
            is_choppy = False
            if has_regime and i < len(regime_probs):
                is_choppy = regime_probs[i, 2] > CHOPPY_THRESHOLD

            # Extract structural levels from current bar
            struct_levels = {
                'vwap': float(row.get('vwap', float('nan'))) if 'vwap' in df.columns else float('nan'),
                'fvg_bull': float(row.get('fvg_bull_price', float('nan'))) if 'fvg_bull_price' in df.columns else float('nan'),
                'fvg_bear': float(row.get('fvg_bear_price', float('nan'))) if 'fvg_bear_price' in df.columns else float('nan'),
                'swing_high': float(row.get('swing_high_price', float('nan'))) if 'swing_high_price' in df.columns else float('nan'),
                'swing_low': float(row.get('swing_low_price', float('nan'))) if 'swing_low_price' in df.columns else float('nan'),
            }

            active = trade_mgr.update(
                active, row['high'], row['low'], row['close'], is_choppy,
                structural_levels=struct_levels,
            )
            if active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):
                _close_trade(active, position_size, entry_bar, bar, row, fold_idx)

                # Liquidation check
                if cumulative_pnl <= -1.0:
                    print(f'LIQUIDATED at fold {fold_idx}, bar {bar}')
                    liquidated = True
                    break

                active = None
            continue  # don't open new trade on same bar

        # ---- execute pending signal from previous bar ----
        if pending_signal is not None:
            ps = pending_signal
            pending_signal = None

            entry_price = row['open']  # NEXT BAR OPEN (realistic fill)
            atr = ps['atr']
            direction = ps['direction']
            confidence = ps['confidence']

            # Enforce Consecutive SL Protection (Matches Live Bot)
            can_enter, skip_reason = trade_mgr.can_enter_trade(
                direction=direction,
                entry_price=entry_price,
                current_atr=atr,
            )
            if not can_enter:
                skip_reasons['sl_block'] = skip_reasons.get('sl_block', 0) + 1
                continue

            # Recompute SL from actual entry price
            if direction == "LONG":
                sl = entry_price - label_cfg.sl_multiplier * atr
                tp = entry_price + trade_mgmt_cfg.full_tp_atr * atr
            else:
                sl = entry_price + label_cfg.sl_multiplier * atr
                tp = entry_price - trade_mgmt_cfg.full_tp_atr * atr

            # Smart Exit Engine: structural adjustments at entry
            original_sl = sl
            adaptive_full_tp_atr = None
            struct_cfg = trade_mgmt_cfg.structural_exit
            if struct_cfg.enabled:
                entry_struct = {
                    'swing_high': float(row.get('swing_high_price', float('nan'))) if 'swing_high_price' in df.columns else float('nan'),
                    'swing_low': float(row.get('swing_low_price', float('nan'))) if 'swing_low_price' in df.columns else float('nan'),
                    'fvg_bull': float(row.get('fvg_bull_price', float('nan'))) if 'fvg_bull_price' in df.columns else float('nan'),
                    'fvg_bear': float(row.get('fvg_bear_price', float('nan'))) if 'fvg_bear_price' in df.columns else float('nan'),
                    'vwap': float(row.get('vwap', float('nan'))) if 'vwap' in df.columns else float('nan'),
                }
                import math
                
                # --- A. FVG TP Stretch ---
                fvg = entry_struct.get('fvg_bear' if direction == 'LONG' else 'fvg_bull')
                if fvg is not None and not math.isnan(fvg):
                    if direction == 'LONG' and fvg > entry_price:
                        if abs(tp - fvg) < struct_cfg.fvg_proximity_atr * atr:
                            tp = fvg
                    elif direction == 'SHORT' and fvg < entry_price:
                        if abs(tp - fvg) < struct_cfg.fvg_proximity_atr * atr:
                            tp = fvg

                # --- B. Sweep-Resistant SL ---
                swing = entry_struct.get('swing_low' if direction == 'LONG' else 'swing_high')
                if swing is not None and not math.isnan(swing):
                    buffer = struct_cfg.sweep_buffer_atr * atr
                    max_stretch = struct_cfg.max_sl_stretch_atr * atr
                    if direction == 'LONG':
                        if abs(sl - swing) < buffer:
                            new_sl = swing - buffer
                            if abs(entry_price - new_sl) - abs(entry_price - sl) <= max_stretch:
                                sl = new_sl
                    else:
                        if abs(sl - swing) < buffer:
                            new_sl = swing + buffer
                            if abs(new_sl - entry_price) - abs(sl - entry_price) <= max_stretch:
                                sl = new_sl

                # --- C. VWAP TP Stretch ---
                vwap = entry_struct.get('vwap')
                if vwap is not None and not math.isnan(vwap):
                    if direction == 'LONG' and vwap > entry_price:
                        if abs(tp - vwap) < struct_cfg.vwap_margin_atr * atr and vwap > tp:
                            tp = vwap
                    elif direction == 'SHORT' and vwap < entry_price:
                        if abs(tp - vwap) < struct_cfg.vwap_margin_atr * atr and vwap < tp:
                            tp = vwap

                # Convert adapted absolute TP back to ATR distance for TradeManager compatibility
                if direction == 'LONG':
                    if abs(tp - (entry_price + trade_mgmt_cfg.full_tp_atr * atr)) > 1e-6:
                        adaptive_full_tp_atr = abs(tp - entry_price) / atr
                else:
                    if abs(tp - (entry_price - trade_mgmt_cfg.full_tp_atr * atr)) > 1e-6:
                        adaptive_full_tp_atr = abs(tp - entry_price) / atr

            # Kelly position sizing
            p = confidence
            q = 1 - p
            kelly = max((p * kelly_b - q) / kelly_b, 0)
            position_size = min(kelly * kelly_fraction, kelly_max)

            # Option A: Risk Normalization — shrink position if SL was widened
            if struct_cfg.enabled and struct_cfg.normalize_risk and sl != original_sl:
                original_risk = abs(entry_price - original_sl)
                new_risk = abs(entry_price - sl)
                if new_risk > original_risk and original_risk > 0:
                    position_size *= original_risk / new_risk

            if position_size < 1e-6:
                continue  # skip if Kelly says don't trade

            active = TradeState(
                direction=direction,
                entry_price=entry_price,
                current_stop_loss=sl,
                take_profit=0.0,
                atr_at_entry=atr,
                adaptive_full_tp_atr=adaptive_full_tp_atr,
            )
            entry_bar = bar
            daily_count += 1
            continue  # don't evaluate signal on entry bar

        # ---- check for new signal ----
        p = preds[i]
        cls = int(np.argmax(p))
        if cls == 1:
            skip_reasons['hold'] += 1
            continue
        if max(p[0], p[2]) < exec_cfg.confidence_threshold:
            skip_reasons['low_conf'] += 1
            continue
        if daily_count >= exec_cfg.max_trades_per_day:
            skip_reasons['daily_cap'] += 1
            continue

        # Time-of-day filter
        tdf = getattr(exec_cfg, 'time_of_day_filter', None)
        if tdf and tdf.enabled:
            hr = row.name.hour if hasattr(row.name, 'hour') else pd.to_datetime(row.name).hour
            if hr in tdf.blocked_hours_utc:
                skip_reasons['time_blocked'] = skip_reasons.get('time_blocked', 0) + 1
                continue

        # Time-of-day filter
        tdf = getattr(exec_cfg, 'time_of_day_filter', None)
        if tdf and tdf.enabled:
            hr = row.name.hour if hasattr(row.name, 'hour') else pd.to_datetime(row.name).hour
            if hr in tdf.blocked_hours_utc:
                skip_reasons['choppy'] += 1  # Reusing choppy skip counter or add new
                continue


        atr = row['atr_14'] if 'atr_14' in df.columns else 0.0
        if atr < 1e-10:
            skip_reasons['no_atr'] += 1
            continue

        # ADX filter
        adx_val = row.get('adx', 999.0) if hasattr(row, 'get') else (
            row['adx'] if 'adx' in df.columns else 999.0)
        if adx_val < getattr(exec_cfg, 'min_adx', MIN_ADX):
            skip_reasons['low_adx'] += 1
            continue

        # ATR percentile filter
        atr_pct_val = row['atr_pct'] if 'atr_pct' in df.columns else 1.0
        if atr_pct_val < getattr(exec_cfg, 'min_atr_percentile', 0.05):
            skip_reasons['low_volatility'] += 1
            continue

        # Choppy regime filter (forward-only probs)
        if has_regime and i < len(regime_probs):
            if regime_probs[i, 2] > CHOPPY_THRESHOLD:
                if adx_val < getattr(exec_cfg, 'choppy_adx_override', 25.0):
                    skip_reasons['choppy'] += 1
                    continue

        # Check next bar exists for entry
        next_bar = pred_offset + i + 1
        if next_bar >= len(df):
            skip_reasons['no_next_bar'] += 1
            continue

        direction = "LONG" if cls == 2 else "SHORT"

        # Set pending signal â€” will execute at next bar open
        pending_signal = {
            'direction': direction,
            'atr': atr,
            'confidence': max(p[0], p[2]),
        }

    # force-close any open trade at fold boundary
    if active is not None:
        last_row = df.iloc[min(test_end - 1, len(df) - 1)]
        if active.direction == "LONG":
            unr = (last_row['close'] - active.entry_price) / active.entry_price
        else:
            unr = (active.entry_price - last_row['close']) / active.entry_price
        gross = active.pnl + unr * active.remaining_size

        _close_trade(active, position_size, entry_bar,
                     min(test_end - 1, len(df) - 1), last_row, fold_idx,
                     status_override='CLOSED_FOLD_END', gross_override=gross)
        active = None
    # Clear pending signal at fold boundary
    pending_signal = None

trades_df = pd.DataFrame(all_trades)
print(f'\nTotal trades: {len(trades_df)}')
print(f'Cumulative net PnL ({LEVERAGE}x levered): {cumulative_pnl*100:.2f}%')
print(f'\nSkip reasons: {skip_reasons}')
if len(trades_df) > 0:
    print(f'Avg position size: {trades_df["position_size"].mean():.4f}')
    print(f'Avg margin util: {trades_df["margin_utilization"].mean()*100:.1f}%')
    print(f'Avg cost/trade (levered): {trades_df["cost"].mean()*10000:.1f} bps')
    print(f'Regime filter active: {has_regime}')
    if liquidated:
        print('*** ACCOUNT LIQUIDATED ***')

# Results summary and visualization
if len(trades_df) == 0:
    print("No trades generated!")
else:
    net = trades_df['net_pnl'].values
    gross = trades_df['gross_pnl'].values
    unit = trades_df['unit_pnl'].values
    n = len(trades_df)

    # Win/loss stats (on leveraged net)
    wins = net[net > 0]
    losses = net[net < 0]
    wr = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) else float('inf')

    # Daily P&L for Sharpe/Sortino (leveraged)
    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
    daily_pnl = trades_df.groupby('date')['net_pnl'].sum()
    full_range = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq='D')
    daily_pnl = daily_pnl.reindex(full_range, fill_value=0.0)

    daily_sharpe = daily_pnl.mean() / (daily_pnl.std() + 1e-10) * np.sqrt(365)
    down = daily_pnl[daily_pnl < 0]
    daily_sortino = daily_pnl.mean() / (down.std() + 1e-10) * np.sqrt(365) if len(down) else 0

    # Max drawdown (on leveraged equity)
    cum = np.cumsum(daily_pnl.values)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    mdd = dd.max()
    calmar = (daily_pnl.mean() * 365) / mdd if mdd > 1e-10 else 0

    # Status distribution
    status_counts = trades_df['status'].value_counts()

    # Per-unit (unleveraged) stats for comparison
    unit_daily = trades_df.groupby('date')['unit_pnl'].sum().reindex(full_range, fill_value=0.0)
    unit_cum = np.cumsum(unit_daily.values)
    unit_peak = np.maximum.accumulate(unit_cum)
    unit_dd = unit_peak - unit_cum
    unit_mdd = unit_dd.max()

    print('=' * 60)
    print('       WALK-FORWARD BACKTEST RESULTS')
    print(f'       (Leverage: {LEVERAGE}x)')
    print('=' * 60)
    print(f'  Total trades       : {n}')
    print(f'  Win rate           : {wr:.4f} ({wr*100:.1f}%)')
    print(f'  Profit factor      : {pf:.4f}')
    print(f'  Expectancy/trade   : {net.mean()*100:.4f}%')
    print(f'  Avg win            : +{avg_win*100:.4f}%')
    print(f'  Avg loss           : {avg_loss*100:.4f}%')
    print(f'  Avg bars held      : {trades_df["bars_held"].mean():.1f}')
    print()
    print(f'  Daily Sharpe       : {daily_sharpe:.4f}')
    print(f'  Daily Sortino      : {daily_sortino:.4f}')
    print(f'  Max Drawdown       : {mdd*100:.4f}%')
    print(f'  Calmar             : {calmar:.4f}')
    print()
    print(f'  Gross PnL          : {gross.sum()*100:.2f}%')
    print(f'  Net PnL            : {net.sum()*100:.2f}%')
    print(f'  Cost impact        : {(gross.sum()-net.sum())*100:.2f}%')
    print(f'  TX cost/trade      : {FLAT_RT_COST*10000:.0f} bps (flat ref)')
    print()
    print('  Close reasons:')
    for status, count in status_counts.items():
        print(f'    {status:20s}: {count:5d} ({count/n*100:.1f}%)')

    # --- Leverage & cost model analysis ---
    print()
    print('-' * 60)
    print('  LEVERAGE & COST MODEL ANALYSIS')
    print('-' * 60)
    print(f'  Leverage            : {LEVERAGE}x')
    print(f'  Net PnL (per-unit)  : {unit.sum()*100:.2f}%')
    print(f'  Net PnL (leveraged) : {net.sum()*100:.2f}%')
    print(f'  MDD (per-unit)      : {unit_mdd*100:.4f}%')
    print(f'  MDD (leveraged)     : {mdd*100:.4f}%')
    print(f'  Avg position size   : {trades_df["position_size"].mean()*100:.2f}%')
    print(f'  Avg margin util     : {trades_df["margin_utilization"].mean()*100:.1f}%')
    print(f'  Max margin util     : {trades_df["margin_utilization"].max()*100:.1f}%')
    if USE_FUNDING:
        total_funding = trades_df['funding_cost'].sum()
        print(f'  Total funding cost  : {total_funding*100:.4f}%')
        n_funded = (trades_df['funding_cost'] > 0).sum()
        print(f'  Trades w/ funding   : {n_funded} ({n_funded/n*100:.1f}%)')
    if USE_VAR_SLIPPAGE:
        print(f'  Avg slippage        : {trades_df["slippage_bps"].mean():.1f} bps')
        print(f'  Max slippage        : {trades_df["slippage_bps"].max():.1f} bps')
    if USE_MARKET_IMPACT:
        print(f'  Avg market impact   : {trades_df["impact_bps"].mean():.2f} bps')
        print(f'  Max market impact   : {trades_df["impact_bps"].max():.2f} bps')
    if liquidated:
        print(f'  *** ACCOUNT LIQUIDATED ***')
    print('=' * 60)

    # --- Charts ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Equity curve (leveraged)
    cum_pct = np.cumsum(daily_pnl.values) * 100
    axes[0].plot(daily_pnl.index, cum_pct, linewidth=1, color='#2196F3')
    axes[0].fill_between(daily_pnl.index, 0, cum_pct, alpha=0.1, color='#2196F3')
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Equity Curve (Cumulative PnL % â€” {LEVERAGE}x Leveraged)')
    axes[0].set_ylabel('Cumulative PnL (%)')
    axes[0].grid(True, alpha=0.3)

    # 2. Drawdown (leveraged)
    axes[1].fill_between(daily_pnl.index, 0, -dd * 100, alpha=0.4, color='red')
    axes[1].set_title(f'Drawdown ({LEVERAGE}x Leveraged)')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True, alpha=0.3)

    # 3. PnL distribution
    axes[2].hist(net * 100, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white')
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[2].set_title('Trade PnL Distribution (Leveraged)')
    axes[2].set_xlabel('PnL per trade (%)')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Monthly returns table (leveraged)
    trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
    monthly = trades_df.groupby('month')['net_pnl'].agg(['sum', 'count'])
    monthly.columns = ['return_pct', 'trades']
    monthly['return_pct'] *= 100
    print('\nMonthly Returns:')
    print(monthly.to_string())

# Dönem Bazlı Performans Analizi — Modelin gerçek edge'i var mı?
if len(trades_df) > 0:
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['year'] = trades_df['timestamp'].dt.year
    trades_df['quarter'] = trades_df['timestamp'].dt.to_period('Q')

    # --- Yıllık Performans ---
    print('=' * 70)
    print('       YILLIK PERFORMANS KARŞILAŞTIRMASI')
    print(f'       (Leverage: {LEVERAGE}x)')
    print('=' * 70)

    yearly_stats = []
    for year, grp in trades_df.groupby('year'):
        net = grp['net_pnl'].values
        n = len(grp)
        wins = net[net > 0]
        losses = net[net < 0]
        wr = len(wins) / n if n > 0 else 0
        pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 else float('inf')

        grp_daily = grp.groupby(grp['timestamp'].dt.date)['net_pnl'].sum()
        dr = pd.date_range(grp_daily.index.min(), grp_daily.index.max(), freq='D')
        grp_daily = grp_daily.reindex(dr, fill_value=0.0)
        sharpe = grp_daily.mean() / (grp_daily.std() + 1e-10) * np.sqrt(365)

        cum = np.cumsum(grp_daily.values)
        peak = np.maximum.accumulate(cum)
        mdd = (peak - cum).max()

        yearly_stats.append({
            'Yıl': year, 'İşlem': n, 'Win Rate': f'{wr*100:.1f}%',
            'Profit Factor': f'{pf:.2f}', 'Sharpe': f'{sharpe:.2f}',
            'Net PnL': f'{net.sum()*100:.2f}%', 'MDD': f'{mdd*100:.2f}%'
        })

    yearly_df = pd.DataFrame(yearly_stats)
    print(yearly_df.to_string(index=False))

    # --- Çeyreklik Performans ---
    print()
    print('=' * 70)
    print('       ÇEYREKLİK PERFORMANS')
    print('=' * 70)

    quarterly_stats = []
    for q, grp in trades_df.groupby('quarter'):
        net = grp['net_pnl'].values
        n = len(grp)
        wins = net[net > 0]
        losses = net[net < 0]
        wr = len(wins) / n if n > 0 else 0
        pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 else float('inf')
        quarterly_stats.append({
            'Çeyrek': str(q), 'İşlem': n, 'Win Rate': f'{wr*100:.1f}%',
            'PF': f'{pf:.2f}', 'Net PnL': f'{net.sum()*100:.2f}%'
        })

    q_df = pd.DataFrame(quarterly_stats)
    print(q_df.to_string(index=False))

    # --- Yıllık Equity Eğrileri (Overlay) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trades_df['year'].unique())))
    for idx, (year, grp) in enumerate(trades_df.groupby('year')):
        grp_daily = grp.groupby(grp['timestamp'].dt.date)['net_pnl'].sum()
        dr = pd.date_range(grp_daily.index.min(), grp_daily.index.max(), freq='D')
        grp_daily = grp_daily.reindex(dr, fill_value=0.0)
        cum_pct = np.cumsum(grp_daily.values) * 100
        # Day-of-year for overlay comparison
        days = np.arange(len(cum_pct))
        axes[0].plot(days, cum_pct, label=str(year), color=colors[idx], linewidth=1.2)

    axes[0].set_title('Yıllık Kümülatif PnL Karşılaştırması')
    axes[0].set_xlabel('Gün (yıl başından)')
    axes[0].set_ylabel('Kümülatif PnL (%)')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)

    # Win rate by year bar chart
    years = [s['Yıl'] for s in yearly_stats]
    wrs = [float(s['Win Rate'].replace('%','')) for s in yearly_stats]
    bar_colors = ['#4CAF50' if w > 55 else '#FF9800' if w > 45 else '#F44336' for w in wrs]
    axes[1].bar(range(len(years)), wrs, color=bar_colors, edgecolor='white')
    axes[1].set_xticks(range(len(years)))
    axes[1].set_xticklabels(years)
    axes[1].set_title('Yıllık Win Rate')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50% (rastgele)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # --- Tutarlılık Değerlendirmesi ---
    print()
    print('=' * 70)
    print('       MODEL TUTARLILIK DEĞERLENDİRMESİ')
    print('=' * 70)
    win_rates = [float(s['Win Rate'].replace('%',''))/100 for s in yearly_stats]
    pfs = [float(s['Profit Factor']) for s in yearly_stats if s['Profit Factor'] != 'inf']

    wr_std = np.std(win_rates)
    wr_mean = np.mean(win_rates)
    pf_std = np.std(pfs) if pfs else 0
    pf_mean = np.mean(pfs) if pfs else 0

    print(f'  Win Rate: Ort={wr_mean*100:.1f}%, Std={wr_std*100:.1f}%')
    print(f'  Profit Factor: Ort={pf_mean:.2f}, Std={pf_std:.2f}')
    print()

    all_positive = all(float(s['Net PnL'].replace('%','')) > 0 for s in yearly_stats)
    low_variance = wr_std < 0.08

    if all_positive and low_variance:
        print('  ✅ GÜÇLÜ: Model TÜM yıllarda pozitif ve tutarlı.')
        print('  → Gerçek tahmin gücü (edge) var, look-ahead bias yok.')
    elif all_positive:
        print('  ⚠️ ORTA: Tüm yıllar pozitif ama performans değişkenliği yüksek.')
        print('  → Edge var ama bazı dönemlerde zayıflıyor (regime-dependent).')
    else:
        neg_years = [s['Yıl'] for s in yearly_stats if float(s['Net PnL'].replace('%','')) <= 0]
        print(f'  ❌ ZAYIF: Negatif yıllar var ({neg_years}).')
        print('  → Model belirli market yapılarına overfit olmuş olabilir.')
    print('=' * 70)
else:
    print("İşlem yok — analiz yapılamadı.")

# ============================================================
# Forward Test Step 1: Download recent data & build features
# ============================================================
import torch
from scalp2.data.preprocessing import clean_ohlcv, resample_ohlcv
from scalp2.features.builder import build_features, drop_warmup_nans, get_feature_columns
from scalp2.data.mtf_builder import build_mtf_dataset

# Need warmup period for features (256 bars wavelet + seq_len)
# Download from 2026-01-15 onward, but only test on bars after 2026-02-20

import ccxt, os, time as _time

EXCHANGES_TO_TRY = [
    ("binanceusdm", "BTC/USDT"),
    ("binance",     "BTC/USDT"),
    ("bybit",       "BTC/USDT"),
    ("okx",         "BTC/USDT"),
]
from datetime import datetime, timezone

FWD_START = "2026-01-15"  # warmup period
FWD_END = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # today
FWD_TEST_CUTOFF = "2026-02-20"  # training data ends here

fwd_cache = '/content/drive/MyDrive/scalp2/data/processed/btc_fwd_15m.parquet'

# Always re-download to get latest data (delete cache if stale)
if os.path.exists(fwd_cache):
    cached_df = pd.read_parquet(fwd_cache)
    if 'timestamp' in cached_df.columns:
        last_ts = pd.to_datetime(cached_df['timestamp']).max()
    else:
        last_ts = cached_df.index.max()
    # Re-download if cache is more than 1 day old
    if (pd.Timestamp.utcnow() - pd.Timestamp(last_ts).tz_localize('UTC')).days > 1:
        print(f"Cache stale (last: {last_ts}), re-downloading...")
        os.remove(fwd_cache)

if os.path.exists(fwd_cache):
    print(f"Loading cached forward data from {fwd_cache}")
    df_fwd_raw = pd.read_parquet(fwd_cache)
    if 'timestamp' in df_fwd_raw.columns:
        df_fwd_raw['timestamp'] = pd.to_datetime(df_fwd_raw['timestamp'])
        df_fwd_raw = df_fwd_raw.set_index('timestamp')
    if df_fwd_raw.index.tz is not None:
        df_fwd_raw.index = df_fwd_raw.index.tz_localize(None)
else:
    df_fwd_raw = None
    for exch_id, symbol in EXCHANGES_TO_TRY:
        print(f"Trying {exch_id} ({symbol})...")
        try:
            exchange = getattr(ccxt, exch_id)({'enableRateLimit': True})
            start_ms = exchange.parse8601(f'{FWD_START}T00:00:00Z')
            end_ms = exchange.parse8601(f'{FWD_END}T23:59:59Z')

            all_candles = []
            since = start_ms
            while since < end_ms:
                candles = exchange.fetch_ohlcv(symbol, '15m', since=since, limit=1000)
                if not candles:
                    break
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                if len(all_candles) % 10000 < 1000:
                    print(f"  {len(all_candles)} candles...")
                _time.sleep(exchange.rateLimit / 1000.0)

            if len(all_candles) > 500:
                df_fwd_raw = pd.DataFrame(all_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_fwd_raw['timestamp'] = pd.to_datetime(df_fwd_raw['timestamp'], unit='ms')
                df_fwd_raw = df_fwd_raw.drop_duplicates('timestamp').sort_values('timestamp')
                df_fwd_raw = df_fwd_raw.set_index('timestamp')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_fwd_raw[col] = df_fwd_raw[col].astype(np.float32)
                df_fwd_raw.to_parquet(fwd_cache)
                print(f"SUCCESS: {exch_id} â€” {len(df_fwd_raw)} bars downloaded & cached")
                break
        except Exception as e:
            print(f"  {exch_id} failed: {e}")
            continue

    if df_fwd_raw is None:
        raise RuntimeError("All exchanges failed for forward data.")

print(f"Forward data: {len(df_fwd_raw)} bars, {df_fwd_raw.index[0]} to {df_fwd_raw.index[-1]}")

# Clean and resample
df_fwd_15m = clean_ohlcv(df_fwd_raw, '15m')
df_fwd_1h = resample_ohlcv(df_fwd_15m, '1h')
df_fwd_4h = resample_ohlcv(df_fwd_15m, '4h')

print(f"Clean 15m: {len(df_fwd_15m)}, 1H: {len(df_fwd_1h)}, 4H: {len(df_fwd_4h)}")

# Build features
print("Building features for forward test...")
df_fwd_15m_feat = build_features(df_fwd_15m, config.features)
df_fwd_1h_feat = build_features(df_fwd_1h, config.features)
df_fwd_4h_feat = build_features(df_fwd_4h, config.features)

df_fwd_full = build_mtf_dataset(df_fwd_15m_feat, df_fwd_1h_feat, df_fwd_4h_feat)
df_fwd_full = drop_warmup_nans(df_fwd_full)

fwd_feature_cols = get_feature_columns(df_fwd_full)
print(f"Forward feature matrix: {len(df_fwd_full)} rows x {len(fwd_feature_cols)} features")
print(f"Test period: {FWD_TEST_CUTOFF} â†’ {df_fwd_full.index[-1]}")

# ============================================================
# Forward Test Step 2: Load LAST fold's model & run inference
# ============================================================
from scalp2.utils.serialization import load_fold_artifacts
from scalp2.models.hybrid import HybridEncoder
from scalp2.models.meta_learner import XGBoostMetaLearner
from scalp2.data.dataset import ScalpDataset
from torch.utils.data import DataLoader
from torch.amp import autocast

CHECKPOINT_DIR = '/content/drive/MyDrive/scalp2/checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

last_fold_idx = len(wf_predictions) - 1
print(f"Using Fold {last_fold_idx} (last fold) for forward test...")

fold_last = load_fold_artifacts(CHECKPOINT_DIR, fold_idx=last_fold_idx, device=device)

scaler_last = fold_last['scaler']
top_indices_last = fold_last['top_feature_indices']
feature_names_last = fold_last['feature_names']
regime_detector_last = fold_last.get('regime_detector', None)

if len(top_indices_last) == 0:
    print(f"  WARNING: top_feature_indices empty, using first {top_k} features")
    top_indices_last = np.arange(min(top_k, len(feature_names_last)), dtype=np.intp)

print(f"  Scaler: {type(scaler_last).__name__}")
print(f"  Top features: {len(top_indices_last)} -> indices: {top_indices_last[:10]}...")
print(f"  Feature names count: {len(feature_names_last)}")
print(f"  Regime detector: {'loaded' if regime_detector_last else 'not found'}")

# Align features
fwd_feature_cols_model = feature_names_last
missing = [c for c in fwd_feature_cols_model if c not in df_fwd_full.columns]
if missing:
    print(f"  WARNING: {len(missing)} features missing, zero-filling:")
    for col in missing:
        print(f"    - {col}")
        df_fwd_full[col] = 0.0
else:
    print(f"  All {len(fwd_feature_cols_model)} features found in forward data")

fwd_raw = df_fwd_full[fwd_feature_cols_model].values.astype(np.float32)
fwd_scaled = scaler_last.transform(fwd_raw).astype(np.float32)
fwd_scaled = np.nan_to_num(fwd_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# === DIAGNOSTIC 1: Feature scaling health ===
print("\n--- DIAGNOSTIC 1: Feature Scaling ---")
print(f"  Raw features: min={fwd_raw.min():.4f}, max={fwd_raw.max():.4f}, "
      f"mean={fwd_raw.mean():.4f}, std={fwd_raw.std():.4f}")
print(f"  Scaled features: min={fwd_scaled.min():.4f}, max={fwd_scaled.max():.4f}, "
      f"mean={fwd_scaled.mean():.4f}, std={fwd_scaled.std():.4f}")
# Check for columns that are all-zero (constant)
zero_cols = np.where(np.abs(fwd_scaled).max(axis=0) < 1e-6)[0]
print(f"  All-zero scaled columns: {len(zero_cols)} / {fwd_scaled.shape[1]}")
if len(zero_cols) > 0:
    zero_names = [fwd_feature_cols_model[i] for i in zero_cols[:10]]
    print(f"    Examples: {zero_names}")
# Check for extreme values
extreme_cols = np.where(np.abs(fwd_scaled).max(axis=0) > 100)[0]
print(f"  Extreme scaled columns (|x|>100): {len(extreme_cols)} / {fwd_scaled.shape[1]}")

encoder_last = HybridEncoder(
    n_features=fwd_scaled.shape[1],
    config=config.model,
).to(device)
encoder_last.load_state_dict(fold_last['model_state'])
encoder_last.eval()
print(f"\n  HybridEncoder loaded: {sum(p.numel() for p in encoder_last.parameters())} params")

# Extract latents
print("Extracting latent vectors...")
dummy_l = np.zeros(len(fwd_scaled), dtype=np.int64)
dummy_r = np.zeros(len(fwd_scaled), dtype=np.float32)
ds_fwd = ScalpDataset(fwd_scaled, dummy_l, dummy_r, seq_len)
loader_fwd = DataLoader(ds_fwd, batch_size=512, shuffle=False, pin_memory=True)

fwd_latents = []
with torch.no_grad():
    for bx, _, _ in loader_fwd:
        bx = bx.to(device)
        with autocast('cuda', enabled=True):
            _, lat = encoder_last(bx)
        fwd_latents.append(lat.cpu().numpy())
fwd_latents = np.concatenate(fwd_latents, axis=0)

# === DIAGNOSTIC 2: Latent vector health ===
print("\n--- DIAGNOSTIC 2: Latent Vectors ---")
print(f"  Shape: {fwd_latents.shape}")
print(f"  Mean: {fwd_latents.mean():.6f}, Std: {fwd_latents.std():.6f}")
print(f"  Min: {fwd_latents.min():.6f}, Max: {fwd_latents.max():.6f}")
lat_var = fwd_latents.var(axis=0)
dead_dims = (lat_var < 1e-8).sum()
print(f"  Dead dimensions (var<1e-8): {dead_dims} / {fwd_latents.shape[1]}")
print(f"  Per-dim variance: min={lat_var.min():.8f}, max={lat_var.max():.8f}, "
      f"median={np.median(lat_var):.8f}")
pairwise_diff = np.abs(fwd_latents[::100] - fwd_latents[0:1]).mean()
print(f"  Avg L1 diff between samples (sampled): {pairwise_diff:.6f}")

# Regime probs (forward-only)
fwd_df_aligned = df_fwd_full.iloc[seq_len:]
if regime_detector_last is not None:
    fwd_regime = regime_detector_last.predict_proba_online(fwd_df_aligned)
else:
    fwd_regime = np.full((len(fwd_df_aligned), 3), 1/3, dtype=np.float32)

# === DIAGNOSTIC 3: Regime probs ===
print("\n--- DIAGNOSTIC 3: Regime Probs ---")
print(f"  Shape: {fwd_regime.shape}")
print(f"  Mean per class: Bull={fwd_regime[:,0].mean():.4f}, "
      f"Bear={fwd_regime[:,1].mean():.4f}, Choppy={fwd_regime[:,2].mean():.4f}")
print(f"  Dominant regime: Bull={np.mean(fwd_regime.argmax(1)==0)*100:.1f}%, "
      f"Bear={np.mean(fwd_regime.argmax(1)==1)*100:.1f}%, "
      f"Choppy={np.mean(fwd_regime.argmax(1)==2)*100:.1f}%")

fwd_hc = fwd_scaled[seq_len:][:, top_indices_last]

# === DIAGNOSTIC 4: Handcrafted features ===
print("\n--- DIAGNOSTIC 4: Handcrafted Features ---")
print(f"  Shape: {fwd_hc.shape}")
print(f"  Mean: {fwd_hc.mean():.4f}, Std: {fwd_hc.std():.4f}")
hc_var = fwd_hc.var(axis=0)
hc_dead = (hc_var < 1e-8).sum()
print(f"  Dead features (var<1e-8): {hc_dead} / {fwd_hc.shape[1]}")

fwd_min = min(len(fwd_latents), len(fwd_hc), len(fwd_regime))
fwd_latents = fwd_latents[:fwd_min]
fwd_hc = fwd_hc[:fwd_min]
fwd_regime = fwd_regime[:fwd_min]

fwd_meta = XGBoostMetaLearner.build_meta_features(fwd_latents, fwd_hc, fwd_regime)

# === DIAGNOSTIC 5: Meta-feature matrix ===
print("\n--- DIAGNOSTIC 5: Meta-Features ---")
n_hc = len(top_indices_last)
print(f"  Shape: {fwd_meta.shape} (expect 128 + {n_hc} + 3 = {128+n_hc+3})")
print(f"  Latent block [0:128]: mean={fwd_meta[:,:128].mean():.4f}, std={fwd_meta[:,:128].std():.4f}")
print(f"  HC block [128:{128+n_hc}]: mean={fwd_meta[:,128:128+n_hc].mean():.4f}, "
      f"std={fwd_meta[:,128:128+n_hc].std():.4f}")
print(f"  Regime block [-3:]: mean={fwd_meta[:,-3:].mean():.4f}, std={fwd_meta[:,-3:].std():.4f}")

xgb_last = XGBoostMetaLearner(config.model.xgboost)
xgb_last.load(f'{CHECKPOINT_DIR}/xgb_fold_{last_fold_idx:03d}.json')

fwd_probs = xgb_last.predict_proba(fwd_meta)

# === DIAGNOSTIC 6: Prediction distribution ===
print("\n--- DIAGNOSTIC 6: Predictions ---")
print(f"  Shape: {fwd_probs.shape}")
print(f"  Class dist: Short={np.mean(fwd_probs.argmax(1)==0)*100:.1f}%, "
      f"Hold={np.mean(fwd_probs.argmax(1)==1)*100:.1f}%, "
      f"Long={np.mean(fwd_probs.argmax(1)==2)*100:.1f}%")
print(f"  Avg max confidence: {fwd_probs.max(1).mean():.4f}")
print(f"  Avg probs: P(Short)={fwd_probs[:,0].mean():.4f}, "
      f"P(Hold)={fwd_probs[:,1].mean():.4f}, P(Long)={fwd_probs[:,2].mean():.4f}")
print(f"  Confidence range: {fwd_probs.max(1).min():.4f} - {fwd_probs.max(1).max():.4f}")

# XGBoost feature importance breakdown
importance = xgb_last.model.feature_importances_
lat_imp = importance[:128].sum()
hc_imp = importance[128:128+n_hc].sum()
reg_imp = importance[-3:].sum()
print(f"\n  XGBoost importance: Latent={lat_imp:.3f}, HC={hc_imp:.3f}, Regime={reg_imp:.3f}")
top5_idx = np.argsort(importance)[-5:]
print(f"  Top 5 features by importance (index): {top5_idx}")

print("\n=== FORWARD DIAGNOSTICS COMPLETE ===")

# ============================================================
# Forward Test Step 3: Backtest only on bars AFTER training cutoff
# ============================================================
# --- Forward test: düşük threshold ile test ---
FWD_CONFIDENCE_THRESHOLD = 0.40  # Normal: 0.70, test amaçlı düşürüldü

df_fwd_bt = df_fwd_full.iloc[seq_len:seq_len + fwd_min].copy()

# Only test bars after the training data cutoff
cutoff_ts = pd.Timestamp(FWD_TEST_CUTOFF)
if df_fwd_bt.index.tz is not None:
    cutoff_ts = cutoff_ts.tz_localize(df_fwd_bt.index.tz)

# Find the index where forward test actually starts
fwd_test_mask = df_fwd_bt.index > cutoff_ts
fwd_test_start_idx = int(np.argmax(fwd_test_mask)) if fwd_test_mask.any() else len(df_fwd_bt)
print(f"Warmup bars (before cutoff): {fwd_test_start_idx}")
print(f"Forward test bars: {fwd_test_mask.sum()}")
print(f"Forward test period: {df_fwd_bt.index[fwd_test_start_idx]} â†’ {df_fwd_bt.index[-1]}")

# ATR percentile
if 'atr_14' in df_fwd_bt.columns:
    df_fwd_bt['atr_pctile'] = df_fwd_bt['atr_14'].rolling(96, min_periods=10).rank(pct=True)
    df_fwd_bt['atr_pctile'] = df_fwd_bt['atr_pctile'].fillna(1.0)
    FWD_MEDIAN_ATR = df_fwd_bt['atr_14'].median()
else:
    df_fwd_bt['atr_pctile'] = 1.0
    FWD_MEDIAN_ATR = 1.0

def fwd_get_slippage_bps(atr_val):
    if not USE_VAR_SLIPPAGE:
        return SLIPPAGE_BPS
    return slip_cfg.base_bps + slip_cfg.volatility_scale * (atr_val / (FWD_MEDIAN_ATR + 1e-10))

# --- Forward Backtest Loop ---
fwd_trades = []
fwd_equity = [0.0]
fwd_cum_pnl = 0.0
fwd_skip = {'low_adx': 0, 'low_volatility': 0, 'low_conf': 0,
            'hold': 0, 'daily_cap': 0, 'no_atr': 0, 'choppy': 0,
            'no_next_bar': 0, 'warmup': 0}

fwd_trade_mgr = TradeManager(trade_mgmt_cfg, label_cfg.max_holding_bars)
fwd_active = None
fwd_pending = None
fwd_daily_count = 0
fwd_prev_date = None
fwd_pos_size = 0.0
fwd_entry_bar_local = 0

n_fwd_bars = len(df_fwd_bt)

for i in tqdm(range(n_fwd_bars), desc='Forward Test'):
    row = df_fwd_bt.iloc[i]
    cur_date = row.name.date() if hasattr(row.name, 'date') else None
    if cur_date != fwd_prev_date:
        fwd_daily_count = 0
        fwd_prev_date = cur_date

    # Manage active trade
    if fwd_active is not None:
        is_choppy = fwd_regime[i, 2] > CHOPPY_THRESHOLD if i < len(fwd_regime) else False
        fwd_active = fwd_trade_mgr.update(fwd_active, row['high'], row['low'], row['close'], is_choppy)
        if fwd_active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):
            gross = fwd_active.pnl
            slip_bps = fwd_get_slippage_bps(fwd_active.atr_at_entry)
            cost = entry_cost_frac(slip_bps)
            if fwd_active.partial_fills:
                cost += exit_cost_frac(TradeStatus.CLOSED_TP, slip_bps) * partial_pct
                cost += exit_cost_frac(fwd_active.status, slip_bps) * (1 - partial_pct)
            else:
                cost += exit_cost_frac(fwd_active.status, slip_bps)
            impact = market_impact_frac(fwd_pos_size, fwd_active.entry_price)
            cost += impact
            entry_ts = df_fwd_bt.index[fwd_entry_bar_local]
            n_fund = count_funding_intervals(entry_ts, row.name)
            funding = n_fund * (funding_cfg.fixed_rate_pct / 100.0) if USE_FUNDING else 0.0
            unit_net = (gross - cost - funding) * fwd_pos_size
            lev_net = unit_net * LEVERAGE
            # Only count trades that ENTERED after cutoff
            if fwd_entry_bar_local >= fwd_test_start_idx:
                fwd_cum_pnl += lev_net
                fwd_trades.append(dict(
                    direction=fwd_active.direction, entry_price=fwd_active.entry_price,
                    bars_held=fwd_active.bars_held, status=fwd_active.status.value,
                    gross_pnl=gross * fwd_pos_size * LEVERAGE, net_pnl=lev_net,
                    cost=cost * fwd_pos_size * LEVERAGE, position_size=fwd_pos_size,
                    slippage_bps=slip_bps, timestamp=row.name,
                ))
                fwd_equity.append(fwd_cum_pnl)
            fwd_active = None
        continue

    # Execute pending signal
    if fwd_pending is not None:
        ps = fwd_pending
        fwd_pending = None
        entry_price = row['open']
        atr = ps['atr']
        direction = ps['direction']
        confidence = ps['confidence']
        if direction == "LONG":
            sl = entry_price - label_cfg.sl_multiplier * atr
        else:
            sl = entry_price + label_cfg.sl_multiplier * atr
        p_k = confidence
        kelly = max((p_k * kelly_b - (1 - p_k)) / kelly_b, 0)
        fwd_pos_size = min(kelly * kelly_fraction, kelly_max)
        if fwd_pos_size < 1e-6:
            continue
        fwd_active = TradeState(
            direction=direction, entry_price=entry_price,
            current_stop_loss=sl, take_profit=0.0, atr_at_entry=atr,
        )
        fwd_entry_bar_local = i
        fwd_daily_count += 1
        continue

    # Only generate signals after cutoff
    if i < fwd_test_start_idx:
        fwd_skip['warmup'] += 1
        continue

    if i >= len(fwd_probs):
        continue

    p = fwd_probs[i]
    cls = int(np.argmax(p))
    if cls == 1:
        fwd_skip['hold'] += 1
        continue
    if max(p[0], p[2]) < FWD_CONFIDENCE_THRESHOLD:  # düşürülmüş threshold
        fwd_skip['low_conf'] += 1
        continue
    if fwd_daily_count >= exec_cfg.max_trades_per_day:
        fwd_skip['daily_cap'] += 1
        continue

    # Time-of-day filter
    tdf = getattr(exec_cfg, 'time_of_day_filter', None)
    if tdf and tdf.enabled:
        hr = row.name.hour if hasattr(row.name, 'hour') else pd.to_datetime(row.name).hour
        if hr in tdf.blocked_hours_utc:
            fwd_skip['choppy'] += 1  # Count as choppy skip or time skip
            continue

    atr = row['atr_14'] if 'atr_14' in df_fwd_bt.columns else 0.0
    if atr < 1e-10:
        fwd_skip['no_atr'] += 1
        continue
    adx_val = row['adx_14'] if 'adx_14' in df_fwd_bt.columns else 999.0
    if adx_val < MIN_ADX:
        fwd_skip['low_adx'] += 1
        continue
    atr_pct = row['atr_pctile'] if 'atr_pctile' in df_fwd_bt.columns else 1.0
    if atr_pct < MIN_ATR_PCTILE:
        fwd_skip['low_volatility'] += 1
        continue
    if i < len(fwd_regime) and fwd_regime[i, 2] > CHOPPY_THRESHOLD:
        fwd_skip['choppy'] += 1
        continue
    if i + 1 >= n_fwd_bars:
        fwd_skip['no_next_bar'] += 1
        continue

    fwd_pending = {
        'direction': "LONG" if cls == 2 else "SHORT",
        'atr': atr,
        'confidence': max(p[0], p[2]),
    }

# Force-close
if fwd_active is not None and fwd_entry_bar_local >= fwd_test_start_idx:
    last_row = df_fwd_bt.iloc[-1]
    if fwd_active.direction == "LONG":
        unr = (last_row['close'] - fwd_active.entry_price) / fwd_active.entry_price
    else:
        unr = (fwd_active.entry_price - last_row['close']) / fwd_active.entry_price
    gross = fwd_active.pnl + unr * fwd_active.remaining_size
    slip_bps = fwd_get_slippage_bps(fwd_active.atr_at_entry)
    cost = entry_cost_frac(slip_bps) + exit_cost_frac('CLOSED_FOLD_END', slip_bps)
    unit_net = (gross - cost) * fwd_pos_size
    lev_net = unit_net * LEVERAGE
    fwd_cum_pnl += lev_net
    fwd_trades.append(dict(
        direction=fwd_active.direction, entry_price=fwd_active.entry_price,
        bars_held=fwd_active.bars_held, status='CLOSED_END',
        gross_pnl=gross * fwd_pos_size * LEVERAGE, net_pnl=lev_net,
        cost=cost * fwd_pos_size * LEVERAGE, position_size=fwd_pos_size,
        slippage_bps=slip_bps, timestamp=last_row.name,
    ))
    fwd_equity.append(fwd_cum_pnl)

# ============================================================
# Forward Test Results
# ============================================================
fwd_df = pd.DataFrame(fwd_trades)

if len(fwd_df) == 0:
    print(f"\n*** NO TRADES in forward test period ({FWD_TEST_CUTOFF} â†’ present) ***")
    print(f"Skip reasons: {fwd_skip}")
else:
    fwd_net = fwd_df['net_pnl'].values
    fwd_n = len(fwd_df)
    fwd_wins = fwd_net[fwd_net > 0]
    fwd_losses = fwd_net[fwd_net < 0]
    fwd_wr = len(fwd_wins) / fwd_n
    fwd_pf = abs(fwd_wins.sum() / fwd_losses.sum()) if len(fwd_losses) else float('inf')

    fwd_df['date'] = pd.to_datetime(fwd_df['timestamp']).dt.date
    fwd_daily = fwd_df.groupby('date')['net_pnl'].sum()
    fwd_range = pd.date_range(fwd_daily.index.min(), fwd_daily.index.max(), freq='D')
    fwd_daily = fwd_daily.reindex(fwd_range, fill_value=0.0)
    fwd_sharpe = fwd_daily.mean() / (fwd_daily.std() + 1e-10) * np.sqrt(365)
    fwd_cum_arr = np.cumsum(fwd_daily.values)
    fwd_peak = np.maximum.accumulate(fwd_cum_arr)
    fwd_dd = fwd_peak - fwd_cum_arr
    fwd_mdd = fwd_dd.max()

    n_days = (fwd_daily.index[-1] - fwd_daily.index[0]).days + 1

    print()
    print('=' * 60)
    print(f'       FORWARD TEST â€” {FWD_TEST_CUTOFF} â†’ Present (Fold {last_fold_idx})')
    print(f'       (Leverage: {LEVERAGE}x, {n_days} days)')
    print('=' * 60)
    print(f'  Total trades       : {fwd_n}')
    print(f'  Win rate           : {fwd_wr:.4f} ({fwd_wr*100:.1f}%)')
    print(f'  Profit factor      : {fwd_pf:.4f}')
    print(f'  Expectancy/trade   : {fwd_net.mean()*100:.4f}%')
    print(f'  Daily Sharpe       : {fwd_sharpe:.4f}')
    print(f'  Max Drawdown       : {fwd_mdd*100:.4f}%')
    print(f'  Net PnL            : {fwd_net.sum()*100:.2f}%')
    print()
    print(f'  Skip reasons: {fwd_skip}')
    print()

    # Interpretation
    print('-' * 60)
    print('  INTERPRETATION')
    print('-' * 60)
    if fwd_wr > 0.60 and fwd_pf > 2.0:
        print('  STRONG: Model performs well on truly unseen recent data.')
        print('  Good candidate for live deployment.')
    elif fwd_wr > 0.50 and fwd_pf > 1.2:
        print('  MODERATE: Edge exists on recent data but with degradation.')
        print('  Consider paper trading before live.')
    elif fwd_wr > 0.40 and fwd_pf > 1.0:
        print('  WEAK: Marginal edge. May not survive live trading costs.')
    else:
        print('  NEGATIVE: Model fails on recent data.')
        print('  DO NOT deploy live. Retrain or investigate.')

    # Comparison table
    print()
    print('-' * 60)
    print('=' * 60)

    # Weekly breakdown
    fwd_df['week'] = pd.to_datetime(fwd_df['timestamp']).dt.to_period('W')
    fwd_weekly = fwd_df.groupby('week')['net_pnl'].agg(['sum', 'count'])
    fwd_weekly.columns = ['return_pct', 'trades']
    fwd_weekly['return_pct'] *= 100
    print('\nForward Test Weekly Returns:')
    print(fwd_weekly.to_string())

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fwd_cum_pct = np.cumsum(fwd_daily.values) * 100
    axes[0].plot(fwd_daily.index, fwd_cum_pct, linewidth=1.5, color='#4CAF50')
    axes[0].fill_between(fwd_daily.index, 0, fwd_cum_pct, alpha=0.1, color='#4CAF50')
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Forward Test Equity ({LEVERAGE}x)')
    axes[0].set_ylabel('Cumulative PnL (%)')
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(fwd_daily.index, 0, -fwd_dd * 100, alpha=0.4, color='red')
    axes[1].set_title('Forward Test Drawdown')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# Sinyal Tablosu: Tüm sinyaller + sonuçları (trade açmadan)
# ============================================================
# Model sinyallerini topla, her biri için TP/SL'ye ulaşılıp ulaşılmadığını kontrol et.

signals = []

df_sig = df_fwd_full.iloc[seq_len:seq_len + fwd_min].copy()
cutoff_ts = pd.Timestamp(FWD_TEST_CUTOFF)
if df_sig.index.tz is not None:
    cutoff_ts = cutoff_ts.tz_localize(df_sig.index.tz)

sig_test_mask = df_sig.index > cutoff_ts
sig_start_idx = int(np.argmax(sig_test_mask)) if sig_test_mask.any() else len(df_sig)

n_sig_bars = len(df_sig)
sig_daily_count = 0
sig_prev_date = None

for i in range(sig_start_idx, n_sig_bars):
    row = df_sig.iloc[i]
    cur_date = row.name.date() if hasattr(row.name, 'date') else None
    if cur_date != sig_prev_date:
        sig_daily_count = 0
        sig_prev_date = cur_date

    if i >= len(fwd_probs):
        continue

    p = fwd_probs[i]
    cls = int(np.argmax(p))
    if cls == 1:  # hold
        continue

    confidence = max(p[0], p[2])
    if confidence < FWD_CONFIDENCE_THRESHOLD:
        continue
    if sig_daily_count >= exec_cfg.max_trades_per_day:
        continue

    atr = row['atr_14'] if 'atr_14' in df_sig.columns else 0.0
    if atr < 1e-10:
        continue

    adx_val = row['adx_14'] if 'adx_14' in df_sig.columns else 999.0
    if adx_val < MIN_ADX:
        continue

    atr_pct = row['atr_pctile'] if 'atr_pctile' in df_sig.columns else 1.0
    if atr_pct < MIN_ATR_PCTILE:
        continue

    if i < len(fwd_regime) and fwd_regime[i, 2] > CHOPPY_THRESHOLD:
        continue

    # Sonraki bar var mı? (entry next bar open)
    if i + 1 >= n_sig_bars:
        continue

    direction = "LONG" if cls == 2 else "SHORT"
    entry_bar = i + 1
    entry_price = df_sig.iloc[entry_bar]['open']

    # SL ve TP hesapla
    if direction == "LONG":
        sl_price = entry_price - label_cfg.sl_multiplier * atr
        tp1_price = entry_price + trade_mgmt_cfg.partial_tp_1_atr * atr
        tp2_price = entry_price + trade_mgmt_cfg.full_tp_atr * atr
    else:
        sl_price = entry_price + label_cfg.sl_multiplier * atr
        tp1_price = entry_price - trade_mgmt_cfg.partial_tp_1_atr * atr
        tp2_price = entry_price - trade_mgmt_cfg.full_tp_atr * atr

    # Kelly pozisyon boyutu
    p_k = confidence
    kelly = max((p_k * kelly_b - (1 - p_k)) / kelly_b, 0)
    pos_size = min(kelly * kelly_fraction, kelly_max)

    # Gerçekte ne oldu? (sonraki barları tara)
    outcome = "?"
    outcome_bar = 0
    outcome_price = entry_price
    max_bars = label_cfg.max_holding_bars

    for j in range(1, max_bars + 1):
        look_idx = entry_bar + j
        if look_idx >= n_sig_bars:
            outcome = "VERİ YOK"
            break

        bar_data = df_sig.iloc[look_idx]
        h, l, c_price = bar_data['high'], bar_data['low'], bar_data['close']

        if direction == "LONG":
            if l <= sl_price:
                outcome = "❌ SL"
                outcome_bar = j
                outcome_price = sl_price
                break
            if h >= tp2_price:
                outcome = "✅ Full TP"
                outcome_bar = j
                outcome_price = tp2_price
                break
            if h >= tp1_price and outcome != "⚡ Partial TP":
                outcome = "⚡ Partial TP"
                outcome_bar = j
        else:  # SHORT
            if h >= sl_price:
                outcome = "❌ SL"
                outcome_bar = j
                outcome_price = sl_price
                break
            if l <= tp2_price:
                outcome = "✅ Full TP"
                outcome_bar = j
                outcome_price = tp2_price
                break
            if l <= tp1_price and outcome != "⚡ Partial TP":
                outcome = "⚡ Partial TP"
                outcome_bar = j

    if outcome == "?":
        outcome = "⏱️ Zaman Aşımı"
        outcome_bar = max_bars
        look_idx = min(entry_bar + max_bars, n_sig_bars - 1)
        outcome_price = df_sig.iloc[look_idx]['close']

    # PnL hesapla
    if direction == "LONG":
        pnl_pct = (outcome_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - outcome_price) / entry_price * 100

    signals.append({
        'Tarih': row.name.strftime('%Y-%m-%d %H:%M') if hasattr(row.name, 'strftime') else str(row.name),
        'Yön': direction,
        'Güven': f'{confidence:.3f}',
        'Entry': f'{entry_price:.1f}',
        'SL': f'{sl_price:.1f}',
        'TP1 (50%)': f'{tp1_price:.1f}',
        'TP2 (Full)': f'{tp2_price:.1f}',
        'Pozisyon': f'{pos_size*100:.2f}%' if pos_size > 0 else '0 (Kelly)',
        'Sonuç': outcome,
        'Bar': outcome_bar,
        'PnL%': f'{pnl_pct:+.3f}%',
    })
    sig_daily_count += 1

sig_df = pd.DataFrame(signals)

if len(sig_df) == 0:
    print("Filtreleri geçen sinyal yok!")
else:
    print(f'Toplam sinyal: {len(sig_df)}')
    print()

    # Sonuç özeti
    outcomes = sig_df['Sonuç'].value_counts()
    print('Sonuç Dağılımı:')
    for outcome, count in outcomes.items():
        print(f'  {outcome}: {count} ({count/len(sig_df)*100:.0f}%)')
    print()

    # Kelly filtresi bilgisi
    kelly_zero = sig_df[sig_df['Pozisyon'] == '0 (Kelly)']
    kelly_trade = sig_df[sig_df['Pozisyon'] != '0 (Kelly)']
    print(f"Kelly'nin trade açtığı: {len(kelly_trade)} / {len(sig_df)}")
    print(f"Kelly'nin reddettiği: {len(kelly_zero)} / {len(sig_df)}")

    if len(kelly_zero) > 0:
        zero_outcomes = kelly_zero['Sonuç'].value_counts()
        print(f"  Reddedilenlerin sonuçları: {dict(zero_outcomes)}")
    print()

    # Tam tablo
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', 20)
    print(sig_df.to_string(index=False))
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_colwidth')

# ============================================================
# FULL REPORT EXPORT — Saves ALL metrics to a single JSON file
# ============================================================
import json as _json
from datetime import datetime as _dt

_report = {
    "generated_at": _dt.now(timezone.utc).isoformat(),
    "leverage": LEVERAGE,
    "config": {
        "confidence_threshold": exec_cfg.confidence_threshold,
        "min_adx": MIN_ADX,
        "min_atr_percentile": MIN_ATR_PCTILE,
        "choppy_threshold": CHOPPY_THRESHOLD,
        "sl_multiplier": label_cfg.sl_multiplier,
        "full_tp_atr": trade_mgmt_cfg.full_tp_atr,
        "partial_tp_1_atr": trade_mgmt_cfg.partial_tp_1_atr,
        "partial_tp_1_pct": trade_mgmt_cfg.partial_tp_1_pct,
        "kelly_fraction": kelly_fraction,
        "kelly_max": kelly_max,
        "kelly_b": kelly_b,
    },
}

# --- Walk-Forward Backtest ---
if len(trades_df) > 0:
    _net = trades_df['net_pnl'].values
    _gross = trades_df['gross_pnl'].values
    _wins = _net[_net > 0]
    _losses = _net[_net < 0]

    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
    _daily = trades_df.groupby('date')['net_pnl'].sum()
    _full_range = pd.date_range(_daily.index.min(), _daily.index.max(), freq='D')
    _daily = _daily.reindex(_full_range, fill_value=0.0)
    _cum = np.cumsum(_daily.values)
    _peak = np.maximum.accumulate(_cum)
    _dd = _peak - _cum

    _report["walkforward"] = {
        "total_trades": len(trades_df),
        "win_rate": round(len(_wins) / len(_net), 4),
        "profit_factor": round(abs(_wins.sum() / _losses.sum()), 4) if len(_losses) > 0 else None,
        "expectancy_pct": round(float(_net.mean() * 100), 4),
        "avg_win_pct": round(float(_wins.mean() * 100), 4) if len(_wins) else 0,
        "avg_loss_pct": round(float(_losses.mean() * 100), 4) if len(_losses) else 0,
        "avg_bars_held": round(float(trades_df['bars_held'].mean()), 1),
        "daily_sharpe": round(float(_daily.mean() / (_daily.std() + 1e-10) * np.sqrt(365)), 4),
        "max_drawdown_pct": round(float(_dd.max() * 100), 4),
        "gross_pnl_pct": round(float(_gross.sum() * 100), 2),
        "net_pnl_pct": round(float(_net.sum() * 100), 2),
        "cost_impact_pct": round(float((_gross.sum() - _net.sum()) * 100), 2),
        "close_reasons": trades_df['status'].value_counts().to_dict(),
        "skip_reasons": skip_reasons,
    }

    # Yearly
    trades_df['year'] = pd.to_datetime(trades_df['timestamp']).dt.year
    _yearly = []
    for _y, _grp in trades_df.groupby('year'):
        _yn = _grp['net_pnl'].values
        _yw = _yn[_yn > 0]
        _yl = _yn[_yn < 0]
        _ywr = len(_yw) / len(_yn) if len(_yn) > 0 else 0
        _ypf = abs(_yw.sum() / _yl.sum()) if len(_yl) > 0 else None
        _yd = _grp.groupby(_grp['timestamp'].apply(lambda x: pd.to_datetime(x).date()))['net_pnl'].sum()
        _ydr = pd.date_range(_yd.index.min(), _yd.index.max(), freq='D')
        _yd = _yd.reindex(_ydr, fill_value=0.0)
        _ysh = float(_yd.mean() / (_yd.std() + 1e-10) * np.sqrt(365))
        _yc = np.cumsum(_yd.values)
        _yp = np.maximum.accumulate(_yc)
        _ymdd = float((_yp - _yc).max() * 100)
        _yearly.append({
            "year": int(_y), "trades": len(_grp),
            "win_rate": round(_ywr, 4), "profit_factor": round(_ypf, 4) if _ypf else None,
            "sharpe": round(_ysh, 2), "net_pnl_pct": round(float(_yn.sum() * 100), 2),
            "max_drawdown_pct": round(_ymdd, 2),
        })
    _report["yearly"] = _yearly

    # Quarterly
    trades_df['quarter'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('Q')
    _quarterly = []
    for _q, _grp in trades_df.groupby('quarter'):
        _qn = _grp['net_pnl'].values
        _qw = _qn[_qn > 0]
        _ql = _qn[_qn < 0]
        _quarterly.append({
            "quarter": str(_q), "trades": len(_grp),
            "win_rate": round(len(_qw) / len(_qn), 4) if len(_qn) > 0 else 0,
            "profit_factor": round(abs(_qw.sum() / _ql.sum()), 4) if len(_ql) > 0 else None,
            "net_pnl_pct": round(float(_qn.sum() * 100), 2),
        })
    _report["quarterly"] = _quarterly

    # Monthly
    trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
    _monthly = []
    for _m, _grp in trades_df.groupby('month'):
        _mn = _grp['net_pnl'].values
        _monthly.append({
            "month": str(_m), "trades": len(_grp),
            "net_pnl_pct": round(float(_mn.sum() * 100), 2),
        })
    _report["monthly"] = _monthly

# --- Forward Test ---
if 'fwd_df' in dir() and len(fwd_df) > 0:
    _fn = fwd_df['net_pnl'].values
    _fw = _fn[_fn > 0]
    _fl = _fn[_fn < 0]
    _report["forward_test"] = {
        "period": f"{FWD_TEST_CUTOFF} → {df_fwd_bt.index[-1].strftime('%Y-%m-%d')}",
        "total_trades": len(fwd_df),
        "win_rate": round(len(_fw) / len(_fn), 4),
        "profit_factor": round(abs(_fw.sum() / _fl.sum()), 4) if len(_fl) > 0 else None,
        "expectancy_pct": round(float(_fn.mean() * 100), 4),
        "daily_sharpe": round(float(fwd_sharpe), 4),
        "max_drawdown_pct": round(float(fwd_mdd * 100), 4),
        "net_pnl_pct": round(float(_fn.sum() * 100), 2),
        "skip_reasons": fwd_skip,
    }
    # Weekly
    fwd_df['week'] = pd.to_datetime(fwd_df['timestamp']).dt.to_period('W')
    _fweekly = []
    for _w, _grp in fwd_df.groupby('week'):
        _wn = _grp['net_pnl'].values
        _fweekly.append({
            "week": str(_w), "trades": len(_grp),
            "net_pnl_pct": round(float(_wn.sum() * 100), 2),
        })
    _report["forward_weekly"] = _fweekly

# Save
_report_path = '/content/drive/MyDrive/scalp2/data/processed/backtest_report.json'
with open(_report_path, 'w', encoding='utf-8') as _f:
    _json.dump(_report, _f, indent=2, ensure_ascii=False, default=str)
print(f'\n✅ Full report saved to: {_report_path}')
print(f'   Size: {os.path.getsize(_report_path) / 1024:.1f} KB')