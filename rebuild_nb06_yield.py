"""Add Yield Maximization to NB06"""
import json

nb_path = "notebooks/06_backtest.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    joined = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "BACKTEST ENGINE" in joined:
        cell["source"] = ["""# ============================================================
#  YIELD-MAXIMIZED BACKTEST ENGINE (ADX + ATR + TP + Conviction Sizing)
# ============================================================
from tqdm import tqdm
import numpy as np

seq_len = config.model.seq_len
exec_cfg = config.execution
label_cfg = config.labeling
order_cfg = config.execution.order_execution

DIR_FILTER = getattr(exec_cfg, 'direction_filter', 'both')
BASE_HOLD_BARS = label_cfg.max_holding_bars  # 10 bars
MAX_HOLD_BARS = 24  # Let winners run up to 24 hours
CONF_THRESHOLD = exec_cfg.confidence_threshold
HIGH_CONF_THRESHOLD = 0.55
LEVERAGE = exec_cfg.position_sizing.leverage
BASE_POSITION_SIZE = 0.10  # 10% base equity
TP_PCT = 0.015  # 1.5% gross take-profit
MAX_CONCURRENT_TRANCHES = 20
MAX_TRADES_PER_DAY = 24

MIN_ADX = exec_cfg.min_adx
MIN_ATR_PCTILE = exec_cfg.min_atr_percentile

if 'atr_14' in df.columns and 'atr_pctile' not in df.columns:
    df['atr_pctile'] = df['atr_14'].rolling(96, min_periods=10).rank(pct=True).fillna(1.0)

# Costs
MAKER_FEE_BPS = order_cfg.maker_fee_bps
SLIPPAGE_BPS = order_cfg.slippage_bps
FLAT_RT_COST = 2 * (SLIPPAGE_BPS + MAKER_FEE_BPS) / 10_000

print(f'Strategy: Yield-Maximized Pyramiding (Max {MAX_CONCURRENT_TRANCHES})')
print(f'Direction: {DIR_FILTER} | Hold {BASE_HOLD_BARS}-{MAX_HOLD_BARS} bars | Conf >= {CONF_THRESHOLD}')
print(f'Filters: ADX >= {MIN_ADX} | ATR Pctile >= {MIN_ATR_PCTILE}')
print(f'Yield Boosters: TP @ {TP_PCT*100}% | 2x Size if Conf > {HIGH_CONF_THRESHOLD}')
print(f'Leverage: {LEVERAGE}x | RT Cost: {FLAT_RT_COST*100:.4f}%')

all_trades = []
cumulative_pnl = 0.0
skip_reasons = {}
fold_stats = []

def _close_tranche(t, exit_price, status, exit_bar, exit_timestamp, fold_idx):
    global cumulative_pnl
    entry_price = t['entry_price']
    direction = t['direction']
    size = t['size']
    
    if direction == 'LONG':
        raw_ret = (exit_price - entry_price) / entry_price
    else:
        raw_ret = (entry_price - exit_price) / entry_price

    net = (raw_ret - FLAT_RT_COST) * size * LEVERAGE
    cumulative_pnl += net

    all_trades.append({
        'fold': fold_idx,
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'bars_held': t['bars_held'],
        'status': status,
        'gross_pnl': raw_ret * size * LEVERAGE,
        'net_pnl': net,
        'cost': FLAT_RT_COST * size * LEVERAGE,
        'position_size': size,
        'entry_bar': t['entry_bar'],
        'exit_bar': exit_bar,
        'timestamp': exit_timestamp,
    })

for fold_data in tqdm(wf_predictions, desc='Backtesting folds'):
    fold_idx = fold_data['fold_idx']
    test_start = fold_data['test_start']
    test_end = fold_data['test_end']
    preds = fold_data['test_probabilities']
    n_preds = len(preds)
    pred_offset = test_start + seq_len

    fold_skips = {}
    fold_trades = 0

    active_tranches = []
    pending_tranches = []
    
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

        # -- Activate pending tranches --
        still_pending = []
        for pt in pending_tranches:
            if pt['entry_bar'] == bar:
                pt['entry_price'] = row['open']
                active_tranches.append(pt)
            else:
                still_pending.append(pt)
        pending_tranches = still_pending

        # -- Evaluate Exits for Active Tranches --
        surviving_tranches = []
        adx_val = row.get('adx', 0)
        
        for t in active_tranches:
            t['bars_held'] += 1
            entry_p = t['entry_price']
            
            # 1. Take Profit Check
            if t['direction'] == 'LONG' and row['high'] >= entry_p * (1 + TP_PCT):
                _close_tranche(t, entry_p * (1 + TP_PCT), 'TAKE_PROFIT', bar, row.name, fold_idx)
                continue
            elif t['direction'] == 'SHORT' and row['low'] <= entry_p * (1 - TP_PCT):
                _close_tranche(t, entry_p * (1 - TP_PCT), 'TAKE_PROFIT', bar, row.name, fold_idx)
                continue
                
            # 2. Time Exit / Let Winners Run Check
            in_profit = (row['close'] > entry_p) if t['direction'] == 'LONG' else (row['close'] < entry_p)
            
            if t['bars_held'] >= BASE_HOLD_BARS:
                if t['bars_held'] >= MAX_HOLD_BARS:
                    _close_tranche(t, row['close'], 'TIME_EXIT_MAX', bar, row.name, fold_idx)
                elif in_profit and adx_val >= MIN_ADX:
                    surviving_tranches.append(t)  # Let it run
                else:
                    _close_tranche(t, row['close'], 'TIME_EXIT', bar, row.name, fold_idx)
            else:
                surviving_tranches.append(t)
                
        active_tranches = surviving_tranches

        # Get model's prediction
        p_arr = preds[i]
        cls = int(np.argmax(p_arr))
        conf = max(p_arr[0], p_arr[2])

        # -- Evaluate Entry --
        if cls == 1:
            fold_skips['hold'] = fold_skips.get('hold', 0) + 1
            continue

        if DIR_FILTER == 'long_only' and cls == 0:
            fold_skips['short_blocked'] = fold_skips.get('short_blocked', 0) + 1
            continue
        if DIR_FILTER == 'short_only' and cls == 2:
            fold_skips['long_blocked'] = fold_skips.get('long_blocked', 0) + 1
            continue

        if conf < CONF_THRESHOLD:
            fold_skips['low_conf'] = fold_skips.get('low_conf', 0) + 1
            continue
            
        # -- Technical Chop Filters --
        if np.isnan(adx_val) or adx_val < MIN_ADX:
            fold_skips['low_adx'] = fold_skips.get('low_adx', 0) + 1
            continue
            
        atr_pct = row.get('atr_pctile', 1.0)
        if np.isnan(atr_pct) or atr_pct < MIN_ATR_PCTILE:
            fold_skips['low_vol'] = fold_skips.get('low_vol', 0) + 1
            continue

        total_active_pending = len(active_tranches) + len(pending_tranches)
        if total_active_pending >= MAX_CONCURRENT_TRANCHES:
            fold_skips['max_tranches'] = fold_skips.get('max_tranches', 0) + 1
            continue

        if daily_count >= MAX_TRADES_PER_DAY:
            fold_skips['daily_cap'] = fold_skips.get('daily_cap', 0) + 1
            continue

        next_bar = bar + 1
        if next_bar >= len(df):
            break
            
        # Conviction Sizing
        is_high_conviction = conf > HIGH_CONF_THRESHOLD
        tranche_size = BASE_POSITION_SIZE * 2 if is_high_conviction else BASE_POSITION_SIZE

        pending_tranches.append({
            'direction': 'LONG' if cls == 2 else 'SHORT',
            'entry_bar': next_bar,
            'bars_held': 0,
            'size': tranche_size
        })
        daily_count += 1
        fold_trades += 1

    # Force close at fold end
    last_bar = min(test_end - 1, len(df) - 1)
    last_row = df.iloc[last_bar]
    for t in active_tranches:
        _close_tranche(t, last_row['close'], 'FOLD_END', last_bar, last_row.name, fold_idx)

    for k, v in fold_skips.items():
        skip_reasons[k] = skip_reasons.get(k, 0) + v

    # Only track folds that actually processed bars to avoid printing 'none' empty folds
    processed_bars = min(n_preds, len(df) - pred_offset)
    if processed_bars > 0:
        start_date = df.index[min(pred_offset, len(df)-1)]
        end_date = df.index[min(pred_offset + processed_bars - 1, len(df)-1)]
        fold_stats.append({
            'fold': fold_idx,
            'start': str(start_date)[:10],
            'end': str(end_date)[:10],
            'bars': processed_bars,
            'trades': fold_trades,
            'top_skip': max(fold_skips, key=fold_skips.get) if fold_skips else 'none',
            'top_skip_n': max(fold_skips.values()) if fold_skips else 0,
        })

trades_df = pd.DataFrame(all_trades)
fold_stats_df = pd.DataFrame(fold_stats)

print(f'\\nTotal trades: {len(trades_df)}')
print(f'Cumulative PnL ({LEVERAGE}x): {cumulative_pnl*100:.2f}%')
print(f'Skip reasons: {skip_reasons}')
"""]
        print(f"Replaced cell {i}: BACKTEST ENGINE -> YIELD-MAXIMIZED BACKTEST ENGINE")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
