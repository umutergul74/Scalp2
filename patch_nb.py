"""Patch Notebook 06 to use the shared backtest engine."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def _lines(source: str) -> list[str]:
    text = textwrap.dedent(source).strip("\n") + "\n"
    return text.splitlines(keepends=True)


NOTEBOOK = Path("notebooks/06_backtest.ipynb")
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

nb["cells"][2]["source"] = _lines(
    """
    from google.colab import drive
    drive.mount('/content/drive')

    import os, sys, json

    REPO_DIR = '/content/scalp2_repo'
    REPO_REF = os.environ.get('SCALP2_GIT_REF', 'main')

    if not os.path.exists(os.path.join(REPO_DIR, '.git')):
        !git clone https://github.com/sergul74/Scalp2.git {REPO_DIR}

    !git -C {REPO_DIR} fetch --all --tags
    !git -C {REPO_DIR} checkout {REPO_REF}
    !git -C {REPO_DIR} pull --ff-only || true

    if not os.path.exists(os.path.join(REPO_DIR, 'scalp2', '__init__.py')):
        raise FileNotFoundError('scalp2 package not found after clone!')

    print(f'Using repo ref: {REPO_REF}')
    !git -C {REPO_DIR} rev-parse --short HEAD

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
    """
)

nb["cells"][3]["source"] = _lines(
    """
    # Load labeled dataset and walk-forward predictions
    import pickle
    from pathlib import Path

    df_path = Path(f'{DATA_DIR}/BTC_USDT_labeled.parquet')
    wf_path = Path(f'{DATA_DIR}/wf_predictions.pkl')
    config_path = Path(f'{REPO_DIR}/config.yaml')

    df = pd.read_parquet(df_path)

    with open(wf_path, 'rb') as f:
        wf_predictions = pickle.load(f)

    def _fmt_mtime(path: Path) -> str:
        return pd.Timestamp(path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')

    print(f'Loaded {len(df)} bars, {len(wf_predictions)} walk-forward folds')
    print(
        f'Artifact timestamps | data: {_fmt_mtime(df_path)} | '
        f'wf_predictions: {_fmt_mtime(wf_path)} | config: {_fmt_mtime(config_path)}'
    )
    if wf_path.stat().st_mtime < df_path.stat().st_mtime:
        print('WARNING: wf_predictions.pkl is older than BTC_USDT_labeled.parquet. Re-run Notebook 05 after refreshing data.')
    if wf_path.stat().st_mtime < config_path.stat().st_mtime:
        print('WARNING: wf_predictions.pkl is older than config.yaml. Re-run Notebook 05 for a fully up-to-date backtest.')
    """
)

nb["cells"][4]["source"] = _lines(
    """
    # Walk-forward backtest aligned with the current shared execution logic
    from scalp2.analysis.backtest_engine import DEFAULT_INITIAL_BALANCE, simulate_walk_forward_backtest

    logging.getLogger('scalp2.execution.trade_manager').setLevel(logging.WARNING)

    def _daily_balance_curve(trades_df, initial_balance):
        if len(trades_df) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        timestamps = pd.to_datetime(trades_df['timestamp'])
        balances = pd.Series(trades_df['balance_after'].values, index=timestamps)
        daily_balance = balances.groupby(balances.index.floor('D')).last().sort_index()
        full_range = pd.date_range(daily_balance.index.min(), daily_balance.index.max(), freq='D')
        daily_balance = daily_balance.reindex(full_range).ffill().fillna(initial_balance)

        prev_balance = daily_balance.shift(1).fillna(initial_balance)
        daily_returns = daily_balance / prev_balance - 1.0
        return daily_balance, daily_returns

    exec_cfg = config.execution
    order_cfg = exec_cfg.order_execution
    trade_mgmt_cfg = exec_cfg.trade_management
    label_cfg = config.labeling

    LEVERAGE = exec_cfg.position_sizing.leverage
    bt_initial_balance = float(globals().get('BACKTEST_INITIAL_BALANCE', DEFAULT_INITIAL_BALANCE))

    FLAT_RT_COST = 2 * (order_cfg.slippage_bps + order_cfg.taker_fee_bps) / 10_000.0
    USE_VAR_SLIPPAGE = bool(getattr(exec_cfg, 'slippage_model', None) and exec_cfg.slippage_model.enabled)
    USE_FUNDING = bool(getattr(exec_cfg, 'funding_rate', None) and exec_cfg.funding_rate.enabled)
    USE_MARKET_IMPACT = bool(getattr(exec_cfg, 'market_impact', None) and exec_cfg.market_impact.enabled)

    print('Notebook backtest now calls the shared strategy/backtest engine from scalp2.')
    print(f'Initial balance: ${bt_initial_balance:,.2f} | Leverage: {LEVERAGE}x')
    print(
        'Config snapshot: '
        f'conf>={exec_cfg.confidence_threshold:.2f}, '
        f'ADX>={exec_cfg.min_adx:.1f}, '
        f'ATR pct>={exec_cfg.min_atr_percentile:.2f}, '
        f'TP1={trade_mgmt_cfg.partial_tp_1_atr:.2f} ATR, '
        f'TP2={trade_mgmt_cfg.full_tp_atr:.2f} ATR, '
        f'SL={label_cfg.sl_multiplier:.2f} ATR'
    )
    print(
        'Cost model: '
        f'taker fee={order_cfg.taker_fee_bps:.1f}bps, '
        f'base slippage={order_cfg.slippage_bps:.1f}bps, '
        f'variable_slippage={USE_VAR_SLIPPAGE}, funding={USE_FUNDING}, impact={USE_MARKET_IMPACT}'
    )

    bt_result = simulate_walk_forward_backtest(
        df=df,
        wf_predictions=wf_predictions,
        config=config,
        initial_balance=bt_initial_balance,
    )

    trades_df = bt_result['trades_df'].copy()
    equity_curve = bt_result['equity_curve']
    bar_equity_curve = bt_result['bar_equity_curve']
    cumulative_pnl = bt_result['cumulative_pnl']
    skip_reasons = bt_result['skip_reasons']
    liquidated = bt_result['liquidated']
    daily_balance, daily_returns = _daily_balance_curve(trades_df, bt_initial_balance)

    print(f'\\nTotal trades: {len(trades_df)}')
    print(f'Cumulative net PnL ({LEVERAGE}x equity impact): {cumulative_pnl*100:.2f}%')
    print(f'Skip reasons: {skip_reasons}')
    if len(trades_df) > 0:
        print(f'Final balance: ${trades_df["balance_after"].iloc[-1]:,.2f}')
        print(f'Avg position size: {trades_df["position_size"].mean()*100:.2f}%')
        print(f'Avg margin util: {trades_df["margin_utilization"].mean()*100:.1f}%')
        print(f'Avg tx cost/trade: {trades_df["cost"].mean()*10000:.1f} bps')
    if liquidated:
        print('*** ACCOUNT LIQUIDATED ***')
    """
)

nb["cells"][5]["source"] = _lines(
    """
    # Results summary and visualization
    if len(trades_df) == 0:
        print("No trades generated!")
    else:
        net = trades_df['net_pnl'].values
        gross = trades_df['gross_pnl'].values
        unit = trades_df['unit_pnl'].values
        n = len(trades_df)

        wins = net[net > 0]
        losses = net[net < 0]
        wr = len(wins) / n
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        pf = abs(wins.sum() / losses.sum()) if len(losses) else float('inf')

        daily_sharpe = daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(365)
        down = daily_returns[daily_returns < 0]
        daily_sortino = daily_returns.mean() / (down.std() + 1e-10) * np.sqrt(365) if len(down) else 0.0

        cum = np.array(bar_equity_curve)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        mdd = dd.max()
        calmar = (daily_returns.mean() * 365) / mdd if mdd > 1e-10 else 0.0

        status_counts = trades_df['status'].value_counts()
        current_balance = float(trades_df['balance_after'].iloc[-1])

        unit_daily = trades_df.assign(date=pd.to_datetime(trades_df['timestamp']).dt.floor('D')).groupby('date')['unit_pnl'].sum()
        unit_full_range = pd.date_range(unit_daily.index.min(), unit_daily.index.max(), freq='D')
        unit_daily = unit_daily.reindex(unit_full_range, fill_value=0.0)
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
        print(f'  Initial balance    : ${bt_initial_balance:,.2f}')
        print(f'  Final balance      : ${current_balance:,.2f}')
        print(f'  Gross PnL          : {gross.sum()*100:.2f}%')
        print(f'  Net PnL            : {cumulative_pnl*100:.2f}%')
        print(f'  Cost impact        : {(gross.sum()-net.sum())*100:.2f}%')
        print(f'  TX cost/trade      : {FLAT_RT_COST*10000:.0f} bps (flat reference)')
        print()
        print('  Close reasons:')
        for status, count in status_counts.items():
            print(f'    {status:20s}: {count:5d} ({count/n*100:.1f}%)')

        print()
        print('-' * 60)
        print('  LEVERAGE & COST MODEL ANALYSIS')
        print('-' * 60)
        print(f'  Leverage            : {LEVERAGE}x')
        print(f'  Net PnL (per-unit)  : {unit.sum()*100:.2f}%')
        print(f'  Net PnL (equity)    : {cumulative_pnl*100:.2f}%')
        print(f'  MDD (per-unit)      : {unit_mdd*100:.4f}%')
        print(f'  MDD (equity)        : {mdd*100:.4f}%')
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
            print('  *** ACCOUNT LIQUIDATED ***')
        print('=' * 60)

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        cum_pct = (daily_balance / bt_initial_balance - 1.0) * 100
        axes[0].plot(daily_balance.index, cum_pct, linewidth=1, color='#2196F3')
        axes[0].fill_between(daily_balance.index, 0, cum_pct, alpha=0.1, color='#2196F3')
        axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
        axes[0].set_title(f'Equity Curve (Actual Balance Path - {LEVERAGE}x)')
        axes[0].set_ylabel('Cumulative PnL (%)')
        axes[0].grid(True, alpha=0.3)

        daily_peak_balance = daily_balance.cummax()
        daily_dd = 1.0 - daily_balance / daily_peak_balance.replace(0, np.nan)
        daily_dd = daily_dd.fillna(0.0)
        axes[1].fill_between(daily_balance.index, 0, -daily_dd * 100, alpha=0.4, color='red')
        axes[1].set_title('Drawdown (Daily Closing Equity)')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(net * 100, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white')
        axes[2].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[2].set_title('Trade PnL Distribution (Equity Impact)')
        axes[2].set_xlabel('PnL per trade (%)')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        monthly_balance = daily_balance.resample('M').last()
        monthly_prev = monthly_balance.shift(1).fillna(bt_initial_balance)
        monthly_returns = (monthly_balance / monthly_prev - 1.0) * 100
        monthly_trades = trades_df.assign(month=pd.to_datetime(trades_df['timestamp']).dt.to_period('M')).groupby('month').size()
        monthly = pd.DataFrame({
            'return_pct': monthly_returns.values,
            'trades': monthly_trades.reindex(monthly_returns.index.to_period('M'), fill_value=0).values,
        }, index=monthly_returns.index.to_period('M'))
        print('\\nMonthly Returns:')
        print(monthly.to_string())
    """
)

nb["cells"][8]["source"] = _lines(
    """
    # Forward Test — Son Gorulmemis Veriler
    **Son fold'un modeli** ile, historical dataset'in bittigi bar sonrasindaki verilerde test.
    Cutoff tarihi sabit degil; notebook bunu `BTC_USDT_labeled.parquet` icindeki son bardan turetir.

    - **Guclu sonuc** (WR > 60%, PF > 2.0) -> Live deployment'a uygun
    - **Orta sonuc** (WR > 50%, PF > 1.2) -> Paper trading ile devam
    - **Zayif sonuc** (WR < 50%) -> Retrain gerekli
    """
)

nb["cells"][9]["source"] = _lines(
    """
    # ============================================================
    # Forward Test Step 1: Download recent data & build features
    # ============================================================
    import torch
    from scalp2.data.preprocessing import clean_ohlcv, resample_ohlcv
    from scalp2.features.builder import build_features, drop_warmup_nans, get_feature_columns
    from scalp2.data.mtf_builder import build_mtf_dataset

    import ccxt, os, time as _time

    EXCHANGES_TO_TRY = [
        ("binanceusdm", "BTC/USDT"),
        ("binance",     "BTC/USDT"),
        ("bybit",       "BTC/USDT"),
        ("okx",         "BTC/USDT"),
    ]
    from datetime import datetime, timezone

    def _timeframe_to_timedelta(tf: str) -> pd.Timedelta:
        tf = tf.strip().lower()
        value = int(tf[:-1])
        unit = tf[-1]
        unit_map = {'m': 'min', 'h': 'h', 'd': 'd'}
        if unit not in unit_map:
            raise ValueError(f'Unsupported timeframe: {tf}')
        return pd.to_timedelta(value, unit=unit_map[unit])

    def _as_utc_naive(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts

    primary_tf = config.data.timeframes.primary.strip().lower()
    mtf_list = [tf.strip().lower() for tf in config.data.timeframes.mtf]
    if primary_tf != '15m':
        raise RuntimeError(
            f"06_backtest.ipynb currently supports primary timeframe '15m'; config has '{primary_tf}'. "
            "Update the shared MTF pipeline before changing the notebook timeframe."
        )
    if mtf_list != ['1h', '4h']:
        raise RuntimeError(
            f"06_backtest.ipynb currently expects MTF timeframes ['1h', '4h']; config has {mtf_list}. "
            "Update build_mtf_dataset/live pipeline/notebooks together to avoid drift."
        )

    history_end_ts = _as_utc_naive(df.index.max())
    max_tf = max(
        [_timeframe_to_timedelta(primary_tf)]
        + [_timeframe_to_timedelta(tf) for tf in mtf_list]
    )
    forward_warmup_days = int(np.ceil((config.features.wavelet.window * max_tf) / pd.Timedelta(days=1))) + 7
    FWD_START_TS = history_end_ts - pd.Timedelta(days=forward_warmup_days)
    FWD_START = FWD_START_TS.strftime("%Y-%m-%d")
    FWD_END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    FWD_TEST_CUTOFF_TS = history_end_ts
    FWD_TEST_CUTOFF = FWD_TEST_CUTOFF_TS.strftime("%Y-%m-%d %H:%M")

    print(f'Historical dataset ends at: {FWD_TEST_CUTOFF_TS}')
    print(f'Forward download warmup start: {FWD_START} ({forward_warmup_days} days)')

    fwd_cache = f'/content/drive/MyDrive/scalp2/data/processed/btc_fwd_{primary_tf}.parquet'

    if os.path.exists(fwd_cache):
        cached_df = pd.read_parquet(fwd_cache)
        if 'timestamp' in cached_df.columns:
            last_ts = pd.to_datetime(cached_df['timestamp']).max()
        else:
            last_ts = cached_df.index.max()
        last_ts = _as_utc_naive(last_ts)
        if (_as_utc_naive(pd.Timestamp.utcnow()) - last_ts).days > 1:
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
                    candles = exchange.fetch_ohlcv(symbol, primary_tf, since=since, limit=1000)
                    if not candles:
                        break
                    all_candles.extend(candles)
                    since = candles[-1][0] + 1
                    if len(all_candles) % 10000 < 1000:
                        print(f"  {len(all_candles)} candles...")
                    _time.sleep(exchange.rateLimit / 1000.0)

                if len(all_candles) > 500:
                    df_fwd_raw = pd.DataFrame(
                        all_candles,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    )
                    df_fwd_raw['timestamp'] = pd.to_datetime(df_fwd_raw['timestamp'], unit='ms')
                    df_fwd_raw = df_fwd_raw.drop_duplicates('timestamp').sort_values('timestamp')
                    df_fwd_raw = df_fwd_raw.set_index('timestamp')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df_fwd_raw[col] = df_fwd_raw[col].astype(np.float32)
                    df_fwd_raw.to_parquet(fwd_cache)
                    print(f"SUCCESS: {exch_id} - {len(df_fwd_raw)} bars downloaded & cached")
                    break
            except Exception as e:
                print(f"  {exch_id} failed: {e}")
                continue

        if df_fwd_raw is None:
            raise RuntimeError("All exchanges failed for forward data.")

    print(f"Forward data: {len(df_fwd_raw)} bars, {df_fwd_raw.index[0]} to {df_fwd_raw.index[-1]}")

    df_fwd_15m = clean_ohlcv(df_fwd_raw, primary_tf)
    df_fwd_1h = resample_ohlcv(df_fwd_15m, mtf_list[0])
    df_fwd_4h = resample_ohlcv(df_fwd_15m, mtf_list[1])

    print(f"Clean {primary_tf}: {len(df_fwd_15m)}, {mtf_list[0]}: {len(df_fwd_1h)}, {mtf_list[1]}: {len(df_fwd_4h)}")

    print("Building features for forward test...")
    df_fwd_15m_feat = build_features(df_fwd_15m, config.features)
    df_fwd_1h_feat = build_features(df_fwd_1h, config.features)
    df_fwd_4h_feat = build_features(df_fwd_4h, config.features)

    df_fwd_full = build_mtf_dataset(df_fwd_15m_feat, df_fwd_1h_feat, df_fwd_4h_feat)
    df_fwd_full = drop_warmup_nans(df_fwd_full)

    fwd_feature_cols = get_feature_columns(df_fwd_full)
    print(f"Forward feature matrix: {len(df_fwd_full)} rows x {len(fwd_feature_cols)} features")
    print(f"Test period: {FWD_TEST_CUTOFF_TS} -> {df_fwd_full.index[-1]}")
    """
)

nb["cells"][10]["source"] = _lines(
    """
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
    seq_len = int(config.model.seq_len)

    last_fold_idx = len(wf_predictions) - 1
    print(f"Using Fold {last_fold_idx} (last fold) for forward test...")

    fold_last = load_fold_artifacts(CHECKPOINT_DIR, fold_idx=last_fold_idx, device=device)
    xgb_path = Path(CHECKPOINT_DIR) / f'xgb_fold_{last_fold_idx:03d}.json'
    fold_dir = Path(CHECKPOINT_DIR) / f'fold_{last_fold_idx:03d}'

    scaler_last = fold_last['scaler']
    top_indices_last = fold_last['top_feature_indices']
    feature_names_last = fold_last['feature_names']
    regime_detector_last = fold_last.get('regime_detector', None)
    top_k_last = int(config.model.handcrafted_top_k)

    if len(top_indices_last) == 0:
        print(f"  WARNING: top_feature_indices empty, using first {top_k_last} features")
        top_indices_last = np.arange(min(top_k_last, len(feature_names_last)), dtype=np.intp)

    print(f"  Scaler: {type(scaler_last).__name__}")
    print(f"  Top features: {len(top_indices_last)} -> indices: {top_indices_last[:10]}...")
    print(f"  Feature names count: {len(feature_names_last)}")
    print(f"  Regime detector: {'loaded' if regime_detector_last else 'not found'}")
    if xgb_path.exists():
        print(f"  XGBoost checkpoint mtime: {pd.Timestamp(xgb_path.stat().st_mtime, unit='s')}")
        if xgb_path.stat().st_mtime < Path(f'{DATA_DIR}/wf_predictions.pkl').stat().st_mtime:
            print('  WARNING: XGBoost checkpoint is older than wf_predictions.pkl.')
    if fold_dir.exists():
        print(f"  Fold artifact dir mtime: {pd.Timestamp(fold_dir.stat().st_mtime, unit='s')}")

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

    print("\\n--- DIAGNOSTIC 1: Feature Scaling ---")
    print(f"  Raw features: min={fwd_raw.min():.4f}, max={fwd_raw.max():.4f}, mean={fwd_raw.mean():.4f}, std={fwd_raw.std():.4f}")
    print(f"  Scaled features: min={fwd_scaled.min():.4f}, max={fwd_scaled.max():.4f}, mean={fwd_scaled.mean():.4f}, std={fwd_scaled.std():.4f}")
    zero_cols = np.where(np.abs(fwd_scaled).max(axis=0) < 1e-6)[0]
    print(f"  All-zero scaled columns: {len(zero_cols)} / {fwd_scaled.shape[1]}")
    if len(zero_cols) > 0:
        zero_names = [fwd_feature_cols_model[i] for i in zero_cols[:10]]
        print(f"    Examples: {zero_names}")
    extreme_cols = np.where(np.abs(fwd_scaled).max(axis=0) > 100)[0]
    print(f"  Extreme scaled columns (|x|>100): {len(extreme_cols)} / {fwd_scaled.shape[1]}")

    if len(fwd_scaled) <= seq_len:
        raise RuntimeError("Not enough forward rows after warmup to build inference windows.")

    encoder_last = HybridEncoder(
        n_features=fwd_scaled.shape[1],
        config=config.model,
    ).to(device)
    encoder_last.load_state_dict(fold_last['model_state'])
    encoder_last.eval()
    print(f"\\n  HybridEncoder loaded: {sum(p.numel() for p in encoder_last.parameters())} params")

    print("Extracting latent vectors...")
    dummy_l = np.zeros(len(fwd_scaled), dtype=np.int64)
    dummy_r = np.zeros(len(fwd_scaled), dtype=np.float32)
    ds_fwd = ScalpDataset(fwd_scaled, dummy_l, dummy_r, seq_len)
    loader_fwd = DataLoader(ds_fwd, batch_size=512, shuffle=False, pin_memory=(device.type == 'cuda'))

    fwd_latents = []
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    amp_enabled = device.type == 'cuda'
    with torch.no_grad():
        for bx, _, _ in loader_fwd:
            bx = bx.to(device)
            with autocast(amp_device, enabled=amp_enabled):
                _, lat = encoder_last(bx)
            fwd_latents.append(lat.cpu().numpy())
    fwd_latents = np.concatenate(fwd_latents, axis=0)

    print("\\n--- DIAGNOSTIC 2: Latent Vectors ---")
    print(f"  Shape: {fwd_latents.shape}")
    print(f"  Mean: {fwd_latents.mean():.6f}, Std: {fwd_latents.std():.6f}")
    print(f"  Min: {fwd_latents.min():.6f}, Max: {fwd_latents.max():.6f}")
    lat_var = fwd_latents.var(axis=0)
    dead_dims = (lat_var < 1e-8).sum()
    print(f"  Dead dimensions (var<1e-8): {dead_dims} / {fwd_latents.shape[1]}")
    print(f"  Per-dim variance: min={lat_var.min():.8f}, max={lat_var.max():.8f}, median={np.median(lat_var):.8f}")
    pairwise_diff = np.abs(fwd_latents[::100] - fwd_latents[0:1]).mean()
    print(f"  Avg L1 diff between samples (sampled): {pairwise_diff:.6f}")

    fwd_df_aligned = df_fwd_full.iloc[seq_len:]
    if regime_detector_last is not None:
        fwd_regime = regime_detector_last.predict_proba_online(fwd_df_aligned)
    else:
        fwd_regime = np.full((len(fwd_df_aligned), 3), 1/3, dtype=np.float32)

    print("\\n--- DIAGNOSTIC 3: Regime Probs ---")
    print(f"  Shape: {fwd_regime.shape}")
    print(f"  Mean per class: Bull={fwd_regime[:,0].mean():.4f}, Bear={fwd_regime[:,1].mean():.4f}, Choppy={fwd_regime[:,2].mean():.4f}")
    print(f"  Dominant regime: Bull={np.mean(fwd_regime.argmax(1)==0)*100:.1f}%, Bear={np.mean(fwd_regime.argmax(1)==1)*100:.1f}%, Choppy={np.mean(fwd_regime.argmax(1)==2)*100:.1f}%")

    fwd_hc = fwd_scaled[seq_len:][:, top_indices_last]

    print("\\n--- DIAGNOSTIC 4: Handcrafted Features ---")
    print(f"  Shape: {fwd_hc.shape}")
    print(f"  Mean: {fwd_hc.mean():.4f}, Std: {fwd_hc.std():.4f}")
    hc_var = fwd_hc.var(axis=0)
    hc_dead = (hc_var < 1e-8).sum()
    print(f"  Dead features (var<1e-8): {hc_dead} / {fwd_hc.shape[1]}")

    fwd_min = min(len(fwd_latents), len(fwd_hc), len(fwd_regime))
    if fwd_min == 0:
        raise RuntimeError("Forward inference arrays are empty after alignment.")
    fwd_latents = fwd_latents[:fwd_min]
    fwd_hc = fwd_hc[:fwd_min]
    fwd_regime = fwd_regime[:fwd_min]

    fwd_meta = XGBoostMetaLearner.build_meta_features(fwd_latents, fwd_hc, fwd_regime)

    print("\\n--- DIAGNOSTIC 5: Meta-Features ---")
    latent_dim = int(fwd_latents.shape[1])
    n_hc = len(top_indices_last)
    hc_start = latent_dim
    hc_end = latent_dim + n_hc
    print(f"  Shape: {fwd_meta.shape} (expect {latent_dim} + {n_hc} + 3 = {latent_dim+n_hc+3})")
    print(f"  Latent block [0:{latent_dim}]: mean={fwd_meta[:,:latent_dim].mean():.4f}, std={fwd_meta[:,:latent_dim].std():.4f}")
    print(f"  HC block [{hc_start}:{hc_end}]: mean={fwd_meta[:,hc_start:hc_end].mean():.4f}, std={fwd_meta[:,hc_start:hc_end].std():.4f}")
    print(f"  Regime block [-3:]: mean={fwd_meta[:,-3:].mean():.4f}, std={fwd_meta[:,-3:].std():.4f}")

    xgb_last = XGBoostMetaLearner(config.model.xgboost)
    xgb_last.load(f'{CHECKPOINT_DIR}/xgb_fold_{last_fold_idx:03d}.json')

    fwd_probs = xgb_last.predict_proba(fwd_meta)

    print("\\n--- DIAGNOSTIC 6: Predictions ---")
    print(f"  Shape: {fwd_probs.shape}")
    print(f"  Class dist: Short={np.mean(fwd_probs.argmax(1)==0)*100:.1f}%, Hold={np.mean(fwd_probs.argmax(1)==1)*100:.1f}%, Long={np.mean(fwd_probs.argmax(1)==2)*100:.1f}%")
    print(f"  Avg max confidence: {fwd_probs.max(1).mean():.4f}")
    print(f"  Avg probs: P(Short)={fwd_probs[:,0].mean():.4f}, P(Hold)={fwd_probs[:,1].mean():.4f}, P(Long)={fwd_probs[:,2].mean():.4f}")
    print(f"  Confidence range: {fwd_probs.max(1).min():.4f} - {fwd_probs.max(1).max():.4f}")

    importance = xgb_last.model.feature_importances_
    lat_imp = importance[:latent_dim].sum()
    hc_imp = importance[latent_dim:latent_dim+n_hc].sum()
    reg_imp = importance[-3:].sum()
    print(f"\\n  XGBoost importance: Latent={lat_imp:.3f}, HC={hc_imp:.3f}, Regime={reg_imp:.3f}")
    top5_idx = np.argsort(importance)[-5:]
    print(f"  Top 5 features by importance (index): {top5_idx}")

    print("\\n=== FORWARD DIAGNOSTICS COMPLETE ===")
    """
)

nb["cells"][11]["source"] = _lines(
    """
    # ============================================================
    # Forward Test Step 3: Backtest only on bars AFTER training cutoff
    # ============================================================
    from scalp2.analysis.backtest_engine import DEFAULT_INITIAL_BALANCE, simulate_forward_backtest

    FWD_CONFIDENCE_THRESHOLD = float(
        globals().get('FWD_CONFIDENCE_THRESHOLD_OVERRIDE', config.execution.confidence_threshold)
    )
    fwd_initial_balance = float(globals().get('FORWARD_INITIAL_BALANCE', DEFAULT_INITIAL_BALANCE))

    df_fwd_bt = df_fwd_full.iloc[seq_len:seq_len + fwd_min].copy()

    cutoff_ts = pd.Timestamp(FWD_TEST_CUTOFF_TS)
    if df_fwd_bt.index.tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(df_fwd_bt.index.tz)

    fwd_test_mask = df_fwd_bt.index > cutoff_ts
    fwd_test_start_idx = int(np.argmax(fwd_test_mask)) if fwd_test_mask.any() else len(df_fwd_bt)
    print(f"Warmup bars (before cutoff): {fwd_test_start_idx}")
    print(f"Forward test bars: {fwd_test_mask.sum()}")
    if fwd_test_start_idx < len(df_fwd_bt):
        print(f"Forward test period: {df_fwd_bt.index[fwd_test_start_idx]} -> {df_fwd_bt.index[-1]}")
    else:
        print("Forward test period: no bars after cutoff")
    print(
        f"Forward confidence threshold: {FWD_CONFIDENCE_THRESHOLD:.2f} "
        f"(config default: {config.execution.confidence_threshold:.2f})"
    )

    fwd_result = simulate_forward_backtest(
        df=df_fwd_bt,
        probs=fwd_probs[:len(df_fwd_bt)],
        regime_probs=fwd_regime[:len(df_fwd_bt)],
        config=config,
        signal_start_bar=fwd_test_start_idx,
        confidence_threshold=FWD_CONFIDENCE_THRESHOLD,
        initial_balance=fwd_initial_balance,
    )

    fwd_df = fwd_result['trades_df'].copy()
    fwd_skip = fwd_result['skip_reasons']
    fwd_equity = fwd_result['equity_curve']
    fwd_bar_equity = fwd_result['bar_equity_curve']
    fwd_cum_pnl = fwd_result['cumulative_pnl']
    fwd_daily_balance, fwd_daily_returns = _daily_balance_curve(fwd_df, fwd_initial_balance)
    fwd_daily = fwd_daily_returns.copy()

    if len(fwd_df) == 0:
        fwd_sharpe = 0.0
        fwd_mdd = 0.0
        print(f"\\n*** NO TRADES in forward test period ({FWD_TEST_CUTOFF} -> present) ***")
        print(f"Skip reasons: {fwd_skip}")
    else:
        fwd_net = fwd_df['net_pnl'].values
        fwd_n = len(fwd_df)
        fwd_wins = fwd_net[fwd_net > 0]
        fwd_losses = fwd_net[fwd_net < 0]
        fwd_wr = len(fwd_wins) / fwd_n
        fwd_pf = abs(fwd_wins.sum() / fwd_losses.sum()) if len(fwd_losses) else float('inf')

        fwd_sharpe = fwd_daily_returns.mean() / (fwd_daily_returns.std() + 1e-10) * np.sqrt(365)
        fwd_curve = np.array(fwd_bar_equity)
        fwd_peak = np.maximum.accumulate(fwd_curve)
        fwd_mdd = (fwd_peak - fwd_curve).max()
        n_days = len(fwd_daily_returns)

        print()
        print('=' * 60)
        print(f'       FORWARD TEST - {FWD_TEST_CUTOFF} -> Present (Fold {last_fold_idx})')
        print(f'       (Leverage: {LEVERAGE}x, {n_days} days)')
        print('=' * 60)
        print(f'  Total trades       : {fwd_n}')
        print(f'  Win rate           : {fwd_wr:.4f} ({fwd_wr*100:.1f}%)')
        print(f'  Profit factor      : {fwd_pf:.4f}')
        print(f'  Expectancy/trade   : {fwd_net.mean()*100:.4f}%')
        print(f'  Daily Sharpe       : {fwd_sharpe:.4f}')
        print(f'  Max Drawdown       : {fwd_mdd*100:.4f}%')
        print(f'  Net PnL            : {fwd_cum_pnl*100:.2f}%')
        print(f'  Final balance      : ${fwd_df["balance_after"].iloc[-1]:,.2f}')
        print()
        print(f'  Skip reasons: {fwd_skip}')
        print()

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

        print()
        print('-' * 60)
        print('=' * 60)

        fwd_df['week'] = pd.to_datetime(fwd_df['timestamp']).dt.to_period('W')
        fwd_weekly = fwd_df.groupby('week')['net_pnl'].agg(['sum', 'count'])
        fwd_weekly.columns = ['return_pct', 'trades']
        fwd_weekly['return_pct'] *= 100
        print('\\nForward Test Weekly Returns:')
        print(fwd_weekly.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fwd_cum_pct = (fwd_daily_balance / fwd_initial_balance - 1.0) * 100
        axes[0].plot(fwd_daily_balance.index, fwd_cum_pct, linewidth=1.5, color='#4CAF50')
        axes[0].fill_between(fwd_daily_balance.index, 0, fwd_cum_pct, alpha=0.1, color='#4CAF50')
        axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
        axes[0].set_title(f'Forward Test Equity ({LEVERAGE}x)')
        axes[0].set_ylabel('Cumulative PnL (%)')
        axes[0].grid(True, alpha=0.3)

        fwd_peak_balance = fwd_daily_balance.cummax()
        fwd_dd_daily = 1.0 - fwd_daily_balance / fwd_peak_balance.replace(0, np.nan)
        fwd_dd_daily = fwd_dd_daily.fillna(0.0)
        axes[1].fill_between(fwd_daily_balance.index, 0, -fwd_dd_daily * 100, alpha=0.4, color='red')
        axes[1].set_title('Forward Test Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    """
)

nb["cells"][13]["source"] = _lines(
    """
    # ============================================================
    # Signal Table: current backtest logic audit view
    # ============================================================
    from scalp2.analysis.backtest_engine import build_signal_audit_table

    df_sig = df_fwd_full.iloc[seq_len:seq_len + fwd_min].copy()
    cutoff_ts = pd.Timestamp(FWD_TEST_CUTOFF_TS)
    if df_sig.index.tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(df_sig.index.tz)

    sig_test_mask = df_sig.index > cutoff_ts
    sig_start_idx = int(np.argmax(sig_test_mask)) if sig_test_mask.any() else len(df_sig)

    sig_df = build_signal_audit_table(
        df=df_sig,
        probs=fwd_probs[:len(df_sig)],
        regime_probs=fwd_regime[:len(df_sig)],
        config=config,
        signal_start_bar=sig_start_idx,
        confidence_threshold=FWD_CONFIDENCE_THRESHOLD,
        initial_balance=fwd_initial_balance,
    )

    if len(sig_df) == 0:
        print("Filtreleri gecen sinyal yok!")
    else:
        print(f'Toplam sinyal: {len(sig_df)}')
        print()

        outcomes = sig_df['Sonuc'].value_counts()
        print('Sonuc Dagilimi:')
        for outcome, count in outcomes.items():
            print(f'  {outcome}: {count} ({count/len(sig_df)*100:.0f}%)')
        print()

        zero_kelly = sig_df[sig_df['Pozisyon'].str.startswith('0.00')]
        tradeable_kelly = sig_df[~sig_df['Pozisyon'].str.startswith('0.00')]
        print(f"Kelly'nin trade actigi: {len(tradeable_kelly)} / {len(sig_df)}")
        print(f"Kelly'nin reddettigi: {len(zero_kelly)} / {len(sig_df)}")
        print()

        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_colwidth', 20)
        print(sig_df.to_string(index=False))
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')
    """
)

NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Patched {NOTEBOOK}")
