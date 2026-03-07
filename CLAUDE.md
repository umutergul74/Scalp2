# Scalp2 — BTC/USDT HFT Scalping Framework

## Project Overview

High-frequency scalping system for BTC/USDT perpetual futures on Binance.
15-minute primary timeframe with 1H/4H multi-timeframe features.
Two-stage ML pipeline trained with purged walk-forward cross-validation.

## Architecture

```
Stage 1: HybridEncoder (TCN + GRU) → 128-dim latent vectors
Stage 2: XGBoost meta-learner on [latent + handcrafted + regime features]
Output:  P(short), P(hold), P(long) → trade signal with Kelly sizing
```

### Walk-Forward CV
- 52 folds, 210-day train, 45-day val, 30-day test, 30-day step
- 48-bar purge + 12-bar embargo between train/val/test
- Each fold gets fresh scaler (RobustScaler fit on train only)
- Each fold gets fresh HMM regime detector (fit on train only)

### Labels
- Triple barrier: `tb_label_cls` → 0=Short, 1=Hold, 2=Long
- `tb_return` = raw price change `(exit - entry) / entry` (negative when price drops, even for short labels)
- ATR-scaled TP/SL with max holding period

## Notebook Pipeline (runs on Google Colab)

```
01_data_prep.ipynb         → Clean OHLCV, resample to 1H/4H, save parquet
02_feature_engineering.ipynb → Technical indicators, wavelet, volatility, smart money
03_labeling.ipynb          → Triple barrier labels
04_train_stage1.ipynb      → Train HybridEncoder per fold, save checkpoints
05_train_stage2.ipynb      → Train XGBoost per fold, save wf_predictions.pkl
06_backtest.ipynb          → Walk-forward backtest with trade management
07_live_inference.ipynb    → Real-time inference (WIP)
```

### Dependency Map (what to re-run when code changes)

| Changed file | Re-run from |
|---|---|
| `scalp2/data/preprocessing.py` | NB 01 |
| `scalp2/features/*` | NB 02 |
| `scalp2/labeling/triple_barrier.py` | NB 03 |
| `scalp2/models/*`, `scalp2/training/trainer.py` | NB 04 |
| `scalp2/training/stage2_trainer.py`, `scalp2/regime/hmm.py` | NB 05 |
| `config.yaml` (labeling/model/training) | NB 03 |
| `config.yaml` (execution only) | NB 06 |
| `scalp2/execution/*` | NB 06 (backtest) or NB 07 (live) |

## Conventions

### Imports
All imports use `from scalp2.<module> import ...`. Never use relative imports.

### Colab Setup
Every notebook cell-2 clones from GitHub:
```python
REPO_DIR = '/content/scalp2_repo'
if os.path.exists(os.path.join(REPO_DIR, '.git')):
    !git -C {REPO_DIR} pull --ff-only
else:
    !git clone https://github.com/sergul74/Scalp2.git {REPO_DIR}
```
Code: GitHub → Colab local filesystem. Data: Google Drive.

### Config
`config.yaml` at repo root → loaded by `scalp2/config.py` into nested dataclasses.
Add new config: define dataclass in `config.py`, add YAML block in `config.yaml`.

## Critical Decisions (DO NOT REVERT)

### Wavelet Denoising
- USE `wavelet_denoise` (causal, 256-bar rolling window) in `features/builder.py`
- NEVER use `wavelet_denoise_fast` — it processes the full series and causes look-ahead bias

### HMM Regime Detection
- `predict_proba()` = forward-backward (uses future data) → OK for training data only
- `predict_proba_online()` = forward-only → MUST use for val/test data
- `stage2_trainer.py` lines 123-125 enforce this

### Triple Barrier Returns
- `tb_return` stores raw price change: `(exit_price - entry_price) / entry_price`
- For short labels, `tb_return` is NEGATIVE (price dropped)
- Fixed in `labeling/triple_barrier.py` lines 203, 209: `combined_returns[i] = -short_ret[i]`

### Cost Model
- Maker orders: 2 bps fee/side + 2 bps slippage/side = 8 bps round-trip
- Config: `config.execution.order_execution` (OrderExecutionConfig dataclass)

### Regime Filters
- `choppy_threshold: 0.4` (config.yaml)
- ADX >= 20 required (`config.execution.min_adx`)
- ATR percentile >= 0.15 required (`config.execution.min_atr_percentile`)

## Key Files

| File | Purpose |
|---|---|
| `config.yaml` | All hyperparameters |
| `scalp2/config.py` | Typed dataclass config loader |
| `scalp2/features/builder.py` | Feature engineering pipeline |
| `scalp2/features/wavelet.py` | Causal + non-causal wavelet functions |
| `scalp2/labeling/triple_barrier.py` | Triple barrier label generator |
| `scalp2/models/hybrid.py` | HybridEncoder (TCN + GRU) |
| `scalp2/models/meta_learner.py` | XGBoost wrapper |
| `scalp2/training/trainer.py` | Stage 1 training loop |
| `scalp2/training/stage2_trainer.py` | Stage 2 training loop |
| `scalp2/training/walk_forward.py` | PurgedWalkForwardCV |
| `scalp2/regime/hmm.py` | HMM regime detector (forward-only + forward-backward) |
| `scalp2/execution/signal_generator.py` | 10-step signal pipeline |
| `scalp2/execution/trade_manager.py` | Partial TP, breakeven, trailing stop |

## Latest Backtest Results (2025-03)

- 3046 trades, 70.2% win rate, profit factor 3.38
- Daily Sharpe 10.68, Max Drawdown 2.72%
- Net PnL 457.28% (8 bps RT cost), 52 months with only 2 negative
- All known look-ahead biases verified clean
