Perform a systematic look-ahead bias audit on the Scalp2 codebase.

## Checklist — Read Each File and Verify

### 1. Wavelet Denoising
- **File:** `scalp2/features/builder.py`
- **Check:** Must use `wavelet_denoise` (causal), NOT `wavelet_denoise_fast` (full-series)
- **Verify:** The import line and function call use the rolling-window version

### 2. Feature Scaler
- **File:** `notebooks/04_train_stage1.ipynb` cell-6
- **Check:** `RobustScaler().fit_transform(train_feat)` — fit on TRAIN only
- **Check:** `scaler.transform(val_feat)` — transform (not fit_transform) on val
- **File:** `notebooks/05_train_stage2.ipynb` cell-4
- **Check:** Scaler loaded from artifacts, `.transform()` used on train/val/test

### 3. HMM Regime Detection
- **File:** `scalp2/training/stage2_trainer.py` lines 120-125
- **Check:** `predict_proba()` used for train only
- **Check:** `predict_proba_online()` used for val AND test
- **File:** `scalp2/regime/hmm.py`
- **Check:** `predict_proba_online()` implements forward-only pass (no backward)

### 4. Walk-Forward CV
- **File:** `scalp2/training/walk_forward.py`
- **Check:** Purge bars > 0 between train and val
- **Check:** Embargo bars > 0 between val and test
- **Check:** Test set NEVER overlaps with training

### 5. Feature Engineering
- **File:** `scalp2/features/technical.py`
- **Check:** All indicators (RSI, EMA, MACD, Bollinger, ADX, ATR) are causal by nature
- **File:** `scalp2/features/volatility.py`
- **Check:** Garman-Klass and Parkinson use rolling windows (causal)

### 6. Triple Barrier Labels
- **File:** `scalp2/labeling/triple_barrier.py`
- **Check:** Labels use FUTURE bars (by design — this is the target, not a feature)
- **Check:** `tb_return` for short labels is NEGATIVE (raw price change convention)
- **Verify:** Lines 203, 209 use `combined_returns[i] = -short_ret[i]`

### 7. Backtest Notebook
- **File:** `notebooks/06_backtest.ipynb` cell-4
- **Check:** Predictions come from `wf_predictions.pkl` (walk-forward test splits)
- **Check:** No future data accessed in the backtest loop
- **Check:** ATR percentile uses rolling rank (causal, not global rank)
- **Check:** ADX filter uses the current bar's ADX (no future bars)

## Output Format

For each check, report:
- PASS ✓ or FAIL ✗
- File:line reference
- Brief evidence (what the code actually does)

End with overall verdict: CLEAN or BIAS FOUND (with details).
