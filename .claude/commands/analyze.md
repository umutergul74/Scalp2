Analyze backtest results that the user pastes from notebook 06.

## Analysis Framework

### 1. Parse Metrics
Extract: total trades, win rate, profit factor, expectancy, avg win/loss, bars held, Sharpe, Sortino, MDD, Calmar, gross PnL, net PnL, cost impact, close reasons, monthly returns.

### 2. Compare with Baseline
Latest validated baseline (2025-03, all biases cleaned):
- 3046 trades, 70.2% WR, PF 3.38, Sharpe 10.68, MDD 2.72%, Net PnL 457.28%
- Cost: 8 bps RT (maker orders)
- Negative months: Aug 2023 (-0.58%), Sep 2023 (-0.17%)

Report delta for each metric. Flag improvements > 20% or degradations > 10%.

### 3. Sanity Checks
- Sharpe > 15 → likely bias or error
- Win rate > 80% → likely bias
- MDD < 1% → likely too few trades or bias
- 0 negative months → suspicious
- Cost impact should be ~8 bps × trade_count

### 4. Monthly Pattern Analysis
- Identify worst months — do they correlate with known choppy periods?
- Check for suspicious runs of only positive months
- Compute monthly Sharpe consistency

### 5. Trade Distribution
- CLOSED_TP ratio ~55-60% is healthy
- CLOSED_SL ratio ~35-40% is normal
- High CLOSED_TIME (>10%) means holding period too short or TP too wide
- Check trades/month consistency (~58-62 expected with 2/day cap)

### 6. Recommendations
Based on the analysis, suggest concrete next steps:
- If cost impact is high → suggest fee optimization
- If MDD increased → check regime filters
- If WR dropped → check model or feature changes
- If everything looks good → suggest out-of-sample test (ETH/USDT)
