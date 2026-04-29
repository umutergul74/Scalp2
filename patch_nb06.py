"""Add pure-signal diagnostic cell to NB06 — no SL/TP, just hold N bars."""
import json

nb_path = "notebooks/06_backtest.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Insert a new cell BEFORE the backtest engine cell
# Find the backtest engine cell index
target_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    joined = "".join(cell["source"])
    if "BACKTEST ENGINE" in joined:
        target_idx = i
        break

if target_idx is None:
    print("ERROR: could not find backtest engine cell")
    exit(1)

pure_signal_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "#  PURE SIGNAL DIAGNOSTIC — No SL/TP, just hold for N bars\n",
        "#  This measures the RAW predictive power of the model\n",
        "#  independent of TradeManager parameters.\n",
        "# ============================================================\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "seq_len = config.model.seq_len\n",
        "HOLD_BARS = 10  # Match the label horizon\n",
        "CONF_THRESHOLD_DIAG = 0.40  # Lower threshold for diagnostic\n",
        "\n",
        "close = df['close'].values\n",
        "diag_trades = []\n",
        "\n",
        "for fold_data in wf_predictions:\n",
        "    fold_idx = fold_data['fold_idx']\n",
        "    test_start = fold_data['test_start']\n",
        "    preds = fold_data['test_probabilities']\n",
        "    n_preds = len(preds)\n",
        "    pred_offset = test_start + seq_len\n",
        "\n",
        "    for i in range(n_preds):\n",
        "        bar = pred_offset + i\n",
        "        exit_bar = bar + HOLD_BARS\n",
        "        if exit_bar >= len(df):\n",
        "            break\n",
        "\n",
        "        p_arr = preds[i]\n",
        "        cls = int(np.argmax(p_arr))\n",
        "        if cls == 1:  # Hold\n",
        "            continue\n",
        "\n",
        "        conf = max(p_arr[0], p_arr[2])\n",
        "        if conf < CONF_THRESHOLD_DIAG:\n",
        "            continue\n",
        "\n",
        "        entry_price = close[bar]\n",
        "        exit_price = close[exit_bar]\n",
        "        raw_ret = (exit_price - entry_price) / entry_price\n",
        "\n",
        "        if cls == 0:  # Short\n",
        "            raw_ret = -raw_ret\n",
        "\n",
        "        diag_trades.append({\n",
        "            'fold': fold_idx,\n",
        "            'bar': bar,\n",
        "            'direction': 'SHORT' if cls == 0 else 'LONG',\n",
        "            'confidence': conf,\n",
        "            'raw_return': raw_ret,\n",
        "            'timestamp': df.index[bar],\n",
        "        })\n",
        "\n",
        "diag_df = pd.DataFrame(diag_trades)\n",
        "\n",
        "if len(diag_df) == 0:\n",
        "    print('No diagnostic trades!')\n",
        "else:\n",
        "    n = len(diag_df)\n",
        "    mean_ret = diag_df['raw_return'].mean()\n",
        "    median_ret = diag_df['raw_return'].median()\n",
        "    wr = (diag_df['raw_return'] > 0).mean()\n",
        "    wins = diag_df.loc[diag_df['raw_return'] > 0, 'raw_return']\n",
        "    losses = diag_df.loc[diag_df['raw_return'] < 0, 'raw_return']\n",
        "    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 else float('inf')\n",
        "    sharpe = mean_ret / diag_df['raw_return'].std() * np.sqrt(252 * 96 / HOLD_BARS) if diag_df['raw_return'].std() > 0 else 0\n",
        "\n",
        "    # Cost analysis\n",
        "    RT_COST = 2 * (2 + 2) / 10_000  # 8bps round-trip\n",
        "    net_ret = mean_ret - RT_COST\n",
        "\n",
        "    print('=' * 70)\n",
        "    print(f'  PURE SIGNAL TEST — Hold {HOLD_BARS} bars, conf >= {CONF_THRESHOLD_DIAG}')\n",
        "    print(f'  No SL, No TP, No TradeManager — raw model signal only')\n",
        "    print('=' * 70)\n",
        "    print(f'  Trades:           {n:,}')\n",
        "    print(f'  Win Rate:         {wr*100:.1f}%')\n",
        "    print(f'  Profit Factor:    {pf:.4f}')\n",
        "    print(f'  Mean Return:      {mean_ret*100:.4f}% (GROSS, no costs)')\n",
        "    print(f'  Median Return:    {median_ret*100:.4f}%')\n",
        "    print(f'  RT Cost:          {RT_COST*100:.4f}%')\n",
        "    print(f'  Net Return/trade: {net_ret*100:.4f}% (after 8bps RT cost)')\n",
        "    print(f'  Annualized Sharpe:{sharpe:.2f}')\n",
        "    print(f'  Cumulative GROSS: {diag_df[\"raw_return\"].sum()*100:.2f}%')\n",
        "    print(f'  Cumulative NET:   {(diag_df[\"raw_return\"].sum() - RT_COST*n)*100:.2f}%')\n",
        "    print()\n",
        "\n",
        "    # By confidence bucket\n",
        "    diag_df['conf_bucket'] = pd.cut(diag_df['confidence'], bins=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0])\n",
        "    bucket_stats = diag_df.groupby('conf_bucket', observed=True).agg(\n",
        "        n=('raw_return', 'count'),\n",
        "        mean_ret=('raw_return', 'mean'),\n",
        "        win_rate=('raw_return', lambda x: (x > 0).mean()),\n",
        "    )\n",
        "    bucket_stats['mean_ret'] = bucket_stats['mean_ret'] * 100\n",
        "    bucket_stats['win_rate'] = bucket_stats['win_rate'] * 100\n",
        "    print('  Confidence Bucket Analysis:')\n",
        "    print(bucket_stats.to_string())\n",
        "    print()\n",
        "\n",
        "    # By direction\n",
        "    dir_stats = diag_df.groupby('direction').agg(\n",
        "        n=('raw_return', 'count'),\n",
        "        mean_ret=('raw_return', 'mean'),\n",
        "        win_rate=('raw_return', lambda x: (x > 0).mean()),\n",
        "    )\n",
        "    dir_stats['mean_ret'] = dir_stats['mean_ret'] * 100\n",
        "    dir_stats['win_rate'] = dir_stats['win_rate'] * 100\n",
        "    print('  Direction Analysis:')\n",
        "    print(dir_stats.to_string())\n",
        "    print()\n",
        "\n",
        "    # By year\n",
        "    diag_df['year'] = pd.to_datetime(diag_df['timestamp']).dt.year\n",
        "    yr = diag_df.groupby('year').agg(\n",
        "        n=('raw_return', 'count'),\n",
        "        gross_pnl=('raw_return', 'sum'),\n",
        "        win_rate=('raw_return', lambda x: (x > 0).mean()),\n",
        "    )\n",
        "    yr['gross_pnl'] = yr['gross_pnl'] * 100\n",
        "    yr['net_pnl'] = yr['gross_pnl'] - yr['n'] * RT_COST * 100\n",
        "    yr['win_rate'] = yr['win_rate'] * 100\n",
        "    print('  Yearly Breakdown (GROSS vs NET):')\n",
        "    print(yr.to_string())\n",
    ]
}

# Insert BEFORE the backtest engine cell
nb["cells"].insert(target_idx, pure_signal_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done - Pure signal diagnostic cell added to NB06.")
