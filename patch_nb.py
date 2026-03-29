import json

with open('notebooks/06_backtest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # 1. Add bar_equity_curve array
        if "equity_curve = [0.0]" in src and "bar_equity_curve" not in src:
            src = src.replace("equity_curve = [0.0]", "equity_curve = [0.0]\nbar_equity_curve = [0.0]")
            
        # 2. Add bar-by-bar tracking of M2M equity
        target_str = "    # force-close any open trade at fold boundary"
        if target_str in src and "bar_equity_curve.append" not in src:
            replacement = """        # M2M equity logic
        if active is not None:
            if active.direction == \"LONG\":
                unr = (row['close'] - active.entry_price) / active.entry_price
            else:
                unr = (active.entry_price - row['close']) / active.entry_price
            m_pnl = (active.pnl + unr * active.remaining_size) * position_size * LEVERAGE
            bar_equity_curve.append(cumulative_pnl + m_pnl)
        else:
            bar_equity_curve.append(cumulative_pnl)

    # force-close any open trade at fold boundary"""
            src = src.replace(target_str, replacement)
            
        # 3. Modify MDD calculation
        mdd_target = '''    # Max drawdown (on leveraged equity)
    cum = np.cumsum(daily_pnl.values)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    mdd = dd.max()'''
        
        mdd_replacement = '''    # Max drawdown (Bar-by-Bar Mark-To-Market)
    cum = np.array(bar_equity_curve)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    mdd = dd.max()'''
        
        if mdd_target in src:
            src = src.replace(mdd_target, mdd_replacement)
            
        # Re-write cell source
        lines = []
        for line in src.split('\n'):
            lines.append(line + '\n')
        if lines and lines[-1].endswith('\n\n'):
            lines[-1] = lines[-1][:-1] # remove trailing newline
        cell['source'] = lines

with open('notebooks/06_backtest.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook parched successfully!")
