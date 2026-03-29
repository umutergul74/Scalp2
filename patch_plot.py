import json

with open('notebooks/06_backtest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # We need to find the cell with plotting
        if "axes[1].fill_between" in src:
            # Recreate daily drawdown specifically for the plot
            plot_dd_logic = '''    # 2. Drawdown (leveraged)
    daily_cum = np.cumsum(daily_pnl.values)
    daily_peak = np.maximum.accumulate(daily_cum)
    daily_dd = daily_peak - daily_cum
    axes[1].fill_between(daily_pnl.index, 0, -daily_dd * 100, alpha=0.4, color='red')
    axes[1].set_title(f'Drawdown ({LEVERAGE}x Leveraged) - End of Day')'''
            
            old_plot_logic = '''    # 2. Drawdown (leveraged)
    axes[1].fill_between(daily_pnl.index, 0, -dd * 100, alpha=0.4, color='red')
    axes[1].set_title(f'Drawdown ({LEVERAGE}x Leveraged)')'''
            
            if old_plot_logic in src:
                src = src.replace(old_plot_logic, plot_dd_logic)
            else:
                # Fallback if text differs slightly
                import re
                src = re.sub(r"    # 2\. Drawdown \(leveraged\)\n    axes\[1\]\.fill_between.*?\n.*?set_title.*?\n", plot_dd_logic + "\n", src)

            # Re-write cell source
            lines = [line + '\n' for line in src.split('\n')]
            if lines and lines[-1].endswith('\n\n'):
                lines[-1] = lines[-1][:-1]
            cell['source'] = lines

with open('notebooks/06_backtest.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Plotting glitch patched successfully!")
