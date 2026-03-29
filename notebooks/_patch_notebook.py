"""Patch 06_backtest.ipynb to integrate Enhancement 1 (SL protection) and Enhancement 5 (risk limits)."""
import json

NB_PATH = r"c:\Users\Umut\Documents\PlatformIO\Projects\Scalp2\notebooks\06_backtest.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def patch_cell_source(cells, cell_id, replacements):
    """Find cell by id and apply text replacements on its joined source."""
    for cell in cells:
        if cell.get("id") == cell_id:
            src = "".join(cell["source"])
            for old, new in replacements:
                if old not in src:
                    print(f"WARNING: pattern not found in {cell_id}:\n  {old[:80]}...")
                    continue
                src = src.replace(old, new, 1)
            # Split back into lines (preserve notebook format)
            lines = src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            return True
    print(f"ERROR: cell {cell_id} not found")
    return False

# ── Cell 002: Add RiskManager import ──
patch_cell_source(nb["cells"], "cell-002", [
    (
        "from scalp2.execution.trade_manager import TradeManager, TradeState, TradeStatus\n"
        "from scalp2.utils.metrics import",
        "from scalp2.execution.trade_manager import TradeManager, TradeState, TradeStatus\n"
        "from scalp2.execution.risk_manager import RiskManager\n"
        "from scalp2.utils.metrics import",
    ),
])

# ── Cell 004: Add RiskManager init, skip reasons, bar advance, record_trade_result, risk checks, size modifier ──
patch_cell_source(nb["cells"], "cell-004", [
    # 1. Suppress risk_manager logs + init RiskManager
    (
        "# Suppress per-trade log spam from TradeManager\n"
        "logging.getLogger('scalp2.execution.trade_manager').setLevel(logging.WARNING)\n",
        "# Suppress per-trade log spam from TradeManager & RiskManager\n"
        "logging.getLogger('scalp2.execution.trade_manager').setLevel(logging.WARNING)\n"
        "logging.getLogger('scalp2.execution.risk_manager').setLevel(logging.WARNING)\n",
    ),
    # 2. Init RiskManager after TradeManager
    (
        "trade_mgr = TradeManager(trade_mgmt_cfg, label_cfg.max_holding_bars)\n",
        "trade_mgr = TradeManager(trade_mgmt_cfg, label_cfg.max_holding_bars)\n"
        "risk_mgr = RiskManager(config=exec_cfg)\n",
    ),
    # 3. Add new skip reasons
    (
        "                'choppy_override': 0, 'time_blocked': 0, 'no_next_bar': 0}",
        "                'choppy_override': 0, 'time_blocked': 0, 'no_next_bar': 0,\n"
        "                'cooldown': 0, 'price_too_close': 0, 'consecutive_sl_cap': 0,\n"
        "                'daily_loss_limit': 0, 'weekly_loss_limit': 0, 'risk_halt': 0}",
    ),
    # 4. Advance bar counter each iteration
    (
        "        # ---- manage active trade ----\n",
        "        # Advance trade manager bar counter (Enhancement 1)\n"
        "        trade_mgr.advance_bar()\n"
        "\n"
        "        # ---- manage active trade ----\n",
    ),
    # 5. After trade closes, record result for protection + risk tracking
    (
        "            if active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):\n"
        "                _close_trade(active, position_size, entry_bar, bar, row, fold_idx)\n",
        "            if active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):\n"
        "                # Record result for SL protection (Enhancement 1)\n"
        "                trade_mgr.record_trade_result(\n"
        "                    status=active.status,\n"
        "                    exit_price=row['close'],\n"
        "                    direction=active.direction,\n"
        "                    atr=active.atr_at_entry,\n"
        "                )\n"
        "                # Record in risk manager (Enhancement 5)\n"
        "                _trade_pnl_pct = active.pnl * position_size * LEVERAGE * 100\n"
        "                risk_mgr.record_trade(\n"
        "                    timestamp=row.name if hasattr(row.name, 'date') else pd.Timestamp(row.name),\n"
        "                    pnl_pct=_trade_pnl_pct,\n"
        "                )\n"
        "                _close_trade(active, position_size, entry_bar, bar, row, fold_idx)\n",
    ),
    # 6. After direction is set, add SL protection + risk checks before pending_signal
    (
        "        direction = \"LONG\" if cls == 2 else \"SHORT\"\n"
        "\n"
        "        # Set pending signal",
        "        direction = \"LONG\" if cls == 2 else \"SHORT\"\n"
        "\n"
        "        # Enhancement 1: SL protection check (cooldown, price distance, consecutive cap)\n"
        "        can_enter, _skip = trade_mgr.can_enter_trade(\n"
        "            direction=direction,\n"
        "            entry_price=row['close'],\n"
        "            current_atr=atr,\n"
        "        )\n"
        "        if not can_enter:\n"
        "            skip_reasons[_skip] = skip_reasons.get(_skip, 0) + 1\n"
        "            continue\n"
        "\n"
        "        # Enhancement 5: Portfolio risk check (daily/weekly loss limits)\n"
        "        _cur_ts = row.name if hasattr(row.name, 'date') else pd.Timestamp(row.name)\n"
        "        _can_risk, _risk_reason = risk_mgr.can_trade(timestamp=_cur_ts)\n"
        "        if not _can_risk:\n"
        "            _short = _risk_reason.split(' ')[0]\n"
        "            skip_reasons[_short] = skip_reasons.get(_short, 0) + 1\n"
        "            continue\n"
        "\n"
        "        # Set pending signal",
    ),
    # 7. Win streak position size reduction
    (
        "            position_size = min(kelly * kelly_fraction, kelly_max)\n"
        "\n"
        "            if position_size < 1e-6:",
        "            position_size = min(kelly * kelly_fraction, kelly_max)\n"
        "\n"
        "            # Enhancement 5: Win streak size reduction\n"
        "            _size_mod = risk_mgr.get_position_size_modifier()\n"
        "            if _size_mod < 1.0:\n"
        "                position_size *= _size_mod\n"
        "\n"
        "            if position_size < 1e-6:",
    ),
])

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook patched successfully!")
print("Changes:")
print("  - Added RiskManager import")
print("  - Initialized RiskManager alongside TradeManager")
print("  - Added new skip reasons: cooldown, price_too_close, consecutive_sl_cap, daily/weekly_loss_limit")
print("  - trade_mgr.advance_bar() called each iteration")
print("  - trade_mgr.record_trade_result() called after trade closes")
print("  - risk_mgr.record_trade() called after trade closes")
print("  - SL protection check before signal creation")
print("  - Portfolio risk check before signal creation")
print("  - Win streak position size reduction applied to Kelly sizing")
