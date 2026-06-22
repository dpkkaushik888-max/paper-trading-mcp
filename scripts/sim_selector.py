#!/usr/bin/env python3
"""S31 — strategy-selection backtest: a named selector picks a sleeve each period,
total profit reported vs HODL. The selector ("agent") uses only past data.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_selector.py --years 4 --capital 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sim_s21_window as oracle
from trading_engine.strategies.connors_swing import precompute_indicators
from trading_engine.selection.harness import (
    SELECTORS, month_boundaries, run_selection)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="S31 strategy-selection backtest")
    p.add_argument("--years", type=float, default=4.0)
    p.add_argument("--capital", type=float, default=1000.0)
    args = p.parse_args(argv)

    bars = oracle.fetch_daily_bars(oracle.UNIVERSE, years=5)   # 5y for SMA warmup
    ind = {s: precompute_indicators(df) for s, df in bars.items()}
    btc = next(bars[k] for k in ("BTC-USD", "BTCUSDT") if k in bars)["Close"]
    dates = sorted(set().union(*[df.index for df in bars.values()]))
    end = dates[-1]; start = end - pd.Timedelta(days=int(args.years * 365))
    start = min(d for d in dates if d >= start)
    bnds = month_boundaries(dates, start, end)

    print(f"Selection backtest — {start.date()} → {end.date()} "
          f"({len(bnds)-1} months), start ${args.capital:,.0f}\n")
    hodl = run_selection(SELECTORS["hodl"], btc, bars, ind, bnds, args.capital)
    print(f"{'selector':<16}{'final':>10}{'return':>9}{'maxDD':>8}{'switches':>9}  {'vs HODL':>9}")
    rows = {}
    for name, sel in SELECTORS.items():
        r = run_selection(sel, btc, bars, ind, bnds, args.capital); rows[name] = r
        vs = r.final_value / hodl.final_value - 1.0
        print(f"{name:<16}${r.final_value:>8,.0f}{r.total_return*100:>+8.0f}%"
              f"{r.max_drawdown*100:>7.0f}%{r.n_switches:>9}  {vs*100:>+8.1f}%")

    # show the monthly choice path for the most interesting selector
    tc = rows["trend_connors"]
    seq = "".join({"btc": "B", "connors": "C", "cash": "_"}[s] for _, s in tc.choices)
    print(f"\ntrend_connors monthly path (B=BTC C=Connors _=cash):\n  {seq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
