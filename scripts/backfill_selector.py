#!/usr/bin/env python3
"""S32 — back-fill the regime-selector tracker over existing journal days.

Replays "bull → BTC (S30), bear → Connors (S18)" across the journal's dates, using the
recorded Connors portfolio path for the bear leg's daily returns and re-fetched BTC
history for the 200d-SMA regime. Idempotent. Going forward, run_daily advances it daily.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_selector.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_engine.paper.benchmarks import BH_BTC_SYMBOL
from trading_engine.paper.data import fetch_klines
from trading_engine.paper.journal import Journal
from trading_engine.paper.selector_track import RegimeSelector, step, serialize


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Back-fill regime-selector tracker")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ma-window", type=int, default=200)
    p.add_argument("--limit", type=int, default=400)
    args = p.parse_args(argv)

    journal = Journal.load()                      # triggers v→v4 migration
    days = journal.days
    if not days:
        print("Journal has no days yet — nothing to back-fill.")
        return 0

    btc = fetch_klines(BH_BTC_SYMBOL, limit=args.limit)
    closes = btc["Close"]; sma = closes.rolling(args.ma_window).mean()

    sel = RegimeSelector(ma_window=args.ma_window)
    prev_pf = journal.starting_capital
    for d in days:
        pf = d["portfolio_value"]
        cdr = (pf / prev_pf - 1.0) if prev_pf > 0 else 0.0
        prev_pf = pf
        ts = pd.Timestamp(d["date"], tz="UTC")
        c_hist = closes[closes.index <= ts]
        bc = float(c_hist.iloc[-1]) if len(c_hist) else None
        s_hist = sma[sma.index <= ts]
        sm = float(s_hist.iloc[-1]) if len(s_hist) and not pd.isna(s_hist.iloc[-1]) else None
        d["selector_value"] = step(sel, bc if bc else 0.0, sm, cdr, d["date"]) if bc else sel.value

    journal.set_benchmark("regime_selector", serialize(sel))
    last = days[-1]
    print(f"Replayed {len(days)} days {days[0]['date']}→{last['date']}")
    print(f"  selector final:  ${last['selector_value']:,.2f}  (mode={sel.mode}, {sel.n_switches} switches)")
    print(f"  connors:         ${last['portfolio_value']:,.2f}")
    print(f"  trend-timed:     ${last.get('trend_timed_value', 0):,.2f}")
    print(f"  BH_BTC:          ${last.get('bh_btc_value', 0):,.2f}")
    if args.dry_run:
        print("DRY RUN — journal NOT saved.")
    else:
        journal.save(); print("Journal saved (v4, regime_selector back-filled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
