#!/usr/bin/env python3
"""S29 — one-time back-fill of the trend-timed BTC tracker over existing journal days.

The 90-day paper-forward is already mid-flight, so to compare trend-timed BTC over the
SAME window we replay the rule (hold BTC above its MA, cash below) across the journal's
existing dates using re-fetched BTC history (≥ ma_window bars before day 1 for the SMA),
writing ``trend_timed_value`` for every past day and setting the tracker's final state.

Idempotent: rebuilds the tracker from scratch each run. Going forward, run_daily.py
advances it daily.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backfill_trend_timed.py [--dry-run]
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
from trading_engine.paper.trend_timed import TrendTimer, step as trend_step, serialize


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Back-fill trend-timed BTC tracker")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ma-window", type=int, default=200)
    p.add_argument("--limit", type=int, default=400, help="BTC bars to fetch (≥ ma_window + days)")
    args = p.parse_args(argv)

    journal = Journal.load()                      # triggers v2→v3 migration
    days = journal.days
    if not days:
        print("Journal has no days yet — nothing to back-fill.")
        return 0

    print(f"Fetching {args.limit} BTC bars for SMA({args.ma_window})...")
    btc = fetch_klines(BH_BTC_SYMBOL, limit=args.limit)
    closes = btc["Close"]
    sma = closes.rolling(args.ma_window).mean()

    timer = TrendTimer(ma_window=args.ma_window)
    n = 0
    for d in days:                                 # chronological (journal appends in order)
        ts = pd.Timestamp(d["date"], tz="UTC")
        c_hist = closes[closes.index <= ts]
        if c_hist.empty:
            d["trend_timed_value"] = timer.cash + timer.shares * (
                closes.iloc[0] if len(closes) else 0.0)
            continue
        close = float(c_hist.iloc[-1])
        s_hist = sma[sma.index <= ts]
        s = float(s_hist.iloc[-1]) if len(s_hist) and not pd.isna(s_hist.iloc[-1]) else None
        d["trend_timed_value"] = trend_step(timer, close, s, d["date"])
        n += 1

    journal.set_benchmark("trend_timed_btc", serialize(timer))

    first, last = days[0], days[-1]
    print(f"Replayed {n} days {first['date']}→{last['date']}")
    print(f"  trend-timed final: ${last['trend_timed_value']:,.2f}  "
          f"({'INVESTED' if timer.invested else 'CASH'}, {timer.n_switches} switches)")
    print(f"  connors:           ${last['portfolio_value']:,.2f}")
    print(f"  BH_BTC:            ${last.get('bh_btc_value', 0):,.2f}")

    if args.dry_run:
        print("DRY RUN — journal NOT saved.")
    else:
        journal.save()
        print("Journal saved (v3, trend_timed_btc back-filled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
