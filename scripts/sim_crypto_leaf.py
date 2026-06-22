#!/usr/bin/env python3
"""S23 — drive the L2 CryptoLeaf + L3 StrategyOrchestrator over a window.

Regression oracle: with regime gating OFF and the S21 caps/sizing, this MUST
reproduce scripts/sim_s21_window.py's numbers exactly (same data, same rules,
same execution math). Reuses sim_s21_window's data layer so both see identical
bars in-process.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_crypto_leaf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sim_s21_window as oracle  # data layer + universe + window
from trading_engine.engine.crypto_leaf import CryptoLeaf
from trading_engine.engine.indicators import build_indicators
from trading_engine.engine.orchestrator import StrategyOrchestrator
from trading_engine.strategies.connors_strategy import ConnorsSwingStrategy, default_config as a_cfg
from trading_engine.strategies.breakout_continuation import BreakoutContinuationStrategy, default_config as b_cfg
from trading_engine.strategies.range_meanrev import RangeMeanRevStrategy, default_config as c_cfg


def build_orchestrator() -> StrategyOrchestrator:
    return StrategyOrchestrator(
        strategies=[ConnorsSwingStrategy(a_cfg()),
                    BreakoutContinuationStrategy(b_cfg()),
                    RangeMeanRevStrategy(c_cfg())],
        regime_filter=None, regime_gating=False,
        global_max_concurrent=8, per_strategy_cap=4, base_pos_size_pct=0.12,
    )


def run() -> None:
    print("Fetching bars (reusing sim_s21_window data layer)...")
    data = oracle.fetch_daily_bars(oracle.UNIVERSE, years=1.5)
    print(f"  {len(data)}/{len(oracle.UNIVERSE)} symbols\n")

    ind = {sym: build_indicators(df) for sym, df in data.items()}
    all_dates = sorted(set().union(*[df.index for df in data.values()]))
    all_dates = pd.DatetimeIndex(
        [d for d in all_dates if oracle.WINDOW_START <= d <= oracle.WINDOW_END]
    )

    leaf = CryptoLeaf(build_orchestrator(), capital=10_000.0)
    for day in all_dates:
        snap = {
            sym: (float(df["Close"].loc[day]), ind[sym].loc[day])
            for sym, df in data.items() if day in df.index
        }
        leaf.run_day(snap, btc_df=None, day=day)
    last_closes = {
        sym: float(df["Close"].loc[all_dates[-1]])
        for sym, df in data.items() if all_dates[-1] in df.index
    }
    leaf.close_all(last_closes, all_dates[-1])
    r = leaf.result()

    print("=" * 60)
    print(f"  CRYPTO LEAF (orchestrator path) — {all_dates[0].date()} → {all_dates[-1].date()}")
    print("=" * 60)
    print(f"  Total return : {r.return_pct*100:+.2f}%   (final ${r.final_value:,.2f})")
    print(f"  Sharpe       : {r.sharpe:.2f}")
    print(f"  Max DD       : {r.max_dd*100:.2f}%")
    print(f"  Trades       : {r.n_trades}  (W {r.wins} / L {r.losses})")
    print(f"  Total costs  : ${r.total_costs:.2f}")
    print("\n  Per-strategy:")
    for name, d in r.per_strategy.items():
        print(f"    {name:<12} trades {d['trades']:>2}  wins {d['wins']:>2}  net ${d['net']:>+8.2f}")
    print("\n  Closed trades:")
    for t in sorted(r.trades, key=lambda x: x.entry_date):
        print(f"    {t.symbol:<9} {t.entry_date.date()}→{t.exit_date.date()} "
              f"{t.pnl_pct*100:+6.2f}%  ${t.pnl:+7.2f}  [{t.reason}]  ({t.strategy})")
    print("=" * 60)
    print("  Compare these to: PYTHONPATH=. .venv/bin/python scripts/sim_s21_window.py")


if __name__ == "__main__":
    run()
