#!/usr/bin/env python3
"""S23 — drive the L2 CryptoLeaf + L3 StrategyOrchestrator.

Two modes:

  --mode window  (regression oracle): with gating OFF + S21 caps/sizing this MUST
                 reproduce scripts/sim_s21_window.py exactly.
  --mode gates   (default, validation): purged walk-forward + LOCKED HOLDOUT split
                 (S21 D8), evaluating gates G1–G9 (S21 D9) ONCE. Exits non-zero on
                 any gate failure (CI-friendly). The holdout is evaluated a single
                 time — do NOT re-run with tweaked parameters (the S17→S19→S20 trap).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_crypto_leaf.py            # gates, 5y
    PYTHONPATH=. .venv/bin/python scripts/sim_crypto_leaf.py --mode window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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

TRAIN_PCT, WALKFWD_PCT = 0.60, 0.20  # remainder = locked holdout (S21 D8)


def all_strategies() -> list:
    return [ConnorsSwingStrategy(a_cfg()),
            BreakoutContinuationStrategy(b_cfg()),
            RangeMeanRevStrategy(c_cfg())]


def _orch(strategies) -> StrategyOrchestrator:
    return StrategyOrchestrator(
        strategies=strategies, regime_filter=None, regime_gating=False,
        global_max_concurrent=8, per_strategy_cap=4, base_pos_size_pct=0.12,
    )


def run_leaf(strategies, data, ind, start, end, capital=10_000.0):
    """Run a CryptoLeaf over [start, end] (inclusive) and return (LeafResult, days)."""
    dates = sorted(set().union(*[df.index for df in data.values()]))
    window = pd.DatetimeIndex([d for d in dates if start <= d <= end])
    leaf = CryptoLeaf(_orch(strategies), capital=capital)
    for day in window:
        snap = {sym: (float(df["Close"].loc[day]), ind[sym].loc[day])
                for sym, df in data.items() if day in df.index}
        if snap:
            leaf.run_day(snap, btc_df=None, day=day)
    if len(window):
        last = window[-1]
        leaf.close_all(
            {sym: float(df["Close"].loc[last]) for sym, df in data.items() if last in df.index}, last)
    return leaf.result(), max(1, len(window))


def bh_btc(data, start, end, capital=10_000.0, cost_pct=0.0020) -> dict:
    """Buy-and-hold BTC over the window with one-time entry cost."""
    btc = None
    for key in ("BTC-USD", "BTCUSDT"):
        if key in data:
            btc = data[key]["Close"]
            break
    if btc is None:
        return {"cagr": 0.0, "sharpe": 0.0, "total_return": 0.0}
    s = btc[(btc.index >= start) & (btc.index <= end)]
    if len(s) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "total_return": 0.0}
    shares = capital * (1 - cost_pct) / float(s.iloc[0])
    vals = shares * s
    total_ret = float(vals.iloc[-1] / capital - 1.0)
    years = len(s) / 365.25
    cagr = (vals.iloc[-1] / capital) ** (1 / years) - 1.0 if years > 0 else 0.0
    dr = vals.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0.0
    return {"cagr": float(cagr), "sharpe": sharpe, "total_return": total_ret}


def cagr_of(result, days) -> float:
    years = days / 365.25
    return (result.final_value / 10_000.0) ** (1 / years) - 1.0 if years > 0 else 0.0


def _split(data):
    dates = sorted(set().union(*[df.index for df in data.values()]))
    n = len(dates)
    return (dates[0], dates[int(n * TRAIN_PCT)],
            dates[int(n * (TRAIN_PCT + WALKFWD_PCT))], dates[-1])


def run_gates(years: float) -> int:
    print(f"Fetching {years}y daily bars for {len(oracle.UNIVERSE)} symbols...")
    data = oracle.fetch_daily_bars(oracle.UNIVERSE, years=years)
    print(f"  {len(data)}/{len(oracle.UNIVERSE)} symbols fetched")
    ind = {sym: build_indicators(df) for sym, df in data.items()}

    _, train_end, wf_end, ho_end = _split(data)
    wf_start, ho_start = train_end, wf_end
    print(f"  Split: walk-forward {wf_start.date()}→{wf_end.date()}, "
          f"LOCKED HOLDOUT {ho_start.date()}→{ho_end.date()}\n")

    results = []  # (gate_id, description, passed, detail)

    # ── Single-strategy gates on walk-forward (G1–G4) ───────────────────────
    print("── Walk-forward (each strategy in isolation) ──")
    wf_bh = bh_btc(data, wf_start, wf_end)
    for strat_factory, label in ((ConnorsSwingStrategy(a_cfg()), "A_connors"),
                                 (BreakoutContinuationStrategy(b_cfg()), "B_breakout"),
                                 (RangeMeanRevStrategy(c_cfg()), "C_range")):
        r, days = run_leaf([strat_factory], data, ind, wf_start, wf_end)
        c = cagr_of(r, days)
        print(f"  {label:<12} ret {r.return_pct*100:+6.2f}%  CAGR {c*100:+6.2f}%  "
              f"Sharpe {r.sharpe:+.2f}  DD {r.max_dd*100:4.1f}%  trades {r.n_trades}")
        results.append((f"G1.{label}", "Sharpe ≥ 0.8", r.sharpe >= 0.8, f"{r.sharpe:.2f}"))
        results.append((f"G2.{label}", "alpha ≥ 0 vs BH_BTC", c >= wf_bh["cagr"],
                        f"{c*100:+.1f}% vs {wf_bh['cagr']*100:+.1f}%"))
        results.append((f"G3.{label}", "max DD < 25%", r.max_dd < 0.25, f"{r.max_dd*100:.1f}%"))
        results.append((f"G4.{label}", "≥ 10 trades", r.n_trades >= 10, str(r.n_trades)))

    # ── Combined-portfolio gates on the LOCKED HOLDOUT (G5–G9) ──────────────
    print("\n── LOCKED HOLDOUT (combined stack, evaluated ONCE) ──")
    r, days = run_leaf(all_strategies(), data, ind, ho_start, ho_end)
    c = cagr_of(r, days)
    ho_bh = bh_btc(data, ho_start, ho_end)
    months = days / 30.44
    tpm = r.n_trades / months if months else 0.0
    print(f"  combined  ret {r.return_pct*100:+.2f}%  CAGR {c*100:+.2f}%  Sharpe {r.sharpe:+.2f}  "
          f"DD {r.max_dd*100:.1f}%  trades {r.n_trades} ({tpm:.1f}/mo)")
    print(f"  BH_BTC    CAGR {ho_bh['cagr']*100:+.2f}%  Sharpe {ho_bh['sharpe']:+.2f}")
    results.append(("G5", "combined Sharpe ≥ 1.0", r.sharpe >= 1.0, f"{r.sharpe:.2f}"))
    results.append(("G6", "combined CAGR ≥ +8%", c >= 0.08, f"{c*100:+.1f}%"))
    results.append(("G7", "combined max DD < 20%", r.max_dd < 0.20, f"{r.max_dd*100:.1f}%"))
    results.append(("G8", "≥ 4 trades/month", tpm >= 4.0, f"{tpm:.1f}/mo"))
    results.append(("G9", "beats BH_BTC on CAGR & Sharpe",
                    c > ho_bh["cagr"] and r.sharpe > ho_bh["sharpe"],
                    f"CAGR {c*100:+.1f}>{ho_bh['cagr']*100:+.1f}, Sharpe {r.sharpe:.2f}>{ho_bh['sharpe']:.2f}"))

    print("\n── Gate results ──")
    n_fail = 0
    for gid, desc, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            n_fail += 1
        print(f"  [{mark}] {gid:<14} {desc:<28} ({detail})")

    print("\n" + "=" * 56)
    if n_fail == 0:
        print("  ALL GATES PASS — S23 stack APPROVED on the holdout.")
    else:
        print(f"  {n_fail} GATE(S) FAILED — S23 REJECTED. Do NOT re-run the holdout")
        print("  with tweaked params; a new spec (S25+) is required to iterate.")
    print("=" * 56)
    return 0 if n_fail == 0 else 1


def run_window() -> int:
    data = oracle.fetch_daily_bars(oracle.UNIVERSE, years=1.5)
    ind = {sym: build_indicators(df) for sym, df in data.items()}
    r, _ = run_leaf(all_strategies(), data, ind, oracle.WINDOW_START, oracle.WINDOW_END)
    print(f"Window {oracle.WINDOW_START.date()}→{oracle.WINDOW_END.date()}: "
          f"{r.return_pct*100:+.2f}%  ${r.final_value:,.2f}  {r.n_trades} trades "
          f"(W{r.wins}/L{r.losses})  costs ${r.total_costs:.2f}")
    print("Compare to: PYTHONPATH=. .venv/bin/python scripts/sim_s21_window.py")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="S23 crypto leaf: gates or oracle window")
    p.add_argument("--mode", choices=["gates", "window"], default="gates")
    p.add_argument("--years", type=float, default=5.0)
    args = p.parse_args(argv)
    return run_gates(args.years) if args.mode == "gates" else run_window()


if __name__ == "__main__":
    sys.exit(main())
