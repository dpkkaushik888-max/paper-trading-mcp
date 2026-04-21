#!/usr/bin/env python3
"""S17/S19 — Rule-based Connors swing backtest (no ML, no training).

Fork of sim_swing_backtest.py (S16) with the ML signal layer replaced by
fixed Connors-style rules. Reuses Binance data pipeline, honest cost model,
timeline split, metrics, and gates so results are directly comparable.

S19 adds --universe flag for expanded-universe test (20 liquid crypto).

Usage:
    # S17 default (6-symbol legacy universe)
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py
    # S19 expanded universe
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe expanded
    # Sensitivity / utilities
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --no-adx
    PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --holdout-only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse S16 infrastructure.
import scripts.sim_swing_backtest as _s16
from scripts.sim_swing_backtest import (
    CAPITAL, COST_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS,
    MAX_CONCURRENT, POS_SIZE_PCT, TRAIN_PCT, WALKFWD_PCT, HOLDOUT_PCT,
    fetch_daily_bars, compute_metrics, compute_benchmark,
    print_metrics, print_gates, fmt_pct, SimResult, DailyStep, ClosedTrade, Position,
)
from trading_engine.strategies.connors_swing import (
    precompute_indicators, long_entry, long_exit,
    SL_PCT, MAX_HOLD_DAYS,
)

# ── Pre-committed universes (S17 + S19 — frozen, do NOT modify without a new spec) ──

UNIVERSE_LEGACY = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
]

# S19 — top-20 liquid crypto, all pre-committed in specs/S19-expanded-universe-connors.md
UNIVERSE_EXPANDED = UNIVERSE_LEGACY + [
    "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "ATOM-USD", "NEAR-USD",
    "LTC-USD",  "TRX-USD", "BCH-USD", "APT-USD", "UNI-USD",  "ARB-USD",
    "OP-USD",   "SUI-USD",
]

# Extra Binance mappings for the 14 new symbols (patched into S16's BINANCE_MAP at import).
_BINANCE_MAP_EXTRA = {
    "DOGE-USD": "DOGEUSDT", "XRP-USD":  "XRPUSDT",  "ADA-USD":  "ADAUSDT",
    "DOT-USD":  "DOTUSDT",  "ATOM-USD": "ATOMUSDT", "NEAR-USD": "NEARUSDT",
    "LTC-USD":  "LTCUSDT",  "TRX-USD":  "TRXUSDT",  "BCH-USD":  "BCHUSDT",
    "APT-USD":  "APTUSDT",  "UNI-USD":  "UNIUSDT",  "ARB-USD":  "ARBUSDT",
    "OP-USD":   "OPUSDT",   "SUI-USD":  "SUIUSDT",
}
_s16.BINANCE_MAP.update(_BINANCE_MAP_EXTRA)

warnings.filterwarnings("ignore")


# ── Rule-based per-symbol data container ───────────────────────────────────

@dataclass
class SymbolBars:
    symbol: str
    bars: pd.DataFrame          # full OHLCV
    indicators: pd.DataFrame    # precomputed (sma_200, sma_5, rsi_2, adx_14)


def prepare_symbols(data_daily: dict[str, pd.DataFrame]) -> dict[str, SymbolBars]:
    out: dict[str, SymbolBars] = {}
    for sym, df in data_daily.items():
        ind = precompute_indicators(df)
        out[sym] = SymbolBars(symbol=sym, bars=df, indicators=ind)
    return out


# ── Simulator (rule-based) ──────────────────────────────────────────────────

@dataclass
class SignalAudit:
    potential_entries: int = 0  # Days × symbols where rule fired
    actual_entries: int = 0     # After position-cap & capital checks
    exits_mr: int = 0
    exits_sl: int = 0
    exits_max_hold: int = 0
    exits_end: int = 0


def run_rules_sim(
    sym_data: dict[str, SymbolBars],
    all_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_adx_filter: bool,
    label: str,
) -> tuple[SimResult, SignalAudit]:
    cash = CAPITAL
    positions: dict[str, Position] = {}
    result = SimResult()
    audit = SignalAudit()
    sim_days = all_dates[(all_dates >= start) & (all_dates <= end)]

    if len(sim_days) == 0:
        print(f"  [{label}] No simulation days in window.")
        return result, audit

    prev_portfolio = CAPITAL

    for today in sim_days:
        # ── Gather today's close + indicators for every symbol ─────────
        today_snapshot: dict[str, tuple[float, pd.Series]] = {}
        for sym, sd in sym_data.items():
            if today not in sd.bars.index:
                continue
            close_today = float(sd.bars["Close"].loc[today])
            ind_row = sd.indicators.loc[today]
            today_snapshot[sym] = (close_today, ind_row)

        # ── Manage exits first ───────────────────────────────────────────
        to_close: list[tuple[str, str]] = []
        for sym, pos in positions.items():
            if sym not in today_snapshot:
                continue
            close_today, ind_row = today_snapshot[sym]
            reason = long_exit(close_today, ind_row, pos.entry_price, pos.entry_date, today)
            if reason:
                to_close.append((sym, reason))

        for sym, reason in to_close:
            pos = positions[sym]
            close_today, _ = today_snapshot[sym]
            exit_slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if reason == "SL" else 0.0)
            fill_price = close_today * (1.0 - exit_slip)  # long close fills at bid
            exit_cost = fill_price * pos.shares * COST_PCT
            proceeds = fill_price * pos.shares - exit_cost
            cash += proceeds
            result.total_costs += exit_cost
            pnl = proceeds - (pos.entry_price * pos.shares + pos.entry_cost)
            pnl_pct = pnl / (pos.entry_price * pos.shares)
            result.trades.append(ClosedTrade(
                symbol=sym, entry_date=pos.entry_date, exit_date=today,
                entry_price=pos.entry_price, exit_price=fill_price,
                shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
                hold_days=(today - pos.entry_date).days, reason=reason,
            ))
            if reason == "SL":
                audit.exits_sl += 1
            elif reason == "MR_EXIT":
                audit.exits_mr += 1
            elif reason == "MAX_HOLD":
                audit.exits_max_hold += 1
            del positions[sym]

        # ── Entry signals ────────────────────────────────────────────────
        candidates: list[tuple[str, float]] = []
        for sym, (close_today, ind_row) in today_snapshot.items():
            if sym in positions:
                continue
            if long_entry(close_today, ind_row, use_adx_filter=use_adx_filter):
                audit.potential_entries += 1
                candidates.append((sym, close_today))

        capacity = MAX_CONCURRENT - len(positions)
        if capacity > 0 and candidates:
            # No ranking signal in rule-based — use alphabetical for determinism.
            candidates.sort(key=lambda c: c[0])
            for sym, price in candidates[:capacity]:
                entry_fill = price * (1.0 + SLIPPAGE_BPS)
                pos_value = CAPITAL * POS_SIZE_PCT
                if pos_value > cash:
                    break
                shares = pos_value / entry_fill
                entry_cost = entry_fill * shares * COST_PCT
                debit = entry_fill * shares + entry_cost
                if debit > cash:
                    continue
                cash -= debit
                result.total_costs += entry_cost
                positions[sym] = Position(
                    symbol=sym, entry_date=today, entry_price=entry_fill,
                    shares=shares, entry_cost=entry_cost, entry_conf=1.0,
                )
                audit.actual_entries += 1

        # ── Portfolio marking ────────────────────────────────────────────
        portfolio = cash
        for sym, pos in positions.items():
            sym_close = sym_data[sym].bars["Close"]
            if today in sym_close.index:
                mark = float(sym_close.loc[today])
            else:
                prior = sym_close[sym_close.index <= today]
                mark = float(prior.iloc[-1]) if not prior.empty else pos.entry_price
            portfolio += mark * pos.shares
        daily_ret = (portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0
        result.steps.append(DailyStep(
            date=today, portfolio_value=portfolio, cash=cash,
            open_positions=len(positions), daily_return=daily_ret,
        ))
        prev_portfolio = portfolio

    # ── End-of-sim close-out ─────────────────────────────────────────────
    if positions:
        last_day = sim_days[-1]
        for sym, pos in list(positions.items()):
            sym_close = sym_data[sym].bars["Close"]
            available = sym_close[sym_close.index <= last_day]
            if available.empty:
                continue
            close_price = float(available.iloc[-1])
            fill_price = close_price * (1.0 - SLIPPAGE_BPS)
            exit_cost = fill_price * pos.shares * COST_PCT
            proceeds = fill_price * pos.shares - exit_cost
            cash += proceeds
            result.total_costs += exit_cost
            pnl = proceeds - (pos.entry_price * pos.shares + pos.entry_cost)
            pnl_pct = pnl / (pos.entry_price * pos.shares)
            result.trades.append(ClosedTrade(
                symbol=sym, entry_date=pos.entry_date, exit_date=last_day,
                entry_price=pos.entry_price, exit_price=fill_price,
                shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
                hold_days=(last_day - pos.entry_date).days, reason="END",
            ))
            audit.exits_end += 1
            del positions[sym]

    result.final_value = cash
    return result, audit


def print_signal_audit(label: str, audit: SignalAudit) -> None:
    blocked = audit.potential_entries - audit.actual_entries
    block_pct = (blocked / audit.potential_entries * 100) if audit.potential_entries else 0.0
    total_exits = audit.exits_mr + audit.exits_sl + audit.exits_max_hold + audit.exits_end
    print(f"\n  ━━━ SIGNAL AUDIT [{label}] ━━━")
    print(f"  Potential entries (rule fired):  {audit.potential_entries:>5}")
    print(f"  Actual entries (taken):          {audit.actual_entries:>5}")
    print(f"  Blocked (cap / cash):            {blocked:>5} ({block_pct:.1f}%)")
    print(f"  Exits — Mean-Reversion:          {audit.exits_mr:>5}")
    print(f"  Exits — Stop Loss:               {audit.exits_sl:>5}")
    print(f"  Exits — Max Hold (10d):          {audit.exits_max_hold:>5}")
    print(f"  Exits — End of sim:              {audit.exits_end:>5}")
    print(f"  Total closed trades:             {total_exits:>5}")


def print_per_symbol_audit(label: str, result: SimResult) -> None:
    """S19 addition: break down trades by symbol so we can see which new
    coins actually produced signal and which are dead weight."""
    if not result.trades:
        print(f"\n  ━━━ PER-SYMBOL AUDIT [{label}] ━━━")
        print("  (no trades)")
        return
    by_sym: dict[str, list] = {}
    for t in result.trades:
        by_sym.setdefault(t.symbol, []).append(t)
    rows = []
    for sym, trades in by_sym.items():
        wins = sum(1 for x in trades if x.pnl > 0)
        n = len(trades)
        wr = wins / n * 100 if n else 0.0
        net = sum(x.pnl for x in trades)
        avg = net / n if n else 0.0
        rows.append((sym, n, wins, wr, net, avg))
    rows.sort(key=lambda r: -r[4])  # sort by net $ desc
    print(f"\n  ━━━ PER-SYMBOL AUDIT [{label}] ━━━")
    print(f"  {'Symbol':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Net $':>10} {'Avg $':>8}")
    for sym, n, wins, wr, net, avg in rows:
        print(f"  {sym:<10} {n:>7} {wins:>5} {wr:>5.1f}% {net:>+10.2f} {avg:>+8.2f}")
    total_n = sum(r[1] for r in rows)
    total_net = sum(r[4] for r in rows)
    print(f"  {'TOTAL':<10} {total_n:>7} {'':>5} {'':>6} {total_net:>+10.2f}")


def _benchmark_from_bars(sym_data: dict[str, SymbolBars], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Adapter: compute_benchmark from S16 expects SymbolData with .close, but we
    have SymbolBars with .bars["Close"]. Build a minimal shim."""
    @dataclass
    class _Adapter:
        close: pd.Series
    shim = {sym: _Adapter(close=sd.bars["Close"]) for sym, sd in sym_data.items()}
    return compute_benchmark(shim, start, end)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S17/S19 — rule-based Connors swing backtest")
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--no-adx", action="store_true", help="Disable ADX>20 filter")
    parser.add_argument("--holdout-only", action="store_true")
    parser.add_argument(
        "--universe", choices=["legacy", "expanded"], default="legacy",
        help="legacy=S17 6-symbol universe; expanded=S19 20-symbol universe",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help=f"Override max concurrent positions (default {MAX_CONCURRENT} from S16)",
    )
    args = parser.parse_args()

    # S20: per-run override of position cap (patched into S16 module so the
    # imported sim helpers see the new value).
    if args.max_concurrent is not None:
        _s16.MAX_CONCURRENT = args.max_concurrent
        globals()["MAX_CONCURRENT"] = args.max_concurrent

    use_adx = not args.no_adx
    if args.universe == "expanded":
        symbols = UNIVERSE_EXPANDED
        spec_tag = "S19 (expanded, 20 crypto)"
    else:
        symbols = UNIVERSE_LEGACY
        spec_tag = "S17 (legacy, 6 crypto)"

    print("=" * 90)
    print(f"  RULE-BASED CONNORS SWING — {spec_tag}")
    print("=" * 90)
    print(f"  Symbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"  Rules: Close>SMA(200) & RSI(2)<10 & Close<SMA(5)"
          f"{' & ADX(14)>=20' if use_adx else ''}")
    print(f"  Exit: Close>SMA(5) | Max hold {MAX_HOLD_DAYS}d | SL {SL_PCT:.0%}")
    print(f"  Costs: {COST_PCT*10000:.0f}bps/side + {SLIPPAGE_BPS*10000:.0f}bps slip + "
          f"{SL_SLIPPAGE_BPS*10000:.0f}bps SL slip")
    print(f"  Capital: ${CAPITAL:,.0f} | Max concurrent: {MAX_CONCURRENT} | "
          f"Position: {POS_SIZE_PCT:.0%}")

    # 1. Fetch data
    print(f"\n  [1/3] Fetching {args.years} years of daily bars...")
    t0 = time.time()
    data_daily = fetch_daily_bars(symbols, years=args.years)
    if not data_daily:
        print("  ERROR: no data fetched")
        return
    print(f"         {len(data_daily)}/{len(symbols)} symbols fetched, {time.time() - t0:.1f}s")

    # 2. Pre-compute indicators
    print(f"\n  [2/3] Pre-computing Connors indicators...")
    t0 = time.time()
    sym_data = prepare_symbols(data_daily)
    print(f"         {len(sym_data)} symbols, {time.time() - t0:.1f}s")

    # 3. Split timeline (same splits as S16 for comparability)
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(sd.indicators.dropna().index for sd in sym_data.values())))
    )
    if len(all_dates) < 500:
        print(f"  ERROR: only {len(all_dates)} valid dates — need at least 500")
        return
    n = len(all_dates)
    train_end_idx = int(n * TRAIN_PCT)
    walkfwd_end_idx = int(n * (TRAIN_PCT + WALKFWD_PCT))
    train_start = all_dates[0]
    train_end = all_dates[train_end_idx]
    walkfwd_end = all_dates[walkfwd_end_idx]
    holdout_end = all_dates[-1]
    print(f"\n  Timeline:")
    print(f"    Warm-up (no sim): {train_start.date()} → {train_end.date()} ({train_end_idx} days)")
    print(f"    Walk-forward:     {train_end.date()} → {walkfwd_end.date()} "
          f"({walkfwd_end_idx - train_end_idx} days)")
    print(f"    LOCKED HOLDOUT:   {walkfwd_end.date()} → {holdout_end.date()} "
          f"({n - walkfwd_end_idx} days)")
    print(f"    NOTE: Rule-based — 'walk-forward' is just the first OOS segment;")
    print(f"          no retraining happens. Both segments evaluated identically.")

    # 4. Run
    print(f"\n  [3/3] Running rule-based sim...")
    benchmark_wf = _benchmark_from_bars(sym_data, train_end, walkfwd_end)
    benchmark_ho = _benchmark_from_bars(sym_data, walkfwd_end, holdout_end)

    adx_tag = "ADX-ON" if use_adx else "ADX-OFF"

    if not args.holdout_only:
        t0 = time.time()
        wf_result, wf_audit = run_rules_sim(
            sym_data, all_dates, train_end, walkfwd_end, use_adx_filter=use_adx,
            label=f"Connors-{adx_tag}-WF",
        )
        wf_metrics = compute_metrics(wf_result)
        print(f"\n  Walk-forward done in {time.time() - t0:.1f}s")
        print_metrics(f"Connors [{adx_tag}] — WALK-FORWARD", wf_metrics, benchmark_wf)
        print_signal_audit(f"{adx_tag} — WF", wf_audit)

    t0 = time.time()
    ho_result, ho_audit = run_rules_sim(
        sym_data, all_dates, walkfwd_end, holdout_end, use_adx_filter=use_adx,
        label=f"Connors-{adx_tag}-HO",
    )
    ho_metrics = compute_metrics(ho_result)
    print(f"\n  Holdout done in {time.time() - t0:.1f}s")
    print_metrics(f"Connors [{adx_tag}] — LOCKED HOLDOUT", ho_metrics, benchmark_ho)
    print_signal_audit(f"{adx_tag} — HOLDOUT", ho_audit)
    print_per_symbol_audit(f"{adx_tag} — HOLDOUT", ho_result)
    print_gates(f"Connors [{adx_tag}] HOLDOUT", ho_metrics, benchmark_ho)

    # S19 success criteria (in addition to G1-G4)
    if args.universe == "expanded":
        print(f"\n  ━━━ S19 SUCCESS CRITERIA (holdout) ━━━")
        s1 = ho_metrics.get("trades", 0) >= 35
        s2 = ho_metrics.get("win_rate", 0) >= 0.60
        s3 = ho_metrics.get("cagr", 0) >= 0.12
        print(f"  {'S1: Trade count >= 35':<45} {'PASS' if s1 else 'FAIL':>6}  "
              f"({ho_metrics.get('trades', 0)})")
        print(f"  {'S2: Win rate >= 60%':<45} {'PASS' if s2 else 'FAIL':>6}  "
              f"({ho_metrics.get('win_rate', 0) * 100:.1f}%)")
        print(f"  {'S3: CAGR >= +12%':<45} {'PASS' if s3 else 'FAIL':>6}  "
              f"({ho_metrics.get('cagr', 0) * 100:+.2f}%)")

    done_tag = "S19" if args.universe == "expanded" else "S17"
    print(f"\n{'=' * 90}")
    print(f"  {done_tag} DONE.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
