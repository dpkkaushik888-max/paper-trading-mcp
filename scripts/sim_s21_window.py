#!/usr/bin/env python3
"""S21 INFORMAL window check — NOT the official locked-holdout gate evaluation.

Runs the S21 regime-stacked engine (3 long-only strategies sharing one capital
pool) over the *same* calendar window as the S18 live paper-forward test so we
can compare apples-to-apples: how would the stacked engine have done over the
exact period where the standalone S20 Connors rule is currently winning live?

Strategies (rules LOCKED in specs/S21-regime-stacked-swing.md):
    A. Uptrend Pullback   = S20 Connors (imported verbatim from connors_swing)
    B. Breakout Continuation (D2)
    C. Range Mean Reversion  (D3)

Orchestration: shared $10k pool, 12% per position, 8 global concurrent,
4 per-strategy cap, first-come-per-symbol conflict resolution (D5/D6),
honest costs (D7: 20bps/side + 5bps slip + 10bps SL slip).

CAVEAT: a 2-month window is NOT S21's mandated 5-year locked-holdout
methodology. This is a directional sanity check, not a pass/fail of the gates.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_s21_window.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sim_swing_backtest as _s16
from scripts.sim_swing_backtest import (
    CAPITAL, COST_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS,
    fetch_daily_bars, compute_metrics, compute_benchmark,
    SimResult, DailyStep, ClosedTrade,
)
from trading_engine.strategies.connors_swing import (
    precompute_indicators as connors_indicators,
    long_entry as connors_entry,
    long_exit as connors_exit,
)

# ── S18 window (same as the live paper-forward run on GitHub) ────────────────
WINDOW_START = pd.Timestamp("2026-04-21")
WINDOW_END = pd.Timestamp("2026-06-22")

# ── S19/S20/S21 frozen 20-coin universe ──────────────────────────────────────
UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
    "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "ATOM-USD", "NEAR-USD",
    "LTC-USD", "TRX-USD", "BCH-USD", "APT-USD", "UNI-USD", "ARB-USD",
    "OP-USD", "SUI-USD",
]

# Resolve every symbol to its binance.com USDT pair (MATIC delisted → POL rebrand).
_s16.BINANCE_MAP.update({s: s.replace("-USD", "USDT") for s in UNIVERSE})
_s16.BINANCE_MAP["MATIC-USD"] = "POLUSDT"

# ── S21 D5 allocation ─────────────────────────────────────────────────────────
POS_SIZE_PCT = 0.12
MAX_CONCURRENT_GLOBAL = 8
PER_STRATEGY_CAP = 4


# ── Indicators for all three strategies (one pass per symbol) ────────────────
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = connors_indicators(df)  # sma_200, sma_5, rsi_2, adx_14
    close = df["Close"]
    # Strategy B inputs
    out["prior_high_20"] = close.shift(1).rolling(20).max()
    out["vol_sma_20"] = df["Volume"].rolling(20).mean()
    out["volume"] = df["Volume"]
    out["sma_50"] = ta.sma(close, length=50)
    out["sma_10"] = ta.sma(close, length=10)
    # Strategy C inputs (Bollinger width on 20 / 2σ)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["bb_width"] = (4.0 * std) / mid  # (upper - lower) / mid = 4σ/mid
    return out


# ── Strategy B — Breakout Continuation (D2) ──────────────────────────────────
def b_entry(close: float, r: pd.Series) -> bool:
    for k in ("prior_high_20", "vol_sma_20", "adx_14", "sma_50"):
        if pd.isna(r.get(k)):
            return False
    return (
        close > r["prior_high_20"]
        and r["volume"] > 1.5 * r["vol_sma_20"]
        and r["adx_14"] >= 25
        and close > r["sma_50"]
    )


def b_exit(close: float, r: pd.Series, entry: float, edate, today) -> str | None:
    if (close - entry) / entry <= -0.08:
        return "SL"
    if not pd.isna(r.get("sma_10")) and close < r["sma_10"]:
        return "MA_BREAK"
    if (today - edate).days >= 15:
        return "MAX_HOLD"
    return None


# ── Strategy C — Range Mean Reversion (D3) ───────────────────────────────────
def c_entry(close: float, r: pd.Series) -> bool:
    for k in ("rsi_2", "adx_14", "bb_width"):
        if pd.isna(r.get(k)):
            return False
    return r["rsi_2"] < 5 and r["adx_14"] < 18 and r["bb_width"] < 0.10


def c_exit(close: float, r: pd.Series, entry: float, edate, today) -> str | None:
    if (close - entry) / entry <= -0.05:
        return "SL"
    if not pd.isna(r.get("rsi_2")) and r["rsi_2"] > 70:
        return "MR_EXIT"
    if (today - edate).days >= 7:
        return "MAX_HOLD"
    return None


# Strategy A wrappers to match (close, row, entry, edate, today) signature
def a_entry(close: float, r: pd.Series) -> bool:
    return connors_entry(close, r, use_adx_filter=True)


def a_exit(close: float, r: pd.Series, entry: float, edate, today) -> str | None:
    return connors_exit(close, r, entry, edate, today)


STRATEGIES = [  # priority order = conflict resolution order (D6)
    ("A_connors", a_entry, a_exit),
    ("B_breakout", b_entry, b_exit),
    ("C_range", c_entry, c_exit),
]


@dataclass
class Pos:
    strategy: str
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_cost: float


def run() -> None:
    print("Fetching ~1.5y daily bars (need 200+ lookback before window start)...")
    t0 = time.time()
    data = fetch_daily_bars(UNIVERSE, years=1.5)
    print(f"  {len(data)}/{len(UNIVERSE)} symbols, {time.time()-t0:.1f}s\n")

    ind = {sym: build_indicators(df) for sym, df in data.items()}
    all_dates = sorted(set().union(*[df.index for df in data.values()]))
    all_dates = pd.DatetimeIndex([d for d in all_dates if WINDOW_START <= d <= WINDOW_END])

    cash = CAPITAL
    positions: dict[str, Pos] = {}  # keyed by symbol (one strategy per symbol, D6)
    result = SimResult()
    by_strategy_trades: dict[str, list[ClosedTrade]] = {s[0]: [] for s in STRATEGIES}
    fire_counts = {s[0]: 0 for s in STRATEGIES}
    prev_value = CAPITAL

    for today in all_dates:
        snap: dict[str, tuple[float, pd.Series]] = {}
        for sym, df in data.items():
            if today in df.index:
                snap[sym] = (float(df["Close"].loc[today]), ind[sym].loc[today])

        # ── Exits (each position uses its owning strategy's exit rule) ──
        exit_fn = {s[0]: s[2] for s in STRATEGIES}
        for sym in list(positions.keys()):
            if sym not in snap:
                continue
            pos = positions[sym]
            close, r = snap[sym]
            reason = exit_fn[pos.strategy](close, r, pos.entry_price, pos.entry_date, today)
            if reason:
                slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if reason == "SL" else 0.0)
                fill = close * (1.0 - slip)
                ec = fill * pos.shares * COST_PCT
                cash += fill * pos.shares - ec
                result.total_costs += ec
                pnl = (fill * pos.shares - ec) - (pos.entry_price * pos.shares + pos.entry_cost)
                trade = ClosedTrade(
                    symbol=sym, entry_date=pos.entry_date, exit_date=today,
                    entry_price=pos.entry_price, exit_price=fill, shares=pos.shares,
                    pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
                    hold_days=(today - pos.entry_date).days, reason=reason,
                )
                result.trades.append(trade)
                by_strategy_trades[pos.strategy].append(trade)
                del positions[sym]

        # ── Entries (strategy priority A→B→C, alpha within strategy) ──
        for sname, entry_fn, _ in STRATEGIES:
            strat_open = sum(1 for p in positions.values() if p.strategy == sname)
            cands = sorted(
                sym for sym, (c, r) in snap.items()
                if sym not in positions and entry_fn(c, r)
            )
            for sym in cands:
                fire_counts[sname] += 1
                if len(positions) >= MAX_CONCURRENT_GLOBAL:
                    break
                if strat_open >= PER_STRATEGY_CAP:
                    break
                close, _ = snap[sym]
                fill = close * (1.0 + SLIPPAGE_BPS)
                pos_value = CAPITAL * POS_SIZE_PCT
                if pos_value > cash:
                    continue
                shares = pos_value / fill
                ec = fill * shares * COST_PCT
                debit = fill * shares + ec
                if debit > cash:
                    continue
                cash -= debit
                result.total_costs += ec
                positions[sym] = Pos(sname, sym, today, fill, shares, ec)
                strat_open += 1

        # ── Mark ──
        value = cash + sum(
            float(data[s].loc[today, "Close"]) * p.shares if today in data[s].index
            else p.entry_price * p.shares
            for s, p in positions.items()
        )
        result.steps.append(DailyStep(
            date=today, portfolio_value=value, cash=cash,
            open_positions=len(positions),
            daily_return=(value / prev_value - 1.0) if prev_value > 0 else 0.0,
        ))
        prev_value = value

    # ── Close-out at window end ──
    last = all_dates[-1]
    for sym, pos in list(positions.items()):
        close = float(data[sym].loc[last, "Close"])
        fill = close * (1.0 - SLIPPAGE_BPS)
        ec = fill * pos.shares * COST_PCT
        cash += fill * pos.shares - ec
        result.total_costs += ec
        pnl = (fill * pos.shares - ec) - (pos.entry_price * pos.shares + pos.entry_cost)
        trade = ClosedTrade(
            symbol=sym, entry_date=pos.entry_date, exit_date=last,
            entry_price=pos.entry_price, exit_price=fill, shares=pos.shares,
            pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
            hold_days=(last - pos.entry_date).days, reason="END",
        )
        result.trades.append(trade)
        by_strategy_trades[pos.strategy].append(trade)
    result.final_value = cash

    # ── Report ──
    m = compute_metrics(result)
    shim = {sym: type("S", (), {"close": df["Close"]})() for sym, df in data.items()}
    bh = compute_benchmark(shim, all_dates[0], all_dates[-1])
    btc_close = data["BTC-USD"]["Close"]
    btc_w = btc_close[(btc_close.index >= all_dates[0]) & (btc_close.index <= all_dates[-1])]
    btc_ret = float(btc_w.iloc[-1] / btc_w.iloc[0] - 1.0)

    print("=" * 64)
    print(f"  S21 STACKED ENGINE — window {all_dates[0].date()} → {all_dates[-1].date()} ({len(all_dates)} days)")
    print("  (INFORMAL window check — not the locked-holdout gate evaluation)")
    print("=" * 64)
    print(f"  Total return : {m['total_return']*100:+.2f}%   (final ${m['final_value']:,.2f})")
    print(f"  Sharpe       : {m['sharpe']:.2f}")
    print(f"  Max DD       : {m['max_dd']*100:.2f}%")
    print(f"  Trades       : {m['trades']}  (W {m['wins']} / L {m['losses']}, WR {m['win_rate']*100:.0f}%)")
    print(f"  Profit factor: {m['profit_factor']:.2f}")
    print(f"  Total costs  : ${m['total_costs']:.2f}")
    print("\n  Per-strategy fire counts & trades taken:")
    for sname, _, _ in STRATEGIES:
        tr = by_strategy_trades[sname]
        w = sum(1 for t in tr if t.pnl > 0)
        net = sum(t.pnl for t in tr)
        print(f"    {sname:<12} fired {fire_counts[sname]:>3}  taken {len(tr):>2}  "
              f"WR {(w/len(tr)*100 if tr else 0):>4.0f}%  net ${net:>+8.2f}")
    if result.trades:
        print("\n  Closed trades:")
        for t in sorted(result.trades, key=lambda x: x.entry_date):
            print(f"    {t.symbol:<9} {t.entry_date.date()}→{t.exit_date.date()} "
                  f"{t.pnl_pct*100:+6.2f}%  ${t.pnl:+7.2f}  [{t.reason}]")
    print("\n  ── Benchmarks (same window) ──")
    print(f"    Buy&Hold BTC    : {btc_ret*100:+.2f}%")
    print(f"    Buy&Hold basket : {bh['total_return']*100:+.2f}%  (Sharpe {bh['sharpe']:.2f})")
    print("\n  ── Live S18 (S20 Connors standalone, GitHub, same window) ──")
    print("    Connors live    : +5.36%  (5 trades, 80% WR, Sharpe 3.57)")
    print("=" * 64)


if __name__ == "__main__":
    run()
