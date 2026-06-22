"""Period-by-period strategy-selection harness (S31).

``run_selection(selector, ...)`` walks the window one period at a time. At each
period boundary the selector sees market data **strictly before** that boundary and
returns a sleeve name; the account runs that sleeve over the period; sleeve changes
pay an honest switching cost. Output: ending value, per-period choices, equity curve.

Sleeves:
  - "btc"     : hold BTC for the period (the trend-participation / HODL sleeve)
  - "cash"    : sit out (flat)
  - "connors" : run Connors dip-scalping across the universe for the period

No-lookahead is structural: the selector is only ever passed ``btc_close[:start]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from trading_engine.paper.config import (
    COST_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS, POS_SIZE_PCT, MAX_CONCURRENT)
from trading_engine.strategies.connors_swing import (
    precompute_indicators, long_entry, long_exit)

# A selector: (btc_close_history_before_period, period_start) -> sleeve name.
Selector = Callable[[pd.Series, pd.Timestamp], str]


@dataclass
class SelectionResult:
    final_value: float
    total_return: float
    max_drawdown: float
    choices: list = field(default_factory=list)   # [(period_start_iso, sleeve)]
    equity: list = field(default_factory=list)     # period-end values
    n_switches: int = 0


# ── sleeves ─────────────────────────────────────────────────────────────────
def _btc_mult(btc_close: pd.Series, b0, b1) -> float:
    """BTC return between two boundary closes (telescopes to a continuous hold)."""
    return float(btc_close.loc[b1] / btc_close.loc[b0])


def _connors_period(bars: dict, ind: dict, a, b, capital: float) -> float:
    """Run Connors over (a, b] starting with ``capital``; close out at period end."""
    cash = capital
    pos: dict = {}
    days = [d for d in sorted(set().union(*[df.index for df in bars.values()])) if a < d <= b]
    for d in days:
        for sym in list(pos):                       # exits first
            if d not in bars[sym].index:
                continue
            c = float(bars[sym]["Close"].loc[d]); r = ind[sym].loc[d]; p = pos[sym]
            why = long_exit(c, r, p["ep"], p["ed"], d)
            if why:
                slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if why == "SL" else 0.0)
                cash += c * (1 - slip) * p["sh"] * (1 - COST_PCT); del pos[sym]
        eq = cash + sum(float(bars[s]["Close"].loc[d]) * p["sh"]
                        for s, p in pos.items() if d in bars[s].index)
        cands = sorted(s for s in bars if s not in pos and d in bars[s].index
                       and not pd.isna(ind[s].loc[d].get("sma_200"))
                       and long_entry(float(bars[s]["Close"].loc[d]), ind[s].loc[d], True))
        for sym in cands[:max(0, MAX_CONCURRENT - len(pos))]:
            c = float(bars[sym]["Close"].loc[d]); fill = c * (1 + SLIPPAGE_BPS); pv = eq * POS_SIZE_PCT
            if pv * (1 + COST_PCT) > cash:
                continue
            cash -= pv * (1 + COST_PCT); pos[sym] = {"ep": fill, "ed": d, "sh": pv / fill}
    if days:                                          # self-contained: flatten at period end
        last = days[-1]
        for sym in list(pos):
            if last in bars[sym].index:
                cash += float(bars[sym]["Close"].loc[last]) * (1 - SLIPPAGE_BPS) * pos[sym]["sh"] * (1 - COST_PCT)
            del pos[sym]
    return cash


# ── period boundaries ───────────────────────────────────────────────────────
def month_boundaries(dates: list, start, end) -> list:
    """Decision boundaries: window start, each month-end trading day, window end.

    Period i runs from boundary[i] to boundary[i+1]; the selector decides at
    boundary[i] (using data through it) and the period's return accrues afterward.
    """
    win = [d for d in dates if start <= d <= end]
    if not win:
        return []
    bnds = [win[0]]
    for i in range(1, len(win)):
        if (win[i].year, win[i].month) != (win[i - 1].year, win[i - 1].month):
            bnds.append(win[i - 1])           # last trading day of the prior month
    if bnds[-1] != win[-1]:
        bnds.append(win[-1])
    return bnds


# ── the harness ─────────────────────────────────────────────────────────────
def run_selection(
    selector: Selector, btc_close: pd.Series, bars: dict, ind: dict,
    boundaries: list, capital: float = 1000.0,
) -> SelectionResult:
    """Walk the boundaries; the selector picks a sleeve per period from past data only.

    At boundary[i] the selector sees ``btc_close`` through boundary[i] and chooses the
    sleeve held over (boundary[i], boundary[i+1]] — so no period's own return leaks in.
    """
    value = capital
    prev: Optional[str] = None
    choices, equity, switches = [], [], 0
    for i in range(len(boundaries) - 1):
        b0, b1 = boundaries[i], boundaries[i + 1]
        history = btc_close[btc_close.index <= b0]    # known at decision time; return is b0→b1
        sleeve = selector(history, b0)
        if prev is not None and sleeve != prev:
            value *= (1 - COST_PCT)                    # honest switching cost
            switches += 1
        if sleeve == "cash":
            pass
        elif sleeve == "connors":
            value = _connors_period(bars, ind, b0, b1, value)
        else:                                          # "btc"
            value *= _btc_mult(btc_close, b0, b1)
        choices.append((b0.date().isoformat(), sleeve)); equity.append(value); prev = sleeve
    eq = pd.Series(equity)
    mdd = float(((eq.cummax() - eq) / eq.cummax()).max()) if len(eq) else 0.0
    return SelectionResult(
        final_value=value, total_return=value / capital - 1.0, max_drawdown=mdd,
        choices=choices, equity=equity, n_switches=switches)


# ── named selectors (deterministic, past-data-only) ─────────────────────────
def _trend_bull(history: pd.Series, ma: int = 200, band: float = 0.0) -> bool:
    if len(history) < ma:
        return False
    sma = float(history.iloc[-ma:].mean()); px = float(history.iloc[-1])
    return px > sma * (1 + band)


def sel_hodl(history, a):            # baseline: always invested = HODL
    return "btc"


def sel_trend(history, a):           # bull → BTC, bear → cash (monthly trend-timing)
    return "btc" if _trend_bull(history) else "cash"


def sel_trend_connors(history, a):   # bull → BTC, bear → Connors
    return "btc" if _trend_bull(history) else "connors"


SELECTORS = {"hodl": sel_hodl, "trend": sel_trend, "trend_connors": sel_trend_connors}
