"""Trend-timed BTC tracker for the S18/S29/S30 paper-forward.

The robust edge from the S25–S28 study: hold BTC while it is above a long moving
average, move to cash when it drops below. S30 adds a **hysteresis band** (default
±2%) around the SMA: enter only when price is a band above it, exit only when a band
below — which kills the whipsaw losses of crossing a flat MA repeatedly (validated:
matched HODL's full 4y return at ~half the drawdown, and never hurt a year vs the
bare rule). Tracked as a phantom portfolio alongside ``bh_btc``/``bh_basket`` —
invested through the entire bull, cash when the trend breaks. Honest 20 bps per switch.

Pure + deterministic: ``step`` takes today's close and SMA and advances the state.
The band is hysteretic, so the in/out decision depends on the current holding state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .config import COST_PCT, STARTING_CAPITAL

DEFAULT_MA = 200
DEFAULT_BAND = 0.02   # S30: ±2% hysteresis around the SMA (anti-whipsaw)


@dataclass
class TrendTimer:
    name: str = "trend_timed_btc"
    initialized: bool = False
    init_date: Optional[str] = None
    invested: bool = False           # currently holding BTC?
    shares: float = 0.0
    cash: float = STARTING_CAPITAL
    ma_window: int = DEFAULT_MA
    band_pct: float = DEFAULT_BAND   # S30 hysteresis band
    n_switches: int = 0
    starting_capital: float = STARTING_CAPITAL


def step(timer: TrendTimer, close: float, sma: Optional[float], date: str) -> float:
    """Advance one day and return the marked portfolio value.

    ``sma`` is BTC's ``ma_window`` SMA today, or None when history is insufficient
    (treated as not-bull → stay/flatten to cash). Hysteresis band (S30): when in cash,
    enter only if close > sma·(1+band); when invested, exit only if close < sma·(1−band).
    Inside the band the position is held — that buffer is what suppresses whipsaws.
    """
    if not timer.initialized:
        timer.initialized = True
        timer.init_date = date
        timer.cash = timer.starting_capital
        timer.shares = 0.0
        timer.invested = False

    if sma is None:
        bull = False
    elif timer.invested:
        bull = close > sma * (1.0 - timer.band_pct)   # stay until a band below
    else:
        bull = close > sma * (1.0 + timer.band_pct)   # enter only a band above
    if bull and not timer.invested:
        timer.shares = timer.cash * (1.0 - COST_PCT) / close
        timer.cash = 0.0
        timer.invested = True
        timer.n_switches += 1
    elif not bull and timer.invested:
        timer.cash = timer.shares * close * (1.0 - COST_PCT)
        timer.shares = 0.0
        timer.invested = False
        timer.n_switches += 1

    return value_of(timer, close)


def value_of(timer: TrendTimer, close: float) -> float:
    """Mark-to-market at ``close`` (cash + BTC held)."""
    return timer.cash + timer.shares * close


def serialize(timer: TrendTimer) -> dict[str, Any]:
    return asdict(timer)


def deserialize(data: dict[str, Any]) -> TrendTimer:
    if not data:
        return TrendTimer()
    fields = TrendTimer.__dataclass_fields__
    return TrendTimer(**{k: v for k, v in data.items() if k in fields})
