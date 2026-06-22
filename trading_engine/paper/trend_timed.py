"""Trend-timed BTC tracker for the S18/S29 paper-forward.

The one robust edge from the S25–S28 study: hold BTC while it is above a long moving
average, move to cash when it drops below. Tracked as a phantom portfolio alongside
``bh_btc``/``bh_basket`` — invested through the entire bull (owner's instruction),
cash only when the trend breaks. Honest 20 bps cost on each regime switch.

Pure + deterministic: ``step`` takes today's close and SMA and advances the state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .config import COST_PCT, STARTING_CAPITAL

DEFAULT_MA = 200


@dataclass
class TrendTimer:
    name: str = "trend_timed_btc"
    initialized: bool = False
    init_date: Optional[str] = None
    invested: bool = False           # currently holding BTC?
    shares: float = 0.0
    cash: float = STARTING_CAPITAL
    ma_window: int = DEFAULT_MA
    n_switches: int = 0
    starting_capital: float = STARTING_CAPITAL


def step(timer: TrendTimer, close: float, sma: Optional[float], date: str) -> float:
    """Advance one day and return the marked portfolio value.

    ``sma`` is BTC's ``ma_window`` SMA today, or None when history is insufficient
    (treated as not-bull → stay/flatten to cash). Bull = close > sma → fully invested.
    """
    if not timer.initialized:
        timer.initialized = True
        timer.init_date = date
        timer.cash = timer.starting_capital
        timer.shares = 0.0
        timer.invested = False

    bull = sma is not None and close > sma
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
