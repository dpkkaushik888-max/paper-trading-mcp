"""Regime-selector tracker for the live paper-forward (S32).

A phantom portfolio that switches strategy by BTC regime, tracked alongside the
others: **bull → hold BTC (trend-timed / S30); bear → mirror Connors (S18)**. The
regime uses BTC's 200-day SMA with the S30 ±2% hysteresis band. In the bear leg it
compounds the *actual* Connors portfolio's daily return; in the bull leg it holds BTC.
This makes "let the agent pick S30 or Connors by regime" a live, auditable series.

Pure + deterministic: ``step`` takes today's BTC close/SMA and Connors' daily return.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .config import COST_PCT, STARTING_CAPITAL


@dataclass
class RegimeSelector:
    name: str = "regime_selector"
    initialized: bool = False
    init_date: Optional[str] = None
    mode: str = "bear"               # "bull" = hold BTC | "bear" = track Connors
    shares: float = 0.0              # BTC shares when in the bull leg
    value: float = STARTING_CAPITAL
    ma_window: int = 200
    band_pct: float = 0.02
    n_switches: int = 0
    starting_capital: float = STARTING_CAPITAL


def step(sel: RegimeSelector, btc_close: float, btc_sma: Optional[float],
         connors_daily_return: float, date: str) -> float:
    """Advance one day and return the selector's portfolio value.

    bull (BTC > SMA·(1±band), hysteretic) → hold BTC; bear → compound Connors' return.
    A regime flip pays an honest 20 bps switch cost.
    """
    if not sel.initialized:
        sel.initialized = True
        sel.init_date = date
        sel.value = sel.starting_capital
        sel.mode = "bear"
        sel.shares = 0.0

    if btc_sma is None:
        bull = False
    elif sel.mode == "bull":
        bull = btc_close > btc_sma * (1.0 - sel.band_pct)
    else:
        bull = btc_close > btc_sma * (1.0 + sel.band_pct)

    if bull and sel.mode != "bull":                 # bear → bull: deploy into BTC
        sel.shares = sel.value * (1.0 - COST_PCT) / btc_close
        sel.mode = "bull"; sel.n_switches += 1
    elif not bull and sel.mode == "bull":           # bull → bear: exit BTC, resume Connors
        sel.value = sel.shares * btc_close * (1.0 - COST_PCT)
        sel.shares = 0.0; sel.mode = "bear"; sel.n_switches += 1

    if sel.mode == "bull":
        sel.value = sel.shares * btc_close
    else:
        sel.value = sel.value * (1.0 + (connors_daily_return or 0.0))
    return sel.value


def serialize(sel: RegimeSelector) -> dict[str, Any]:
    return asdict(sel)


def deserialize(data: dict[str, Any]) -> RegimeSelector:
    if not data:
        return RegimeSelector()
    fields = RegimeSelector.__dataclass_fields__
    return RegimeSelector(**{k: v for k, v in data.items() if k in fields})
