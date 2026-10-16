"""S34 shadow variants: compatibility shim over the strategy pool (S35 C1).

The four shadows are registered in ``paper/pool/registry.py`` with unchanged ids and journal
keys; ``step`` delegates to ``pool.book.step``. Kept so S34 tests and build_dashboard work unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Optional

import pandas as pd

from trading_engine.strategies.connors_swing import ConnorsOverrides

from .pool import book as _book
from .pool.context import DayContext
from .pool.families.connors import ATR_COL, ConnorsStrategy
from .pool.registry import LINES, BookParams, Hypothesis, LineSpec

ShadowBook = _book.Book
serialize = _book.serialize
deserialize = _book.deserialize
_OV_FIELDS = tuple(f.name for f in fields(ConnorsOverrides))


@dataclass(frozen=True)
class ShadowVariant:
    name: str
    overrides: ConnorsOverrides = ConnorsOverrides()
    btc_sma_gate: bool = False   # open new positions only when BTC close > SMA200

    def to_spec(self) -> LineSpec:
        return LineSpec(self.name, "connors",
                        MappingProxyType({**asdict(self.overrides), "btc_sma_gate": self.btc_sma_gate}),
                        book=BookParams(data_guard=False), registered_on="2026-10-16",
                        hypothesis=Hypothesis("beats_live"), note="S34 shim")


SHADOW_VARIANTS: tuple[ShadowVariant, ...] = tuple(
    ShadowVariant(s.id, ConnorsOverrides(**{k: s.params[k] for k in _OV_FIELDS if k in s.params}),
                  bool(s.params.get("btc_sma_gate", False)))
    for s in LINES if s.family == "connors" and s.id.startswith("shadow_"))
SHADOW_VALUE_KEYS: tuple[str, ...] = tuple(f"{v.name}_value" for v in SHADOW_VARIANTS)


def step(book: ShadowBook, variant: ShadowVariant,
         snapshot: dict[str, tuple[float, pd.Series]], atr_today: dict[str, float],
         btc_close: Optional[float], btc_sma: Optional[float], today: str) -> float:
    """S34 signature: advance one day and return the value (delegates to pool.book.step)."""
    spec = variant.to_spec()
    snap = {}
    for sym, (c, row) in snapshot.items():
        row = row.copy()
        row[ATR_COL] = atr_today.get(sym)
        snap[sym] = (c, row)
    ctx = DayContext(btc_close=btc_close, btc_sma200=btc_sma)
    return _book.step(book, spec, ConnorsStrategy(spec.params), snap, ctx, today)["value"]
