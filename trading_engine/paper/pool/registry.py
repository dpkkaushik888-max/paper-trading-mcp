"""Declarative line registry (S35 §1). A param change is a new id; never edit a line in place."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from types import MappingProxyType
from typing import Mapping, Optional

LIVE_ID = "live_s18"
MAX_ACTIVE = 10
# journal.benchmarks keys written by run_daily for the live benchmarks (mirrored; a line id may never shadow one).
LIVE_BENCHMARK_KEYS = ("bh_btc", "bh_basket", "trend_timed_btc", "regime_selector")
HYPOTHESIS_KINDS = ("beats_live", "reduces_dd_in", "earns_when_live_idle")


@dataclass(frozen=True)
class BookParams:
    pos_size_pct: float = 0.15
    max_concurrent: int = 6
    catastrophe_sl: Optional[float] = None   # close <= entry * (1 - x) -> "SL", applied by the book
    data_guard: bool = True                  # skip ENTRIES in stale / gap200 symbols


@dataclass(frozen=True)
class Hypothesis:
    kind: str                     # beats_live | reduces_dd_in | earns_when_live_idle
    cond: Optional[str] = None    # e.g. "below200" for reduces_dd_in


@dataclass(frozen=True)
class LineSpec:
    id: str
    family: str
    params: Mapping
    book: BookParams = BookParams()
    registered_on: str = ""
    status: str = "active"        # active | retired | promoted
    hypothesis: Optional[Hypothesis] = None
    baseline: Optional[str] = None   # state/pool_baselines/<id>.json
    retired_on: Optional[str] = None
    note: str = ""
    baseline_waiver: Optional[str] = None   # explicit reason an active line may run without a baseline

    def __post_init__(self):
        # Immutable params even when a plain dict is passed.
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


def _p(**kw) -> Mapping:
    return MappingProxyType(dict(kw))


_S34 = dict(family="connors", registered_on="2026-10-16", baseline=None,
            baseline_waiver="S34 shadow registered before the pool; full-history numbers in specs/S34",
            # S34 shadows keep live data handling (no data guard) so their series is unchanged.
            book=BookParams(data_guard=False))
BEATS_LIVE = Hypothesis("beats_live")

LINES: tuple[LineSpec, ...] = (
    LineSpec("shadow_atr_stop", params=_p(sl_pct=0.25, atr_stop_mult=3.0), hypothesis=BEATS_LIVE, **_S34),
    LineSpec("shadow_btc200", params=_p(btc_sma_gate=True),
             hypothesis=Hypothesis("reduces_dd_in", "below200"), **_S34),
    LineSpec("shadow_rsi15", params=_p(rsi_threshold=15), hypothesis=BEATS_LIVE, **_S34),
    LineSpec("shadow_combo", params=_p(rsi_threshold=15, sl_pct=0.25, atr_stop_mult=3.0, btc_sma_gate=True),
             hypothesis=BEATS_LIVE, **_S34),
)


def _value_fields() -> set[str]:
    from ..journal import DaySnapshot   # lazy: avoids an import cycle
    return {f.name for f in fields(DaySnapshot) if f.name.endswith("_value")}


def _is_iso_date(s) -> bool:
    try:
        return isinstance(s, str) and date.fromisoformat(s).isoformat() == s
    except ValueError:
        return False


def validate(lines: tuple[LineSpec, ...]) -> None:
    ids = [l.id for l in lines]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate line ids: {dupes}")
    value_fields = _value_fields()
    for l in lines:
        if l.id == LIVE_ID:
            raise ValueError(f"{LIVE_ID} is the live line and cannot be registered")
        if l.id in LIVE_BENCHMARK_KEYS:
            raise ValueError(f"{l.id}: reserved live benchmark key")
        # A line's id may not be, or write over, a DaySnapshot *_value field (S34 shadow_* keys are declared for them).
        if l.id in value_fields or (f"{l.id}_value" in value_fields and not l.id.startswith("shadow_")):
            raise ValueError(f"{l.id}: collides with a DaySnapshot *_value field")
        if not _is_iso_date(l.registered_on):
            raise ValueError(f"{l.id}: registered_on {l.registered_on!r} is not an ISO date")
    active = [l for l in lines if l.status == "active"]
    if len(active) > MAX_ACTIVE:
        raise ValueError(f"{len(active)} active lines > MAX_ACTIVE={MAX_ACTIVE}")
    for l in active:
        if l.hypothesis is None or l.hypothesis.kind not in HYPOTHESIS_KINDS:
            raise ValueError(f"{l.id}: active line needs a primary hypothesis")
        if l.baseline is None and not l.baseline_waiver:
            raise ValueError(f"{l.id}: active line registered {l.registered_on} has no baseline")


def active_lines(lines: tuple[LineSpec, ...] = LINES) -> tuple[LineSpec, ...]:
    return tuple(l for l in lines if l.status == "active")


validate(LINES)
