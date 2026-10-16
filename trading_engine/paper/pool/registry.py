"""Declarative line registry (S35 §1). A param change is a new id; never edit a line in place."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

LIVE_ID = "live_s18"
MAX_ACTIVE = 10
# Lines registered on/before this date (the S34 shadows) are waived from the baseline rule by note.
BASELINE_WAIVER_UNTIL = "2026-10-16"
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


def _p(**kw) -> Mapping:
    return MappingProxyType(dict(kw))


_S34 = dict(family="connors", registered_on="2026-10-16", baseline=None,
            note="S34 shadow; full-history numbers in specs/S34",
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


def validate(lines: tuple[LineSpec, ...]) -> None:
    ids = [l.id for l in lines]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate line ids: {dupes}")
    if LIVE_ID in ids:
        raise ValueError(f"{LIVE_ID} is the live line and cannot be registered")
    active = [l for l in lines if l.status == "active"]
    if len(active) > MAX_ACTIVE:
        raise ValueError(f"{len(active)} active lines > MAX_ACTIVE={MAX_ACTIVE}")
    for l in active:
        if l.hypothesis is None or l.hypothesis.kind not in HYPOTHESIS_KINDS:
            raise ValueError(f"{l.id}: active line needs a primary hypothesis")
        if l.baseline is None and (l.registered_on > BASELINE_WAIVER_UNTIL or not l.note):
            raise ValueError(f"{l.id}: active line registered {l.registered_on} has no baseline")


def active_lines(lines: tuple[LineSpec, ...] = LINES) -> tuple[LineSpec, ...]:
    return tuple(l for l in lines if l.status == "active")


validate(LINES)
