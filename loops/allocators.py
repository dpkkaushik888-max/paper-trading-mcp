"""Deterministic fallback allocators (S22 D6).

A parent splits its capital_budget across children from their last Reports.
These run whenever no agent is wired (or the agent's output is rejected), so the
whole hierarchy works headless. A halted child gets zero weight (S22 D7) and its
budget is reallocated to siblings by renormalization.
"""

from __future__ import annotations

from typing import Optional

EPS = 0.05  # drawdown floor so a zero-DD child doesn't get infinite weight


def _equal(child_ids: list[str]) -> dict[str, float]:
    n = len(child_ids)
    return {cid: 1.0 / n for cid in child_ids} if n else {}


def confidence_weighted(children, prior_reports, mandate=None) -> dict[str, float]:
    """weight ∝ confidence × (1 + max(0, return)) / (max_drawdown + EPS)."""
    ids = [c.loop_id for c in children]
    scores: dict[str, float] = {}
    for cid in ids:
        r = prior_reports.get(cid)
        if r is None:
            scores[cid] = 1.0                       # no history → neutral prior
        elif r.halted:
            scores[cid] = 0.0                       # D7: halted → zero, reallocated
        else:
            scores[cid] = (
                max(0.0, r.confidence)
                * (1.0 + max(0.0, r.period_return))
                / (r.max_drawdown + EPS)
            )
    total = sum(scores.values())
    if total <= 0:
        return _equal(ids)
    return {cid: scores[cid] / total for cid in ids}


def risk_parity(children, prior_reports, mandate=None) -> dict[str, float]:
    """weight ∝ 1 / (max_drawdown + EPS); halted children excluded."""
    ids = [c.loop_id for c in children]
    inv: dict[str, float] = {}
    for cid in ids:
        r = prior_reports.get(cid)
        if r is not None and r.halted:
            inv[cid] = 0.0
        else:
            dd = r.max_drawdown if r is not None else 0.0
            inv[cid] = 1.0 / (dd + EPS)
    total = sum(inv.values())
    if total <= 0:
        return _equal(ids)
    return {cid: inv[cid] / total for cid in ids}
