"""Read-only LineDay for the live S18 line, derived from that day's decisions and fills.

Never steps or writes live state: it only reads the values run_daily already computed.
"""

from __future__ import annotations

from typing import Optional

from .book import MAX_BLOCKED_SYMS

_BLOCK_REASON = {"max_concurrent": "cap", "insufficient_cash": "cash", "debit>cash": "cash"}


def live_line_day(decisions: list[dict], fills: list[dict], portfolio_value: float, cash: float,
                  n_open: int, strengths: Optional[dict[str, float]] = None) -> dict:
    strengths = strengths or {}
    blocked = dict(cap=0, gate=0, cash=0, data=0)
    blocked_syms = []
    entries = exits = 0
    for d in decisions:
        if d["action"] == "enter":
            entries += 1
        elif d["action"] == "exit":
            exits += 1
        elif d["action"] == "blocked" and d["reason"] in _BLOCK_REASON:
            blocked[_BLOCK_REASON[d["reason"]]] += 1
            blocked_syms.append([d["symbol"], strengths.get(d["symbol"])])
    invested = portfolio_value - cash
    return dict(value=portfolio_value, n_open=n_open,
                exposure=invested / portfolio_value if portfolio_value > 0 else 0.0,
                signals=entries + blocked["cap"] + blocked["cash"], entries=entries, exits=exits,
                blocked=blocked, blocked_syms=blocked_syms[:MAX_BLOCKED_SYMS],
                n_fills=len(fills))
