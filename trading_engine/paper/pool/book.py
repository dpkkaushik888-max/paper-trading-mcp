"""Generalised paper book for pool lines (S35 §2), generalised from the S34 shadow step.

Order per day: exits on close, entries on close (cap, ``continue`` on cash shortfall as live),
mark at close with entry-price fallback. Fees, slippage and notional come from paper/config.py.
A book is its own phantom portfolio in ``journal.benchmarks[<id>]``; the live line is never stepped.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

import pandas as pd

from ..config import COST_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS, STARTING_CAPITAL
from .registry import LIVE_ID, LineSpec

MAX_BLOCKED_SYMS = 10


@dataclass
class Signal:
    strength: Optional[float]
    meta: dict = field(default_factory=dict)


class LineStrategy(Protocol):
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def gate(self, ctx) -> bool: ...
    def entry(self, sym: str, close: float, ind_row: pd.Series, ctx) -> Optional[Signal]: ...
    def exit(self, pos: dict, close: float, ind_row: pd.Series, ctx, today: str) -> Optional[str]: ...
    def rank_key(self, sym: str, sig: Signal) -> tuple: ...


@dataclass
class Book:
    name: str
    initialized: bool = False
    init_date: Optional[str] = None
    cash: float = STARTING_CAPITAL
    starting_capital: float = STARTING_CAPITAL
    open_positions: list[dict] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)
    value: float = STARTING_CAPITAL


def serialize(book: Book) -> dict[str, Any]:
    return asdict(book)


def deserialize(data: dict[str, Any], name: str) -> Book:
    if not data:
        return Book(name=name)
    # Deep copy: a step failing half-way must not leave a half-applied book in the journal.
    known = Book.__dataclass_fields__
    return Book(**{k: copy.deepcopy(v) for k, v in data.items() if k in known})


def _num(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


def _ctx_fields(ctx) -> dict:
    return {k: getattr(ctx, k, None) for k in ("bar_date", "dow", "btc_trend", "btc_vol20", "btc_vol20_bucket",
                                       "basket_ret20", "basket_ret20_bucket")}


def step(book: Book, spec: LineSpec, strat: LineStrategy,
         snapshot: dict[str, tuple[float, pd.Series]], ctx, today: str) -> dict:
    """Advance one day. Returns the LineDay dict (S35 §3); ``book`` is mutated in place."""
    if spec.id == LIVE_ID:
        raise ValueError("pool book.step refuses the live line")
    if not book.initialized:
        book.initialized = True
        book.init_date = today
        book.cash = book.starting_capital
        book.open_positions, book.closed_trades = [], []
    bp = spec.book
    today_ts = pd.Timestamp(today)
    flags = getattr(ctx, "data_flags", None) or {}
    flagged = set(flags.get("stale", [])) | set(flags.get("gap200", []))
    day = dict(value=0.0, n_open=0, exposure=0.0, signals=0, entries=0, exits=0,
               blocked=dict(cap=0, gate=0, cash=0, data=0), blocked_syms=[])

    prepare = getattr(strat, "prepare", None)   # optional: cross-sectional families rank the day first
    if prepare is not None:
        prepare(snapshot, ctx, today)

    # 1. Exits (data guard never blocks an exit).
    for pos in list(book.open_positions):
        if pos["symbol"] not in snapshot:
            continue
        close_today, ind_row = snapshot[pos["symbol"]]
        reason = None
        if bp.catastrophe_sl is not None and close_today <= pos["entry_price"] * (1.0 - bp.catastrophe_sl):
            reason = "SL"
        if reason is None:
            reason = strat.exit(pos, close_today, ind_row, ctx, today)
        if reason is None:
            continue
        exit_slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if reason == "SL" else 0.0)
        fill_price = close_today * (1.0 - exit_slip)
        exit_cost = fill_price * pos["shares"] * COST_PCT
        proceeds = fill_price * pos["shares"] - exit_cost
        pnl = proceeds - (pos["entry_price"] * pos["shares"] + pos["entry_cost"])
        lo = min(pos.get("min_close", pos["entry_price"]), close_today)
        hi = max(pos.get("max_close", pos["entry_price"]), close_today)
        hold = (today_ts - pd.Timestamp(pos["entry_date"])).days
        book.closed_trades.append(dict(
            symbol=pos["symbol"], entry_date=pos["entry_date"], exit_date=today,
            entry_price=pos["entry_price"], exit_price=fill_price, shares=pos["shares"],
            pnl=pnl, pnl_pct=pnl / (pos["entry_price"] * pos["shares"]),
            hold_days=hold, reason=reason,
            mae_pct=lo / pos["entry_price"] - 1.0, mfe_pct=hi / pos["entry_price"] - 1.0,
            entry_ctx=pos.get("entry_ctx"), exit_ctx={**_ctx_fields(ctx), "reason": reason, "hold_days": hold}))
        book.open_positions = [p for p in book.open_positions if p["symbol"] != pos["symbol"]]
        book.cash += proceeds
        day["exits"] += 1

    # 2. Entries. A closed gate / data guard still counts the signals that would have fired.
    gate_open = strat.gate(ctx)
    open_syms = {p["symbol"] for p in book.open_positions}
    sigs = []
    for sym, (c, row) in snapshot.items():
        if sym in open_syms:
            continue
        sig = strat.entry(sym, c, row, ctx)
        if sig is not None:
            sigs.append((sym, sig))
    sigs.sort(key=lambda s: strat.rank_key(*s))
    day["signals"] = len(sigs)
    blocked_syms = []
    candidates = []
    for sym, sig in sigs:
        if bp.data_guard and sym in flagged:
            day["blocked"]["data"] += 1
            blocked_syms.append([sym, sig.strength])
        elif not gate_open:
            day["blocked"]["gate"] += 1
            blocked_syms.append([sym, sig.strength])
        else:
            candidates.append((sym, sig))
    capacity = bp.max_concurrent - len(book.open_positions)
    taken = candidates[:capacity] if capacity > 0 else []
    for rank, (sym, sig) in enumerate(taken):
        close_today, row = snapshot[sym]
        entry_fill = close_today * (1.0 + SLIPPAGE_BPS)
        pos_value = STARTING_CAPITAL * bp.pos_size_pct
        shares = pos_value / entry_fill
        entry_cost = entry_fill * shares * COST_PCT
        debit = entry_fill * shares + entry_cost
        if pos_value > book.cash or debit > book.cash:
            day["blocked"]["cash"] += 1
            blocked_syms.append([sym, sig.strength])
            continue
        book.cash -= debit
        sma200 = _num(row.get("sma_200"))
        book.open_positions.append(dict(
            symbol=sym, entry_date=today, entry_price=entry_fill, shares=shares,
            entry_cost=entry_cost, entry_atr=sig.meta.get("entry_atr"),
            min_close=entry_fill, max_close=entry_fill,
            entry_ctx={**_ctx_fields(ctx), "symbol": sym, "strength": sig.strength,
                       "adx": _num(row.get("adx_14")),
                       "dist_sma200_pct": (close_today / sma200 - 1.0) if sma200 else None,
                       "n_signals_today": len(sigs), "rank_today": rank,
                       "data_flags_symbol": sorted(k for k, v in flags.items() if sym in v)}))
        day["entries"] += 1
    for sym, sig in candidates[len(taken):]:
        day["blocked"]["cap"] += 1
        blocked_syms.append([sym, sig.strength])
    day["blocked_syms"] = blocked_syms[:MAX_BLOCKED_SYMS]

    # 3. Mark at today's close (entry-price fallback when a close is missing, as live).
    value = book.cash
    for pos in book.open_positions:
        c = snapshot[pos["symbol"]][0] if pos["symbol"] in snapshot else pos["entry_price"]
        if pos["symbol"] in snapshot and pos["entry_date"] != today:
            pos["min_close"] = min(pos.get("min_close", pos["entry_price"]), c)
            pos["max_close"] = max(pos.get("max_close", pos["entry_price"]), c)
        value += c * pos["shares"]
    book.value = value
    day.update(value=value, n_open=len(book.open_positions),
               exposure=(value - book.cash) / value if value > 0 else 0.0)
    return day
