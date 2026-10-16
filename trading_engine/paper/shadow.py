"""Shadow variant portfolios (paper-only) run beside the frozen S18 Connors series.

Each shadow is a phantom book (own cash, positions, closed trades) persisted in
``journal["benchmarks"][<name>]`` like ``trend_timed_btc``. It starts at
STARTING_CAPITAL on the first day it runs and uses the same data, fees, slippage,
sizing, position cap and exit rules as the live path in run_daily.run(), except
for one variant expressed as ConnorsOverrides and/or the BTC SMA200 entry gate.
Shadows never read or write the live journal cash/positions.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from trading_engine.strategies.connors_swing import ConnorsOverrides, long_entry, long_exit

from .config import (
    STARTING_CAPITAL, POS_SIZE_PCT, MAX_CONCURRENT, COST_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS,
)


@dataclass(frozen=True)
class ShadowVariant:
    name: str
    overrides: ConnorsOverrides = ConnorsOverrides()
    btc_sma_gate: bool = False   # open new positions only when BTC close > SMA200


SHADOW_VARIANTS: tuple[ShadowVariant, ...] = (
    ShadowVariant("shadow_atr_stop", ConnorsOverrides(sl_pct=0.25, atr_stop_mult=3.0)),
    ShadowVariant("shadow_btc200", ConnorsOverrides(), btc_sma_gate=True),
    ShadowVariant("shadow_rsi15", ConnorsOverrides(rsi_threshold=15)),
    ShadowVariant("shadow_combo", ConnorsOverrides(rsi_threshold=15, sl_pct=0.25, atr_stop_mult=3.0),
                  btc_sma_gate=True),
)
SHADOW_VALUE_KEYS: tuple[str, ...] = tuple(f"{v.name}_value" for v in SHADOW_VARIANTS)


@dataclass
class ShadowBook:
    name: str
    initialized: bool = False
    init_date: Optional[str] = None
    cash: float = STARTING_CAPITAL
    starting_capital: float = STARTING_CAPITAL
    open_positions: list[dict] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)
    value: float = STARTING_CAPITAL


def step(book: ShadowBook, variant: ShadowVariant,
         snapshot: dict[str, tuple[float, pd.Series]], atr_today: dict[str, float],
         btc_close: Optional[float], btc_sma: Optional[float], today: str) -> float:
    """Advance one day (exits on close, then entries on close, then mark). Returns value."""
    if not book.initialized:
        book.initialized = True
        book.init_date = today
        book.cash = book.starting_capital
        book.open_positions, book.closed_trades = [], []
    ov = variant.overrides
    today_ts = pd.Timestamp(today)

    # 1. Exits — same order, fills, costs as run_daily.run().
    for pos in list(book.open_positions):
        if pos["symbol"] not in snapshot:
            continue
        close_today, ind_row = snapshot[pos["symbol"]]
        entry_date = pd.Timestamp(pos["entry_date"])
        reason = long_exit(close_today, ind_row, pos["entry_price"], entry_date, today_ts,
                           overrides=ov, entry_atr=pos.get("entry_atr"))
        if reason is None:
            continue
        exit_slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if reason == "SL" else 0.0)
        fill_price = close_today * (1.0 - exit_slip)
        exit_cost = fill_price * pos["shares"] * COST_PCT
        proceeds = fill_price * pos["shares"] - exit_cost
        pnl = proceeds - (pos["entry_price"] * pos["shares"] + pos["entry_cost"])
        book.closed_trades.append(dict(
            symbol=pos["symbol"], entry_date=pos["entry_date"], exit_date=today,
            entry_price=pos["entry_price"], exit_price=fill_price, shares=pos["shares"],
            pnl=pnl, pnl_pct=pnl / (pos["entry_price"] * pos["shares"]),
            hold_days=(today_ts - entry_date).days, reason=reason))
        book.open_positions = [p for p in book.open_positions if p["symbol"] != pos["symbol"]]
        book.cash += proceeds

    # 2. Entries.
    gate_open = (not variant.btc_sma_gate) or (
        btc_close is not None and btc_sma is not None and btc_close > btc_sma)
    open_syms = {p["symbol"] for p in book.open_positions}
    candidates = sorted(
        (sym for sym, (c, row) in snapshot.items()
         if sym not in open_syms and long_entry(c, row, use_adx_filter=True, overrides=ov)))
    if not gate_open:
        candidates = []
    capacity = MAX_CONCURRENT - len(book.open_positions)
    for sym in candidates[:capacity] if capacity > 0 else []:
        close_today = snapshot[sym][0]
        entry_fill = close_today * (1.0 + SLIPPAGE_BPS)
        pos_value = STARTING_CAPITAL * POS_SIZE_PCT
        if pos_value > book.cash:
            continue
        shares = pos_value / entry_fill
        entry_cost = entry_fill * shares * COST_PCT
        debit = entry_fill * shares + entry_cost
        if debit > book.cash:
            continue
        book.cash -= debit
        atr = atr_today.get(sym)
        book.open_positions.append(dict(
            symbol=sym, entry_date=today, entry_price=entry_fill, shares=shares,
            entry_cost=entry_cost,
            entry_atr=None if atr is None or pd.isna(atr) else float(atr)))

    # 3. Mark at today's close (entry-price fallback when a close is missing, as live).
    value = book.cash
    for pos in book.open_positions:
        c = snapshot[pos["symbol"]][0] if pos["symbol"] in snapshot else pos["entry_price"]
        value += c * pos["shares"]
    book.value = value
    return value


def serialize(book: ShadowBook) -> dict[str, Any]:
    return asdict(book)


def deserialize(data: dict[str, Any], name: str) -> ShadowBook:
    if not data:
        return ShadowBook(name=name)
    # Deep copy: journal.get_benchmark returns the stored dict itself, so a step that
    # fails half-way must not leave a half-applied book in the journal.
    fields = ShadowBook.__dataclass_fields__
    return ShadowBook(**{k: copy.deepcopy(v) for k, v in data.items() if k in fields})
