"""Persistent journal for S18 paper-forward.

Journal is a single JSON file committed to the repo each day. Schema is
versioned (version field) so future migrations are explicit.

Every mutation goes through the Journal class — direct dict access is not
supported so we can enforce invariants (cash never negative, positions
match cash debits, etc).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    JOURNAL_PATH,
    STARTING_CAPITAL,
    MAX_CONCURRENT,
    POS_SIZE_PCT,
    UNIVERSE,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass
class OpenPosition:
    symbol: str
    entry_date: str          # ISO date
    entry_price: float       # fill price (post-slippage)
    shares: float
    entry_cost: float        # commission paid


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    hold_days: int
    reason: str              # "MR_EXIT" | "MAX_HOLD" | "SL" | "END"


@dataclass
class Fill:
    symbol: str
    side: str                # "BUY" | "SELL"
    modeled_price: float     # what backtest would have used (close)
    fill_price: float        # after slippage
    shares: float
    cost: float


@dataclass
class Decision:
    symbol: str
    action: str              # "enter" | "exit" | "hold" | "blocked"
    reason: str


@dataclass
class DaySnapshot:
    date: str
    portfolio_value: float
    cash: float
    n_open: int
    closes: dict[str, float]
    decisions: list[dict]
    fills: list[dict]
    skipped: bool = False    # True if cron failed / bars unavailable


class Journal:
    """Stateful accessor for `state/journal.json`."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # ── Load / save ──────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path = JOURNAL_PATH) -> "Journal":
        p = Path(path)
        if not p.exists():
            log.info("No journal at %s — initializing fresh", p)
            return cls.init_fresh()
        with p.open() as f:
            return cls(json.load(f))

    def save(self, path: str | Path = JOURNAL_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self._data, f, indent=2, sort_keys=False)

    @classmethod
    def init_fresh(cls) -> "Journal":
        return cls({
            "version": SCHEMA_VERSION,
            "start_date": datetime.now(timezone.utc).date().isoformat(),
            "starting_capital": STARTING_CAPITAL,
            "cash": STARTING_CAPITAL,
            "config": {
                "max_concurrent": MAX_CONCURRENT,
                "pos_size_pct": POS_SIZE_PCT,
                "universe": list(UNIVERSE),
                "rules": "connors_swing_v1",
            },
            "days": [],
            "open_positions": [],
            "closed_trades": [],
        })

    # ── Accessors ────────────────────────────────────────────────────────
    @property
    def cash(self) -> float:
        return float(self._data["cash"])

    @property
    def start_date(self) -> str:
        return self._data["start_date"]

    @property
    def open_positions(self) -> list[OpenPosition]:
        return [OpenPosition(**p) for p in self._data["open_positions"]]

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return [ClosedTrade(**t) for t in self._data["closed_trades"]]

    @property
    def days(self) -> list[dict]:
        return self._data["days"]

    @property
    def starting_capital(self) -> float:
        return float(self._data["starting_capital"])

    def days_elapsed(self) -> int:
        return len(self._data["days"])

    def open_symbols(self) -> set[str]:
        return {p["symbol"] for p in self._data["open_positions"]}

    # ── Mutations ────────────────────────────────────────────────────────
    def add_position(self, pos: OpenPosition, debit: float) -> None:
        if debit > self._data["cash"] + 1e-6:
            raise ValueError(f"Debit {debit:.2f} exceeds cash {self._data['cash']:.2f}")
        self._data["cash"] -= debit
        self._data["open_positions"].append(asdict(pos))

    def close_position(self, symbol: str, trade: ClosedTrade, proceeds: float) -> None:
        self._data["open_positions"] = [
            p for p in self._data["open_positions"] if p["symbol"] != symbol
        ]
        self._data["cash"] += proceeds
        self._data["closed_trades"].append(asdict(trade))

    def append_day(self, snap: DaySnapshot) -> None:
        # Idempotency: if today already recorded, replace it (allows re-run).
        existing_idx = next(
            (i for i, d in enumerate(self._data["days"]) if d["date"] == snap.date),
            None,
        )
        if existing_idx is not None:
            self._data["days"][existing_idx] = asdict(snap)
            log.info("Replaced existing day %s in journal (re-run)", snap.date)
        else:
            self._data["days"].append(asdict(snap))

    # ── Derived metrics ──────────────────────────────────────────────────
    def last_portfolio_value(self) -> float:
        if not self._data["days"]:
            return float(self._data["starting_capital"])
        return float(self._data["days"][-1]["portfolio_value"])

    def max_drawdown(self) -> float:
        values = [d["portfolio_value"] for d in self._data["days"]]
        if not values:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def consecutive_losses(self) -> int:
        count = 0
        for t in reversed(self._data["closed_trades"]):
            if t["pnl"] < 0:
                count += 1
            else:
                break
        return count

    def days_since_last_trade(self) -> int:
        """Days elapsed since the last entry or exit. 0 if trade happened today."""
        if not self._data["days"]:
            return 0
        last_trade_date = None
        if self._data["closed_trades"]:
            last_trade_date = self._data["closed_trades"][-1]["exit_date"]
        if self._data["open_positions"]:
            entry_dates = [p["entry_date"] for p in self._data["open_positions"]]
            latest_entry = max(entry_dates)
            if last_trade_date is None or latest_entry > last_trade_date:
                last_trade_date = latest_entry
        if last_trade_date is None:
            # No trades ever — count days since start.
            return self.days_elapsed()
        today = self._data["days"][-1]["date"]
        return (
            datetime.fromisoformat(today).date()
            - datetime.fromisoformat(last_trade_date).date()
        ).days

    # ── Raw access ───────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return self._data
