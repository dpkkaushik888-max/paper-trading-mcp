"""Passive buy-and-hold benchmarks for S18.

Two phantom portfolios to contextualize Connors' performance:

  BH_BTC     : 100% BTC bought at initialization, held until day 90.
  BH_BASKET  : Equal-weight across the full 20-coin universe, held.

Both pay realistic entry costs (20 bps) so the comparison is apples-to-apples
with Connors' honest cost model. No rebalancing — true passive.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from .config import STARTING_CAPITAL, COST_PCT, UNIVERSE


BH_BTC_SYMBOL = "BTCUSDT"


@dataclass
class BenchmarkPosition:
    """Single-symbol B&H position."""
    symbol: str
    shares: float
    entry_price: float       # actual fill (post entry cost)
    allocated: float         # dollars allocated to this symbol


@dataclass
class Benchmark:
    name: str                                    # "bh_btc" | "bh_basket"
    initialized: bool = False
    init_date: str | None = None
    positions: list[BenchmarkPosition] = field(default_factory=list)
    # Book-keeping
    starting_capital: float = STARTING_CAPITAL


def init_bh_btc(todays_closes: dict[str, float], date: str) -> Benchmark:
    """Allocate 100% of $10k to BTC at today's close minus 20 bps entry cost."""
    if BH_BTC_SYMBOL not in todays_closes:
        raise ValueError("BTCUSDT close missing — cannot initialize BH_BTC")
    px = todays_closes[BH_BTC_SYMBOL]
    capital = STARTING_CAPITAL
    # Entry cost: buy at close, pay 20 bps commission → effective shares = capital * (1 - COST_PCT) / px
    shares = (capital * (1.0 - COST_PCT)) / px
    return Benchmark(
        name="bh_btc",
        initialized=True,
        init_date=date,
        positions=[BenchmarkPosition(
            symbol=BH_BTC_SYMBOL, shares=shares, entry_price=px, allocated=capital,
        )],
    )


def init_bh_basket(todays_closes: dict[str, float], date: str) -> Benchmark:
    """Equal-weight across the 20-coin universe at today's closes."""
    available = [s for s in UNIVERSE if s in todays_closes]
    if len(available) < len(UNIVERSE):
        missing = [s for s in UNIVERSE if s not in available]
        # Don't fail — just skip missing symbols (they get zero allocation).
        # This is pre-committed behavior so a flaky fetch doesn't block init.
        print(f"[BH_BASKET] WARNING: missing {missing} at init; allocating zero")
    per_coin = STARTING_CAPITAL / len(UNIVERSE)   # equal $500 each
    positions = []
    for sym in available:
        px = todays_closes[sym]
        shares = (per_coin * (1.0 - COST_PCT)) / px
        positions.append(BenchmarkPosition(
            symbol=sym, shares=shares, entry_price=px, allocated=per_coin,
        ))
    return Benchmark(
        name="bh_basket",
        initialized=True,
        init_date=date,
        positions=positions,
    )


def mark_benchmark(bench: Benchmark, todays_closes: dict[str, float]) -> float:
    """Return current portfolio value of a benchmark at today's closes.

    If a symbol's close is missing, fall back to its entry price (stale mark —
    better than crashing). This should be rare with Binance's 99%+ uptime.
    """
    if not bench.initialized:
        return bench.starting_capital
    value = 0.0
    for p in bench.positions:
        px = todays_closes.get(p.symbol, p.entry_price)
        value += p.shares * px
    # Unallocated cash (for basket if symbols were missing at init)
    allocated_total = sum(p.allocated for p in bench.positions)
    value += (bench.starting_capital - allocated_total)
    return value


def serialize(bench: Benchmark) -> dict[str, Any]:
    return {
        "name": bench.name,
        "initialized": bench.initialized,
        "init_date": bench.init_date,
        "starting_capital": bench.starting_capital,
        "positions": [asdict(p) for p in bench.positions],
    }


def deserialize(data: dict[str, Any]) -> Benchmark:
    return Benchmark(
        name=data["name"],
        initialized=data.get("initialized", False),
        init_date=data.get("init_date"),
        starting_capital=data.get("starting_capital", STARTING_CAPITAL),
        positions=[BenchmarkPosition(**p) for p in data.get("positions", [])],
    )
