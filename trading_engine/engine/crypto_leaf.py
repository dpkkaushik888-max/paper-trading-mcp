"""L2 Crypto leaf engine (S23).

Owns capital + execution + accounting for the crypto asset-class loop. Runs the
L3 StrategyOrchestrator each day and produces a LeafResult. Execution math
(fixed-base sizing, honest costs, slippage, exit-before-entry ordering, end-of-run
close-out) mirrors scripts/sim_s21_window.py verbatim so the orchestrator path
reproduces that regression oracle exactly.

Stays in the execution plane: imports NOTHING from `loops/`. The L2 loop adapter
(loops/l2_crypto.py) converts a LeafResult into a control-plane Report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from trading_engine.engine.orchestrator import Order, PortfolioView, StrategyOrchestrator

# Honest cost model defaults (S21 D7) — overridable via Mandate.
COST_PCT = 0.0020
SLIPPAGE_BPS = 0.0005
SL_SLIPPAGE_BPS = 0.0010


@dataclass
class ClosedTrade:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    reason: str


@dataclass
class LeafResult:
    final_value: float
    return_pct: float
    sharpe: float
    max_dd: float
    n_trades: int
    wins: int
    losses: int
    total_costs: float
    trades: list[ClosedTrade] = field(default_factory=list)
    per_strategy: dict = field(default_factory=dict)
    regime_log: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    halted: bool = False
    halt_reason: Optional[str] = None


class CryptoLeaf:
    def __init__(
        self,
        orchestrator: StrategyOrchestrator,
        capital: float = 10_000.0,
        cost_pct: float = COST_PCT,
        slippage_bps: float = SLIPPAGE_BPS,
        sl_slippage_bps: float = SL_SLIPPAGE_BPS,
        max_drawdown_pct: float = 0.15,
    ) -> None:
        self.orch = orchestrator
        self.capital = capital            # fixed sizing base (matches oracle)
        self.cost_pct = cost_pct
        self.slippage_bps = slippage_bps
        self.sl_slippage_bps = sl_slippage_bps
        self.max_drawdown_pct = max_drawdown_pct
        self.cash = capital
        self.positions: dict[str, dict] = {}
        self.closed: list[ClosedTrade] = []
        self.total_costs = 0.0
        self.steps: list[dict] = []
        self.regime_log: list = []
        self._prev_value = capital
        self._peak = capital

    # ── view ──────────────────────────────────────────────────────────────
    def _view(self, snapshot: dict) -> PortfolioView:
        equity = self.cash + sum(
            snapshot[s][0] * p["shares"] for s, p in self.positions.items() if s in snapshot
        )
        return PortfolioView(cash=self.cash, equity=equity, positions=self.positions)

    # ── execution ──────────────────────────────────────────────────────────
    def _exec_exit(self, order: Order, close: float, day) -> None:
        pos = self.positions[order.symbol]
        slip = self.slippage_bps + (self.sl_slippage_bps if order.reason == "SL" else 0.0)
        fill = close * (1.0 - slip)
        ec = fill * pos["shares"] * self.cost_pct
        self.cash += fill * pos["shares"] - ec
        self.total_costs += ec
        pnl = (fill * pos["shares"] - ec) - (pos["entry_price"] * pos["shares"] + pos["entry_cost"])
        self.closed.append(ClosedTrade(
            symbol=order.symbol, strategy=pos["strategy"], entry_date=pos["entry_date"],
            exit_date=day, entry_price=pos["entry_price"], exit_price=fill,
            shares=pos["shares"], pnl=pnl, pnl_pct=pnl / (pos["entry_price"] * pos["shares"]),
            reason=order.reason,
        ))
        del self.positions[order.symbol]

    def _exec_enter(self, order: Order, close: float, day) -> None:
        fill = close * (1.0 + self.slippage_bps)
        pos_value = self.capital * order.size_pct
        if pos_value > self.cash:
            return
        shares = pos_value / fill
        ec = fill * shares * self.cost_pct
        debit = fill * shares + ec
        if debit > self.cash:
            return
        self.cash -= debit
        self.total_costs += ec
        self.positions[order.symbol] = {
            "strategy": order.strategy, "entry_price": fill, "entry_date": day,
            "shares": shares, "entry_cost": ec,
        }

    # ── one day: exits (free capital) THEN entries on the updated view ──────
    def run_day(self, snapshot: dict, btc_df: Optional[pd.DataFrame], day) -> None:
        rr, policy = self.orch.regime_policy(btc_df, day)
        for o in self.orch.decide_exits(self._view(snapshot), snapshot, day):
            self._exec_exit(o, snapshot[o.symbol][0], day)
        for o in self.orch.decide_entries(self._view(snapshot), snapshot, policy, day):
            self._exec_enter(o, snapshot[o.symbol][0], day)
        value = self.cash + sum(
            snapshot[s][0] * p["shares"] for s, p in self.positions.items() if s in snapshot
        )
        daily_ret = (value / self._prev_value - 1.0) if self._prev_value > 0 else 0.0
        self._prev_value = value
        self._peak = max(self._peak, value)
        self.steps.append({"date": day, "value": value, "cash": self.cash,
                           "n_open": len(self.positions), "daily_return": daily_ret})
        if rr is not None:
            self.regime_log.append({"date": day, "state": rr.state.value,
                                    "confidence": rr.confidence})

    def close_all(self, last_closes: dict, day) -> None:
        """End-of-run close-out at last close (slippage, no SL slip), reason END."""
        for sym in list(self.positions):
            if sym not in last_closes:
                continue
            pos = self.positions[sym]
            fill = last_closes[sym] * (1.0 - self.slippage_bps)
            ec = fill * pos["shares"] * self.cost_pct
            self.cash += fill * pos["shares"] - ec
            self.total_costs += ec
            pnl = (fill * pos["shares"] - ec) - (pos["entry_price"] * pos["shares"] + pos["entry_cost"])
            self.closed.append(ClosedTrade(
                symbol=sym, strategy=pos["strategy"], entry_date=pos["entry_date"],
                exit_date=day, entry_price=pos["entry_price"], exit_price=fill,
                shares=pos["shares"], pnl=pnl, pnl_pct=pnl / (pos["entry_price"] * pos["shares"]),
                reason="END",
            ))
            del self.positions[sym]

    # ── metrics roll-up ─────────────────────────────────────────────────────
    def result(self) -> LeafResult:
        final = self.cash  # after close_all, all in cash
        rets = np.array([s["daily_return"] for s in self.steps]) if self.steps else np.array([])
        sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.size and rets.std() > 0 else 0.0
        values = np.array([s["value"] for s in self.steps]) if self.steps else np.array([self.capital])
        peak = np.maximum.accumulate(values)
        max_dd = float(((peak - values) / peak).max()) if values.size else 0.0
        wins = sum(1 for t in self.closed if t.pnl > 0)
        losses = sum(1 for t in self.closed if t.pnl <= 0)
        per_strat: dict = {}
        for t in self.closed:
            d = per_strat.setdefault(t.strategy, {"trades": 0, "wins": 0, "net": 0.0})
            d["trades"] += 1
            d["wins"] += 1 if t.pnl > 0 else 0
            d["net"] += t.pnl
        return LeafResult(
            final_value=final, return_pct=final / self.capital - 1.0, sharpe=sharpe,
            max_dd=max_dd, n_trades=len(self.closed), wins=wins, losses=losses,
            total_costs=self.total_costs, trades=self.closed, per_strategy=per_strat,
            regime_log=self.regime_log, steps=self.steps,
        )
