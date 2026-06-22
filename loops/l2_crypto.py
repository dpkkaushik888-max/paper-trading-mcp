"""L2 Crypto asset-class loop (S23) — adapts CryptoLeaf to the Loop contract.

This is the control-plane wrapper: it accepts a Mandate, drives the
execution-plane CryptoLeaf over its data for the period, and converts the
LeafResult into a Report that flows up. The execution engine stays unaware of
the loop hierarchy; the dependency points one way (loops → trading_engine).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from trading_engine.engine.crypto_leaf import CryptoLeaf, LeafResult
from trading_engine.engine.indicators import build_indicators
from trading_engine.engine.orchestrator import StrategyOrchestrator

from .base import Loop
from .contracts import Cadence, Report


class CryptoAssetLoop(Loop):
    """Daily crypto leaf as an L2 loop.

    Data is injected (``bars`` = {symbol: OHLCV df}) so the loop is testable
    without network; the live runner (paper_v2) supplies freshly-fetched bars.
    """

    def __init__(
        self,
        loop_id: str = "L2.crypto",
        strategies: Optional[list] = None,
        bars: Optional[dict[str, pd.DataFrame]] = None,
        state: Any = None,
        agent: Any = None,
        regime_filter: Any = None,
        regime_gating: bool = False,
    ) -> None:
        super().__init__(loop_id, Cadence.DAILY, state=state, agent=agent)
        self._strategies = strategies or []
        self._bars = bars or {}
        self._regime_filter = regime_filter
        self._regime_gating = regime_gating
        self._leaf: Optional[CryptoLeaf] = None
        self._result: Optional[LeafResult] = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def observe(self, period: str) -> dict:
        ind = {sym: build_indicators(df) for sym, df in self._bars.items()}
        dates = sorted(set().union(*[df.index for df in self._bars.values()])) if self._bars else []
        return {"period": period, "bars": self._bars, "ind": ind, "dates": dates}

    def decide(self, observation: dict) -> dict:
        m = self.mandate
        orch = StrategyOrchestrator(
            strategies=self._strategies,
            regime_filter=self._regime_filter,
            regime_gating=self._regime_gating,
            global_max_concurrent=int(m.risk_limits.get("max_concurrent", 8)) if m else 8,
            per_strategy_cap=int(m.risk_limits.get("per_strategy_cap", 4)) if m else 4,
            base_pos_size_pct=float(m.risk_limits.get("pos_size_pct", 0.12)) if m else 0.12,
        )
        capital = float(m.capital_budget) if m else 10_000.0
        return {"orch": orch, "capital": capital, **observation}

    def act(self, decision: dict) -> dict:
        leaf = CryptoLeaf(decision["orch"], capital=decision["capital"])
        bars, ind, dates = decision["bars"], decision["ind"], decision["dates"]
        for day in dates:
            snap = {
                sym: (float(df["Close"].loc[day]), ind[sym].loc[day])
                for sym, df in bars.items() if day in df.index
            }
            if not snap:
                continue
            leaf.run_day(snap, btc_df=None, day=day)
        if dates:
            last = dates[-1]
            leaf.close_all(
                {sym: float(df["Close"].loc[last]) for sym, df in bars.items() if last in df.index},
                last,
            )
        self._leaf = leaf
        self._result = leaf.result()
        return {}

    def measure(self, period: str) -> dict:
        return {"result": self._result}

    def report(self, measurement: dict) -> Report:
        r: LeafResult = measurement["result"]
        capital = self._leaf.capital
        regime = self._leaf.regime_log[-1]["state"] if self._leaf and self._leaf.regime_log else None
        return Report(
            loop_id=self.loop_id,
            period=self.mandate.period if self.mandate else "",
            starting_value=capital,
            ending_value=r.final_value,
            period_return=r.return_pct,
            max_drawdown=r.max_dd,
            realized_pnl=sum(t.pnl for t in r.trades),
            n_trades=r.n_trades,
            regime=regime,
            confidence=min(1.0, max(0.0, r.sharpe / 3.0)),  # rough conviction proxy
            capital_utilization=max(s["n_open"] for s in r.steps) / 8.0 if r.steps else 0.0,
            halted=r.halted,
            halt_reason=r.halt_reason,
            trades=[{"symbol": t.symbol, "strategy": t.strategy, "pnl": t.pnl,
                     "reason": t.reason} for t in r.trades],
            diagnostics={"per_strategy": r.per_strategy, "sharpe": r.sharpe,
                         "wins": r.wins, "losses": r.losses},
        )
