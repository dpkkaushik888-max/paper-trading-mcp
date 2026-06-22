"""L3 regime-aware strategy orchestrator (S23).

Pure decision logic: given a PortfolioView + per-symbol indicator snapshot +
(optional) regime, decide which strategies fire, resolve per-symbol conflicts
(first-come, priority A>B>C), allocate the shared pool under global + per-strategy
caps, and emit Orders. Owns no cash or persistence — the L2 leaf executes Orders.

Regime wiring (the "missing link") is built but DEFAULT OFF (regime_gating=False)
so the S21-locked unconditional-stack methodology is reproduced exactly. With
gating on, regime_map scales/gates per-strategy weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from trading_engine.regime.regime_filter import RegimeFilter, RegimeResult, RegimeState


@dataclass(frozen=True)
class Order:
    action: str          # "enter" | "exit"
    symbol: str
    strategy: str
    side: str            # "long" (S21 D1: long-only)
    size_pct: float      # fraction of leaf equity (post-weighting)
    sl_pct: float
    tp_pct: float
    reason: str
    confidence: float


@dataclass
class PortfolioView:
    cash: float
    equity: float
    positions: dict[str, dict] = field(default_factory=dict)
    # positions: symbol -> {strategy, entry_price, entry_date, shares, ...}


@dataclass
class RegimePolicy:
    """Per-strategy weight multiplier + open-gate derived from the regime."""
    weights: dict[str, float]
    allow_new: dict[str, bool]


# Suggested gating table — only used when regime_gating=True (a future spec's
# one-shot holdout). Long-only ⇒ mostly cash in bear; only deep-oversold range
# fades fire. Keyed by strategy .name.
DEFAULT_REGIME_MAP: dict[RegimeState, dict[str, float]] = {
    RegimeState.BULL:    {"A_connors": 1.0, "B_breakout": 1.5, "C_range": 0.5},
    RegimeState.NEUTRAL: {"A_connors": 1.0, "B_breakout": 0.5, "C_range": 1.0},
    RegimeState.BEAR:    {"A_connors": 0.0, "B_breakout": 0.0, "C_range": 1.0},
}


class StrategyOrchestrator:
    def __init__(
        self,
        strategies: list,
        regime_filter: Optional[RegimeFilter] = None,
        regime_map: Optional[dict] = None,
        regime_gating: bool = False,
        global_max_concurrent: int = 8,
        per_strategy_cap: int = 4,
        base_pos_size_pct: float = 0.12,
    ) -> None:
        self.strategies = strategies
        self.regime_filter = regime_filter
        self.regime_map = regime_map or DEFAULT_REGIME_MAP
        self.regime_gating = regime_gating
        self.global_max = global_max_concurrent
        self.per_strategy_cap = per_strategy_cap
        self.base_pos_size_pct = base_pos_size_pct

    def _ordered(self) -> list:
        return sorted(self.strategies, key=lambda s: getattr(s, "priority", 99))

    # ── regime → policy (the missing link) ───────────────────────────────
    def regime_policy(
        self, btc_df: Optional[pd.DataFrame], current_day
    ) -> tuple[Optional[RegimeResult], RegimePolicy]:
        rr: Optional[RegimeResult] = None
        if self.regime_filter is not None and btc_df is not None:
            try:
                rr = self.regime_filter.evaluate(btc_df, current_day=current_day)
            except Exception:
                rr = None

        names = [s.name for s in self.strategies]
        if not self.regime_gating:
            # Reproduces the unconditional S21 stack: all fire, weight 1.0.
            return rr, RegimePolicy({n: 1.0 for n in names}, {n: True for n in names})

        state = rr.state if rr else RegimeState.NEUTRAL
        base = self.regime_map.get(state, {})
        weights = {n: base.get(n, 1.0) for n in names}
        if rr and rr.is_caution:
            weights = {n: w * 0.5 for n, w in weights.items()}
        # regime_map is the sole per-strategy gate: weight 0 = blocked. We do NOT
        # also AND rr.allows_new_longs (a coarse "no longs in bear") — the map
        # already encodes which long strategies fire in each regime (e.g. range
        # fades are still allowed in bear).
        allow = {n: weights[n] > 0.0 for n in names}
        return rr, RegimePolicy(weights, allow)

    # ── exits (each position uses its owning strategy's exit) ─────────────
    def decide_exits(self, view: PortfolioView, snapshot: dict, current_day) -> list[Order]:
        by_name = {s.name: s for s in self.strategies}
        orders: list[Order] = []
        for sym, pos in view.positions.items():
            if sym not in snapshot:
                continue
            close, ind_row = snapshot[sym]
            strat = by_name[pos["strategy"]]
            reason = strat.exit_reason(close, ind_row, pos["entry_price"],
                                       pos["entry_date"], current_day)
            if reason:
                orders.append(Order("exit", sym, pos["strategy"], "long",
                                    0.0, 0.0, 0.0, reason, 1.0))
        return orders

    # ── entries (priority A>B>C, alpha within, shared-pool caps, D6) ──────
    def decide_entries(
        self, view: PortfolioView, snapshot: dict, policy: RegimePolicy, current_day
    ) -> list[Order]:
        orders: list[Order] = []
        held = set(view.positions)
        strat_open = {s.name: 0 for s in self.strategies}
        for pos in view.positions.values():
            strat_open[pos["strategy"]] = strat_open.get(pos["strategy"], 0) + 1
        n_open = len(view.positions)

        for strat in self._ordered():
            name = strat.name
            if not policy.allow_new.get(name, True):
                continue
            cands = sorted(
                sym for sym, (close, r) in snapshot.items()
                if sym not in held and strat.entry(close, r)
            )
            for sym in cands:
                if n_open >= self.global_max:
                    break
                if strat_open[name] >= self.per_strategy_cap:
                    break
                size = self.base_pos_size_pct * policy.weights.get(name, 1.0)
                orders.append(Order("enter", sym, name, "long", size,
                                    strat.sl_pct, 0.0, strat.reason, 1.0))
                held.add(sym)
                n_open += 1
                strat_open[name] += 1
        return orders

    # ── one daily step ────────────────────────────────────────────────────
    def step(
        self, view: PortfolioView, snapshot: dict,
        btc_df: Optional[pd.DataFrame], current_day,
    ) -> tuple[Optional[RegimeResult], list[Order]]:
        rr, policy = self.regime_policy(btc_df, current_day)
        exits = self.decide_exits(view, snapshot, current_day)
        entries = self.decide_entries(view, snapshot, policy, current_day)
        return rr, exits + entries
