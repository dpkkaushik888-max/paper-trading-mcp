"""Parent loop that composes children (S22).

decide() allocates the parent's capital_budget across children (via an agent if
wired, else a deterministic allocator); act() issues each child a fresh Mandate
and runs it; measure()/report() aggregate child Reports into a roll-up. A halted
child does NOT halt the parent (S22 D7) — the allocator simply gives it zero
weight next period.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .allocators import confidence_weighted
from .base import Loop
from .contracts import Cadence, Mandate, Report


class CompositeLoop(Loop):
    def __init__(
        self,
        loop_id: str,
        cadence: Cadence | str,
        children: list[Loop],
        state: Any = None,
        agent: Any = None,
        allocator: Optional[Callable] = None,
    ) -> None:
        super().__init__(loop_id, cadence, state=state, agent=agent, children=children)
        self.allocator = allocator or confidence_weighted
        self._child_reports: list[Report] = []

    def observe(self, period: str) -> dict:
        prior: dict[str, Report] = {}
        if self.state is not None:
            prior = self.state.last_reports([c.loop_id for c in self.children])
        return {"period": period, "prior_reports": prior}

    def decide(self, observation: dict) -> dict:
        prior = observation["prior_reports"]
        weights = None
        if self.agent is not None:
            # Agent picks among engine-enumerated children; may return None →
            # deterministic fallback (S22 D6).
            weights = self.agent.allocate(self.mandate, prior, self.children)
        if weights is None:
            weights = self.allocator(self.children, prior, self.mandate)
        return {"weights": weights, "period": observation["period"]}

    def act(self, decision: dict) -> dict:
        self._child_reports = []
        budget = self.mandate.capital_budget if self.mandate else 0.0
        weights = decision["weights"]
        for child in self.children:
            w = weights.get(child.loop_id, 0.0)
            child.set_mandate(Mandate(
                loop_id=child.loop_id, issued_by=self.loop_id,
                period=decision["period"], capital_budget=budget * w,
                risk_limits=self.mandate.risk_limits if self.mandate else {},
                constraints=self.mandate.constraints if self.mandate else {},
                horizon_days=self.mandate.horizon_days if self.mandate else 30,
                notes=f"weight={w:.3f} from {self.loop_id}",
            ))
            self._child_reports.append(child.run(decision["period"]))
        return {}

    def measure(self, period: str) -> dict:
        start = sum(r.starting_value for r in self._child_reports)
        end = sum(r.ending_value for r in self._child_reports)
        return {"start": start, "end": end}

    def report(self, measurement: dict) -> Report:
        start, end = measurement["start"], measurement["end"]
        return Report.from_values(
            self.loop_id,
            self.mandate.period if self.mandate else "",
            start, end,
            max_drawdown=max((r.max_drawdown for r in self._child_reports), default=0.0),
            realized_pnl=sum(r.realized_pnl for r in self._child_reports),
            n_trades=sum(r.n_trades for r in self._child_reports),
            confidence=(
                sum(r.confidence for r in self._child_reports) / len(self._child_reports)
                if self._child_reports else 0.0
            ),
            children=self._child_reports,
            diagnostics={"weights": {r.loop_id: r.starting_value / start if start else 0.0
                                     for r in self._child_reports}},
        )
