"""Stub leaf loop — a placeholder for not-yet-built leaves (e.g. L2.equity).

Returns a canned Report so parent loops (L1/L0) can run end-to-end before every
real leaf exists. The mandate's capital flows through unchanged so roll-ups stay
arithmetically honest.
"""

from __future__ import annotations

from typing import Any

from .base import Loop
from .contracts import Cadence, Report


class StubLeaf(Loop):
    def __init__(
        self, loop_id: str, cadence: Cadence | str = Cadence.DAILY,
        period_return: float = 0.0, confidence: float = 0.5,
        max_drawdown: float = 0.0, regime: str | None = None,
        halted: bool = False, state: Any = None,
    ) -> None:
        super().__init__(loop_id, cadence, state=state)
        self._ret = period_return
        self._conf = confidence
        self._dd = max_drawdown
        self._regime = regime
        self._halted = halted

    def observe(self, period): return {"period": period}
    def decide(self, observation): return {}
    def act(self, decision): return {}
    def measure(self, period): return {}

    def report(self, measurement) -> Report:
        capital = self.mandate.capital_budget if self.mandate else 0.0
        period = self.mandate.period if self.mandate else ""
        return Report.from_values(
            self.loop_id, period, capital, capital * (1.0 + self._ret),
            max_drawdown=self._dd, confidence=self._conf, regime=self._regime,
            halted=self._halted, diagnostics={"stub": True},
        )
