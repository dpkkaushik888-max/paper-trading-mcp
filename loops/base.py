"""The Loop contract (S22 D2).

Every loop implements the lifecycle observe → decide → act → measure → report,
orchestrated by the ``run()`` template method. A parent injects a Mandate via
``set_mandate``; ``run()`` returns a Report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .contracts import Cadence, Mandate, Report


class Loop(ABC):
    def __init__(
        self,
        loop_id: str,
        cadence: Cadence | str,
        state: Any = None,            # LoopState | None
        agent: Any = None,            # AgentClient | None (deterministic if None)
        children: Optional[list["Loop"]] = None,
    ) -> None:
        self.loop_id = loop_id
        self.cadence = cadence if isinstance(cadence, Cadence) else Cadence(cadence)
        self.state = state
        self.agent = agent
        self.children: list["Loop"] = children or []
        self.mandate: Optional[Mandate] = None

    def set_mandate(self, mandate: Mandate) -> None:
        self.mandate = mandate
        if self.state is not None:
            self.state.record_mandate(mandate)

    # ── lifecycle (subclasses implement) ─────────────────────────────────
    @abstractmethod
    def observe(self, period: str) -> dict:
        """Gather inputs: market data, child Reports, calibration, state."""

    @abstractmethod
    def decide(self, observation: dict) -> dict:
        """Produce the period's plan. Judgment phases may call self.agent."""

    @abstractmethod
    def act(self, decision: dict) -> dict:
        """Deterministic execution + accounting."""

    @abstractmethod
    def measure(self, period: str) -> dict:
        """Mark-to-market, drawdown, halt checks."""

    @abstractmethod
    def report(self, measurement: dict) -> Report:
        """Build the Report that flows up."""

    # ── orchestration (template method) ──────────────────────────────────
    def run(self, period: str, dry_run: bool = False) -> Report:
        observation = self.observe(period)
        decision = self.decide(observation)
        self.act(decision)
        measurement = self.measure(period)
        rpt = self.report(measurement)
        if self.state is not None and not dry_run:
            self.state.record_report(rpt)
        return rpt
