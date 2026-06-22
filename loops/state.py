"""Per-loop JSON state ledger (S22 D5).

Generalizes the proven ``trading_engine/paper/journal.py`` pattern: versioned,
git-committable JSON under ``state/loops/<loop_id>/`` with append-mostly
mandate / report / agent-call ledgers. Positions/cash stay in their existing
inner stores (this is the control-plane ledger, not the execution store).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .contracts import Mandate, Report

SCHEMA_VERSION = 1
DEFAULT_BASE_DIR = "state/loops"


class LoopState:
    def __init__(self, loop_id: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> None:
        self.loop_id = loop_id
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / loop_id
        self.mandates_dir = self.root / "mandates"
        self.reports_dir = self.root / "reports"
        self.agent_dir = self.root / "agent"

    def _ensure(self) -> None:
        for d in (self.mandates_dir, self.reports_dir, self.agent_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── mandates (down) ──────────────────────────────────────────────────
    def record_mandate(self, mandate: Mandate) -> None:
        self._ensure()
        (self.mandates_dir / f"{mandate.period}.json").write_text(mandate.to_json())

    def current_mandate(self) -> Optional[Mandate]:
        files = sorted(self.mandates_dir.glob("*.json")) if self.mandates_dir.exists() else []
        return Mandate.from_json(files[-1].read_text()) if files else None

    # ── reports (up) ─────────────────────────────────────────────────────
    def record_report(self, report: Report) -> None:
        self._ensure()
        (self.reports_dir / f"{report.period}.json").write_text(report.to_json())

    def last_report(self) -> Optional[Report]:
        files = sorted(self.reports_dir.glob("*.json")) if self.reports_dir.exists() else []
        return Report.from_json(files[-1].read_text()) if files else None

    def last_reports(self, child_ids: list[str]) -> dict[str, Report]:
        """Latest report for each child loop_id, read from sibling state dirs."""
        out: dict[str, Report] = {}
        for cid in child_ids:
            rdir = self.base_dir / cid / "reports"
            files = sorted(rdir.glob("*.json")) if rdir.exists() else []
            if files:
                out[cid] = Report.from_json(files[-1].read_text())
        return out

    # ── agent calls (audit trail) ────────────────────────────────────────
    def record_agent_call(
        self, period: str, decision_type: str, request: dict, response: dict
    ) -> None:
        self._ensure()
        payload = {"schema_version": SCHEMA_VERSION, "request": request, "response": response}
        (self.agent_dir / f"{period}_{decision_type}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
