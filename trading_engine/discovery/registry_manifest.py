"""Versioned strategy registry manifest (S25 D7).

Promotions are the *output* of the discovery loop. Each one appends an entry to a
git-committable JSON manifest carrying full provenance — the candidate's rules, all
parameters, walk-forward + holdout metrics, the trial count, and the deflated
Sharpe bar it had to clear. Promotions are reversible: ``rollback`` flips an entry's
status to ``rolled_back`` without deleting its audit record.

``active_specs()`` / ``load_registry_strategies()`` reconstruct live strategies for
the orchestrator (which stays persistence-free — it just receives the list).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from trading_engine.discovery.candidate import CandidateSpec, GeneratedStrategy

SCHEMA_VERSION = 1
DEFAULT_PATH = "state/discovery/registry.json"


@dataclass
class PromotionEntry:
    version: int                 # monotonic, 1-based
    candidate: dict              # CandidateSpec.to_dict()
    provenance: dict             # wf metrics, holdout verdict, trial count, deflated bar
    period: str                  # ISO date of the discovery run that promoted it
    status: str = "active"       # "active" | "rolled_back"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromotionEntry":
        return cls(version=d["version"], candidate=d["candidate"],
                   provenance=d.get("provenance", {}), period=d.get("period", ""),
                   status=d.get("status", "active"), notes=d.get("notes", ""))


@dataclass
class _Manifest:
    schema_version: int = SCHEMA_VERSION
    entries: list = field(default_factory=list)   # list[PromotionEntry]


class RegistryManifest:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._m = self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> _Manifest:
        if not self.path.exists():
            return _Manifest()
        d = json.loads(self.path.read_text())
        return _Manifest(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            entries=[PromotionEntry.from_dict(e) for e in d.get("entries", [])],
        )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": self._m.schema_version,
                   "entries": [e.to_dict() for e in self._m.entries]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # ── queries ──────────────────────────────────────────────────────────
    @property
    def entries(self) -> list:
        return list(self._m.entries)

    def active_entries(self) -> list:
        return [e for e in self._m.entries if e.status == "active"]

    def active_specs(self) -> list:
        """Reconstruct CandidateSpec for every active promotion."""
        return [CandidateSpec.from_dict(e.candidate) for e in self.active_entries()]

    def find(self, candidate_id: str) -> Optional[PromotionEntry]:
        # latest entry for this id wins (a re-promotion supersedes a rollback)
        matches = [e for e in self._m.entries if e.candidate.get("id") == candidate_id]
        return matches[-1] if matches else None

    # ── mutations ────────────────────────────────────────────────────────
    def promote(self, spec: CandidateSpec, provenance: dict, period: str,
                notes: str = "") -> PromotionEntry:
        """Append a promotion with full provenance. Returns the new entry."""
        version = (self._m.entries[-1].version + 1) if self._m.entries else 1
        entry = PromotionEntry(version=version, candidate=spec.to_dict(),
                               provenance=provenance, period=period, notes=notes)
        self._m.entries.append(entry)
        self._save()
        return entry

    def rollback(self, candidate_id: str, reason: str = "") -> bool:
        """Mark the latest active promotion of ``candidate_id`` rolled_back. Reversible
        (the entry stays for audit). Returns True if something was rolled back."""
        for e in reversed(self._m.entries):
            if e.candidate.get("id") == candidate_id and e.status == "active":
                e.status = "rolled_back"
                e.notes = (e.notes + " | " if e.notes else "") + f"rollback: {reason}"
                self._save()
                return True
        return False


def load_registry_strategies(
    path: str | Path = DEFAULT_PATH, priority_start: int = 10,
) -> list[GeneratedStrategy]:
    """Build live ``GeneratedStrategy`` instances from active promotions.

    Discovered strategies rank after the fixed A/B/C stack (priority 0–2), so
    ``priority_start`` defaults to 10. The orchestrator receives this list via its
    existing ``strategies=`` argument — it never touches the manifest itself.
    """
    manifest = RegistryManifest(path)
    return [spec.to_strategy(priority=priority_start + i)
            for i, spec in enumerate(manifest.active_specs())]
