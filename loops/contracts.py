"""The down/up payloads every loop exchanges (S22 D3).

``Mandate`` flows DOWN (parent → child): capital budget + risk limits +
constraints for one period. ``Report`` flows UP (child → parent): performance +
regime read + confidence, nesting child Reports for recursive roll-up.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Cadence(str, Enum):
    """How often a loop runs. Inherits from str so it serializes as its value."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class Mandate:
    """Issued by a parent to a child for one period. Immutable per period."""

    loop_id: str                          # child this mandate is addressed to
    issued_by: str                        # parent loop_id
    period: str                           # ISO date governing this mandate
    capital_budget: float                 # $ the child may deploy this period
    risk_limits: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    horizon_days: int = 30
    objective: str = "risk_adjusted_return"
    notes: str = ""                       # rationale (often agent-authored)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Mandate":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Mandate":
        return cls.from_dict(json.loads(s))


@dataclass
class Report:
    """Returned by a loop's run(); aggregated by its parent. Reports nest."""

    loop_id: str
    period: str
    starting_value: float
    ending_value: float
    period_return: float = 0.0
    max_drawdown: float = 0.0
    realized_pnl: float = 0.0
    n_trades: int = 0
    # judgment surface (what flows up for allocation decisions)
    regime: Optional[str] = None
    regime_confidence: float = 0.0
    confidence: float = 0.0               # loop's self-assessed conviction
    capital_utilization: float = 0.0      # deployed / budget
    halted: bool = False
    halt_reason: Optional[str] = None
    children: list["Report"] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        # asdict recurses into nested Report dataclasses → plain dicts.
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        d = dict(d)
        d["children"] = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Report":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_values(
        cls, loop_id: str, period: str, starting_value: float,
        ending_value: float, **kw: Any,
    ) -> "Report":
        """Convenience: derive period_return from start/end."""
        ret = (ending_value / starting_value - 1.0) if starting_value else 0.0
        return cls(
            loop_id=loop_id, period=period, starting_value=starting_value,
            ending_value=ending_value, period_return=ret, **kw,
        )
