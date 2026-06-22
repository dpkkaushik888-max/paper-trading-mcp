"""Loop hierarchy wiring (S22).

REGISTRY.build(loop_id) constructs a loop and its subtree from config.
bootstrap_mandate(loop_id) supplies a top-level Mandate when none is persisted.

For now the asset-class leaves are stubs (offline-runnable) so L1/L0 work
end-to-end before every real leaf has live data wired. The real L2.crypto leaf
(CryptoAssetLoop) plugs in by passing bars — done by the paper_v2 runner.
"""

from __future__ import annotations

from typing import Any, Optional

from .contracts import Cadence, Mandate
from .l0_personal import PersonalFinanceLoop
from .l1_allocator import AllocatorLoop
from .state import LoopState
from .stub import StubLeaf

DEFAULT_TOP_CAPITAL = 10_000.0
DEFAULT_RISK_LIMITS = {"max_drawdown_pct": 0.20, "max_concurrent": 8,
                       "per_strategy_cap": 4, "pos_size_pct": 0.12}


def _l2_children(state_base: Optional[str]) -> list:
    def st(loop_id):
        return LoopState(loop_id, base_dir=state_base) if state_base else None
    # Stubs until real leaves are wired. Crypto carries a positive prior so the
    # allocator demonstrably tilts toward it; equity is the not-built placeholder.
    return [
        StubLeaf("L2.crypto", Cadence.DAILY, period_return=0.05, confidence=0.8,
                 max_drawdown=0.02, regime="BULL", state=st("L2.crypto")),
        StubLeaf("L2.equity", Cadence.DAILY, period_return=-0.01, confidence=0.3,
                 max_drawdown=0.06, state=st("L2.equity")),
    ]


class _Registry:
    def build(self, loop_id: str, state_base: Optional[str] = None, agent: Any = None):
        def st(lid):
            return LoopState(lid, base_dir=state_base) if state_base else None

        if loop_id == "L1.allocator":
            return AllocatorLoop(_l2_children(state_base), state=st("L1.allocator"), agent=agent)
        if loop_id == "L0.personal":
            l1 = AllocatorLoop(_l2_children(state_base), state=st("L1.allocator"), agent=agent)
            return PersonalFinanceLoop([l1], state=st("L0.personal"), agent=agent)
        if loop_id == "L2.crypto":
            return _l2_children(state_base)[0]
        raise ValueError(f"unknown loop_id: {loop_id}")


REGISTRY = _Registry()


def bootstrap_mandate(loop_id: str, period: str, capital: float = DEFAULT_TOP_CAPITAL) -> Mandate:
    issuer = {"L0.personal": "owner", "L1.allocator": "L0.personal",
              "L2.crypto": "L1.allocator"}.get(loop_id, "owner")
    return Mandate(
        loop_id=loop_id, issued_by=issuer, period=period, capital_budget=capital,
        risk_limits=dict(DEFAULT_RISK_LIMITS), constraints={"long_only": True},
        horizon_days=30, notes="bootstrap",
    )
