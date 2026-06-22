"""L1 Investment / Allocator loop (S22 stub).

Weekly: splits the personal-finance investment budget across asset-class leaves
(L2.crypto, L2.equity, ...) from their last Reports. A thin CompositeLoop — the
real allocation judgment is the deterministic allocator (or a wired agent).
"""

from __future__ import annotations

from typing import Any, Optional

from .composite import CompositeLoop
from .contracts import Cadence


class AllocatorLoop(CompositeLoop):
    def __init__(self, children: list, state: Any = None, agent: Any = None,
                 allocator: Optional[Any] = None) -> None:
        super().__init__("L1.allocator", Cadence.WEEKLY, children,
                         state=state, agent=agent, allocator=allocator)
