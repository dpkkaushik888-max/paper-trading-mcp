"""L0 Personal Finance loop (S22 stub).

Monthly/quarterly: the top of the hierarchy. Decides how much capital flows to
the Investment loop (L1) vs cash/other. Stubbed here to pass a fixed fraction of
its budget to L1; real income/expense/savings/debt logic is a later spec.
"""

from __future__ import annotations

from typing import Any, Optional

from .composite import CompositeLoop
from .contracts import Cadence


class PersonalFinanceLoop(CompositeLoop):
    def __init__(self, children: list, state: Any = None, agent: Any = None,
                 allocator: Optional[Any] = None) -> None:
        # children is typically [AllocatorLoop]; the deterministic allocator
        # sends the whole budget to the single child until L0 logic exists.
        super().__init__("L0.personal", Cadence.MONTHLY, children,
                         state=state, agent=agent, allocator=allocator)
