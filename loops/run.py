"""CLI entrypoint for the loop hierarchy (S22).

    python -m loops.run --loop L1.allocator --dry-run
    python -m loops.run --loop L0.personal --period 2026-06-22

The cron target for each cadence. With --dry-run, agents are disabled and nothing
is persisted, so it runs deterministically headless.
"""

from __future__ import annotations

import argparse
import sys

from .agent import AgentClient
from .contracts import Report
from .registry import REGISTRY, bootstrap_mandate


def _print_report(rpt: Report, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}{rpt.loop_id}  {rpt.period}  "
          f"${rpt.starting_value:,.0f} → ${rpt.ending_value:,.0f}  "
          f"({rpt.period_return * 100:+.2f}%)  "
          f"trades={rpt.n_trades} conf={rpt.confidence:.2f}"
          + (f"  HALTED:{rpt.halt_reason}" if rpt.halted else ""))
    for child in rpt.children:
        _print_report(child, indent + 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a loop in the hierarchy")
    p.add_argument("--loop", required=True, help="loop_id, e.g. L1.allocator")
    p.add_argument("--period", default=None, help="ISO date; default a fixed placeholder")
    p.add_argument("--dry-run", action="store_true", help="no agents, no persistence")
    p.add_argument("--state-base", default=None, help="base dir for loop state ledgers")
    args = p.parse_args(argv)

    # NOTE: date helpers are intentionally explicit — the period is passed in
    # (cron supplies it) rather than read from the clock.
    period = args.period or "1970-01-01"
    agent = None if args.dry_run else AgentClient(enabled=True)
    state_base = None if args.dry_run else (args.state_base or "state/loops")

    loop = REGISTRY.build(args.loop, state_base=state_base, agent=agent)
    mandate = None
    if loop.state is not None:
        mandate = loop.state.current_mandate()
    if mandate is None:
        mandate = bootstrap_mandate(args.loop, period)
    loop.set_mandate(mandate)

    rpt = loop.run(period, dry_run=args.dry_run)
    print(f"\n=== Loop run: {args.loop} (dry_run={args.dry_run}) ===")
    _print_report(rpt)
    return 2 if rpt.halted else 0


if __name__ == "__main__":
    sys.exit(main())
