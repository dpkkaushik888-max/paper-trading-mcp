"""One-shot: backfill the paper-forward days missed during the pandas-3.0 cron
outage (S33). The S18 cron segfaulted 2026-06-23 → 2026-06-29, so the journal
jumped 2026-06-22 → 2026-06-29 and never recorded the Connors trades that fired
in between (NEARUSDT ~06-23, TRXUSDT ~06-26).

Approach: pop the already-committed 2026-06-29 day (today's green run wrote it
from the flat, trade-less state), then replay 06-23 .. 06-29 in order through the
real ``run(as_of=...)`` code path so the result is exactly what the live cron
would have produced. Binance daily klines for past dates are immutable.

Usage:
    PYTHONPATH=. python scripts/backfill_gap.py            # apply
    PYTHONPATH=. python scripts/backfill_gap.py --dry-run  # log only, no save
"""
from __future__ import annotations

import argparse
import logging

from trading_engine.paper.journal import Journal
from trading_engine.paper.run_daily import run

log = logging.getLogger("s33.backfill")

LAST_GOOD_DATE = "2026-06-22"
POP_DATE = "2026-06-29"
REPLAY_DATES = [
    "2026-06-23", "2026-06-24", "2026-06-25",
    "2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill S18 cron-outage gap (S33)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Replay without saving the journal (preview only).")
    args = ap.parse_args()

    j = Journal.load()
    days = j.days
    if not days:
        log.error("Empty journal — nothing to backfill.")
        return 1
    last = days[-1]["date"]
    if last != POP_DATE:
        log.error("Expected last day %s, found %s — refusing to backfill.", POP_DATE, last)
        return 1
    if days[-2]["date"] != LAST_GOOD_DATE:
        log.error("Expected prior day %s, found %s — gap shape unexpected, aborting.",
                  LAST_GOOD_DATE, days[-2]["date"])
        return 1

    log.info("Baseline: popping %s day to restore the %s end-state.", POP_DATE, LAST_GOOD_DATE)
    if not args.dry_run:
        days.pop()              # remove the trade-less 06-29 snapshot
        j.save()

    for d in REPLAY_DATES:
        rc = run(dry_run=args.dry_run, as_of=d)
        log.info("Replayed %s → rc=%d", d, rc)

    j2 = Journal.load()
    log.info("Done. days=%d closed_trades=%d open=%d last_date=%s",
             len(j2.days), len(j2.closed_trades), len(j2.open_positions),
             j2.days[-1]["date"] if j2.days else "—")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
