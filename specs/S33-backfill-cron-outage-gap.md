# S33: Backfill the paper-forward gap from the pandas-3.0 cron outage

**Status:** VERIFIED
**Branch:** `master` (hotfix follow-up)
**Priority:** P1 (live track-record integrity)

## Overview
The S18 Paper Forward cron segfaulted for 7 consecutive daily runs (2026-06-23 → 2026-06-29)
because `requirements.txt` pinned `pandas>=2.0.0` and the run picked up the yanked pandas 3.0.4
(datetime segfaults). The journal therefore jumped from 2026-06-22 straight to 2026-06-29,
skipping 6 trading days. A read-only probe showed real Connors entry signals fired in that
window (NEARUSDT ~06-23, TRXUSDT ~06-26), so the gap is not cosmetic — the live record is
missing trades. This spec backfills the 6 missed days faithfully.

## Design Decisions
- **Reuse the real `run()` code path**, not a re-implementation, so the backfill is byte-for-byte
  what the live cron would have produced. Achieved via a new optional `as_of` parameter that
  (a) uses the given date as "today" and (b) truncates each symbol's bars to `index <= as_of`.
- **Baseline = current journal minus the 2026-06-29 day.** Today's green run left `cash`,
  `open_positions`, `closed_trades`, and every benchmark's internal state unchanged (flat day,
  zero trades, BTC still below its 200d SMA), so popping the last day yields the exact 2026-06-22
  end-state — no need to restore a pre-S32 git snapshot (which would reset the regime_selector).
- **Replay order:** 2026-06-23, 24, 25, 26, 27, 28, then re-run 29 so its portfolio/benchmark
  marks reflect any positions opened during the gap.
- Binance daily klines for past dates are immutable, so the replay is deterministic.

## Research
- `run_daily.run()` acts only on the latest bar; failed runs wrote nothing → a true gap.
- `MAX_NO_SIGNAL_DAYS` was removed 2026-06-22; `update_issue.py` still imported it (fixed alongside).
- Probe (open=0 baseline): entries Jun23 NEAR, Jun24 NEAR, Jun25 NEAR, Jun26 NEAR+TRX, Jun27 none, Jun28 none.

## Acceptance Criteria
- [x] `run(as_of=DATE)` replays a historical day using only bars up to that date, identical logic otherwise.
- [x] Backfill script reconstructs days 2026-06-23 → 2026-06-29 in order from the popped baseline.
- [x] Resulting journal contains the NEARUSDT (and TRXUSDT) trade(s) with sane entry/exit/P&L.
- [x] Day count and dates are contiguous business-as-usual (no remaining gap, no duplicate 06-29).
- [x] Existing tests still pass; `run()` with no `as_of` behaves exactly as before.

## Technical Design
### Files to Create/Modify
| File | Change |
|------|--------|
| `trading_engine/paper/run_daily.py` | Add optional `as_of` param to `run()`: inject `today`, truncate bars to `<= as_of`. |
| `scripts/backfill_gap.py` | One-shot driver: pop the 06-29 day, replay 06-23..06-29 via `run(as_of=...)`. |
| `state/journal.json` | Regenerated with the 6 backfilled days (committed). |

### Data Model Changes
None — same `DaySnapshot` schema (v4).

## Verification
- Run `scripts/backfill_gap.py` against a backup copy; inspect closed_trades and the new days.
- Confirm NEAR entry ~06-23 and its exit; confirm contiguous dates 06-22..06-29.

## UAT
- [x] Backfilled journal reviewed: 61 contiguous days (06-21..06-29, no dups), closed_trades 5→7
      (NEAR 06-23→06-26 SL −$145.09; TRX 06-26→06-28 MR +$17.84), NEAR re-entry open from 06-26.
      Corrected 06-29 portfolio $10,441.43 (+4.41%) vs phantom $10,535.69 (+5.36%) — the gap had
      masked a net loss. 19 paper tests pass.

## Notes
- Pairs with the requirements pandas cap and the `update_issue.py` import fix (already committed).
