# S30: Anti-Whipsaw Band for the Trend-Timed Core

**Status:** VERIFIED
**Branch:** `feature/s30-trend-band`
**Priority:** P2 (hardens the S29 core; "fixing the agent")
**Depends on:** S29 (trend-timed tracker)

## Overview

The year-by-year study exposed S29's weakness: it whipsaws when BTC oscillates around
a *flat* 200-day SMA, and crossing costs erode returns. S30 adds a **hysteresis band**
(±2% default) around the SMA — enter only when price is a band *above* it, exit only
when a band *below*, hold inside the band. This is a standard, a-priori anti-whipsaw
technique (not tuned to a known window). Validated: it **matched buy-and-hold's full
4-year return (+205%) at ~half the drawdown (31% vs ~70%)** and **never hurt a single
year** vs the bare rule.

## User Story

As the system owner, I want the trend-timed core to stop bleeding from repeated MA
crossings in choppy markets, so it keeps its bull-capture + crash-protection without
churning, using a principled buffer rather than a curve-fit regime model.

## Design Decisions

### D1 — Hysteresis band, chosen on principle (NOT fit to the data)
We tested two textbook anti-whipsaw techniques honestly, on all four years, and
reported both:
- **Slope filter (only hold when the SMA is rising): REJECTED.** It backfired —
  cut the 4y return nearly in half (+101% vs +184%) by sitting out the entire 2022–23
  recovery (the SMA stays falling long after price bottoms). The "obvious" fix was wrong;
  the test caught it.
- **±2% band: ADOPTED.** Matched HODL (+205%) at 31% max DD, improved the bear year
  (2025–26: −6%→0%), and was ≥ the bare rule in every year. Robust, not cherry-picked.

### D2 — The band does NOT "fix" 2024–25, and we say so
2024–25 (BTC +66%, S29 +4%) is **MA lag, not whipsaw**: BTC rose in bursts while still
*below* its slow average, so trend-timed was correctly in cash and missed it. The band
(which kills whipsaws) still made only +3% that year — confirming the cause is lag.
Lag is only "fixable" with a faster signal, which reintroduces whipsaw — the inherent
trade-off. S30 does not pretend to solve it; chasing it is where overfitting begins.

### D3 — Default band = 2%, an a-priori sensible buffer
2% is a standard MA-band width, picked before optimizing; not swept-and-best-fit. The
exact width should be re-confirmed forward, not tuned on this (already-observed) sample.

### D4 — Backward-compatible; band is the new default for the live tracker
`TrendTimer.band_pct` (default 0.02). The journal scaffold carries it; an existing v3
tracker without the field deserializes to the 0.02 default. Re-backfill applies it.
The bare-rule (band=0) remains expressible for reproducing the S29 baseline.

## Research
- Year-by-year (memory `trend-timed-btc-beats-hodl`): S29 whipsawed in 2024–25.
- On-data test of slope vs band variants (this spec, D1) — band robustly best, slope worst.

## Acceptance Criteria
- [x] `TrendTimer.band_pct` + hysteretic `step()` (enter above +band, exit below −band, hold inside).
- [x] Journal scaffold carries `band_pct`; backward-compatible; live journal re-backfilled.
- [x] Tests: band blocks marginal entry, holds through minor dips, suppresses whipsaw switch count.
- [x] All prior tests pass; deterministic.
- [x] Honest report: slope rejected, band adopted, 2024–25 lag acknowledged as unfixed.

## Technical Design
| File | Change |
|------|--------|
| `trading_engine/paper/trend_timed.py` | `band_pct` field + hysteresis in `step()` |
| `trading_engine/paper/journal.py` | `_fresh_trend_timed` carries `band_pct` |
| `tests/test_trend_timed.py` | band behavior tests |
| `scripts/backfill_trend_timed.py` | (unchanged) re-run to apply band |

## Verification
- `pytest tests/test_trend_timed.py -q` (12 tests).
- Backtest (in-sample, 4y): band matches HODL +205% at 31% DD; ≥ bare rule every year.

## UAT — 2026-06-22
- [x] 231 tests green (12 trend-timed incl. 3 new band tests).
- [x] Live journal re-backfilled: tracker now `band_pct=0.02`; still CASH over the
      downtrend window (band keeps it even more firmly out — correct).
- [x] 4y backtest: **+205% / 31% maxDD** (band) vs +184%/32% (bare S29) vs +205%/~70% (HODL).

**Verdict:** the principled fix (band) is shipped; the intuitive fix (slope) was tested
and rejected; the lag-driven 2024–25 miss is honestly left unsolved. In-sample only —
forward test is the real proof.

## Notes & Risks
- **No free lunch on lag.** The band trades nothing away but also can't recover lag-missed bulls.
- **Still in-sample.** Same discipline: the live forward test decides, not this backtest.
- **Promotion ≠ live capital.** Paper-tracked, like everything in this stack.
