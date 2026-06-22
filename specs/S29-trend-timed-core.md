# S29: Trend-Timed BTC Core (added to the 90-day paper-forward)

**Status:** VERIFIED
**Branch:** `feature/s29-trend-timed-core`
**Priority:** P1 (the one robust edge found — now tracked live)
**Depends on:** S18 (paper-forward harness), S28 (the finding)

## Overview

The S25–S28 investigation found one robust edge: **hold BTC while it is above a long
moving average, go to cash when it drops below** — beating HODL on return *and*
drawdown across all MA windows (memory: trend-timed-btc-beats-hodl). This spec adds
that strategy as a **third tracked series in the running 90-day paper-forward**, so it
is compared live, day-by-day, against Connors and the passive benchmarks. Per the
owner's instruction, the rule **stays fully invested in BTC during the bull** (does
not sit in cash) and only moves to cash when the trend breaks.

## User Story

As the system owner, I want the trend-timed BTC strategy tracked alongside Connors in
the live 90-day test — staying invested through bull trends — so I can see, on the same
window and cost model, whether trend participation beats both Connors and buy-and-hold.

## Design Decisions

### D1 — A tracked series, not a new trading strategy in the leaf
Trend-timed BTC is a single-asset timing overlay, not a per-symbol entry/exit strategy
that fits the multi-symbol orchestrator. So it joins the paper-forward the same way
`bh_btc`/`bh_basket` do — a phantom portfolio marked daily — not as a Connors-style
position book. Clean, isolated, and directly comparable.

### D2 — The rule: invested in bull, cash in bear (200-day SMA default)
Each day: `bull = BTC_close > BTC_SMA(ma_window)` with `ma_window=200`. If bull and
flat → buy BTC with all cash (pay 20 bps). If bear and invested → sell to cash (pay 20
bps). **Stays fully invested for the entire bull** (owner's instruction); switches only
on a regime flip. Same honest cost model as everything else (`COST_PCT=0.0020`).

### D3 — Back-fill the existing 54 days for a fair comparison
The test is on day 54. To compare over the *same* window, replay the rule across the
journal's existing dates using re-fetched BTC history (≥200 bars before day 1 for the
SMA), populating `trend_timed_value` for every past day and the tracker's current
state. One-time, idempotent, re-runnable. Going forward, `run_daily` advances it daily.

### D4 — Journal schema v2 → v3, backward-compatible
Add a `trend_timed_btc` benchmark scaffold + a `trend_timed_value` day field
(defaulting to starting capital), exactly as v1→v2 added the BH benchmarks. Existing
state is preserved; the migration is idempotent.

### D5 — No lookahead; decision at the close
The SMA on day *t* uses bars up to and including *t*; the switch is applied at *t*'s
close and the position is marked at *t*'s close. Standard close-to-close timing,
consistent with the benchmarks.

## Research
- S28 / memory `trend-timed-btc-beats-hodl`: full-5y, all MA windows 50–250 beat HODL
  on return + drawdown + Sharpe (200d best: +183% vs +86%, DD 32% vs 77%).
- `trading_engine/paper/{run_daily,journal,benchmarks,data,config}.py`: the S18 harness;
  `KLINE_LIMIT=300` already fetches enough history for SMA(200).
- Live journal: day 54, Connors +5.4% vs BH_BTC −16.7% (this window is a downtrend).

## Acceptance Criteria
- [ ] `trading_engine/paper/trend_timed.py`: `TrendTimer` + pure `step()` (invested in
      bull, cash in bear) + serialize/deserialize.
- [ ] Journal v2→v3: `trend_timed_btc` scaffold + `trend_timed_value` day field; idempotent.
- [ ] `run_daily.py` advances the tracker each day and records `trend_timed_value`.
- [ ] `scripts/backfill_trend_timed.py` replays accurately over the existing days.
- [ ] No lookahead; honest 20 bps cost per switch; deterministic.
- [ ] Tests: bull→invested, bear→cash, switch costs, no double-switch, value marking.
- [ ] Report: 3-way live comparison (Connors / trend-timed / BH_BTC) over the window.

## Technical Design
### Files
| File | Change |
|------|--------|
| `trading_engine/paper/trend_timed.py` | new — TrendTimer + step + (de)serialize |
| `trading_engine/paper/journal.py` | SCHEMA_VERSION=3, v2→v3 migrate, DaySnapshot.trend_timed_value, init_fresh |
| `trading_engine/paper/run_daily.py` | step tracker daily, log line |
| `scripts/backfill_trend_timed.py` | new — one-time historical replay |
| `tests/test_trend_timed.py` | new |

### Data Model Changes
Journal `benchmarks.trend_timed_btc` (cash/shares/invested state) + day field
`trend_timed_value`. v2→v3 migration; backward-compatible.

## Dependencies
S18 harness (live). No new packages.

## Verification
- `pytest tests/test_trend_timed.py -q`.
- `python scripts/backfill_trend_timed.py` → 54 days populated; tracker state set.
- Inspect journal: 3 series present; print Connors vs trend-timed vs BH_BTC.

## UAT — 2026-06-22

- [x] `trend_timed.py` `TrendTimer` + pure `step()` (invested in bull, cash in bear) +
      (de)serialize. 9 tests pass.
- [x] Journal v2→v3 migration adds `trend_timed_btc` + `trend_timed_value`; idempotent;
      live journal migrated cleanly (now `version 3`, 3 benchmarks).
- [x] `run_daily.py` advances the tracker daily and logs INVESTED/CASH + value.
- [x] `backfill_trend_timed.py` replayed the rule across all 54 existing days.
- [x] No lookahead; 20 bps per switch; deterministic. Full suite 228 green.

**3-way live result over the SAME window (day 54, 2026-04-21→2026-06-22):**

| series | value | return | note |
|--------|-------|--------|------|
| Connors | $10,535.69 | **+5.4%** | actively scalped the chop |
| **Trend-timed BTC** | $10,000.00 | **0.0%** | **CASH the whole window — preserved capital** |
| BH_BTC | $8,330.88 | **−16.7%** | rode the drawdown down |

**Honest read (as predicted in D-notes):** this window has been a pure downtrend — BTC
stayed *below* its 200-day SMA the entire 54 days, so trend-timed correctly sat in cash
(0 switches) and avoided BTC's −16.7% loss. Its value here is capital preservation, not
the bull-capture that drove the full-cycle outperformance (S28). Connors edged it by
actively trading the chop. Going forward, if BTC reclaims its 200-day SMA the tracker
flips to INVESTED automatically and we observe its bull behavior live.

**Verdict:** the one robust edge is now tracked live, on the same window + cost model as
Connors and the passive benchmarks. The downtrend window can't yet show its upside, but
the comparison is fair and running.

## Notes & Risks
- **Live window is short (90d) and currently a downtrend** — trend-timed will likely be
  mostly in cash here (BTC below its 200d SMA), so it should resemble "flat/cash"
  this window, not its full-cycle outperformance. That is expected and honest.
- **MA-timing's weakness is chop** — a sideways market whipsaws it. The 90-day test is
  a small sample; the real evidence remains the full-cycle study (S28) + future windows.
- **Still not live capital.** This is paper tracking to build evidence, exactly as S18.
