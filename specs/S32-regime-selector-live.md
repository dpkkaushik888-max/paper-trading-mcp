# S32: Live Regime-Selector in the 90-Day Paper-Forward

**Status:** VERIFIED
**Branch:** `feature/s32-regime-selector-live`
**Priority:** P2 (watch the agent's strategy choice live)
**Depends on:** S18 (Connors), S29/S30 (trend-timed), S31 (selection idea)

## Overview

Adds a 5th tracked series to the running 90-day paper-forward: a **regime selector**
that holds **BTC (trend-timed, S30) in a bull and Connors (S18) in a bear**, switching
on BTC's 200-day SMA (with the S30 ±2% band). It answers two live questions directly:
*does the trend (S30) leg ever fire in this window, and does combining lower the profit
vs pure Connors?*

## User Story

As the system owner, I want the bull→BTC / bear→Connors selector tracked live next to
Connors and the benchmarks, so I can see — on real forward data — whether S30 activates
and whether the combination beats or drops below my current Connors result.

## Design Decisions

### D1 — Phantom series, like the other trackers
Tracked alongside `bh_btc`/`bh_basket`/`trend_timed_btc`, not as a real position book.
Bull leg holds BTC (its own shares); bear leg **compounds the actual Connors portfolio's
daily return** (so the selector's bear performance equals Connors by construction).

### D2 — Same regime signal as S30 (hysteresis band)
Bull = BTC > 200d SMA·(1+band) to enter, < ·(1−band) to exit (band 2%). Regime flip pays
20 bps. Identical mechanism to the live trend-timed tracker, so the two are comparable.

### D3 — Back-fill the existing 54 days for a same-window comparison
Replay over the journal's dates using the recorded Connors path (bear-leg returns) and
re-fetched BTC (regime). Idempotent. Going forward `run_daily` advances it daily.

### D4 — Journal v3 → v4, backward-compatible
Adds `regime_selector` benchmark + `selector_value` day field (default starting capital).
Existing v3 state migrates cleanly; idempotent.

## Acceptance Criteria
- [x] `trading_engine/paper/selector_track.py` — `RegimeSelector` + pure `step()` (bull→BTC, bear→Connors) + (de)serialize.
- [x] Journal v3→v4 migration + `selector_value`; `run_daily` advances it daily (Connors daily return feeds the bear leg).
- [x] `scripts/backfill_selector.py` replays over existing days; live journal back-filled.
- [x] Tests: bear mirrors Connors, bull tracks BTC (ignores Connors), hysteresis, switch cost, serialization.
- [x] All prior tests pass (245 green); deterministic.

## Technical Design
| File | Change |
|------|--------|
| `trading_engine/paper/selector_track.py` | new — RegimeSelector + step + (de)serialize |
| `trading_engine/paper/journal.py` | v3→v4 migrate, `selector_value`, init_fresh |
| `trading_engine/paper/run_daily.py` | compute Connors daily return, step selector daily |
| `scripts/backfill_selector.py` | new — one-time replay |
| `tests/test_selector_track.py` | new |

## UAT — 2026-06-22 (day 54)

Live journal now v4 with 5 series. Back-fill result:

| series | value | return |
|--------|------:|-------:|
| Connors (S18) | $10,535.69 | +5.36% |
| **Regime selector** | **$10,535.69** | **+5.36%** |
| Trend-timed (S30) | $10,000.00 | +0.00% |
| BH_BTC | $8,330.88 | −16.69% |

**Answers to the two questions:**
- **Does S30 fire?** *Not yet* — BTC stayed below its 200-day SMA the entire window
  (selector `mode=bear`, **0 switches**), so the selector held the Connors leg throughout.
  S30 activates only when BTC reclaims ~$77k.
- **Does combining drop the profit?** *No* — in a bear the selector **equals Connors
  exactly** (+5.36%). It can only diverge once BTC turns bull and it switches to BTC;
  then we observe live whether catching S30 helps or hurts. No downside in the meantime.

**Verdict:** the selector is armed and tracking live at zero cost vs Connors; the trend
leg is waiting on a regime turn.

## Notes & Risks
- **The bear-leg = Connors by construction** — so in down/chop regimes the selector
  carries no penalty vs Connors; the only open question is the bull handoff.
- **Forward, not in-sample.** This is the clean evidence: the switch will happen (or not)
  on unseen data. Paper only — not real capital.
