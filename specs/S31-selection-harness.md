# S31: Strategy-Selection Backtest Harness

**Status:** VERIFIED
**Branch:** `feature/s31-selection-harness`
**Priority:** P2 (makes the "agent picks strategy vs HODL" test explicit + reusable)
**Depends on:** S18 (Connors), S29/S30 (trend-timed)

## Overview

Reusable harness for the question "let a selector choose a strategy each period, then
compare total profit to HODL." Walks the window period-by-period (monthly default); at
each boundary a pluggable **selector** sees data *through that boundary only* and picks
one sleeve from {btc, connors, cash}; the account runs it for the period; sleeve changes
pay an honest switching cost. The selector is the "agent" — a deterministic rule today,
a bounded agent later — with no-lookahead enforced by construction.

## User Story

As the system owner, I want a clean, repeatable test where a named selector picks the
strategy for each period using only past data and the result is reported against HODL,
so "the agent chooses strategies" is an explicit, auditable experiment — not a one-off.

## Design Decisions

### D1 — Boundary-chained periods (telescopes to a true hold)
Decisions occur at month-end boundaries; period i return accrues over
(boundary[i], boundary[i+1]]. BTC return is measured boundary-to-boundary, so an
always-BTC selector reproduces a continuous hold *exactly* (verified) — no gap leakage.

### D2 — No-lookahead by construction
The selector is only ever passed ``btc_close[:boundary]`` (through the decision point);
the period's own return is strictly after it. A test asserts the selector can never see
beyond its decision boundary.

### D3 — Three sleeves, single-sleeve per period
{btc (hold), cash (flat), connors (dip-scalp the universe)}. The account commits to one
sleeve per period — matching "the agent picks *a* strategy." Connors is self-contained
per period (flattens at period end); switching sleeves costs 20 bps.

### D4 — Pluggable selectors; named registry
``SELECTORS = {hodl, trend, trend_connors}`` — deterministic, past-data-only. A bounded
agent (or learned policy) drops into the registry with the same signature; the harness
and no-lookahead guarantee are unchanged.

## Research / Result (4y, $1,000, 49 months)
| selector | final | return | maxDD (month-end) | vs HODL |
|----------|------:|-------:|------------------:|--------:|
| hodl     | $3,052 | +205% | 44% | — |
| trend (bull→BTC, bear→cash) | $2,118 | +112% | 23% | −31% |
| trend_connors (bull→BTC, bear→Connors) | $2,023 | +102% | 22% | −34% |

Findings: (1) naive monthly selection does NOT beat HODL on return — it buys ~half the
drawdown for ~half the return; (2) monthly cadence is too slow vs daily S30 (+205%);
(3) Connors-in-bear < cash-in-bear again. Beating HODL needs finer cadence (→ S30) or
genuine regime detection (unsolved). The harness is the deliverable; the verdict is honest.

## Acceptance Criteria
- [x] `trading_engine/selection/harness.py` — sleeves, boundary periods, `run_selection`, named selectors.
- [x] `scripts/sim_selector.py` — run all named selectors over a window vs HODL.
- [x] No-lookahead enforced + tested; HODL selector reproduces buy-and-hold exactly.
- [x] Deterministic; switching cost charged on sleeve change; tests cover mechanics.
- [x] All prior tests pass (238 green).

## Technical Design
| File | Change |
|------|--------|
| `trading_engine/selection/{__init__,harness}.py` | new — harness + selectors |
| `scripts/sim_selector.py` | new — CLI |
| `tests/test_selection_harness.py` | new |

## Verification
- `pytest tests/test_selection_harness.py -q` (7 tests).
- `python scripts/sim_selector.py --years 4` → table vs HODL + monthly choice path.

## UAT — 2026-06-22
- [x] 238 tests green incl. 7 harness tests (HODL-reproduction exact, no-lookahead asserted).
- [x] 4y run: naive selectors underperform HODL on return, ~halve drawdown; monthly path sensible.

## Notes & Risks
- **maxDD is month-end sampled** — understates true intra-month drawdown, but fair across rows.
- **Dumb selectors only.** A bounded agent is the next plug-in; the hard part (regime
  detection good enough to beat HODL on return) remains unsolved — by design, not tuned away.
- **In-sample.** Forward/holdout discipline still applies before any of this is real capital.
