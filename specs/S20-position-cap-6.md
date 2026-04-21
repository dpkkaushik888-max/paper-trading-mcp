# S20: Raise Position Cap from 4 → 6 Concurrent

**Status:** VERIFIED (all gates + S20.1 pass; treat results as provisional pending paper-forward)
**Branch:** `feature/s20-position-cap-6`
**Priority:** P1
**Ticket:** —
**Depends on:** S19 (20-symbol expanded universe)

## Overview

S19 holdout audit revealed a mechanical constraint: **40.7% of potential
entries (24 of 59) were blocked by the `MAX_CONCURRENT = 4` position cap**.
The strategy is now signal-saturated during correlation-clustering events.

Hypothesis: raising the cap to **6** (single pre-committed value) will
reclaim roughly 10–15 additional trades of similar quality, lifting CAGR
and restoring Sharpe above the G1 threshold (>1.0).

## Data Snooping Disclosure (read before interpreting results)

This is the **second look at the S19 holdout**. Pure science requires a
fresh holdout after any parameter change. We are accepting mild
contamination because:

1. The change is a **diagnostic response** (fix the observed block rate),
   not a parameter sweep (we test 6, not 4/5/6/7/8/10)
2. The value is pre-committed **before** rerunning the sim
3. If 6 fails, we do **not** try 7 — we accept S17 as the ceiling and
   move to paper-forward

Any positive result must be treated as **provisional** until validated by
rolling-window robustness test (S17.1) or paper-forward (S18).

## User Story

As the account owner, I want to know whether relaxing the 4-concurrent
cap turns S19's "almost passing" result into a clean pass, so that I can
make an informed deploy / no-deploy decision.

## Design Decisions

### Single-value test, not a sweep
- **Test exactly `MAX_CONCURRENT = 6`** — no sweep, no multiple attempts
- Reasoning: the 4→6 jump aligns with empirical precedent (Connors
  himself used 4–8 max positions on similar-size universes); it's a 50%
  increase that approximates the observed signal overflow (40.7%)

### Position size stays 15%
- At 6 concurrent × 15% = 90% max deployed capital
- Leaves 10% cash buffer for slippage, stop fills
- Keeping size flat isolates the cap effect

### Everything else unchanged
- Same 20-symbol universe (S19 pre-committed list)
- Same Connors rules, same costs, same holdout dates
- No other parameter touched

## Research

### Why the cap was 4 in S17
- Arbitrary default copied from the S16 ML simulator
- Never rationalized — it was a reasonable conservative choice for
  capital allocation on 6 symbols
- With 6 symbols, cap=4 rarely binds (0% in S17 holdout)
- With 20 symbols, cap=4 binds 41% of the time

### Expected impact
If blocked trades were of similar quality to taken trades (reasonable
assumption — first-come-first-served is alphabetical, not signal-ranked):
- Reclaim ~10–14 of the 24 blocked entries (some still block at cap=6)
- WR ~70% × ~12 new trades × ~$50 avg → ~$350–500 additional P&L
- Expected CAGR: **~+12 to +14%**, Sharpe **~1.1 to 1.3**

### Risk: correlation clustering
More concurrent positions during a cluster = larger simultaneous drawdown.
Expected MaxDD increase: 5.2% → ~7–9% (still well under G3 30% limit).

## Acceptance Criteria

### Must-have
- [ ] **AC1:** `sim_swing_rules.py` accepts `--max-concurrent N` flag
- [ ] **AC2:** Default stays 4 (S17/S19 reproducibility preserved)
- [ ] **AC3:** Expanded universe run with `--max-concurrent 6` completes
- [ ] **AC4:** Signal audit block rate reported
- [ ] **AC5:** Side-by-side comparison: S19 (cap=4) vs S20 (cap=6)

### Go/no-go gates (same as S17/S19)
- [ ] **G1:** Holdout Sharpe > 1.0
- [ ] **G2:** CAGR > benchmark + 2%
- [ ] **G3:** MaxDD < 30%
- [ ] **G4:** Profitable ≥2/3 yrs (or 1/2 partial)

### S20 success criterion
- [ ] **S20.1:** G1 PASSES cleanly (Sharpe ≥ 1.05, giving 5% buffer above threshold)

A marginal pass (Sharpe 1.00–1.04) is **not** considered a success —
it would mean we barely moved the needle after tuning, which is
consistent with noise.

## Technical Design

### Files to Modify
| File | Change |
|------|--------|
| `scripts/sim_swing_rules.py` | ADD `--max-concurrent` CLI flag, override `MAX_CONCURRENT` at runtime |
| `specs/S20-position-cap-6.md` | THIS FILE |
| `STATE.md` | UPDATE after UAT |

### Implementation
Single-line simulator change: read `args.max_concurrent` and pass to
`run_rules_sim`. The existing code already references `MAX_CONCURRENT` as
a module constant — we override it locally.

## Verification

```bash
# Baseline (S19 reproduction — must match previous run)
PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe expanded

# S20 test
PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe expanded --max-concurrent 6
```

### Expected comparison
| Metric | S19 (cap=4) | S20 target (cap=6) |
|--------|-------------|---------------------|
| Trades | 35 | 45–50 |
| Block rate | 40.7% | 15–25% |
| CAGR | +9.06% | +12–14% |
| Sharpe | 0.988 | ≥1.05 |
| MaxDD | 5.23% | 6–9% |

## UAT (completed 2026-04-21)

### Gates & success criteria — ALL PASS

| Check | Target | Actual | Status |
|-------|--------|--------|--------|
| G1 Sharpe > 1.0 | >1.0 | **1.064** | ✅ PASS |
| G2 CAGR > BM+2% | +2% | +63.91% alpha | ✅ PASS |
| G3 MaxDD < 30% | <30% | 7.68% | ✅ PASS |
| G4 Profitable ≥2/3 yrs | 1/2 partial | 1/2 | ✅ PASS |
| S20.1 Sharpe ≥ 1.05 | 1.05 | **1.064** | ✅ PASS (not marginal) |

### Holdout vs prior iterations

| Metric | S17 (6 coins, cap=4) | S19 (20, cap=4) | **S20 (20, cap=6)** |
|--------|----------------------|------------------|---------------------|
| CAGR | +7.55% | +9.06% | **+13.39%** |
| Sharpe | 1.079 | 0.988 | **1.064** |
| Trades | 18 | 35 | **44** |
| Win rate | 72.2% | 71.4% | **70.5%** |
| MaxDD | 3.88% | 5.23% | **7.68%** |
| Block rate | 0% | 40.7% | **21.4%** |

### Per-symbol breakdown (holdout, cap=6)

- **Top 3:** ETH (+$395, 100% WR), BCH (+$328), LINK (+$196, 100%) — unblocked by cap
- **Losers** (unchanged from S19, pre-committed, not removed): DOGE, ADA, ARB (~−$325 combined)
- Dead symbols (0 trades): DOT, ATOM, NEAR, APT, OP, SUI, MATIC (delisted)

### Honest assessment

Three successive passes on the same holdout (S17 → S19 → S20), each
"better" than the last, is a classic snooping trajectory. The reported
+13.39% CAGR and Sharpe 1.064 must be treated as **upper-bound
estimates** — real-world paper-forward will likely land at:
- CAGR: +6–9%
- Sharpe: 0.7–1.0
- MaxDD: similar (5–10%)

The STRUCTURAL finding holds independent of the tuning:
1. Rules-based beats ML on same data (confirmed twice)
2. Position cap was binding — that was a real mechanical fix
3. Win rate preserved across changes (~70% consistently)

## Lessons Learned

1. **Diagnostics should drive parameter changes, not optimization.**
   S20 fixed an observed bottleneck (40% block rate); that's a principled
   change. If we had instead tried cap=3, 4, 5, 6, 7, 8 and picked the
   best — that's snooping. Always ask "am I responding to a mechanism,
   or fitting the curve?"

2. **Three looks at one holdout is two too many.** Even principled
   changes accumulate snooping risk. After S17 we should have paper-
   forwarded immediately. S19/S20 sharpened the config but we've now
   exhausted the honest information content of this particular window.

3. **Expansion was not a linear multiplier.** Going from 6 → 20 symbols
   doubled trade count but only lifted CAGR by ~1.5pp (before S20's
   unlocking). Correlation clustering + position cap masked most of
   the gain. This is the kind of thing only a backtest can reveal.

4. **Half the symbols are dead weight.** DOT, ATOM, NEAR, APT, OP, SUI
   produced zero trades. That's useful intelligence: if we were to
   shrink the universe, we know where to cut. But doing so now would be
   snooping.

## Next Step

**→ S18: paper-forward validation.** Wire the cap=6, expanded-universe
Connors rules into a daily cron and run for 90 days live. Compare live
P&L, slippage, and trade count against backtest projections. Only after
that signal is clean do we consider real capital.

## Notes

### Frozen configuration for S18 paper-forward
| Parameter | Value |
|-----------|-------|
| Universe | S19 pre-committed 20 crypto |
| Rules | Connors: Close>SMA(200) & RSI(2)<10 & Close<SMA(5) & ADX(14)≥20 |
| Exit | Close>SMA(5) \| 10d max hold \| −7% SL |
| Position size | 15% per trade |
| Max concurrent | **6** (S20) |
| Costs | 20bps/side + 5bps slip + 10bps SL slip |

This configuration is **frozen for paper-forward**. No more backtest tuning.

### Decision after results (reference — executed above)

| Outcome | Action |
|---------|--------|
| S20.1 PASS (Sharpe ≥ 1.05) | Draft S18 paper-forward using cap=6 |
| G1 marginal pass (Sharpe 1.00–1.04) | Treat as noise; revert to S17 config, start paper-forward |
| G1 fails again | Accept S17 as ceiling; edge is genuinely weaker than hoped; move on |
| DD explodes (>15%) | Revert to cap=4; clustering risk is real |

### Why we stop here regardless of result
After S20, further tuning on this same holdout is pure snooping. The next
evidence-gathering must come from **new data** (paper-forward) or
**new windows** (rolling-holdout test).
