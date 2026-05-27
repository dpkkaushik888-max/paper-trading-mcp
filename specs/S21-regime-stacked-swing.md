# S21: Regime-Stacked Swing Engine

**Status:** DRAFT
**Branch:** `feature/s21-regime-stacked`
**Priority:** P1 (replaces S20-as-standalone after S18 FAIL)
**Depends on:** S20 (kept as one of three stacked strategies), S18 FAIL findings

## Overview

Combine three uncorrelated long-side swing strategies — each tuned for a
specific market regime — into a single capital-shared portfolio. Goal:
deliver at least one trade per week on average across the 20-crypto
universe, while keeping S18's honest-cost discipline and pre-committed
rule methodology.

## User Story

As a personal-capital compounder who wants weekly trading activity, I want a
portfolio of regime-specific strategies so that the engine produces signals
in every market environment (uptrend, range, downtrend) rather than going
idle for 30+ days when a single rule's regime is absent.

## Why this exists (S18 lessons)

The S18 paper-forward run produced **0 trades in 29 days** with the S20
Connors rule alone. Backtest replay (2026-05-27) confirmed this was not a
bug — it was the rule's honest verdict on the regime:

| Connors filter | Pass rate in S18 window | Status |
|---|---:|---|
| F1 Close > SMA(200) | 8.0% | 🔴 Binding — broad downtrend |
| F2 RSI(2) < 10 | 13.3% | Normal |
| F3 Close < SMA(5) | 51.2% | Healthy |
| F4 ADX(14) ≥ 20 | 55.3% | Healthy |

S20 is fundamentally a *single-regime* strategy. To trade weekly, we need
*multiple* strategies covering different regimes.

## Design Decisions

All decisions below are LOCKED before any backtest is run. Any change after
backtest = tampering = invalidated results. **Do not modify mid-backtest.**

### D1 — Three strategies, all long-side, all spot-tradeable
- **A. Uptrend Pullback** = current S20 Connors rule (kept verbatim)
- **B. Breakout Continuation** = momentum strategy, fires on new highs
- **C. Range Mean Reversion** = low-trend mean revert, fires when ADX is low

All three are **long-only**. No shorts. Rationale: shorts require margin
account / futures venue / extra regulation, and for a personal compounder
on spot the simpler "cash in downtrends" stance is honest. Strategies B
and C fire in regimes where the all-three-passive BH_BTC also struggles,
so the "beat doing nothing" comparison stays fair.

### D2 — Strategy B (Breakout Continuation) rule set
**Long entry (all must be true):**
1. `Close > rolling_max(Close[1:21])` — new 20-day high (closing basis, prior bar)
2. `Volume > 1.5 × SMA(Volume, 20)` — confirmed by volume expansion
3. `ADX(14) >= 25` — strong trend confirmation (higher bar than Connors' 20)
4. `Close > SMA(50)` — short-term trend up (relaxed vs SMA(200))

**Long exit (first true wins):**
1. `Close < SMA(10)` — short-term trend break
2. Max hold = 15 trading days
3. Hard stop = −8% from entry

### D3 — Strategy C (Range Mean Reversion) rule set
**Long entry (all must be true):**
1. `RSI(2) < 5` — *deeper* oversold than Connors (avoids overlap)
2. `ADX(14) < 18` — low-trend regime (anti-Connors filter)
3. `bb_width = (BB_upper - BB_lower) / BB_mid < 0.10` — tight Bollinger range
4. No SMA(200) trend filter

**Long exit (first true wins):**
1. `RSI(2) > 70` — mean reversion complete
2. Max hold = 7 trading days (shorter than Connors — ranges decay fast)
3. Hard stop = −5% from entry (tighter — no trend buffer to lean on)

### D4 — Frozen universe (same as S19/S20)
20 liquid crypto: BTC, ETH, SOL, AVAX, LINK, MATIC, DOGE, XRP, ADA, DOT,
ATOM, NEAR, LTC, TRX, BCH, APT, UNI, ARB, OP, SUI. No additions, no removals.

### D5 — Capital allocation
- **Shared pool**, not partitioned. All three strategies draw from the
  same $10,000 starting capital.
- **Per-position size**: 12% of equity (down from S20's 15% — to fit more
  concurrent positions across strategies).
- **Max concurrent**: 8 positions globally (up from S20's 6).
- **Per-strategy cap**: max 4 concurrent positions per strategy (prevents
  one strategy from saturating capital during its preferred regime).

### D6 — Conflict resolution: first-come, first-served per symbol
If strategy A's exit triggers on a position currently held by strategy A,
trivially exit. A symbol cannot simultaneously have positions from multiple
strategies — once strategy A opens BTCUSDT, B and C ignore BTCUSDT until
A closes it. Strategy that *first* generates a fire signal owns that
symbol-slot.

### D7 — Honest cost model (unchanged from S16/S20)
- COST_PCT = 0.0020 (20 bps/side)
- SLIPPAGE_BPS = 0.0005 (5 bps per fill)
- SL_SLIPPAGE_BPS = 0.0010 (extra 10 bps on stops)
- Total realistic round-trip: ~50 bps

### D8 — Backtest methodology
**Same purged-walk-forward + locked-holdout split as S16/S17/S19/S20.**
- 60% training/warm-up (no sim, indicators only)
- 20% walk-forward (out-of-sample run 1)
- **20% LOCKED HOLDOUT** — evaluated **exactly once**, after walk-forward
  metrics are reviewed
- 5 years of daily bars (cross-cycle coverage: bull/bear/sideways)

**Hard rule:** the holdout is evaluated *one time only*. If gates fail,
S21 is REJECTED and a NEW spec (S22+) is required for any iteration —
no re-running the holdout with tweaked parameters. This is the S17→S19→
S20 trap we are actively avoiding.

### D9 — Pre-committed gates (must pass to APPROVE)

Single-strategy gates (each strategy in isolation, walk-forward):
- **G1**: Sharpe ≥ 0.8 (lower bar than S17's 1.0 since each is one regime)
- **G2**: Beats BH_BTC by ≥ 0% (no negative alpha)
- **G3**: Max drawdown < 25%
- **G4**: ≥ 10 trades over the walk-forward window

Combined-portfolio gates (all three running together, holdout):
- **G5**: Combined Sharpe ≥ 1.0
- **G6**: Combined CAGR ≥ +8% (after honest costs)
- **G7**: Combined max DD < 20%
- **G8**: **Average ≥ 4 trades per month** across the universe (the
  "personal-compounder engagement" metric — the whole reason S21 exists)
- **G9**: Beats BH_BTC on **both** CAGR and Sharpe (S18 addendum gate)

Failing G8 alone (low frequency) is treated as a hard fail even if returns
look good. Frequency is a primary requirement, not a secondary one.

### D10 — No mid-flight retuning
If any strategy individually fails its G1-G4 single-strategy gates during
walk-forward, that strategy is **dropped**, not retuned. The portfolio
proceeds with the remaining strategies. If 2 of 3 drop, S21 is REJECTED.

### D11 — No mid-flight strategy addition
Resist the temptation to add a 4th strategy if G8 (trade frequency) misses.
Each addition is a new spec. Keep this engine focused on three.

## Research

- **S17/S19/S20** demonstrated rule-based + honest costs + pre-committed
  rules can produce a defensible holdout result. That methodology is reused
  verbatim.
- **S18** demonstrated single-regime strategies have month+ idle stretches.
  Direct motivation for the three-strategy stack.
- **Connors (2009)** — strategy A is unchanged from his published rules.
- **Donchian breakout (1960s) + ADX confirmation** — basis for strategy B,
  modified for crypto's higher volatility (8% stop vs traditional 2-3%).
- **Bollinger squeeze + RSI(2) extreme** — basis for strategy C, drawn
  from low-volatility mean reversion literature.

## Acceptance Criteria

### Backtest phase
- [ ] `scripts/sim_swing_stacked.py` runs end-to-end on 5y daily bars
- [ ] Each strategy's per-symbol audit produced separately
- [ ] Walk-forward results table for A, B, C, and combined
- [ ] Holdout result computed exactly once
- [ ] All gates G1-G9 evaluated and reported pass/fail
- [ ] Per-strategy and combined equity curves saved to `docs/s21/`

### If APPROVED (all gates pass)
- [ ] `trading_engine/strategies/breakout_continuation.py` (strategy B)
- [ ] `trading_engine/strategies/range_meanrev.py` (strategy C)
- [ ] `trading_engine/strategies/connors_swing.py` — unchanged (strategy A)
- [ ] `trading_engine/engine/multi_strategy.py` — capital-shared orchestrator
- [ ] Tests covering: per-strategy entry logic, exit logic, conflict
      resolution, per-strategy cap
- [ ] S22 drafted as next paper-forward (with stricter D10 — halt at 14
      days of inactivity since stacked engine should never be that idle)

### If REJECTED (any gate fails)
- [ ] Document which gate(s) failed and root cause in this spec
- [ ] Mark S21 as REJECTED, do NOT iterate this spec
- [ ] Decide: drop the failing strategy(s) and proceed (S22), or rethink
      from scratch (no S22 from this draft)

## Technical Design

### Files to create
| File | Role |
|------|------|
| `trading_engine/strategies/breakout_continuation.py` | Strategy B rule functions |
| `trading_engine/strategies/range_meanrev.py` | Strategy C rule functions |
| `trading_engine/engine/multi_strategy.py` | Capital-shared orchestrator |
| `scripts/sim_swing_stacked.py` | Backtest harness — extends sim_swing_rules.py |
| `tests/test_breakout.py` | Unit tests for strategy B |
| `tests/test_range.py` | Unit tests for strategy C |
| `tests/test_multi_strategy.py` | Orchestrator tests (conflict resolution, caps) |

### Files to modify
| File | Change |
|------|--------|
| `trading_engine/strategies/connors_swing.py` | None — kept verbatim |
| `STATE.md` | Track S21 progress |
| `ROADMAP.md` | Add S21 + planned S22 paper-forward |
| `BACKLOG.md` | Mark "stacked regime engine" as in-progress |

### Combined backtest output schema
Per strategy: trades, WR, profit factor, Sharpe, max DD, CAGR, signal audit
Combined: equity curve, trade overlap matrix, regime contribution
(% of trades from each strategy by month)

## Dependencies

- No new external deps. All indicators (RSI, SMA, ADX, BB) already in
  pandas-ta which is used by `connors_swing.py`.
- 5 years of Binance daily bars (same data path as S17/S20).
- Existing test infra (pytest).

## Verification

### Per-strategy unit tests (TDD before implementation)
- Strategy B: entry fires on synthetic new-high+volume bar; doesn't fire
  without volume; SL/MAX_HOLD/exit-by-MA cases each tested
- Strategy C: entry fires on synthetic RSI<5 + low-ADX bar; doesn't fire
  when ADX is high; exit-by-RSI / SL / max-hold each tested
- Orchestrator: capital cap, per-strategy cap, conflict resolution all
  verified with synthetic multi-symbol fixtures

### Backtest verification
- Reproducibility: two runs of the same backtest produce identical metrics
  (deterministic numpy random seed where applicable)
- Per-strategy isolation: running strategy A alone via the orchestrator
  reproduces the S17/S20 standalone result exactly (sanity check on
  orchestrator overhead)
- Audit: signal-audit log shows per-strategy fire counts that match the
  per-symbol equity changes

### Gate evaluation
- All G1-G9 results printed at end of backtest run
- Reject script exits with non-zero if any gate fails (CI-friendly)

## Notes & Risks

- **Capital starvation under correlated regimes.** All three strategies
  could simultaneously have no setups if the market is in a deep tight
  downtrend with low volatility. Mitigation: G8 measures this directly
  over the holdout — if combined trades < 4/month average, the engine
  itself is not viable for the personal-compounder use case.
- **Strategy correlation.** A and C both involve RSI-style oversold logic.
  If they fire on the same symbols on the same days they are not actually
  uncorrelated. Per-strategy audit + D6 conflict resolution mitigate this
  but the holdout result is the honest test.
- **Overlapping universe risk.** All three strategies share the same 20
  symbols. A bad month for crypto = bad month for the engine. Future S23+
  could add equity index ETFs or commodities for true asset-class diversity.
- **Frequency vs win rate trade-off.** Strategy B (breakout) historically
  has lower WR (~40-50%) but larger winners. Combined portfolio WR may
  look lower than S17/S20's 70% — that's expected and OK if PF holds up.
- **No live deployment from this spec.** S21 ends at the backtest gate.
  A separate S22 paper-forward is required before any live deployment,
  with stricter early-termination triggers (14 days of inactivity, not 30).
