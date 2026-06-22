# S23: Crypto Leaf + L3 Regime-Aware Orchestrator

**Status:** HOLDOUT REJECTED (2026-06-22) — framework/engine VALIDATED, but the
long-only 3-strategy config failed combined gates G5/G6/G7/G9 on the 5y locked
holdout. Do NOT iterate this spec (D10/D11); a new spec (S25+) is required. See
"Holdout result" below.
**Branch:** `feature/s23-crypto-leaf`
**Priority:** P1 (first compliant L2 leaf for the loop hierarchy)
**Depends on:** S22 (Loop framework), S20 (validated single-strategy baseline)
**Absorbs:** S21 (Regime-Stacked Swing Engine) — its locked rules D1–D11 are
carried in verbatim; S21 is marked SUPERSEDED-BY-S23, not rejected.

## Overview

Refactor the existing crypto trading logic into the first compliant **L2 leaf**
of the loop hierarchy, with an internal **L3 `StrategyOrchestrator`** that wires
market regime into strategy selection (the missing link today), runs a
capital-shared stack of long-only strategies, and returns a `Report`. Built
**alongside** the live S18 run with full physical isolation — the live loop is
never modified.

## User Story

As the crypto-loop owner, I want a regime-aware orchestrator that runs multiple
pre-committed strategies over a shared capital pool, so the leaf produces signals
across uptrend/range/downtrend regimes instead of going idle for a month — while
keeping S18's honest-cost discipline and the live run untouched.

## Design Decisions

### D-iso — Physical isolation of the live S18 run (HARD CONSTRAINT)
Never modify `trading_engine/paper/{run_daily,config,journal}.py`,
`state/journal.json`, `docs/`, the existing live workflow, or `connors_swing.py`.
The new leaf writes to `state_v2/` + `docs_v2/` via a separate workflow and the
`trading_engine/paper_v2/` module path. Shared code is limited to pure,
side-effect-free helpers (`paper/data.py`, `paper/benchmarks.py`, the unchanged
`connors_swing.py` rules).

### D1–D11 — Carried verbatim from S21
Three long-only strategies (A=Connors uptrend pullback, B=breakout continuation,
C=range mean-reversion); rule sets per S21 D2/D3; frozen 20-coin universe (D4);
shared capital pool, 12% per position, 8 global / 4 per-strategy caps (D5);
first-come-per-symbol conflict resolution, priority A>B>C (D6); honest costs
(D7); purged walk-forward + locked-holdout, evaluated once (D8); pre-committed
gates G1–G9 (D9); no mid-flight retuning / no 4th strategy (D10/D11). See
`specs/S21-regime-stacked-swing.md` for the full locked text.

### D-orch — Orchestrator is pure decision logic
`StrategyOrchestrator` takes a `PortfolioView` (positions, cash, equity) +
per-symbol snapshot + regime, and returns `list[Order]`. It owns no cash or
persistence — the L2 leaf executes orders against its journal. This lets the
same orchestrator run in backtest, paper-forward, and unit tests (the S21
"strategy A alone reproduces standalone" sanity check).

### D-regime — Regime wired but OFF for the first validated holdout
Build `regime_map: dict[RegimeState, dict[str, float]]` (the requested missing
link) but DEFAULT to all-weights-1.0 / gating disabled for the S23 holdout, so
S21's locked unconditional-stack methodology is not invalidated (the
S17→S19→S20 data-snoop trap). Regime-gated weights are evaluated only as a
separate future spec's one-shot holdout.

### D-exit — Native strategy exits authoritative
Each strategy's native exit (Connors MR/MAX_HOLD/SL; B's MA-break/8% stop; C's
RSI70/5% stop) is authoritative. `ExitManager` is the shared SL/time-stop spine
+ per-strategy config registry, passed in via constructor (not edited).

## Research

- `scripts/sim_s21_window.py` — already implements B/C entry/exit + the
  shared-pool + conflict-resolution loop; that logic is lifted into strategy
  classes + the orchestrator and the script is kept as a regression oracle.
- `trading_engine/strategies/__init__.py` — `BaseStrategy`/`StrategySignal`/
  `StrategyConfig` is the stable contract the orchestrator depends on.
- `trading_engine/regime/regime_filter.py` — `RegimeFilter.evaluate()` returns
  `RegimeResult(state, confidence, allows_new_longs, is_caution)`; currently
  computed in `time_machine.py` but only logged. The orchestrator consumes it.
- `scripts/sim_swing_rules.py` — the purged walk-forward + locked-holdout split
  (`TRAIN_PCT/WALKFWD_PCT/HOLDOUT_PCT`, `--holdout-only`) reused for validation.

## Acceptance Criteria

### Backtest phase
- [ ] `trading_engine/engine/orchestrator.py` — `StrategyOrchestrator` (Order,
      PortfolioView, RegimePolicy, regime_map) with shared-pool allocation, caps,
      A>B>C conflict resolution, regime wiring (default off).
- [ ] `ConnorsSwingStrategy` wraps `connors_swing.py` verbatim; B and C promoted
      from `sim_s21_window.py` into strategy classes.
- [ ] Orchestrator with **only Connors** reproduces standalone S17/S20 numbers.
- [ ] Full stack reproduces `sim_s21_window.py` numbers for the same window
      (regression oracle).
- [ ] `trading_engine/engine/crypto_leaf.py` — `CryptoLeaf` accepts a Mandate,
      runs the orchestrator per day, returns a Report.
- [ ] `loops/l2_crypto.py` adapts `CryptoLeaf` to the S22 `Loop` contract.
- [ ] `scripts/sim_crypto_leaf.py` runs walk-forward + locked-holdout, evaluates
      gates G1–G9 once, exits non-zero on any gate fail.

### Paper-forward phase (delivered as the validation track; full run is S24)
- [ ] `trading_engine/paper_v2/` isolated runner writes only to `state_v2/`.
- [ ] `.github/workflows/paper-forward-v2.yml` — separate cron, 14-day halt.
- [ ] Isolation proof: `paper_v2` dry-run touches neither `state/journal.json`
      nor `docs/`; live `paper/run_daily.py` byte-unchanged.

## Technical Design

### Files to Create
| File | Change |
|------|--------|
| `trading_engine/engine/__init__.py` | New package |
| `trading_engine/engine/orchestrator.py` | L3 `StrategyOrchestrator`, `Order`, `PortfolioView`, `RegimePolicy`, `regime_map` |
| `trading_engine/engine/crypto_leaf.py` | L2 `CryptoLeaf` (owns capital/journal/execution) |
| `trading_engine/strategies/connors_strategy.py` | `ConnorsSwingStrategy` (wraps `connors_swing.py`) |
| `trading_engine/strategies/breakout_continuation.py` | Strategy B (S21 D2) |
| `trading_engine/strategies/range_meanrev.py` | Strategy C (S21 D3) |
| `trading_engine/paper_v2/{__init__,config,journal,run_daily,build_dashboard}.py` | Isolated paper-forward runner |
| `loops/l2_crypto.py` | `CryptoAssetLoop(CompositeLoop)` adapter |
| `scripts/sim_crypto_leaf.py` | Backtest harness (walk-forward + holdout + gates) |
| `.github/workflows/paper-forward-v2.yml` | Isolated cron |
| `tests/test_orchestrator.py`, `test_connors_strategy.py`, `test_breakout.py`, `test_range.py`, `test_crypto_leaf.py` | TDD per S21 verification |

### Data Model Changes
New `state_v2/journal.json` (extends the live journal schema with `strategy` and
`regime` fields). New `docs_v2/`. Existing `state/`, `docs/` untouched.

### API Changes
New CLI: `python -m trading_engine.paper_v2.run_daily [--dry-run]`;
`python scripts/sim_crypto_leaf.py [--holdout-only]`.

## Dependencies
- S22 (Loop framework) for the `Loop`/`Mandate`/`Report` contract.
- 5y Binance daily bars (same source as S17/S20).
- All indicators already in `pandas-ta`.

## Verification
- `pytest tests/test_orchestrator.py tests/test_connors_strategy.py tests/test_breakout.py tests/test_range.py tests/test_crypto_leaf.py -v`
- `python scripts/sim_crypto_leaf.py --holdout-only` prints G1–G9 and exits non-zero on fail.
- Regression: orchestrator path reproduces `sim_s21_window.py`; Connors-only reproduces S17/S20.
- Isolation: `python -m trading_engine.paper_v2.run_daily --dry-run` writes nothing to `state/` or `docs/`.

## UAT
<Filled during Phase 4.>

## Holdout result (2026-06-22) — STACK REJECTED, framework VALIDATED

Ran `scripts/sim_crypto_leaf.py --mode gates --years 5` (purged walk-forward +
locked holdout, evaluated **once** per D8). Split: walk-forward 2024-06→2025-06,
holdout 2025-06→2026-06 (a brutal crypto bear — BH_BTC −38.7%).

**Walk-forward (each strategy in isolation):**
| Strategy | CAGR | Sharpe | Max DD | Trades | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|:--:|:--:|:--:|:--:|
| A_connors | +3.3% | 0.31 | 8.2% | 80 | ✗ | ✗ | ✓ | ✓ |
| B_breakout | +22.9% | 1.29 | 13.8% | 38 | ✓ | ✗ | ✓ | ✓ |
| C_range | +0.3% | 0.35 | 0.7% | 2 | ✗ | ✗ | ✓ | ✗ |

**Locked holdout (combined stack):** ret −18.15%, CAGR −18.2%, Sharpe −1.18,
max DD 24.4%, 83 trades (6.9/mo). BH_BTC: CAGR −38.7%, Sharpe −0.93.

| Gate | Target | Result | |
|---|---|---|:--:|
| G5 | combined Sharpe ≥ 1.0 | −1.18 | ✗ |
| G6 | combined CAGR ≥ +8% | −18.2% | ✗ |
| G7 | combined max DD < 20% | 24.4% | ✗ |
| G8 | ≥ 4 trades/month | 6.9/mo | ✓ |
| G9 | beats BH_BTC on CAGR & Sharpe | CAGR yes, Sharpe no | ✗ |

**Verdict: 10 gates failed → S23 stack REJECTED on the holdout (per D9/D10).**

Root cause: the holdout was a sustained downtrend. The long-only stack lost less
than buy-and-hold (−18% vs −39%) but came nowhere near the absolute gates
(+8% CAGR / Sharpe 1.0) — long-only strategies cannot hit those bars in a bear
year. B_breakout was the only walk-forward standout (Sharpe 1.29); A and C were
weak. Trade frequency (G8) was the only combined gate met.

**Per D10/D11: do NOT iterate S23.** Options for a new spec (S25+): (a) drop A/C
and run breakout-only, (b) add a short/hedge sleeve so the engine isn't purely
long, (c) regime-gate entries (the wired-but-off `regime_map`) and re-validate
on a *fresh* holdout. The **framework, orchestrator, leaf, gates, and agent are
validated and reusable** — only this particular long-only 3-strategy config is
rejected. The methodology did its job: it caught an unviable config honestly
before any capital, instead of data-snooping it into looking good.

## Notes & Risks
- **S21 is DRAFT and gated on a one-shot holdout that may REJECT it.** The
  framework depends on `BaseStrategy`, not S21's specific strategies — if the
  stack fails its gates, the leaf falls back to single Connors (today's
  behavior) with zero framework change.
- **No live cutover in this spec.** Promotion of `paper_v2` over the live S18
  loop is a future spec, only after S23 gates pass AND a ≥90-day S24
  paper-forward beats BH on CAGR+Sharpe with ≥4 trades/month.
