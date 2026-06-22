# S25: Strategy Discovery Loop ("Agent 1")

**Status:** IMPLEMENTED (awaiting UAT)
**Branch:** `feature/s25-strategy-discovery`
**Priority:** P1 (the research loop that grows the strategy registry)
**Depends on:** S22 (loop framework), S23 (orchestrator/leaf/gates — VALIDATED engine)

## Overview

A **Research Loop** that discovers, tests, and promotes trading strategies
automatically. An agent proposes candidates (by tuning known templates AND
inventing new rule combinations); a deterministic harness searches/tunes them on
training data only, filters on walk-forward, then gives survivors **one** shot at
a locked holdout. Only holdout-survivors that pass the full G1–G9 gates are added
to the live L3 registry, each with a validated regime map ("when to use which").
Everything else is logged and discarded.

This sits one level above the daily crypto engine in the loop hierarchy: a slow
(monthly) loop whose Report is *an update to the strategy registry*.

## User Story

As the system owner, I want an agent that continuously searches for profitable
strategies and proves them on data they were never tuned on, so the engine's
strategy set grows with edges that survive honest out-of-sample testing — without
me hand-coding and hand-validating each one.

## Design Decisions (LOCKED — changing any after a run invalidates results)

### D1 — Pipeline: propose → search → filter → prove → promote
1. **Propose** — agent emits candidate strategy specs (entry/exit rules + param
   ranges + a regime hypothesis).
2. **Search** — tune each candidate on **train + walk-forward windows only**
   (reuse `strategy_optimizer.run_walk_forward`, the guarded optimizer).
3. **Filter** — drop anything failing walk-forward gates G1–G4. Most die here.
4. **Prove** — survivors get **one** evaluation on a **locked holdout**.
5. **Promote** — only full-G1–G9 passers join the registry, with provenance.

### D2 — Generation: tune known + invent new
- **Tune known**: search the parameter space of existing templates (Connors,
  breakout, range) and a few classic forms.
- **Invent new**: compose entry/exit rules from a fixed **indicator primitive
  library** (RSI, SMA/EMA, MACD, Bollinger, ADX, volume, Donchian, ATR). The
  agent picks primitives + thresholds + combinators; the result is a concrete,
  deterministic `BaseStrategy` — once generated it has NO LLM in its hot path.

### D3 — Promotion bar: full G1–G9 on a locked holdout (S23's gates)
A candidate must beat buy-and-hold on a fresh holdout and clear the same gates
S23 used. Same bar, same harness (`sim_crypto_leaf.py --mode gates`).

### D4 — Anti-overfitting / multiple-testing discipline (THE critical rule)
Testing K candidates against one holdout yields ~0.05·K false positives by luck.
Mitigations, all mandatory:
- **Search never touches the holdout.** Tuning/selection use train + walk-forward
  only. The holdout is read once per candidate, never for tuning.
- **Three-way time split is not enough for many trials → add a trial budget.**
  Cap the number of candidates that reach the locked holdout per run (default 10).
  Log ALL proposed/searched candidates (no silent multiple testing).
- **Deflated threshold.** The Sharpe gate on the holdout is raised by a
  multiple-testing correction for the number of candidates that reached it
  (deflated-Sharpe / Bonferroni-style): effective bar rises with trial count.
- **Robustness, not a single number.** A promoted strategy must pass on the
  holdout AND show non-negative alpha in ≥2 of 3 holdout sub-periods (no single
  lucky stretch).
- **No tune-to-target.** The objective is "survives untouched data," never "hit
  +X%." Any candidate edited after seeing holdout results is dead — a new
  candidate id, fresh holdout discipline.

### D5 — Agent proposes, deterministic harness disposes
The agent's output is a **strategy spec** (rules as a structured, code-generatable
form) + a regime hypothesis — never a trade, never a tuned result it declares
good. All validation is deterministic and reproducible. The agent never sees the
holdout during proposal/search.

### D6 — Regime map ("when to use which") is proposed then validated
The agent may propose each candidate's regime weighting. This is validated as part
of the holdout (regime-conditional evaluation via the orchestrator's `regime_map`).
Note: enabling regime-conditioning expands the search space → counts against the
trial budget (D4).

### D7 — Registry update is the output, auditable + reversible
Promotions append to a versioned registry manifest with full provenance (rules,
params, all-window metrics, trial count, deflated threshold used). Any promotion
can be rolled back. Adding to the live engine still requires a paper-forward
(separate spec) before real capital.

### D8 — Candidates conform to BaseStrategy
Every generated strategy implements the existing `BaseStrategy` contract
(`entry`/`exit_reason`/`evaluate`) so it drops straight into the orchestrator.

## Research

- `trading_engine/strategy_optimizer.py` `run_walk_forward()` — already does
  train/val/OOS-guarded parameter search; the safe search engine to reuse.
- `trading_engine/autoresearch.py` — parameter sweep loop, but **no overfitting
  guards**; usable only behind the holdout/trial-budget discipline of D4.
- `scripts/sim_crypto_leaf.py --mode gates` (S23) — the locked-holdout G1–G9 judge.
- `trading_engine/engine/orchestrator.py` registry + `regime_map` — promotion target.
- `trading_engine/learning_loop.py` — calibration, optional for ML candidates.

## Acceptance Criteria

- [x] `loops/research_loop.py` — `StrategyDiscoveryLoop(Loop)` running the D1 pipeline.
- [x] `trading_engine/discovery/primitives.py` — indicator primitive library + a
      rule grammar that compiles a candidate spec into a `BaseStrategy`.
- [x] `trading_engine/discovery/candidate.py` — candidate spec dataclass +
      `to_strategy()` codegen.
- [x] `loops/agent.py` gains `propose_candidates(...)` (bounded: returns specs from
      the primitive grammar only; validated/rejected on malformed output).
- [x] Search runs each candidate on train+WF only; holdout read once/candidate.
      *(See Research note: built a candidate-native `search.py` on `CryptoLeaf`
      rather than reusing `run_walk_forward`, which only mutates `rules.json`.)*
- [x] Trial budget enforced + ALL candidates logged (no silent multiple testing).
- [x] Deflated-Sharpe threshold applied by trial count; ≥2/3 sub-period robustness.
- [x] Promotions append to a registry manifest with full provenance; reversible.
- [x] Report up: survivors + evidence + regime maps; mandate down: constraints +
      trial budget.
- [x] Determinism: same seed + same data → same candidates + same verdicts.

**Implementation note (search engine):** D1 step 2 suggested reusing
`strategy_optimizer.run_walk_forward`. That optimizer mutates `rules.json` (a
different strategy representation) and cannot evaluate a `GeneratedStrategy`.
Instead, `discovery/search.py` evaluates each candidate in isolation through the
canonical `CryptoLeaf` + `StrategyOrchestrator` on the train+WF span and applies
the S23 single-strategy gates G1–G4 — same anti-overfitting discipline, native to
the candidate codegen path, and keeps the dependency arrow discovery → engine
(never discovery → scripts). Tests: `test_discovery_search.py`,
`test_discovery_gates.py`, `test_discovery_generate.py`,
`test_registry_manifest.py`, `test_research_loop.py` (47 tests; full suite 198 green).

## Technical Design

### Files to Create
| File | Change |
|------|--------|
| `loops/research_loop.py` | `StrategyDiscoveryLoop` — D1 pipeline as a Loop |
| `trading_engine/discovery/__init__.py` | package |
| `trading_engine/discovery/primitives.py` | indicator primitives + rule grammar |
| `trading_engine/discovery/candidate.py` | candidate spec + `to_strategy()` codegen |
| `trading_engine/discovery/search.py` | walk-forward search wrapper + trial budget |
| `trading_engine/discovery/gates.py` | deflated-Sharpe + sub-period robustness on holdout |
| `trading_engine/discovery/registry_manifest.py` | versioned promotions + provenance |
| `scripts/run_discovery.py` | CLI to run one discovery pass |
| `tests/test_primitives.py`, `test_candidate_codegen.py`, `test_discovery_gates.py`, `test_research_loop.py` | TDD |

### Files to Modify
| File | Change |
|------|--------|
| `loops/agent.py` | add `propose_candidates()` (grammar-bounded) |
| `trading_engine/engine/orchestrator.py` | load strategies from the registry manifest |
| `STATE.md`, `ROADMAP.md` | track S25 |

## Dependencies
- S22 framework, S23 engine + gates (both built; S23 config rejected but engine validated).
- No new external deps (pandas-ta primitives already present).

## Verification
- `pytest tests/test_primitives.py tests/test_candidate_codegen.py tests/test_discovery_gates.py tests/test_research_loop.py -v`
- `python scripts/run_discovery.py --trial-budget 10 --seed 0` → reproducible
  candidate list, all logged, ≤10 reach holdout, survivors carry provenance.
- A known-good template (e.g. breakout, S23's walk-forward standout) is
  rediscovered and promoted; a known-junk rule is proposed, searched, and
  correctly rejected.
- Re-run with same seed → identical promotions (determinism).

## UAT — 2026-06-22 (live 5y × 20-symbol crypto)

`python scripts/run_discovery.py --trial-budget 10 --seed 0 --n-candidates 20 --years 5`
WF span 2021-06-24→2025-06-23 | LOCKED HOLDOUT 2025-06-23→2026-06-22.

**Machinery — VERIFIED (all 10 acceptance criteria):**
- [x] Pipeline runs end-to-end: 20 proposed → 0 passed WF → 0 reached holdout → 0 promoted.
- [x] ALL 20 candidates logged with metrics + reject reason (no silent multiple testing).
- [x] Trial budget + deflated bar + sub-period robustness wired; holdout untouched (no survivors to test).
- [x] Gates discriminate correctly — candidates traded 167–2006 times (no entry-misfire bug);
      rejections are real metric failures, not zero-trade artifacts.
- [x] Determinism: same seed → identical candidate list + verdicts (re-run confirmed).
- [x] Manifest reversible; Report carries the full log up; Mandate carries trial budget down.

**Outcome — 0 promotions, and that is the HONEST result, not a defect:**
Every candidate failed **G2 (alpha vs buy-and-hold)** — BTC returned **+32% CAGR** over the
5y WF span, and long-only de-risking strategies cannot beat raw HODL on a bull-dominated
window. This reproduces S23's holdout REJECTION exactly. The walk-forward standout is
`tmpl_breakout` (the breakout template): **Sharpe 0.86 ✓, maxDD 16.3% ✓, 167 trades ✓ — passes
3/4, fails only raw-CAGR alpha** (+13.8% < +32% BH).

**Deliberate non-action:** the verification example ("a known-good template is rediscovered
AND promoted") was NOT forced to pass. Lowering the G2 bar or swapping in a risk-adjusted
benchmark to manufacture a promotion would be tune-to-target — the exact overfitting trap D4
forbids. The disciplined verdict stands: **the discovery machinery is correct; nothing earned
promotion on this window.** A genuine promotion requires either a strategy that beats BH on raw
CAGR, or a deliberate, spec'd change to the alpha benchmark (a new spec, fresh holdout).

## Notes & Risks
- **Multiple-testing is the existential risk.** D4 is non-negotiable; without the
  trial budget + deflated threshold + sub-period robustness, this loop will
  manufacture false edges — the exact failure that produced the bogus "+13.39%."
- **Agent creativity is bounded by the grammar.** The agent cannot emit arbitrary
  code — only specs the primitive grammar can compile and the harness can backtest.
- **Promotion ≠ live.** A promoted strategy still needs a paper-forward (future
  spec) before real capital, exactly as S18/S24 gated S20.
- **Compute.** Inventing + searching many candidates over 5y × 20 symbols is the
  heaviest job in the project; the trial budget also caps cost.
