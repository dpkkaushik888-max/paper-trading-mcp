# S22: Recursive Loop + Agent Framework

**Status:** DRAFT
**Branch:** `feature/s22-loop-framework`
**Priority:** P1 (foundation for the loop-engineering redesign)
**Depends on:** None (new control-plane package)

## Overview

Introduce a recursive feedback-loop framework — a new top-level `loops/`
package — that lets the project be structured as a hierarchy of control loops:
Personal Finance (L0) ⊃ Investment/Allocator (L1) ⊃ {Equity, Crypto} (L2) ⊃
strategy loops (L3). Every loop obeys one contract: a **Mandate** flows down
(capital budget + risk limits + constraints), a **Report** flows up
(performance + regime read + confidence). Loops compose recursively.

## User Story

As the system owner, I want every level of my financial automation — from a
single strategy up to whole-portfolio allocation — to expose the same
observe→decide→act→measure→report cycle, so I can compose, schedule, and audit
them uniformly and add new asset classes or strategies as drop-in loops.

## Design Decisions

These are locked for this spec. (Numbering matches the approved plan's locked
decisions; see `/Users/deepakkaushik/.claude/plans/starry-dancing-scone.md`.)

### D1 — Two planes
Control plane = `loops/` (this spec): contracts, loop lifecycle, composition,
state, agent client. Execution plane = `trading_engine/` (existing + S23).
`loops/` depends on `trading_engine`, never the reverse. No import cycles.

### D2 — The Loop contract
`Loop` ABC with lifecycle methods `observe → decide → act → measure → report`,
orchestrated by a `run(period)` template method. `set_mandate(Mandate)` injects
the parent's mandate. `run()` returns a `Report`.

### D3 — Mandate (down) / Report (up) dataclasses
`Mandate(loop_id, issued_by, period, capital_budget, risk_limits, constraints,
horizon_days, objective, notes)`. `Report(loop_id, period, starting_value,
ending_value, period_return, max_drawdown, realized_pnl, n_trades, regime,
regime_confidence, confidence, capital_utilization, halted, halt_reason,
children, trades, diagnostics)`. Reports nest (`children`) for recursive roll-up.

### D4 — Decoupled cadence (mandate/report bus)
Children run on their own cron and read the latest published mandate at the top
of `observe()`. Parents publish mandates + aggregate already-written child
reports on their own cadence. No synchronous fan-out. A missed parent run →
child keeps last mandate (graceful degradation).

### D5 — JSON ledger, git-committed
Per-loop state is versioned JSON under `state/loops/<loop_id>/` (mandates/,
reports/, agent/), generalizing the proven `trading_engine/paper/journal.py`
pattern. Positions/cash stay in their existing inner stores. No SQLite for the
mandate/report ledger (append-mostly, benefits from git history + PR review).

### D6 — Bounded agents, always-on deterministic fallback
This spec defines the `AgentClient` contract and the deterministic fallback
allocators; wiring real agent calls into L1/L3 is delivered here but every
decision MUST have a deterministic fallback so the whole hierarchy runs headless
under `--dry-run`. Agents select only among engine-enumerated `options` and are
re-validated against `risk_limits` before `act()`. Invoked via `claude --print`
subprocess (the CodeGraph-AI `parallel_review.py` pattern) — no new dependency.

### D7 — Halt is local
`Report.halted` is informational to the parent. A halted child is treated by the
allocator as zero-confidence and its budget reallocated to siblings/cash. The
parent does not halt because a child halted. Halt-the-cron semantics stay local
to the leaf.

## Research

- `trading_engine/paper/journal.py` — versioned, migration-aware, invariant-
  enforcing JSON state store; the template `LoopState` generalizes.
- `trading_engine/paper/run_daily.py` — already an observe→decide→act→measure→
  report→halt loop in disguise; the lifecycle shape is lifted from it.
- CodeGraph-AI `parallel_review.py` — the `claude --print` subprocess pattern
  (unset CLAUDECODE, JSON in/out, timeout, debug-dump on parse error) reused for
  `AgentClient`.
- No LLM SDK in `requirements.txt` — subprocess pattern avoids adding one.

## Acceptance Criteria

- [ ] `loops/` package created: `contracts.py`, `base.py`, `state.py`,
      `composite.py`, `allocators.py`, `registry.py`, `run.py`, `agent.py`,
      plus `l0_personal.py`, `l1_allocator.py` stubs.
- [ ] `Mandate` and `Report` round-trip to/from JSON; `Report` nests children.
- [ ] `Loop.run()` calls lifecycle methods in order and persists the Report.
- [ ] `CompositeLoop` allocates a parent's `capital_budget` across children from
      their last Reports (deterministic risk-parity / confidence-weighted),
      clamped to `risk_limits`, and aggregates child Reports into one roll-up.
- [ ] `LoopState` writes/reads per-loop mandate, report, and agent-call ledgers
      under `state/loops/<loop_id>/`.
- [ ] `AgentClient` invokes `claude --print`, validates the response against
      constraints, falls back deterministically on any error, and persists the
      request/response packet.
- [ ] `python -m loops.run --loop L1.allocator --dry-run` completes with agents
      disabled, producing a valid roll-up Report (uses L0/L1 stubs + a stub L2).
- [ ] No change to any existing `trading_engine/paper/` runtime behavior.

## Technical Design

### Files to Create
| File | Change |
|------|--------|
| `loops/__init__.py` | Package marker |
| `loops/contracts.py` | `Mandate`, `Report`, `Cadence` dataclasses + JSON (de)serialization |
| `loops/base.py` | `Loop` ABC + `run()` template method |
| `loops/state.py` | `LoopState` — per-loop JSON ledgers (generalizes `journal.py`) |
| `loops/composite.py` | `CompositeLoop` — allocate across children, aggregate reports |
| `loops/allocators.py` | Deterministic fallback allocators (risk-parity, confidence-weighted) |
| `loops/agent.py` | `AgentClient`, `AgentRequest/Response`, subprocess + validate/fallback |
| `loops/registry.py` | `REGISTRY.build(loop_id)`, `bootstrap_mandate` |
| `loops/run.py` | CLI entrypoint `python -m loops.run --loop <id> [--dry-run]` |
| `loops/l0_personal.py` | `PersonalFinanceLoop` — thin stub |
| `loops/l1_allocator.py` | `AllocatorLoop(CompositeLoop)` — thin stub |
| `tests/test_loops_contracts.py` | Mandate/Report serialization + recursive roll-up |
| `tests/test_loop_lifecycle.py` | Lifecycle ordering, halt propagation, dry-run |
| `tests/test_composite_allocation.py` | Reallocation from reports, constraint clamping |
| `tests/test_agent_contract.py` | Fallback on parse error, constraint-violation rejection |

### Data Model Changes
New JSON state tree under `state/loops/<loop_id>/{state.json, mandates/, reports/, agent/}`.
Existing `state/journal.json` untouched (S23 wires the crypto leaf's inner store).

### API Changes
New CLI: `python -m loops.run --loop <id> [--period YYYY-MM-DD] [--dry-run]`.

## Dependencies
- None new. Reuses stdlib + existing `trading_engine` imports. `claude` CLI is
  optional at runtime (deterministic fallback if absent).

## Verification
- `pytest tests/test_loops_contracts.py tests/test_loop_lifecycle.py tests/test_composite_allocation.py tests/test_agent_contract.py -v`
- `python -m loops.run --loop L1.allocator --dry-run` → valid roll-up Report, no agent calls.
- Confirm `trading_engine/paper/run_daily.py --dry-run` output is byte-identical to pre-S22.

## UAT
<Filled during Phase 4.>

## Notes
- L2 crypto leaf and L3 orchestrator are delivered in **S23**, which depends on
  this framework. This spec is correct with a single stub leaf.
- Agents are bounded by construction (select-from-options + revalidate); this is
  the safety property that lets the hierarchy run unattended on paper money.
