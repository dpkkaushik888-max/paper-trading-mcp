# S28: Multi-Window (Full-Cycle) Walk-Forward

**Status:** VERIFIED (mechanism) — hypothesis it tested was DISPROVEN (see UAT)
**Branch:** `feature/s28-fullcycle-wf`
**Priority:** P1 (the discovery filter currently rejects the strategies that work)
**Depends on:** S25 (loop + gates), S26 (alpha modes), S27 (grammar)

## Overview

The S27 trend-following experiment exposed a flaw in the discovery *filter*: the
walk-forward span (2021-2025) is judged as **one bull-dominated aggregate**, so a
strategy whose edge is downside protection scores a mediocre 4-year Sharpe and is
filtered out at G1/G2 — *before* it can reach the holdout where it actually wins.
The cycle-robust strategy (e.g. `TF_50_200`: the only candidate positive in the
2025-26 crash) never gets evaluated. S28 fixes the filter: slice the WF span into N
sub-windows (the 2022 bear becomes its own window) and gate on **consistency across
windows** (≥ K of N show non-negative risk-adjusted alpha), not on one aggregate
number that the bull dominates.

## User Story

As the system owner, I want the walk-forward filter to reward strategies that hold
up across *different market regimes* — credit for protecting capital in a bear
sub-window, not only for matching HODL in a bull — so cycle-robust trend-following
can survive the filter and reach the holdout for honest judgement.

## Design Decisions

### D1 — Gate on per-window consistency, not one aggregate
Split [wf_start, wf_end] into N contiguous windows (default 4 → ~1y each over a 5y
run, so 2021 top / 2022 bear / 2023 recovery / 2024-25 bull are separated). Evaluate
each candidate per window; a window is a "pass" when it shows **non-negative
risk-adjusted alpha** (Sharpe ≥ BH AND CAGR ≥ 0; raw-CAGR in raw mode). The candidate
clears WF iff it passes **≥ K of N** windows (default 3/4). This is walk-forward
*analysis* (robustness across windows), the WF-stage analogue of S25's holdout
sub-period robustness.

### D2 — Aggregate guards still apply
- **Worst-window max drawdown < ceiling** (default 30%) — no single window blow-up.
- **Total trades ≥ min** (default 12) — enough activity to trust.

### D3 — Opt-in mode; single-span stays the default
`wf_mode ∈ {"single", "windowed"}`. Library + S25/S26 defaults stay `single` so
prior results reproduce. The loop/CLI opt into `windowed`. No behavior change unless
asked for.

### D4 — Ranking by consistency, then quality
Survivors rank by (windows-passed desc, mean window CAGR desc, id) so the most
regime-robust candidates get the scarce holdout trial-budget slots.

### D5 — Holdout discipline unchanged; reused holdout is still contaminated
S28 changes only the WF *filter*. The locked-holdout judge (S25 deflated-Sharpe +
S26 risk-adjusted + sub-period robustness) is untouched. The 5y holdout has been
observed repeatedly during this research; any promotion from re-running it stays
`clean_oos=false` and is NOT live-eligible. A clean promotion still needs S24's
forward paper-trade or unseen data.

## Research
- S27 experiment (memory: trend-following-edge-is-downside-protection): `TF_50_200`
  passes the holdout risk-adjusted (+0.3%, Sharpe +0.11 in the −38.8% crash) but
  FAILS the single-span bull WF (Sharpe 0.39 < BH 0.78) → filtered out today.
- WF span 2021-2025 *does* contain the 2022 bear; windowing surfaces it instead of
  letting the recovery wash it out.
- `gates.subperiod_alphas` already proves the per-window pattern works on the holdout.

## Acceptance Criteria
- [ ] `WindowedWFConfig` + `window_bounds()` + `gate_candidate_windowed()` +
      `search_walk_forward_windows()` in `search.py`.
- [ ] `CandidateResult` carries per-window breakdown; `to_dict` includes it.
- [ ] Consistency gate (≥K/N), worst-window DD ceiling, total-trades floor; rank by
      windows-passed.
- [ ] `research_loop` + `run_discovery.py` thread `wf_mode` (+ n_windows, min_pass);
      default `single`.
- [ ] All S25/S26/S27 tests pass unchanged.
- [ ] New tests cover window splitting, consistency gating, determinism.
- [ ] UAT: windowed WF produces a per-regime consistency verdict on the real 5y data
      and is used to **empirically test** the cycle-robustness hypothesis.
      *(Outcome: hypothesis DISPROVEN — see UAT. The mechanism is validated; the
      prediction that TF would pass windowed WF was wrong, and that is the finding.)*

## Technical Design
### Files to Modify
| File | Change |
|------|--------|
| `trading_engine/discovery/search.py` | windowed WF config + gating + search; per-window `CandidateResult` |
| `loops/research_loop.py` | `wf_mode`/`n_wf_windows`/`min_windows_pass`; windowed branch in `decide` |
| `scripts/run_discovery.py` | `--wf-mode`, `--wf-windows`, `--min-windows-pass` |
| `tests/test_windowed_wf.py` | new |

### Data Model Changes
`CandidateResult.windows` (optional list of per-window dicts). No persisted-schema break.

## Dependencies
S25–S27 (built). No new packages.

## Verification
- `pytest tests/ -q` → all green incl. new windowed-WF tests.
- Experiment: TF_50_200 fails `single` WF, passes `windowed` (≥3/4) — the documented fix.
- `run_discovery.py --wf-mode windowed --wf-windows 4 --min-windows-pass 3` runs end-to-end.

## UAT — 2026-06-22

**Mechanism — VERIFIED.** `window_bounds` splits the WF span into N contiguous
windows; `gate_candidate_windowed` returns a per-window breakdown and gates on
consistency (≥K/N) + worst-window DD + total trades; survivors rank by windows
passed. 219 tests green (7 new in `test_windowed_wf.py`); all S25–S27 unchanged;
`run_discovery.py --wf-mode windowed` runs end-to-end; deterministic.

**Hypothesis — DISPROVEN (the real finding).** Ran the four trend-followers from the
S27 experiment on the real 5y WF span (2021-06→2025-06) sliced into 4 ~1y windows:

| strat | single-WF (risk-adj) | windowed (4 windows) |
|-------|----------------------|----------------------|
| TF_50_200 | fail | **0/4 → fail** |
| TF_200ride | fail | 0/4 → fail |
| TF_turtle55 | fail | 1/4 → fail |
| TF_mom | fail | 1/4 → fail |

Per-window truth (BTC by window: −39%, +44%, **+110%**, +66%):
- TFs get **crushed in the big bull windows** — they cannot match a doubling BTC.
- `TF_50_200` even **lost the 2022 bear window** (−23%, Sharpe −1.78) — caught long
  at the top; the SMA-200 exit lagged the crash.

So the S27 "TF_50_200 wins the holdout" was a **single-window artifact** (the 2025-26
crash shape), not cycle-robustness. The multi-window view caught exactly the
overfit-to-one-window conclusion it exists to catch. **Trend-following is correctly
filtered out** — it is not consistently risk-adjusted-superior to HODL across regimes.

**Verdict:** mechanism validated and now the standard tool for judging cross-regime
robustness; the cycle-robustness hypothesis is rejected on this universe/era. The
structural gap is more robust than thought: no long-only variant tested beats HODL
risk-adjusted consistently across regimes. Honest conclusion — for this 20-alt crypto
universe the edge is at best *crash-specific downside protection*, not general alpha.

## Notes & Risks
- **More windows → fewer per-window bars → noisier per-window Sharpe.** Default 4
  keeps ~1y/window over 5y; don't over-slice.
- **Consistency ≠ profit.** A strategy can pass 3/4 windows yet still fail the
  holdout (correctly). S28 widens what *reaches* the holdout; the holdout still decides.
- **Promotion still needs S24.** S28 fixes the filter, not the live-readiness bar.
