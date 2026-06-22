# S26: Risk-Adjusted Alpha Gate

**Status:** VERIFIED
**Branch:** `feature/s26-risk-adjusted-alpha`
**Priority:** P2 (unblocks S25 promotions / S24 paper-forward)
**Depends on:** S25 (discovery loop + gates — machinery VERIFIED, 0 promotions)

## Overview

S25's discovery loop promoted 0 strategies on the live 5y crypto window. The
binding constraint was **G2/G9 — "beat buy-and-hold BTC on raw CAGR"** — because
BTC returned +32% CAGR over a bull-dominated window, which a long-only,
de-risking book structurally cannot beat on raw return. S26 changes the alpha
benchmark from **raw CAGR** to **risk-adjusted** (Sharpe-based) excess return, the
correct yardstick for a risk-managed long-only strategy, and makes it a
configurable mode so S25's strict bar is preserved and reproducible.

## User Story

As the system owner, I want the discovery gates to reward strategies that beat
buy-and-hold *on a risk-adjusted basis* — better Sharpe, smaller drawdown — not
only on raw CAGR, so a strategy that earns +13.8% at Sharpe 0.86 / 16% max-DD is
recognized as superior to holding BTC (huge CAGR, huge drawdown) when it is.

## Design Decisions

### D1 — The benchmark change is decided ON PRINCIPLE, not on this run's result
Raw-CAGR-vs-BTC has been the wrong benchmark for a risk-managed long-only book
since before S25 ran — a buy-and-hold of a 3×-ing asset is nearly unbeatable on
raw return yet carries ~70%+ drawdowns no risk mandate would accept. The fix is
justified independently of the S25 numbers. We are NOT reverse-engineering a bar
to make the breakout template pass (which would be the S17→S19→S20 tune-to-target
trap). We do not inspect BTC's window Sharpe before fixing the rule.

### D2 — Risk-adjusted alpha = Sharpe-based excess, with a profitability floor
- **WF G2 (per strategy):** `candidate.sharpe ≥ BH.sharpe` AND `candidate.CAGR ≥ 0`.
  Sharpe is already computed everywhere (it is G1/G5) — the minimal, consistent
  change. The CAGR ≥ 0 floor keeps "must actually make money" so a low-volatility
  flat strategy can't pass on Sharpe alone.
- **Holdout G9 (combined):** `candidate.sharpe ≥ BH.sharpe` AND `candidate.CAGR ≥ 0`
  (replaces "CAGR > BH AND Sharpe > BH"). The deflated-Sharpe absolute bar (D4 of
  S25) is unchanged — it still rises with trial count.
- **Sub-period robustness:** per-slice excess becomes Sharpe-difference
  (`candidate.sharpe − BH.sharpe`) in risk-adjusted mode; still requires ≥ 2/3
  slices non-negative. Raw mode keeps CAGR-difference.

### D3 — Configurable mode; raw stays the default at the function layer
`alpha_mode ∈ {"raw_cagr", "risk_adjusted"}`. The library default is `raw_cagr`
so S25's gates and tests reproduce exactly. The discovery loop / CLI opt into
`risk_adjusted`. Every promotion records the mode used in its provenance.

### D4 — Re-running the existing 5y holdout under the new gate is NOT a clean OOS
**The S25 holdout was already observed.** Re-evaluating it under a changed rule is
methodology validation, not an out-of-sample promotion. Any promotion produced
this way is written to a **demo manifest** (`registry_s26_demo.json`), flagged
`clean_oos=false` in provenance, and is NOT eligible for live capital. A clean
promotion requires one of: (a) the S24 forward paper-trade, or (b) a genuinely
unseen data window. This is the non-negotiable honesty boundary of S26.

## Research
- S25 UAT: breakout template = WF standout (Sharpe 0.86, maxDD 16.3%, 167 trades,
  +13.8% CAGR) — passed G1/G3/G4, failed only G2 vs BTC's +32% CAGR.
- `bh_btc`/`buy_hold` already return `sharpe` alongside `cagr` — no new metric
  plumbing needed.

## Acceptance Criteria
- [ ] `WFGateConfig.alpha_mode` added; `gate_candidate` applies Sharpe-based G2 in
      `risk_adjusted` mode, CAGR-based in `raw_cagr` (default).
- [ ] `gates.evaluate_holdout(alpha_mode=...)` applies Sharpe-based G9 + Sharpe-diff
      sub-period robustness in `risk_adjusted` mode; deflated-Sharpe bar unchanged.
- [ ] `research_loop` + `run_discovery.py` thread `alpha_mode`; default `raw_cagr`.
- [ ] Provenance records `alpha_mode` and `clean_oos`.
- [ ] All existing S25 tests pass unchanged (raw default preserved).
- [ ] New tests cover the `risk_adjusted` branch in search + gates.
- [ ] Methodology-demo run is clearly labelled contaminated (reused holdout), writes
      only to the demo manifest.

## Technical Design
### Files to Modify
| File | Change |
|------|--------|
| `trading_engine/discovery/search.py` | `WFGateConfig.alpha_mode`; mode-aware G2 in `gate_candidate` |
| `trading_engine/discovery/gates.py` | `alpha_mode` in `evaluate_holdout` + `subperiod_alphas`; mode-aware G9 |
| `loops/research_loop.py` | `alpha_mode` ctor arg threaded to search + gates; in provenance |
| `scripts/run_discovery.py` | `--alpha-mode` flag; demo-manifest + `clean_oos` labelling |
| `tests/test_discovery_search.py`, `tests/test_discovery_gates.py` | risk_adjusted cases |

### Data Model Changes
Promotion provenance gains `alpha_mode` and `clean_oos` keys. No schema break.

### API Changes
None external.

## Dependencies
S25 (built). No new packages.

## Verification
- `pytest tests/ -q` → all green, S25 tests unchanged.
- `python scripts/run_discovery.py --alpha-mode risk_adjusted --manifest state/discovery/registry_s26_demo.json`
  → demonstrates the gate mechanism on the reused holdout (labelled contaminated).
- Determinism: same seed + mode → identical verdicts.

## UAT — 2026-06-22 (methodology demo, reused 5y holdout, contaminated)

`run_discovery.py --alpha-mode risk_adjusted --reused-holdout --seed 0 --n-candidates 20`

- [x] **Acceptance criteria 1–7 met.** 205 tests green (7 new in `test_risk_adjusted_alpha.py`);
      all S25 tests pass unchanged (raw default preserved); provenance carries
      `alpha_mode` + `clean_oos`; run printed the contamination warning and targeted the
      demo manifest only.
- [x] **The gate change works on principle.** Under `risk_adjusted`, the breakout
      template now **passes walk-forward** (1/20, vs 0/20 under raw) — Sharpe 0.86 ≥ BH,
      +13.8% CAGR ≥ 0 — exactly the strategy raw-CAGR-vs-BTC was wrongly rejecting.
- [x] **The holdout still honestly rejects it.** On the unseen holdout year
      (2025-06→2026-06) breakout **collapsed: −25.3% CAGR, Sharpe −2.16, 0/3 sub-periods
      positive.** Promoted: **0.** The WF filter let the genuine standout through; the
      locked holdout + deflated-Sharpe + sub-period robustness killed it.

**Verdict:** The benchmark fix is correct and the full apparatus is validated end-to-end —
WF promotes the right candidate, the holdout disposes of it on real OOS performance. The
contamination risk (D4) never materialized: nothing was written to any registry. **0
promotions remains the honest answer**, now for the right reason (out-of-sample failure,
not an unbeatable raw-CAGR bar). A clean promotion still requires S24's forward
paper-trade — S26 only corrects the yardstick.

## Notes & Risks
- **Holdout reuse is the live risk.** D4 is the guard: demo-only, never live.
- **Sharpe on short sub-periods is noisy** — that is why robustness needs only
  ≥ 2/3 slices and the absolute deflated-Sharpe bar still gates the full holdout.
- **A real edge still needs S24.** S26 changes the yardstick; it does not certify a
  strategy for capital. The forward paper-trade remains the final gate.
