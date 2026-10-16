# S35: Strategy Pool

**Status:** BUILT (C1–C4 on feat/shadow-variants; awaiting first Action run; PR pending)
**Branch:** feat/shadow-variants
**Priority:** P2 (observation only; the frozen S18 forward test is untouched)
**Depends on:** S18 (paper-forward), S33 (backfill), S34 (shadow variants; C1–C3 detail lives in its "Pool registry" subsections)
**Design:** POOL-DESIGN-2026-10-16 (§1–§14), owner rulings §14 of 2026-10-16

## Purpose
Run several pre-registered paper-only strategy lines beside the frozen S18 line, score each against live
with honest confidence statements, and give the owner monthly keep / retire / promote **recommendations**.

## Non-goals
- Never changes live S18: no line reads or writes live state (read-only adapter, golden live-path fixture,
  step refuses `live_s18`, mutator static test). `trading_engine/paper/config.py` stays frozen.
- Recommendations only. Nothing in code changes a line's `status`; the owner rules.
- No parameter may come from the 100-day live window (§7 registration gate).

## Registry and lines (`trading_engine/paper/pool/registry.py`)
A parameter change is a new id; lines are never edited in place. `MAX_ACTIVE = 10` (owner ruling §14.1).
All lines registered 2026-10-16. Pool lines use `data_guard=True` (skip entries in stale / gap200 symbols);
the four S34 shadows keep live data handling (`data_guard=False`) so their series is unchanged.

| id | family | hypothesis | baseline | years > S18 | status |
|---|---|---|---|---|---|
| shadow_atr_stop | connors | beats_live | state/pool_baselines/shadow_atr_stop.json | 6/7 | active (S34) |
| shadow_btc200 | connors | reduces_dd_in(below200) | shadow_btc200.json | 3/7 (hypothesis supported) | active (S34) |
| shadow_rsi15 | connors | beats_live | shadow_rsi15.json | 4/7 | active (S34) |
| shadow_combo | connors | beats_live | shadow_combo.json | 5/7 | active (S34) |
| breakout_s21 | breakout | earns_when_live_idle | breakout_s21.json | 5/7 | active |
| xsec_mom_top3 | xsec_momentum | beats_live | xsec_mom_top3.json | 5/7 | active |
| basket_sma50_band | basket_trend | reduces_dd_in(below200); secondary beats_live | basket_sma50_band.json | 5/7 | active |
| connors_guarded | connors | control | connors_guarded.json | 0/7 (control waiver) | active (control) |
| connors_exit_sma10 | connors | beats_live | connors_exit_sma10.json | 5/7 | active |

S34 shadows carry a `baseline_waiver` (registered before the pool); their baselines are still committed and
their `years_beating_s18` is read for promotion eligibility.

## Context dimensions (frozen constants, `pool/context.py`)
`btc_trend` (above_both / above200_below50 / below200_above50 / below_both vs SMA50/200), `btc_vol20_bucket` (terciles frozen at
**0.4330 / 0.6134** annualised 20d vol: low / mid / high), `basket_ret20_bucket` (edges **−10 %, 0, +10 %**),
`dow`. Stale symbols are excluded from the basket. The design's `symbol` dimension is not built.

## Scoring (`pool/scoring.py`; detail in S34 "C3 scoring")
Unit = closed trade, net `pnl_pct`. Cluster bootstrap by entry date, **B = 5000** (2000 for condition cells),
**90 %** percentile CI. Win probability Beta posterior with prior strength **k = 10** pseudo-trades at the
baseline win rate (Beta(1,1) without one). Paired line − live daily returns: stationary block bootstrap,
**mean block 5** days, p = 2 min((#≤0 + 1)/(B + 1), (#≥0 + 1)/(B + 1)). **Holm** at family **α = 0.10** across
active lines with `hypothesis.kind != "control"`. Tiers: < 8 trades or < 5 entry days insufficient,
8–14 indicative, ≥ 15 usable. Seeds = hash(line id, purpose[, bucket]) — stable across days.

## Verdict rules (`pool/verdicts.py`; recommendations, in precedence order)
1. LIVE (`live_s18`), NOT-STEPPED (status candidate), CONTROL (control line; shows `guard_effect` =
   control − live return over the common window; precedes HARD-KILL because the live halt covers it; never promotes).
2. HARD-KILL: max DD > 25 % or ≥ `streak_threshold` consecutive losing trades, where
   `streak_threshold = max(10, min{k : (1 − p_backtest)^k < 0.01})`, p_backtest = baseline full-period win rate
   (0.5 default → k = 7 → 10). Current values: basket_sma50_band 15 (p 0.279), breakout_s21 12 (p 0.339),
   all others 10. See D2.
3. RETIRE-cand: ≥ 60 days, ≥ 15 trades, Holm p_adj < 0.10 with mean d < 0.
4. DORMANT: rows ≥ 30 and signals < 5 · rows / 90 (rows = LineDay rows in the trailing 90 journal days;
   skipped days have no row). The design's "waiting for condition" exemption is applied by the owner, not code.
5. PROMOTE-cand: ≥ 90 days, ≥ 25 trades, Holm-significant positive (p_adj < 0.10, mean d > 0), mean d > 0 in
   both halves, baseline `years_beating_s18` ≥ 4.
6. KEEP; at ≥ 180 days rule INCONCLUSIVE-180 with flag "inconclusive 180d; owner rules monthly" (ruling §14.3).

**Hypothesis guard (D3):** for reduces_dd_in and earns_when_live_idle lines (`primary_test_available = False`)
RETIRE-cand and PROMOTE-cand are suppressed with reason "primary hypothesis test not built (C5)";
HARD-KILL, DORMANT and INCONCLUSIVE-180 still apply.

## Registration gate and backtest hook
A line becomes active only with a committed baseline whose full history beats S18 in ≥ 4/7 years or that
supports its stated hypothesis; the control passes via `baseline_waiver`; a line failing the gate is
`candidate` (not stepped). `scripts/pool_backtest.py --line <id>|all` reuses the real `book.step` and family
code over cached daily bars (`data/bars/{com,us}/`, gitignored), writes `state/pool_baselines/<id>.json` with
`generated_from` (git sha + dirty flag), per-year chained returns (`years_method = "chained"`), buckets and the
gate verdict. It runs locally or manually, **never in the daily Action**.

## Daily flow and report
`run_daily` steps each active line after the live day (exceptions isolated into `pool_errors`), then writes
`state/pool_report.json` atomically (temp file + `os.replace`); a scoring failure lands in
`pool_errors["_scoring"]` and never fails the run. CLI: `scripts/pool_report.py --print`.

## Dashboard (`scripts/build_dashboard.py` + `pool/dashboard.py`)
Server-rendered static HTML block inserted after the S34 "Shadow variants" table (kept for continuity; shadow
rows in the pool table carry an "S34 shadow" tag). Source: `state/pool_report.json`, else `score_pool` over
the journal; a malformed report or any failure renders a "pool report unavailable" box and the build
continues. Live sections are byte-identical to the pre-S35 page (tested). `data.json` gets key `pool`
(verdict aggregates only). Sections:
- **Lines table**: id, family, hypothesis, status, days, value, return, vs live (pp), trades / entry days,
  win % with 90 % CI, posterior P(win), profit factor, MaxDD, exposure, tier, Holm p_adj, verdict badge
  (KEEP grey, PROMOTE-cand green, RETIRE-cand orange, HARD-KILL red, DORMANT blue, CONTROL purple + guard
  effect, NOT-STEPPED light grey, INCONCLUSIVE-180 yellow) with the auto-sentence as tooltip and "why" expander.
  Header: "Verdicts are recommendations; the owner rules. Live S18 is never changed by this table."
- **Condition × line heatmap**, one table per dimension: cell = posterior mean win probability on a diverging
  red/green scale around 0.5, text "n=.. w=..%", insufficient cells hatched ("n<8"). Caption "Exploratory;
  never drives verdicts".
- **Weekly triggering**: last 8 ISO weeks × lines: signals / entries / blocked, with cap · gate · cash · data.
- **Pool errors and data flags**: today's `pool_errors`, `stale_symbols`, `data_flags`, control guard effect.
The pinned progress issue (`scripts/update_issue.py`) gets one line per pool line with verdict and tier.

## Risks (design §9)
- journal growth (~+0.8 MB/year): if `state/journal.json` > 5 MB, move `lines` history to `state/pool_days.jsonl`
  (pre-agreed trigger; 580 KB today).
- Action runtime (15 min timeout): pool + scoring target < 60 s, logged; ~0.1 s scoring on the 120-day replay.
- Correlated lines: Connors variants share entries; Holm stays valid under dependence but is conservative.
- Survivorship: the 20-coin universe is today's survivors; baselines are optimistic for every line equally.
- 2021 dominance of the .com 2020–2025 baselines; priors capped at k = 10. Venue: baselines .com, live .us.

## Decisions

### D1 — Strategy pool beside the frozen S18 line (OWNER TO ACCEPT; design §12 draft)
Paper-only lines run beside S18 from a declarative registry (`paper/pool/registry.py`). Each line is
pre-registered with one primary hypothesis and a committed full-history baseline (≥ 4/7 years beating S18, or
supporting its hypothesis). Lines never read or write live state, and a line's parameters are immutable (a
change means a new id). Verdicts: RETIRE-candidate after ≥ 60 days and ≥ 15 trades when the Holm-adjusted
(alpha 0.10, all active non-control lines) block-bootstrap paired test of daily (line − live) return is
significantly negative. Hard kill at DD > 25 % or a losing streak ≥ the D2 threshold. PROMOTE-candidate after
≥ 90 days and ≥ 25 trades when it is significantly positive and has the same sign in both window halves.
Promotion requires an owner decision, a new spec and a restart of the 90-day clock (D4). The condition heatmap
is exploratory and never drives a verdict. Max 10 active lines (owner ruling §14.1). The four S34 shadows run
unguarded (live data handling); `connors_guarded` is the control that measures the data-guard effect
(ruling §14.2).

### D2 — HARD-KILL streak threshold scales with the backtest win rate (OWNER TO CONFIRM)
`streak_threshold = max(10, smallest k with (1 − p_backtest)^k < 0.01)`, p_backtest = baseline full-period win
rate, default 0.5. Reason: in the verifier's 120-day replay the fixed 10 killed basket_sma50_band on 15 straight
losses at only 13.6 % DD; trend families have low win rates by construction. This is a design consequence, not
an owner ruling yet. Note: with p 0.279 the threshold is 15 and the replay still shows HARD-KILL for
basket_sma50_band (15 ≥ 15); the owner may prefer a stricter alpha or a strict `>`.

### D3 — Hypothesis-specific primary tests deferred to C5, with a guard (built)
reduces_dd_in(cond) and earns_when_live_idle tests (design §4) are not built. Until they are, those lines
cannot become RETIRE-cand or PROMOTE-cand (reason "primary hypothesis test not built (C5)"); safety verdicts
(HARD-KILL, DORMANT, INCONCLUSIVE-180) still apply. Affected: shadow_btc200, breakout_s21, basket_sma50_band.

### D4 — Promotion is an owner act (accepted, design §5 / S34 Graduation)
PROMOTE-cand is a recommendation. Promotion requires an owner decision, a new spec, and a restart of the 90-day
clock (config.py header). Never automatic.

## Owner-pending
- D1 acceptance into this spec as the decision of record.
- D2 streak-threshold formula (and whether basket_sma50_band's replay kill at exactly 15 is intended).
- C5: build the reduces_dd_in / earns_when_live_idle tests (lifts the D3 guard).
- DORMANT "waiting for condition" exemption: applied by hand at the monthly ruling.

## Acceptance Criteria
- Live S18 journal path identical with and without the pool (golden fixture test).
- `state/pool_report.json` regenerated daily, atomically; scoring failure never fails the run.
- Dashboard renders with the replay fixture (all 9 lines, badges, 4 heatmap tables, 8-week view), with no pool
  data, and with a malformed report (fallback box); live sections unchanged; no "None"/"nan" in cells.
