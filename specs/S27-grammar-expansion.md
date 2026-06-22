# S27: Primitive Grammar Expansion (Patterns, Candlesticks, Sequences)

**Status:** VERIFIED
**Branch:** `feature/s27-grammar-expansion`
**Priority:** P2 (widens what the discovery loop can discover)
**Depends on:** S25 (discovery grammar + loop), S26 (alpha gate)

## Overview

The discovery loop can only invent strategies its grammar can express. Today that
grammar is **threshold comparisons on ~23 trend/momentum indicators** — so cranking
the candidate count just resamples the same template family. S27 widens the
*vocabulary*: it adds candlestick patterns, multi-bar price sequences, Donchian
support/resistance levels, and a trend-slope primitive, so the loop can express the
chartist/price-action strategies found in trading books and on barchart-style
screeners — not just indicator thresholds.

## User Story

As the system owner, I want the discovery loop to compose strategies from chart
patterns and candlestick/price-action signals — not only RSI/SMA thresholds — so it
searches a genuinely broader space of market-reading ideas, while the same
anti-overfitting discipline (S25 D4) decides which survive.

## Design Decisions

### D1 — Add only VECTORIZED primitives; defer O(n²) pivot patterns
`chart_patterns.detect_candlestick_patterns()` is fully vectorized (O(n)). But
`build_chart_features_series()` (pivot-clustered S/R, double-top, H&S, Fibonacci)
recomputes over every prefix → **O(n²)**, which would make a 5y×20 search crawl.
S27 folds in only the cheap, vectorized families. Pivot-clustered patterns (H&S,
double-top, fib) are deferred to a later opt-in (precompute-once) spec. Honest
scope: we add breadth where it is cheap and correct, and say what we left out.

### D2 — New primitive families (all vectorized, deterministic, no lookahead)
- **Candlesticks (10):** hammer, hanging_man, bullish/bearish_engulfing,
  morning_star, evening_star, dragonfly_doji, shooting_star, piercing_line,
  dark_cloud_cover — reused verbatim from `detect_candlestick_patterns` (+1 bullish,
  −1 bearish, 0 none). Conditions: `bullish_engulfing > 0`, `shooting_star < 0`.
- **Multi-bar sequences:** `up_streak`, `down_streak` (consecutive up/down closes),
  `roc_5`, `roc_10` (extra momentum windows).
- **Donchian S/R:** `prior_high_55`, `prior_low_55` (55-day channel) → enables
  longer-horizon breakout / breakdown rules over the existing 20-day.
- **Trend slope:** `sma20_slope` (5-bar % change of SMA-20) — a cheap trend proxy.

### D3 — No lookahead, decided at the bar's close
Every primitive uses only current/past bars (`shift`/rolling). A candlestick uses
that day's OHLC, known at close — consistent with how `GeneratedStrategy` already
decides entry on `current_day`'s close. No future leak.

### D4 — Grammar stays bounded; the agent gains vocabulary, not freedom
New columns are appended to `FEATURE_COLUMNS`, so `Condition.__post_init__` validates
them and the agent may reference them — but it still emits only bounded conditions
that compile to a deterministic `BaseStrategy`. No new code-execution surface.

### D5 — Generator explores the new space
`generate.py` pools gain pattern/sequence conditions and one pattern-based template,
so a default discovery run actually proposes candlestick/sequence strategies.
Determinism (seed → same specs) is preserved.

## Research
- `trading_engine/chart_patterns.py`: `detect_candlestick_patterns()` (vectorized,
  reused) and the O(n²) `build_chart_features_series()` (deferred per D1).
- `primitives.py` `build_features` / `FEATURE_COLUMNS`: the single extension point;
  `candidate.py` codegen and the agent both read `FEATURE_COLUMNS`, so additions
  propagate automatically.

## Acceptance Criteria
- [ ] `build_features` emits the new columns; `FEATURE_COLUMNS` lists them; all
      vectorized (build_features stays ~O(n) per symbol).
- [ ] No NaN surprises: pattern/sequence columns are 0-filled before warmup.
- [ ] No lookahead — primitives use only current/past bars.
- [ ] `generate.py` composes the new primitives (+ a pattern template); determinism held.
- [ ] A candlestick/sequence candidate compiles via `to_strategy()` and backtests
      through `CryptoLeaf` (entry/exit fire on feature rows).
- [ ] All existing S25/S26 tests pass unchanged; new tests cover the new primitives.
- [ ] Demo run on live data shows pattern-based candidates being proposed + searched.

## Technical Design
### Files to Modify
| File | Change |
|------|--------|
| `trading_engine/discovery/primitives.py` | new columns in `build_features` + `FEATURE_COLUMNS`; reuse `detect_candlestick_patterns`; add `_streak` helper |
| `trading_engine/discovery/generate.py` | pattern/sequence conditions in pools + a pattern template |
| `tests/test_pattern_primitives.py` | new — primitive correctness, no-NaN, no-lookahead, codegen+backtest |

### Data Model Changes
`FEATURE_COLUMNS` grows ~23 → ~40. No persisted-schema break (manifests store specs,
which reference column names as strings).

## Dependencies
S25, S26 (built). `chart_patterns.py` (present). No new packages.

## Verification
- `pytest tests/ -q` → all green incl. new pattern tests.
- `python scripts/run_discovery.py --seed 0 --n-candidates 30` → pattern/sequence
  candidates appear in the logged `all_candidates`.
- Determinism: same seed → identical candidate list.

## UAT — 2026-06-22

- [x] **New primitives present + grammar-valid + vectorized.** `FEATURE_COLUMNS`
      grew 23 → 40 (10 candlesticks + roc_5/10 + up/down_streak + Donchian-55 +
      sma20_slope). `build_features` measured **43 ms/symbol on 1825 bars** — flat,
      no O(n²) regression.
- [x] **No NaN / no lookahead.** Candlestick cols are ternary {−1,0,1}, 0-filled;
      prefix-stability test confirms a row never changes when future bars are added.
- [x] **Generator explores the new space, deterministically.** seed 0 / n=30 →
      **17/30 candidates reference new-family columns**; the engulfing-reversal
      template is candidate #2; same seed → identical specs.
- [x] **Pattern candidate compiles + backtests** through `CryptoLeaf` (test +
      live run).
- [x] **All prior tests pass unchanged** (212 green, +7 new in
      `test_pattern_primitives.py`).
- [x] **Live run:** `run_discovery.py --seed 0 --n-candidates 30 --years 5` —
      30 candidates (incl. pattern/sequence) proposed + searched on real 5y×20 data;
      0 passed WF under default raw_cagr (the +32% BTC bar, as expected), 0 promoted,
      nothing written to any registry.

**Verdict:** Grammar breadth delivered where it is cheap and correct. The loop now
*expresses* candlestick/price-action/sequence strategies; the gates still decide
which survive (none did on this window under the raw bar — breadth ≠ winners, as
designed). Deferred (per D1): O(n²) pivot patterns (H&S, double-top, Fibonacci) need
an O(n) precompute first — a future spec.

## Notes & Risks
- **Breadth ≠ winners.** More expressible strategies means more honest OOS
  *rejections*, not more promotions (S26 showed even the WF standout dies OOS). The
  value is searching a wider space under the same discipline.
- **Deferred patterns.** H&S / double-top / Fibonacci need an O(n) precompute before
  they belong in the hot grammar — explicitly out of S27 scope.
- **Candlestick daily-bar caveat.** Single-bar candlesticks on daily crypto are weak
  edges; their worth is decided by the gates, not assumed.
