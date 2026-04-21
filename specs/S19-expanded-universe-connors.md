# S19: Expanded-Universe Test for Connors Rules (20 Crypto)

**Status:** VERIFIED (3/4 gates + 2/3 success criteria; Sharpe 0.988 misses G1 by 0.012 → triggered S20)
**Branch:** `feature/s19-expanded-universe-connors`
**Priority:** P1
**Ticket:** —
**Depends on:** S17 (verified Connors rule set on 6-symbol universe)

## Overview

The S17 locked-holdout test (CAGR +7.55%, Sharpe 1.08, 18 trades) showed a
real but weak edge. Signal audit confirmed the bottleneck is **signal
rarity, not execution** — only 18 entries fired across 6 symbols × 326
days, with zero blocked by capital or position caps.

Hypothesis: adding **14 more liquid crypto symbols** (same asset class,
same Connors behavioral bias) should produce ~3–4× the trade count with
similar win rate, meaningfully lifting CAGR without structural change.

## User Story

As the account owner, I want to know whether the S17 edge scales with
universe size so that I can decide between deploying the strategy live,
adding ensemble layers, or abandoning trading entirely.

## Design Decisions

### What changes vs S17
- **Universe: 6 → 20 crypto symbols** (pre-committed below)
- Everything else identical: rules, costs, timeline splits, gates, holdout dates

### What does NOT change
- **Rules are locked** — no parameter tuning. Same Connors entry/exit + ADX-ON
- **Same holdout dates** (2025-05-30 → 2026-04-20) for direct comparability with S17
- **Same position sizing** (15%/trade, max 4 concurrent)
- **Same honest cost model** (50 bps round-trip)
- **Capital stays $10,000** for apples-to-apples comparison

### Why this is not data snooping
- Symbol list is pre-committed in this spec before any results are seen
- All new symbols are objectively liquid (top-20 by market cap + long-standing Binance listing)
- No backtest run yet — the list is frozen as of this commit
- If this expansion fails, we do NOT retry with a different list

### Pre-committed universe (frozen — do not modify after this point)

**Original (S17):** BTC, ETH, SOL, AVAX, LINK, MATIC (6)

**New additions (14):** DOGE, XRP, ADA, DOT, ATOM, NEAR, LTC, TRX, BCH,
APT, UNI, ARB, OP, SUI

**Selection rationale:**
- Top ~25 by market cap as of 2026-04-20, filtered for:
  - Listed on Binance with ≥3 years of daily history (covers warm-up + holdout)
  - Daily volume > $10M (liquidity for $1,500 position sizes)
  - Not a stablecoin, not a wrapped asset
- Final 14: all satisfy criteria. APT/ARB/OP/SUI listed after 2021 but
  have enough history for 200-SMA warm-up before holdout starts.

## Research

### Expected mechanism
Connors edge comes from **retail buy-strength / sell-weakness bias** —
this is behavioral and universal across liquid tradeable assets. Any crypto
with a large retail holder base should exhibit similar mean-reversion after
sharp pullbacks.

### Risk: correlation
Crypto majors correlate ~0.70–0.85 during drawdowns, ~0.40–0.60 in normal
markets. Expect:
- Positive regimes: ~3× more independent signals → 3× more trades
- Negative regimes: signals cluster; all 20 fire on the same dip day

Net expectation: **~2.5–3.5× the S17 trade count**, ~2× CAGR (not 3×
because of clustering and position cap).

### What would invalidate the hypothesis
- Trade count < 2× (suggests signal saturation or poor universe quality)
- Win rate drops below 55% (suggests thin-coin trades are systematically worse)
- Max DD > 15% (suggests clustering risk is worse than modeled)

## Acceptance Criteria

### Must-have
- [ ] **AC1:** `scripts/sim_swing_rules.py` supports a `--universe` flag:
      `legacy` (S17's 6-coin list) or `expanded` (20-coin list)
- [ ] **AC2:** Expanded universe fetches cleanly — all 20 symbols have
      ≥3 years of daily bars on Binance (except APT/ARB/SUI which have
      since-listing data, min ~2.5 years)
- [ ] **AC3:** Pre-committed symbol list is hard-coded in the spec and
      in `sim_swing_rules.py` — no runtime override allowed
- [ ] **AC4:** Same purged walk-forward + locked holdout window as S17
- [ ] **AC5:** Signal audit reports per-symbol entry count (visibility
      into which new symbols actually produced trades)
- [ ] **AC6:** Report includes A/B summary: S17 (legacy) vs S19 (expanded)
      side-by-side on identical holdout

### Go/no-go gates (same as S17 for apples-to-apples)
- [ ] **G1:** Holdout Sharpe > 1.0
- [ ] **G2:** Holdout CAGR > benchmark + 2%
- [ ] **G3:** Max drawdown < 30%
- [ ] **G4:** Profitable in ≥2 of 3 holdout years (or 1/2 on partial windows)

### Success criteria (beyond gates — for deploy decision)
- [ ] **S1:** Trade count ≥ 35 on holdout (vs 18 on S17) — confirms signal scaling
- [ ] **S2:** Win rate ≥ 60% (vs 72% on S17) — allows some degradation
- [ ] **S3:** CAGR ≥ +12% — meaningful lift over S17's +7.55%

If all 4 gates + S1/S2 pass but S3 fails (e.g. CAGR +8%) → still deployable,
just underwhelming. If gates pass but S1 fails (trades < 2×) → universe
expansion didn't deliver the core hypothesis; investigate per-symbol
audit before next step.

## Technical Design

### Files to Create/Modify

| File | Change |
|------|--------|
| `scripts/sim_swing_rules.py` | ADD `UNIVERSE_LEGACY` + `UNIVERSE_EXPANDED` constants, `--universe` CLI flag, per-symbol audit rollup |
| `specs/S19-expanded-universe-connors.md` | THIS FILE |
| `STATE.md` | UPDATE at end of S19 cycle |

### Implementation

```python
# In scripts/sim_swing_rules.py

UNIVERSE_LEGACY = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
]

UNIVERSE_EXPANDED = UNIVERSE_LEGACY + [
    "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "ATOM-USD", "NEAR-USD",
    "LTC-USD",  "TRX-USD", "BCH-USD", "APT-USD", "UNI-USD",  "ARB-USD",
    "OP-USD",   "SUI-USD",
]

BINANCE_MAP_EXTRA = {
    "DOGE-USD": "DOGEUSDT", "XRP-USD": "XRPUSDT",  "ADA-USD": "ADAUSDT",
    "DOT-USD":  "DOTUSDT",  "ATOM-USD": "ATOMUSDT", "NEAR-USD": "NEARUSDT",
    "LTC-USD":  "LTCUSDT",  "TRX-USD":  "TRXUSDT",  "BCH-USD": "BCHUSDT",
    "APT-USD":  "APTUSDT",  "UNI-USD":  "UNIUSDT",  "ARB-USD": "ARBUSDT",
    "OP-USD":   "OPUSDT",   "SUI-USD":  "SUIUSDT",
}
```

CLI:
```bash
# Explicit flag, default = legacy (matches S17)
PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe expanded
```

Per-symbol audit: after the sim, print a table:
```
Symbol     Entries   Wins   WR%    Net $    Avg $/trade
BTC-USD    ...
```

## Dependencies
- S17 verified — this is strictly additive

## Verification

### Commands
```bash
# Baseline (reproduce S17 exactly)
PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe legacy

# Expanded universe (S19 main test)
PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe expanded

# Expanded, ADX-OFF sensitivity
PYTHONPATH=. .venv/bin/python scripts/sim_swing_rules.py --universe expanded --no-adx
```

### Expected output
- Both universes report side-by-side
- Per-symbol audit reveals which new coins produced signal

## UAT

(Filled in after implementation)

## Notes

### Decision tree after results

| Outcome | Next step |
|---------|-----------|
| All 4 gates PASS + S1/S2/S3 all pass | Draft S20 (paper-forward) — this is ready for 90-day paper test |
| Gates pass, S3 misses (CAGR +8-12%) | Deploy to paper anyway; expand universe further in S20 |
| Trade count OK, WR drops below 60% | New symbols are lower quality; restrict list in next iteration |
| Gates fail in expanded but passed in legacy | Original 6-coin result was regime-specific; reconsider entire thesis |
| Signals cluster badly (DD > 15%) | Add correlation-based position cap or symbol diversification constraint |

### Risks
- **Correlation clustering** — one bad day could stop out 6+ positions simultaneously.
  Mitigated partially by 4-concurrent cap, but the cap may itself be throttling gains.
  If DD is fine but trade count is capped, loosen to 6 concurrent in S20.
- **Thin-coin slippage** — smaller coins may have worse real-world fills than the
  5 bps model. Large-caps dominate our universe so this should be minor.
- **APT/ARB/SUI short history** — these listed after 2022. Warm-up for 200-SMA
  pushes their effective sim start later; may reduce their trade count.

### Open questions (pre-locked, no changes permitted)
1. **Position cap 4 concurrent** — kept identical to S17 for controlled comparison.
   If throttling shows up in audit (blocked > 20% of signals), address in S20.
2. **Position size 15% flat** — kept identical. Kelly-lite sizing is S21 material.
3. **No correlation filter** — kept identical. If clustering hurts, address in S20.
