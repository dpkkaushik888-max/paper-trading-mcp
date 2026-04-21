# S16: Swing-Timeframe ML on Crypto (Daily Bars)

**Status:** VERIFIED (failed — verdict: DO NOT DEPLOY, pivot to S17 rule-based)
**Branch:** `feature/s16-swing-ml-crypto`
**Priority:** P1
**Ticket:** —
**Depends on:** S05 (time-machine learning), S06 (model calibration), S15 (short model)

## Overview
Pivot the ML trading engine from 1-minute day-trading (which the honest-cost
sim proved unprofitable — see STATE.md cost-audit on 2026-04-20) to daily-bar
swing trading with 2–10 day holds. Keeps the existing Logistic/XGBoost
classifiers and feature engineering, but changes the label, timeframe, and
risk parameters so the per-trade edge is large enough to survive real
transaction costs.

## User Story
As the account owner, I want a swing-trading simulation on daily bars so that
I can evaluate whether my ML infrastructure can produce a positive-expectancy
strategy at realistic costs, targeting €3–5/day of coffee money on €10k capital.

## Design Decisions

### Why swing, not day-trading
The `sim_1min_replay.py` honest-cost run on 2026-04-20 showed:
- ADAPTIVE mode: +0.011%/day average (~4%/yr, worse than Tagesgeld)
- REGIME mode: −0.80%/day average (~−90%/yr, wipeout trajectory)
Day-trading round-trip cost (50 bps) exceeds the model's gross edge per trade
(~20 bps). Moving to swing timeframes makes cost a small fraction of the target
move, letting the same predictive signal become economically viable.

### Why crypto, not equities (for Phase 1)
- All infrastructure (Binance data, funding rates, MCP tooling) already in place
- 24/7 markets — no overnight-gap risk on stops
- Higher volatility → moves are large enough to overcome cost even on swing
- Single data source (Binance) → deterministic backtests
- Phase 2 could add equity ETFs for diversification

### Why ML, not rule-based (user decision, with caveats)
User selected "Crypto + ML" over rule-based Connors-style on 2026-04-20.
**Overfitting is the primary risk.** Mitigations baked into the spec:
1. Feature selection — reduce from ~30 features to a curated ~12 (see below)
2. Heavier L2 regularization — `C = 0.05` (was 0.15 on 1-min)
3. Purged walk-forward CV — 3-day embargo between train/test to prevent leakage
4. Hard holdout — last 20% of data, never touched during model selection
5. Sharpe gate — must hit Sharpe > 1.0 on holdout, not just positive return
6. Benchmark gate — must beat equal-weight buy-and-hold of the same universe

### Why 5-day label
- Median swing-trade hold in published Connors/Clenow strategies: 3–7 days
- 5-day forward return has enough signal-to-noise on daily bars
- Short enough that the model retrains frequently (monthly)
- Long enough that 50 bps cost is <10% of a typical 5% target move

## Research

### Known crypto swing edges (public literature)
| Setup | Documented return | Decay risk |
|-------|-------------------|------------|
| RSI(2) < 10 + 200SMA up (Connors) | 8–15%/yr alpha, 2008–2018 | High — widely published |
| 52-week breakout + ATR trail | 10–20%/yr (Clenow) | Medium |
| Pairs trade BTC/ETH z > 2σ | 5–10%/yr | Low — stat-arb persistent |
| Funding rate > 0.05% daily → short perp | 5–15%/yr | Medium — scales with TVL |

### ML on daily crypto
Academic consensus (Fischer & Krauss 2018, Huang et al 2023): neural nets and
gradient-boosted trees show out-of-sample Sharpe 0.8–1.4 on daily crypto
**after costs**, but with high variance across random seeds. Reproducible
edges are small (0.3–0.6%/mo) and regime-sensitive.

### Trade count concern
- ML needs samples to avoid overfitting
- Daily bars × 10 symbols × 2 years = 7,300 training rows — sufficient
- Actual **trades** per year: 30–80 (low, hence overfitting mitigations above)

## Acceptance Criteria

### Must-have (blocks merge)
- [ ] **AC1:** New simulator `scripts/sim_swing_backtest.py` exists and runs with
      `PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py`
- [ ] **AC2:** Fetches 3 years of **daily** crypto bars from Binance for 6+ symbols
- [ ] **AC3:** Uses the honest-cost model (50 bps round-trip, slippage, SL slip)
- [ ] **AC4:** Label = `close.shift(-5) > close` (5-day forward direction)
- [ ] **AC5:** Walk-forward retrain with purged 3-day embargo
- [ ] **AC6:** Reserves the last 20% of the timeline as a **locked holdout**
      that is only evaluated once
- [ ] **AC7:** Reports on holdout: annualized return, Sharpe, max DD,
      win rate, trades/year, vs buy-and-hold benchmark

### Go/no-go gates (determines next step)
- [ ] **G1 (must pass for live):** Holdout Sharpe > 1.0
- [ ] **G2 (must pass for live):** Holdout annualized return > buy-and-hold of
      equal-weighted crypto basket + 2%
- [ ] **G3 (must pass for live):** Max drawdown < 30%
- [ ] **G4 (must pass for live):** Profitable in at least 2 of 3 holdout years
      (no single-year luck)

### Nice-to-have
- [ ] Compares Logistic vs XGBoost side-by-side on holdout
- [ ] Outputs a per-trade CSV for external analysis
- [ ] Integrates with existing `SimJournal` (SQLite)

## Technical Design

### Files to Create/Modify

| File | Change |
|------|--------|
| `scripts/sim_swing_backtest.py` | NEW — daily-bar swing simulator |
| `trading_engine/features/swing.py` | NEW — curated feature set (12 features) |
| `trading_engine/labels/swing.py` | NEW — 5-day forward return label |
| `scripts/sim_journal.py` | MINOR — add `run_mode='swing'` tag |
| `specs/S16-swing-ml-crypto.md` | THIS FILE |
| `STATE.md` | UPDATE — log S16 as active spec, deprecate 1-min sim |

### Feature Set (12 curated)

Reduced from the 30+ in `build_bar_features()` to avoid overfitting on low
trade counts. Chosen for orthogonal signal:

**Trend (3):**
1. `close_vs_sma_50` — position vs 50-day SMA (trend)
2. `close_vs_sma_200` — position vs 200-day SMA (regime)
3. `adx_14` — trend strength (new)

**Momentum (3):**
4. `rsi_14` — classic oscillator
5. `return_5d` — 5-day return
6. `return_20d` — 20-day return

**Volatility (2):**
7. `atr_pct_14` — normalized ATR
8. `bb_pct_20` — Bollinger %B

**Volume (2):**
9. `volume_ratio_20` — volume vs 20-day avg
10. `obv_slope_10` — on-balance volume trend

**Microstructure (2):**
11. `funding_rate` — Binance perp funding (sentiment)
12. `dist_from_20d_high` — distance from recent high (breakout / pullback)

### Label

```python
target_dir = (close.shift(-5) > close).astype(int)
```

Binary: 1 if price is higher 5 days from now, 0 otherwise. No magnitude target
in v1 (keep it simple; magnitude models overfit worse).

### Entry / Exit Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Entry confidence | `up_prob > 0.62` | Lower than 1-min's 0.70 (more trades needed for swing) |
| Position size | 15% of capital per position | Max 4 concurrent → 60% deployed, 40% cash buffer |
| SL | −7% from entry | Wider than 1-min; daily noise ~2–3% |
| TP | +15% | 2:1 R:R; typical swing target on crypto |
| Trailing stop | activate at +5%, offset 3% | Locks in swings that don't hit TP |
| Max hold | 10 days | Forces exit if signal stale |
| Min hold | 1 day (no intraday re-entry) | Prevents whipsaw |

### Walk-Forward Protocol

```
Timeline: 2023-01-01 ────────────────────── 2026-04-20 (3+ years)

┌───── Training (60%) ─────┬── Walk-forward (20%) ──┬── Locked holdout (20%) ──┐
2023-01-01    →    2024-12-31     2025-01-01 → 2025-09-30     2025-10-01 → 2026-04-20

Walk-forward loop (monthly retrain):
  For month M in walk-forward window:
    Train on all data up to (M − 3 days)  ← 3-day embargo
    Test on month M
    Record trades

Locked holdout:
  After all model choices are frozen, run ONCE on holdout.
  This is the only number we trust for go/no-go.
```

### Cost Model (inherited from honest-sim)

```python
COST_PCT = 0.0020          # 20 bps per side
SLIPPAGE_BPS = 0.0005      # 5 bps per fill
SL_SLIPPAGE_BPS = 0.0010   # extra 10 bps on stops
# Round-trip = ~50 bps
```

At +10% average winner / −5% average loser, the 50 bps cost is 5% of a winner
— no longer edge-killing.

### Universe

Phase 1: BTC-USD, ETH-USD, SOL-USD, AVAX-USD, LINK-USD, MATIC-USD (6 symbols)
Phase 2 (if AC + gates pass): add DOT, ADA, ATOM, NEAR for 10 symbols.

## API / Data Changes

### Data fetching
- Reuse `fetch_binance_1h()` pattern, change interval to `"1d"`, period to
  `3 * 365` days
- Reuse `fetch_binance_funding_rates()` — already daily-ish resolution

### No new external dependencies

## Dependencies
- S05 (time-machine infrastructure) — reused
- S06 (calibration) — reused
- S15 (short model) — **deferred**, long-only Phase 1

## Verification

### Commands to run
```bash
# Build + run full backtest
PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py --all

# Quick smoke test (1 year)
PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py --years 1

# Holdout-only final evaluation
PYTHONPATH=. .venv/bin/python scripts/sim_swing_backtest.py --holdout-only
```

### Expected output
- Per-day portfolio value time series
- Per-trade log (entry/exit, PnL, reason)
- Summary metrics: Sharpe, max DD, CAGR, win rate, trades/year
- Benchmark comparison: equal-weight buy-and-hold of the same universe
- Walk-forward validation chart (return per OOS month)

## UAT (completed 2026-04-20)

Acceptance criteria:
- [x] AC1: Simulator runs end-to-end without errors (2 bugfixes during implementation)
- [x] AC2: Fetches daily data — BTC/ETH/SOL/AVAX/LINK: 1,825 bars each; MATIC: 1,238 (delisted 2024-09)
- [x] AC3: Honest cost model active — logged `20bps/side + 5bps slip + 10bps SL slip`
- [x] AC4: Label verified — smoke test mean ≈ 0.50 on synthetic data
- [x] AC5: Purged CV active — 3-day embargo between train cutoff and today
- [x] AC6: Holdout reserved — 2025-05-30 → 2026-04-20, evaluated exactly once
- [x] AC7: All holdout metrics printed (CAGR, Sharpe, MaxDD, WR, trades/yr, year-by-year, vs benchmark)

Go/no-go gates:
| Gate | Logistic | XGBoost | Status |
|------|----------|---------|--------|
| G1: Sharpe > 1.0 | −0.77 | 0.00 | FAIL both |
| G2: CAGR > benchmark + 2% | +37.0% alpha | +48.9% alpha | PASS both |
| G3: Max DD < 30% | 16.4% | 0% | PASS both |
| G4: Profitable 2 of 3 years | 0/2 | 0/2 | FAIL both |

**Verdict: DO NOT DEPLOY.** Logistic walk-forward was +3.95% (Sharpe 0.40)
showing weak signal; holdout (BTC crash −48.9%) was −10.64% — beat benchmark
but lost money. XGBoost over-regularized to zero trades.

## Lessons learned
1. **Daily ML on 12 features finds weak signal in flat/bull regimes, none in crashes.**
   The walk-forward (+3.95% in flat market) confirms the features aren't useless;
   the holdout loss confirms the edge is small and regime-dependent.
2. **Heavy regularization on XGBoost can completely kill trading activity.**
   `min_child_weight=40 + reg_lambda=3.0` pushed all predictions toward 0.5,
   never crossing 0.62 entry threshold. Trade-off between overfit protection
   and signal strength is delicate on low trade counts.
3. **Locked holdout discipline paid off.** Without it, we'd be tempted to tune
   until things looked good — exactly the kind of self-deception that creates
   the "+31% 2-year backtest" illusions we saw in the 1-min sim.
4. **Benchmark-relative framing (G2 passes) is misleading during crashes.**
   Beating a disaster is not the same as making money. Absolute-return tests
   (G1 Sharpe) are the honest filter.

## Next step
→ **S17: Rule-based Connors-style swing on same infrastructure.**
   Rules are regime-robust, no overfitting risk, well-documented in literature.

## Notes

### Risks
- **Overfitting:** 30–80 trades/year × 12 features → watch validation curve.
  If OOS/IS gap > 2%, reduce features further or move to rule-based (S17).
- **Regime change:** 2024 was a bull year. If holdout is 2025-late + 2026 and
  that's flat/bear, the model may look worse than reality.
- **Data quality:** Binance daily bars go back to 2017 for BTC, less for newer
  coins. Universe may shrink in early years.

### Open questions
- Should we filter by correlation at entry time (skip new entries if portfolio
  correlation > 0.85)? Decision: **no** for v1 — adds complexity, defer to v2.
- Should we use XGBoost with monotonic constraints on some features (e.g.,
  RSI should be monotonically negative for the target)? Decision: **try both
  Logistic and XGBoost**, report side-by-side.
- Should short side be included? Decision: **no**. Crypto shorts are hard;
  borrow fees on spot shorts, funding on perps. Long-only Phase 1.

### Success means
- **G1–G4 all pass:** deploy paper trading live for 90 days, then €500 real
- **G1 passes but G2/G3 fail:** refine (S17) with rule-based overlay
- **G1 fails:** abandon ML-on-daily-crypto; consider pivot to rule-based or
  the Personal Finance OS pivot discussed on 2026-04-20

---

*This spec is the result of the honest-cost sim audit (2026-04-20). The
user's goal is €3–5/day on €10k = 11–18% annualized. Swing trading on daily
bars is the lowest-risk, highest-reuse path to evaluate if that goal is
achievable with the existing ML infrastructure.*
