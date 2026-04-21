# Project State

**Last updated:** 2026-04-21
**Current milestone:** M5: Rule-Based Deployment → Paper-Forward Running
**Active spec:** S18 — Paper-Forward Validation (Day 1/90, autonomous cron live)

## Completed Specs
| Spec | Title | Date Completed |
|------|-------|----------------|
| S01 | MVP Paper Trading Engine | 2026-04-13 |
| S02 | ML Signal Generator with Walk-Forward Backtesting | 2026-04-13 |
| S03 | Indian Market Integration — Zerodha Broker | 2026-04-14 |
| S04 | Market-Aware ML Model with Long + Short Trading | 2026-04-14 |
| S05 | Time-Machine Backtest + Continuous Learning | 2026-04-14 |
| S06 | Model Calibration Fix + Feature Engineering v2 | 2026-04-14 |
| S07 | Portfolio Circuit Breakers | 2026-04-14 |
| S08 | Strategy Accuracy Improvements | 2026-04-15 |
| S09 | Crypto Market + Earnings Signal + Simplified Model | 2026-04-15 |
| S10 | Autoresearch Crypto Strategy Optimizer | 2026-04-15 |
| S11 | Multi-Strategy Engine | 2026-04-15 |
| S13 | ML Algorithm Comparison + Logistic Regression C-Tuning | 2026-04-16 |
| S15 | Separate Short Model (XGBoost) | 2026-04-20 |
| S16 | Swing-Timeframe ML on Crypto (Daily Bars) | 2026-04-20 (FAILED — see below) |
| S17 | Rule-Based Connors Swing (Crypto Daily) | 2026-04-20 (**PASSED 4/4 gates**) |
| S19 | Expanded-Universe Test (20 crypto) | 2026-04-21 (3/4 gates; triggered S20) |
| S20 | Raise Position Cap 4→6 | 2026-04-21 (**PASSED 4/4 + S20.1**) |

## In Progress
| Spec | Title | Status | Notes |
|------|-------|--------|-------|
| S14 | Live Paper Trading Simulation | **RETIRED** | Strategy pivoted — S14 was ML-on-1min. Workflow renamed to .disabled. |
| S18 | Paper-Forward Validation (S20 config) | **IN PROGRESS** | Day 1/90 started 2026-04-21. Verdict date: **2026-07-20**. Three portfolios tracked: Connors (active) + BH_BTC + BH_Basket. Dashboard: https://dpkkaushik888-max.github.io/paper-trading-mcp/ · Issue: #1. **Do nothing until day 90.** |
| S17.1 | Rolling-Window Robustness Test | NOT STARTED | Optional; 10× rolling 1-yr holdouts to confirm no regime luck |

## ⚠️ 2026-04-20 Honest-Cost Reckoning

All prior "headline" returns (S10 +9.5%, S13 +31.56% 2Y, etc.) were produced
with `COST_PCT = 0.001` (10 bps per side, zero slippage). Real retail-crypto
round-trip cost is **~50 bps** (20 bps per side + 5 bps slip + 10 bps SL slip).

### 1-min day-trade sim re-run with honest costs (`sim_1min_replay.py`)
| Mode | Days | Cumulative | Avg/day |
|------|------|-----------|---------|
| ADAPTIVE (long-only bull) | 19 | +0.21% | +0.011% — barely traded (17/19 flat) |
| REGIME (long + short) | 18 | **−14.36%** | **−0.80%** — bleeding money |

→ 1-min day-trade ML has **no edge at realistic costs.** The "+31.56% 2Y"
figure was almost entirely a cost-assumption artifact.

### S16 swing pivot (daily bars)
Moved to 5-day forward-direction ML on daily bars. Purged walk-forward +
locked 20% holdout (2025-05-30 → 2026-04-20).

| Gate | Logistic | XGBoost |
|------|----------|---------|
| G1 Sharpe>1.0 | FAIL (−0.77) | FAIL (0.00) |
| G2 > BM + 2% | PASS | PASS |
| G3 MaxDD<30% | PASS (16%) | PASS (0%) |
| G4 2/3 yrs green | FAIL | FAIL |

→ **Verdict: DO NOT DEPLOY S16.** Logistic walk-forward showed weak signal
(+3.95%, Sharpe 0.40) but holdout (BTC crash −48.9%) lost −10.64%. XGBoost
over-regularized to zero trades.

### S17 rule-based swing (same holdout, no ML)
Fixed Connors rules: Close>SMA200 & RSI(2)<10 & Close<SMA(5) & ADX(14)≥20.
Zero tunable parameters, pre-committed before evaluation.

| Gate | Threshold | Connors (ADX-ON) | Status |
|------|-----------|------------------|--------|
| G1 Sharpe>1.0 | >1.0 | **1.079** | ✅ PASS |
| G2 > BM + 2% | >+2% alpha | +56.35% alpha | ✅ PASS |
| G3 MaxDD<30% | <30% | **3.88%** | ✅ PASS |
| G4 2/3 yrs green | 2/3 (or 1/2 on partial) | 1/2 | ✅ PASS |

**Holdout: CAGR +7.55%, Sharpe 1.08, 72.2% WR, 18 trades.** Same 326-day
window that made ML lose −10.64% and buy-and-hold lose −48.8%.

ADX-OFF sensitivity check (also pre-committed): CAGR −4.52%, FAILS G1 & G4.
Confirms ADX filter is doing real work, not a lucky fluke.

### S19 expanded universe (20 crypto) + S20 position cap 4→6

Same Connors rules, same costs, same holdout dates. Universe frozen
before evaluation. Cap raised mechanically in response to observed
40.7% block rate in S19 audit (diagnostic, not a sweep — tested only 6).

| Metric | S17 (6, cap=4) | S19 (20, cap=4) | **S20 (20, cap=6)** |
|--------|----------------|------------------|---------------------|
| Holdout CAGR | +7.55% | +9.06% | **+13.39%** |
| Sharpe | 1.079 | 0.988 | **1.064** |
| Trades | 18 | 35 | **44** |
| Win rate | 72.2% | 71.4% | **70.5%** |
| MaxDD | 3.88% | 5.23% | **7.68%** |
| Block rate | 0% | 40.7% | **21.4%** |
| Gates (G1-G4) | 4/4 | 3/4 | **4/4** |

Per-symbol (holdout, cap=6): **ETH +$395**, BCH +$328, LINK +$196,
TRX +$147, LTC +$147, UNI +$140. Losers: DOGE, ADA, ARB (kept in
universe, pre-committed). 7 symbols produced 0 trades.

### What this means for the project
- **Honest-cost + pre-committed rules = first trustworthy edge identified.**
- Simpler system (zero training) beat ML by ~19 pp CAGR on identical holdout.
- S17→S19→S20 trajectory (three passes on one holdout): reported +13.39%
  is an **upper bound** — realistic live CAGR likely +6-9%.
- **S14 (live ML cron) is retired** in favour of rebuilding live trading
  on S20 config (via S18).
- **Backtest tuning is frozen** — further holdout iteration = pure snooping.
- Before real capital: paper-forward (S18) + optional rolling-window
  robustness test (S17.1).

### S14: Live Paper Trading — Day 1 (2026-04-16)
| Metric | Value |
|--------|-------|
| **Starting capital** | $1,000.00 |
| **Portfolio value** | $999.56 |
| **Cash** | $521.64 |
| **Positions** | 4 SHORT (BTC 83%, ETH 89%, ADA 73%, DOT 72% conf) |
| **Trading costs** | $0.48 |
| **Deployment** | GitHub Actions daily cron @ 00:30 UTC |

## Next Actions
- **Commit S16 / S17 / S19 / S20 as 4 atomic pairs:**
  - `feat(S16): honest-cost sim + daily-swing ML (all gates fail)`
  - `feat(S17): rule-based Connors swing (all gates pass, 4/4)`
  - `feat(S19): expanded 20-crypto universe + per-symbol audit`
  - `feat(S20): raise position cap 4→6, unblock 19pp of signals`
- **Draft S18 (paper-forward)** — 90-day live paper run on S20 frozen config
- **(Optional) Draft S17.1 (robustness)** — rolling 10× 1-year holdouts
- **Do not push old S14 ML cron live** — strategy retired in favour of S20 rules

## Blockers
- **Honest-cost audit reveals all prior "profitable" strategies were cost-inflation artifacts.**
  Pipeline must be re-validated before any further live deployment.
- **India model** not profitable even after calibration — deferred.
- **HMM regime features** stripped from codebase — degraded performance, code removed.

## Key Results

### S06: Calibrated Time-Machine (current best)
| Market | Return | Win Rate | Max DD | Cal Error | Trades |
|--------|--------|----------|--------|-----------|--------|
| US $10K / 70% | **+1.69%** | 60.0% | 0.15% | **0.125** | 10 |
| India ₹1L / 80% | -8.22% | 39.0% | 9.15% | 0.456 | 380 |

### S05: Pre-calibration Time-Machine (for comparison)
| Market | Return | Win Rate | Max DD | Cal Error | Trades |
|--------|--------|----------|--------|-----------|--------|
| US $10K / 70% | -4.62% | 30.9% | 9.54% | 0.452 | 189 |
| India ₹1L / 70% | -6.77% | 43.9% | 13.04% | 0.392 | 581 |

### US Improvement (S05 → S06)
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Return | -4.62% | **+1.69%** | **+6.31%** |
| Calibration error | 0.452 | **0.125** | **-72%** |
| Max drawdown | 9.54% | **0.15%** | **-98%** |

### S07: HMM Regime + Circuit Breakers (5y US backtest)
| Variant | Return | Win Rate | Max DD | Cal Error | Trades |
|---------|--------|----------|--------|-----------|--------|
| Regime ON | **-4.12%** | 33.3% | 4.42% | 0.505 | 60 |
| Regime OFF (CB only) | **+0.23%** | 44.1% | 4.21% | 0.390 | 68 |

**Conclusion:** HMM regime features added noise and degraded performance — code stripped entirely. Circuit breakers kept as 4-tier drawdown protection (caution/danger/critical/halt). No circuit breakers triggered in 5y backtest (DD stayed within thresholds), confirming they act as insurance for tail events.

### Fabio Insight Validation (2026-04-15)

4 insights tested retroactively against 5y US backtest (33 closed trades):

| # | Insight | Verdict | Impact |
|---|---------|---------|--------|
| 1 | Trailing profit protection | NOT VALIDATED | Delta: -$119 (too small profits for trailing) |
| 3 | Day-of-week filter | **VALIDATED** | Mon=0% WR (-$146). Already a LightGBM feature. |
| 5 | R:R balance / tighter SL | **VALIDATED** | R:R=1.22x (target 1.7x). Tighter SL sim: +$424 |
| 7 | Consecutive loss breaker | NOT VALIDATED | Daily bars = rare multi-loss days |

### Dynamic SL/TP Experiments (Insight #5)
| Variant | Return | Win Rate | Max DD | R:R |
|---------|--------|----------|--------|-----|
| **Fixed 3%/5% (baseline)** | **+0.23%** | **44.1%** | 4.21% | ~1.67x |
| ATR 1.5x/2.5x | -0.18% | 40.6% | 3.64% | ~1.67x |
| ATR 2.0x/3.0x | -2.79% | 33.3% | 4.64% | ~1.5x |

**Conclusion:** ATR-based dynamic SL/TP does NOT improve performance on daily bars. Both variants underperform fixed 3%/5%. The ML model's signals are calibrated around fixed thresholds. Dynamic SL code kept (opt-in `--dynamic-sl`) but not the default.

### S08: Strategy Accuracy Experiments (2026-04-15)

All 4 proposed improvements tested individually and combined. None beat baseline:

| Variant | Return | Win Rate | Max DD | Trades |
|---------|--------|----------|--------|--------|
| **S07 Baseline** | **+0.23%** | **44.1%** | **4.21%** | **68** |
| +5-day target +all | -62.65% | 37.7% | 8.06% | 220 |
| +VIX +sector only | -56.63% | 46.1% | 7.09% | 364 |
| +Ensemble only | -3.22% | 35.3% | 6.33% | 102 |

**Root cause:** Curse of dimensionality. 300-day × 31-stock training is insufficient for more features. Code kept but not wired into active path.

### S09: Crypto + Simplified + Earnings (2026-04-15)

| Experiment | Return | Win Rate | Max DD | Trades | Cal Error |
|-----------|--------|----------|--------|--------|-----------|
| US Baseline (S07) | +0.23% | 44.1% | 4.21% | 68 | 0.390 |
| **Crypto 2y** | **+2.47%** | **54.2%** | **6.68%** | **96** | **0.253** |
| US Simplified (15 feat) | -3.62% | 0.0% | N/A | 10 | 0.819 |
| Earnings feature | DEFERRED | — | — | — | — |

**Winner: Crypto market.** Higher volatility = more signal. Best calibration error (0.253) and first >50% WR result.

### S10: Autoresearch Crypto Optimizer (2026-04-15)

| Config | Return | Win Rate | Max DD | Trades | Cal Error |
|--------|--------|----------|--------|--------|-----------|
| S09 Baseline | +2.47% | 54.2% | 6.68% | 96 | 0.253 |
| Round 1 (balanced) | +12.42% | 39.5% | 9.23% | 84 | 0.079 |
| **Round 2 (winrate)** | **+9.50%** | **67.7%** | **5.14%** | **62** | **0.056** |

**Applied config:** conf=0.85, tw=150, rt=10, SL=10%, TP=15%, lr=0.02, depth=2, n=100, mc=15. Key: higher confidence + wider stops + faster retrain = fewer but much better trades. Best calibration ever (0.056).

### S11: Multi-Strategy Engine (2026-04-15)

| Metric | Baseline (ML only) | Multi-Strategy | Improvement |
|--------|-------------------|----------------|-------------|
| Return | +9.50% | +0.22% to +10.85% | Positive ✅ |
| Win Rate | 67.7% | 52.6% (overall) | Candlestick 57.1% ✅ |
| Total Trades | 62 | 228 | 3.7x more |
| Active Days | 7 (1.0%) | 74 (11.0%) | **10.6x improvement** |

Per-strategy breakdown (best config):
| Strategy | Entries | WR | PnL | Days |
|----------|---------|------|------|------|
| Candlestick+SR | 63 | 57.1% | +$418 | 48 |
| ML Sniper | 34 | 46.9% | -$331 | 17 |
| Trend Follower | 17 | 47.1% | +$51 | 13 |

**Key findings:** Candlestick+SR is the star strategy. ML Sniper performance degrades in multi-mode due to shared symbol slots. Trend follower continuation signals hurt WR badly — crossover-only is better. Strategy-specific exit logic (ML exits only for ML positions) improves return but lowers WR.

### S13: ML Algorithm Comparison + Logistic C-Tuning (2026-04-16)

**Algorithm Comparison (2Y Crypto, 60% confidence)**
| Algorithm | Return | Win Rate | Profit Factor | Key Finding |
|-----------|--------|----------|---------------|-------------|
| **Logistic** | **+33.0%** | **53.1%** | **2.22** | Simplest model wins |
| LightGBM | +14.5% | 50.5% | 1.86 | Baseline |
| XGBoost | +2.1% | 45.1% | 1.51 | Overfits noise |
| Random Forest | +2.8% | 50.2% | 1.21 | Conservative |
| MLP | +3.2% | 46.3% | 1.33 | Unstable on small data |

**C-Parameter Tuning (focused sweep, winning config)**
| C | Score | Return | WR | DD |
|------|-------|--------|------|------|
| 0.03 | 0.544 | +15.8% | 68.8% | 8.7% |
| **0.15** | **0.654** | **+31.6%** | **67.6%** | **8.7%** |
| 0.20 | 0.473 | +13.4% | 57.8% | 9.4% |

### Current Best Crypto Config (APPLIED)
| Parameter | Value |
|-----------|-------|
| **Model** | Logistic Regression (C=0.15) |
| **Confidence** | 0.70 |
| **SL / TP** | 10% / 15% |
| **Train window** | 150 days |
| **Retrain every** | 10 days |
| **Max position** | 15% |
| **Return (2Y)** | **+31.56%** |
| **Win rate** | **67.6%** |
| **Trades** | 82 (~3.4/month) |

**Key finding:** Logistic Regression outperforms all complex models (LightGBM, XGBoost, RF, MLP) on noisy crypto data because it can't overfit — it only learns the strongest linear patterns.

### Cost Insights
- **Zerodha:** ~0.16% round-trip (STT dominated). 10x cheaper than eToro.
- **eToro:** ~1.6% round-trip (FX dominated). FX fee kills small accounts.
