# Project State

**Last updated:** 2026-04-15
**Current milestone:** M3: Strategy Refinement + Live Readiness
**Active spec:** S11 — Multi-Strategy Engine (IMPLEMENTED)

## Completed Specs
| Spec | Title | Date Completed |
|------|-------|----------------|
| S01 | MVP Paper Trading Engine | 2026-04-13 |
| S02 | ML Signal Generator with Walk-Forward Backtesting | 2026-04-13 |
| S03 | Indian Market Integration — Zerodha Broker | 2026-04-14 |
| S04 | Market-Aware ML Model with Long + Short Trading | 2026-04-14 |
| S05 | Time-Machine Backtest + Continuous Learning | 2026-04-14 |

## In Progress
| Spec | Title | Status | Notes |
|------|-------|--------|-------|
| S06 | Model Calibration Fix + Feature Engineering v2 | IMPLEMENTED | US fixed, India needs more work |
| S07 | Portfolio Circuit Breakers | IMPLEMENTED | 4-tier drawdown protection; HMM regime code stripped (hurt performance) |

## Next Actions
- Commit S05 + S06 + S07 + Fabio validation work
- India feature engineering — model is not predictive enough (separate spec)
- Consider: per-symbol calibration, domain-specific India features

## Blockers
- **India model** not profitable even after calibration — needs feature engineering work
- **HMM regime features** stripped from codebase — degraded performance, code removed

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

### Cost Insights
- **Zerodha:** ~0.16% round-trip (STT dominated). 10x cheaper than eToro.
- **eToro:** ~1.6% round-trip (FX dominated). FX fee kills small accounts.
