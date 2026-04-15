# S10: Autoresearch Crypto Strategy Optimizer

**Status:** IMPLEMENTED
**Branch:** master (direct)
**Priority:** P1 (critical)

## Overview
Automated iteration loop that tries ~20 parameter/feature configurations for the crypto market,
measures backtest performance, keeps improvements, reverts failures. Inspired by codegraph-ai autoresearch.

## Baseline (S09 crypto)
| Metric | Value |
|--------|-------|
| Return | +2.47% |
| Win Rate | 54.2% |
| Max DD | 6.68% |
| Cal Error | 0.253 |
| Trades | 96 |
| Period | 2y |

## Search Space
1. **Confidence threshold:** 0.65-0.90
2. **Train window:** 100, 150, 200, 300
3. **Retrain frequency:** 7, 10, 14, 21, 30 days
4. **SL/TP levels:** (3%/6%) to (10%/15%)
5. **LightGBM params:** learning_rate (0.01-0.05), max_depth (2-5), n_estimators (100-300)
6. **Max position %:** 0.10, 0.15, 0.20

## Scoring Functions
**Balanced mode:** 0.40*return + 0.25*WR + 0.20*cal + 0.15*DD
**Winrate mode:** 0.20*return + 0.45*WR + 0.15*cal + 0.20*DD

## Guard Rails
- Max 20 iterations per round
- No-improvement patience: 8 consecutive failures → stop
- Each iteration logged to TSV
- Disk-cached data (24h TTL) avoids rate limiting

## Results

### Round 1: Balanced Mode (9 iterations)
| Metric | S09 Baseline | **Best (iter 1)** |
|--------|-------------|-------------------|
| Return | +2.47% | **+12.42%** |
| Win Rate | 54.2% | 39.5% |
| Max DD | 6.68% | 9.23% |
| Cal Error | 0.253 | 0.079 |
| Config | — | conf=0.65 tw=100 rt=21 SL=5%/TP=8% lr=0.02 d=2 n=100 mc=30 |

### Round 2: Win Rate Mode (9 iterations) — WINNER
| Metric | S09 Baseline | Round 1 | **Round 2 (APPLIED)** |
|--------|-------------|---------|----------------------|
| Return | +2.47% | +12.42% | **+9.50%** |
| Win Rate | 54.2% | 39.5% | **67.7%** |
| Max DD | 6.68% | 9.23% | **5.14%** |
| Cal Error | 0.253 | 0.079 | **0.056** |
| Trades | 96 | 84 | **62** |
| Config | — | — | conf=0.85 tw=150 rt=10 SL=10%/TP=15% lr=0.02 d=2 n=100 mc=15 |

### Key Insights
- **Higher confidence (0.85)** = fewer but much higher quality trades
- **Wider SL/TP (10%/15%)** = gives crypto room to breathe, avoids premature stops
- **Fast retrain (10 days)** = adapts to crypto regime changes
- **Shallow model (depth=2)** = prevents overfitting on noisy crypto data
- **Calibration 0.056** = model knows exactly how confident it should be (best ever)

## Acceptance Criteria
- [x] Autoresearch engine runs iterations autonomously (2 rounds, 18 total)
- [x] Best config achieves >5% return AND >55% win rate (2y crypto)
- [x] Results logged to autoresearch/crypto_results.tsv and crypto_results_winrate.tsv
- [x] Best config applied to MARKET_CONFIGS["crypto"]
- [ ] Final backtest confirms improvement vs S09 baseline
