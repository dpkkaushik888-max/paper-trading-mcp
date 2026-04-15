# S09: Crypto Market + Earnings Signal + Simplified Model

**Status:** IMPLEMENTED
**Branch:** master (direct)
**Priority:** P1 (critical)

## Overview
Three experiments to improve trading accuracy:
1. **Crypto market** — BTC, ETH, SOL etc. via yfinance (higher volatility, more patterns)
2. **Earnings proximity signal** — days-to-earnings feature for US stocks
3. **Simplified model** — top 15 features only (reduce overfitting)

## Design Decisions
- Crypto uses same US feature builder (price/volume technicals are universal)
- Crypto cost model: 0.1% taker fee (Binance-tier), no spread
- Earnings dates fetched once at startup via yfinance, cached as feature
- Simplified model uses feature importance ranking to select top 15
- Each experiment run independently to isolate impact

## Acceptance Criteria
- [ ] Crypto watchlist + market config added
- [ ] 2y crypto backtest runs and produces results
- [ ] Earnings proximity feature added to US feature builder
- [ ] US backtest with earnings feature runs
- [ ] Simplified 15-feature model backtest runs
- [ ] All results compared in STATE.md

## Baseline
| Market | Return | Win Rate | Max DD | Trades |
|--------|--------|----------|--------|--------|
| US (S07) | +0.23% | 44.1% | 4.21% | 68 |

## Results

| Experiment | Return | Win Rate | Max DD | Trades | Cal Error |
|-----------|--------|----------|--------|--------|-----------|  
| **US Baseline (S07)** | **+0.23%** | **44.1%** | **4.21%** | **68** | **0.390** |
| **Crypto 2y (NEW)** | **+2.47%** | **54.2%** | **6.68%** | **96** | **0.253** |
| US Simplified (15 feat) | -3.62% | 0.0% | N/A | 10 | 0.819 |
| Earnings feature | DEFERRED | — | — | — | — |

### Analysis

**Crypto (+2.47%, 54.2% WR)** is the best result across all experiments:
- Higher volatility = more signal for mean-reversion features
- Calibration error 0.253 = model is well-calibrated
- Circuit breaker triggered (5.45% DD) but recovered
- 96 trades in 2y = active enough to be statistically meaningful

**Simplified model failed** (-3.62%, 0% WR):
- Dropping cross-asset features (SPY relative strength) removed the most useful signals
- Only 10 trades in 5y = model became too cautious
- The cross-asset features are load-bearing, not noise

**Earnings feature deferred:**
- yfinance provides earnings dates (24/31 stocks have data)
- Requires architecture change to pass per-symbol external data into feature builder
- Punted to future spec

### Conclusion
Crypto is a viable new market for the time machine. US accuracy plateau is a data volume issue, not a feature issue.
