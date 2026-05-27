# Roadmap

## M1: MVP Paper Trading Engine

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 1 | S01 | MVP Paper Trading Engine | DONE | — |

## M2: Multi-Market ML Trading

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 2 | S02 | ML Signal Generator with Walk-Forward Backtesting | DONE | S01 |
| 3 | S03 | Indian Market Integration — Zerodha Broker | DONE | S01, S02 |
| 4 | S04 | Market-Aware ML Model with Long + Short Trading | DONE | S02, S03 |
| 5 | S05 | Time-Machine Backtest + Continuous Learning | DONE | S04 |
| 6 | S06 | Model Calibration Fix + Feature Engineering v2 | DONE | S05 |

## M3: Strategy Refinement + Live Readiness

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 7 | S07 | Portfolio Circuit Breakers (HMM stripped, CB kept) | DONE | S06 |
| 8 | S08 | Strategy Accuracy Improvements (all experiments failed) | DONE | S07 |
| 9 | S09 | Crypto Market + Earnings Signal + Simplified Model | DONE | S07 |
| 10 | S10 | Autoresearch Crypto Strategy Optimizer | DONE | S09 |
| 11 | S11 | Multi-Strategy Engine (Candlestick+SR + ML Sniper) | DONE | S10 |
| 12 | S13 | ML Algorithm Comparison + Logistic Regression C-Tuning | DONE | S10 |

## M4: Honest-Cost Pivot + Paper-Forward (closed)

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 13 | S14 | Live Paper Trading Simulation (S14 ML-on-1min) | RETIRED | — |
| 14 | S16 | Honest-cost swing-ML backtest | FAIL (all gates fail) | S13 |
| 15 | S17 | Rule-based Connors swing | DONE (4/4 gates) | S16 |
| 16 | S19 | Expanded 20-crypto universe | DONE (3/4 gates) | S17 |
| 17 | S20 | Raise position cap 4→6 | DONE (4/4 gates) | S19 |
| 18 | S18 | Paper-Forward Validation of S20 | **FAIL** — closed day 29/90, 0 trades, regime mismatch | S20 |

## M5: Regime-Stacked Personal-Compounder Engine

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 19 | S21 | Regime-Stacked Swing Engine (3 strategies × 3 regimes) | DRAFT | S18 lessons |
| 20 | S22 | S21 Paper-Forward Validation (planned, stricter D10) | NOT STARTED | S21 APPROVED |

## Deferred / Backlogged

| Spec | Title | Status |
|------|-------|--------|
| Live Broker Automation (Zerodha / IBKR) | BACKLOG | Gated on S22 PASS |
| Web Dashboard (P&L charts, multi-strategy attribution) | BACKLOG | After S21 APPROVED |
| Alert System (email/Telegram on signals) | BACKLOG | After live deployment |
| Indian F&O Integration | BACKLOG | After live US/crypto deployment |
| Alternative Broker Profiles | BACKLOG | After live deployment |

## Backlog (Unscheduled)
See BACKLOG.md
