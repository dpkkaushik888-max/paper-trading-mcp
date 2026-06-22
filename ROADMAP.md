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
| 18 | S18 | Paper-Forward Validation of S20 | **IN PROGRESS** (reopened 2026-06-22; +5.36% at day 54/90, beating both BH) | S20 |

## M5: Loop-Engineering Redesign — recursive loop+agent hierarchy

Personal Finance ⊃ Investment ⊃ {Equity, Crypto}. Crypto engine becomes the
first compliant L2 leaf. Built alongside the live S18 run (physical isolation).
Plan: `/Users/deepakkaushik/.claude/plans/starry-dancing-scone.md`.

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 19 | S22 | Loop+Agent Framework (`loops/` pkg, Mandate↓/Report↑, composite, agent client) | DRAFT | S18 lessons |
| 20 | S23 | Crypto Leaf + L3 Regime Orchestrator (absorbs S21 rules) | DRAFT | S22 |
| 21 | S25 | Strategy Discovery Loop ("Agent 1") — propose/search/prove/promote | IMPLEMENTED (awaiting UAT) | S23 engine+gates |
| 21b | S26 | Risk-Adjusted Alpha Gate (Sharpe-based G2/G9, configurable) | VERIFIED | S25 |
| 21c | S27 | Primitive Grammar Expansion (candlesticks, sequences, Donchian, slope) | VERIFIED | S25 |
| 22 | S24 | Crypto-Leaf Paper-Forward Validation (planned, stricter 14-day halt) | BLOCKED | a config passes S25/S26 gates on a CLEAN holdout |
| — | S21 | Regime-Stacked Swing Engine (3 strategies × 3 regimes) | **SUPERSEDED-BY-S23** (rules absorbed, not rejected) | — |

## Deferred / Backlogged

| Spec | Title | Status |
|------|-------|--------|
| Live Broker Automation (Zerodha / IBKR) | BACKLOG | Gated on S24 PASS |
| Web Dashboard (P&L charts, multi-strategy attribution) | BACKLOG | After S23 APPROVED |
| Alert System (email/Telegram on signals) | BACKLOG | After live deployment |
| Indian F&O Integration | BACKLOG | After live US/crypto deployment |
| Alternative Broker Profiles | BACKLOG | After live deployment |

## Backlog (Unscheduled)
See BACKLOG.md
