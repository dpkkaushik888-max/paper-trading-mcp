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
| 6 | S06 | Model Calibration Fix + Feature Engineering v2 | IMPLEMENTED | S05 |

## M3: Strategy Refinement + Live Readiness

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 7 | S07 | Portfolio Circuit Breakers | IMPLEMENTED | S06 |
| 8 | S08 | Session Volatility + Trailing Profit Protection | BACKLOG | S07 |
| 9 | S09 | Dynamic SL/TP + Consecutive Loss Breaker | BACKLOG | S07 |
| 10 | S10 | Indian F&O Integration for Live Short Selling | BACKLOG | S03, S06 |
| 11 | S11 | Multi-Session: Run US + India Strategies in Parallel | BACKLOG | S06 |
| 12 | S12 | Alternative Broker Profiles (IBKR, Trading 212, Degiro) | BACKLOG | S01 |

## M4: Automation + Live Bridge

| Order | Spec | Title | Status | Depends On |
|-------|------|-------|--------|------------|
| 13 | S13 | Scheduled Auto-Scan + Auto-Trade (cron/daemon mode) | BACKLOG | S06 |
| 14 | S14 | Live Broker Automation (Zerodha API / cheapest US broker) | BACKLOG | S10, S12 |
| 15 | S15 | Web Dashboard (P&L charts, market comparison) | BACKLOG | S11 |
| 16 | S16 | Alert System (email/Telegram on signals) | BACKLOG | S06 |

## Backlog (Unscheduled)
See BACKLOG.md
