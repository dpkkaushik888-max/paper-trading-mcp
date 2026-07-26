# Paper Trading MCP — AI-Driven Trading Engine

An MCP server + standalone engine for AI-driven paper trading across US stocks, Indian stocks (NSE), and crypto markets. Uses ML models with walk-forward backtesting, automated parameter optimization, and multi-strategy signal generation.

## Current Best Results (Crypto, 2Y backtest)
| Metric | Value |
|--------|-------|
| **Model** | Logistic Regression (C=0.15) |
| **Return** | **+31.56%** |
| **Win Rate** | **67.6%** |
| **Max Drawdown** | 8.65% |
| **Trades** | 82 (~3.4/month) |

## Architecture
```
                    ┌─────────────────────┐
  Market Data ────► │ Feature Engineering │ ──► 60+ technical features
  (Yahoo Finance)   │ (pandas-ta)         │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ ML Model            │ ──► Calibrated probabilities
                    │ (Logistic/LightGBM) │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ Strategy Engine      │ ──► Signals (ML Sniper +
                    │ (Multi-Strategy)     │     Candlestick/SR)
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ Risk Management     │ ──► Position sizing,
                    │ (Circuit Breakers)  │     SL/TP, drawdown limits
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ Paper Portfolio      │ ──► SQLite + MCP Tools
                    │ (Cost Engine)        │
                    └─────────────────────┘
```

## Markets Supported
| Market | Assets | Model | Status |
|--------|--------|-------|--------|
| **Crypto** | BTC, ETH, SOL + 9 more | Logistic Regression (C=0.15) | ✅ Best performing |
| **US** | 31 stocks (SPY, AAPL, MSFT...) | LightGBM | ✅ Working (+0.23%) |
| **India** | 19 NSE stocks | LightGBM | ⚠️ Needs improvement |

## ML Models
Pluggable model architecture — any `SmartClassifier` can be swapped in:
| Model | Status | Best Use |
|-------|--------|----------|
| **Logistic Regression** | ✅ Default (crypto) | Noisy/small datasets — can't overfit |
| **LightGBM** | ✅ Default (US/India) | Larger datasets with complex patterns |
| **XGBoost** | ✅ Available | Alternative to LightGBM |
| **Random Forest** | ✅ Available | Conservative predictions |
| **MLP (Neural Net)** | ✅ Available | Experimental |

## MCP Tools
| Tool | Description |
|------|-------------|
| `scan_signals` | Scan watchlist for buy/sell signals |
| `place_trade` | Execute a paper trade (buy/sell) |
| `get_portfolio` | Current positions, cash, total value |
| `daily_report` | Today's P&L, win rate, streak |
| `trade_history` | All trades with outcomes |
| `backtest` | Run strategy against historical data |
| `get_quote` | Real-time price + indicators for a symbol |
| `cost_summary` | Cumulative cost breakdown |

## Quick Start
```bash
cd paper-trading-mcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Paper trading
python -m trading_engine.cli scan          # scan for signals
python -m trading_engine.cli portfolio     # check portfolio
python -m trading_engine.cli report        # daily P&L

# Backtesting
python time_machine_run.py --market crypto --period 2y
python time_machine_run.py --market us --period 5y

# Autoresearch (parameter optimization)
python -m trading_engine.autoresearch --mode logistic --iterations 20
python -m trading_engine.autoresearch --mode balanced --iterations 20

# Algorithm comparison
python scripts/model_comparison.py

# MCP server
python -m trading_engine.mcp_server
```

## Key Features
- **Time-Machine Backtest** — Day-by-day replay with strict temporal isolation (no future data leakage)
- **Walk-Forward Training** — Model retrains every N days on trailing window
- **Platt Calibration** — All models output calibrated probabilities (what "70% confident" really means)
- **Multi-Strategy** — ML Sniper + Candlestick/Support-Resistance run in parallel
- **Circuit Breakers** — 4-tier portfolio drawdown protection (caution → danger → critical → halt)
- **Autoresearch** — Automated parameter sweep with composite scoring
- **Realistic Costs** — Spread, slippage, FX conversion, tax withholding per broker profile

## Project Evolution (12 specs, S01→S13)
| Phase | Specs | Key Achievement |
|-------|-------|----------------|
| **M1: MVP** | S01 | Paper trading engine + MCP server |
| **M2: ML** | S02-S06 | ML signals, India/Crypto markets, calibration fix |
| **M3: Refinement** | S07-S13 | Circuit breakers, autoresearch, multi-strategy, Logistic Regression wins |
| **M4: Live** | S14+ | Scheduled auto-trade, live broker, dashboard (planned) |

See `specs/` for detailed specs, `STATE.md` for current results, `ROADMAP.md` for the full plan.

## Not Financial Advice
This is a paper trading simulation for testing strategies. Past performance
does not predict future results. Never risk money you can't afford to lose.
