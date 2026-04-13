# Paper Trading MCP — AI Coffee Money Engine

An MCP server + standalone engine for AI-driven paper trading of ETFs.
Validates trading rules against real-time market data and tracks P&L.

## Goal
Start with $1,000 paper capital. Let AI evaluate technical rules, paper-trade ETFs,
and measure daily profit. Target: consistent $5-10/day (coffee money).

## Architecture
```
rules.json (strategy)  →  Rule Evaluator  →  Paper Portfolio (SQLite)
                             ↑                      ↓
                        Price Engine          P&L Tracker
                      (Yahoo Finance +        (daily report,
                       pandas-ta)              win rate, CSV)
                             ↑
                         MCP Server
                      (Cascade tools)
```

## MCP Tools
| Tool | Description |
|------|-------------|
| `scan_signals` | Scan watchlist for buy/sell signals based on rules.json |
| `place_trade` | Execute a paper trade (buy/sell) |
| `get_portfolio` | Current positions, cash, total value |
| `daily_report` | Today's P&L, win rate, streak |
| `trade_history` | All trades with outcomes |
| `backtest` | Run strategy against historical data |
| `get_quote` | Real-time price + indicators for a symbol |

## Watchlist (Default ETFs)
- **SPY** — S&P 500
- **QQQ** — Nasdaq 100
- **IWM** — Russell 2000
- **DIA** — Dow Jones
- **XLF** — Financials
- **XLK** — Technology
- **GLD** — Gold
- **TLT** — 20+ Year Treasury

## Quick Start
```bash
cd paper-trading-mcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m trading_engine.cli scan          # scan for signals
python -m trading_engine.cli portfolio     # check portfolio
python -m trading_engine.cli report        # daily P&L
python -m trading_engine.mcp_server        # start MCP server
```

## Strategy (rules.json)
Uses machine-evaluable conditions + human-readable descriptions:
```json
{
  "entry_rules": {
    "long": [
      {"condition": "rsi_14 < 30", "description": "RSI oversold"},
      {"condition": "price > ema_50", "description": "Above 50 EMA — uptrend"}
    ]
  }
}
```

## Fees & Realism
- Spread: 0.05% per trade (realistic for ETFs)
- No commission (matches eToro/most brokers)
- Slippage: 0.02% simulated
- Market hours only: 09:30-16:00 ET (15:30-22:00 Berlin)

## Daily Workflow
```
1. Morning scan: AI evaluates rules against watchlist
2. Signals found: paper-trade with position sizing (max 10% per position)
3. Evening: daily_report shows P&L, win rate, running total
4. Weekly: review which rules work, adjust strategy
```

## Not Financial Advice
This is a paper trading simulation for testing strategies. Past performance
does not predict future results. Never risk money you can't afford to lose.
