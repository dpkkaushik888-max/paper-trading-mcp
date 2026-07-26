# S14: Live Paper Trading Simulation (GitHub Actions)

**Status:** IN PROGRESS
**Branch:** master (direct)
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Run the crypto Logistic Regression model against real market prices every day, simulating trades with a virtual $1,000 portfolio. Deployed as a GitHub Actions cron job — no laptop required. After 30 days, we see what's left of the $1,000.

## User Story
As a trader, I want to paper-trade with real daily prices without running my laptop, so I can validate the model's performance in a forward-looking test before risking real money.

## Design Decisions
1. **Standalone script** — `scripts/live_paper_trader.py` runs independently. No MCP server or CLI dependency.
2. **JSON state file** — `paper_trading/portfolio_state.json` persists the portfolio between runs. Git-committed after each run.
3. **CSV trade log** — `paper_trading/trade_log.csv` is append-only. One row per action (BUY/SELL/HOLD/CHECK_EXIT).
4. **GitHub Actions cron** — Runs daily at 00:30 UTC (crypto trades 24/7, no market hours). Free for public repos.
5. **Git commit from CI** — After each run, the action commits updated state + log back to the repo.
6. **Same model as backtest** — Uses `SmartLogistic(C=0.15)` with the same feature builder and crypto config from S13.
7. **Walk-forward retrain** — Retrains on the last 150 days of data every 10 days (same as backtest config).
8. **No external APIs** — Only yahoo finance (free, no key needed).
9. **$1,000 starting capital** — Small enough to be realistic, large enough to make meaningful trades.

## Acceptance Criteria
- [ ] `scripts/live_paper_trader.py` runs end-to-end locally
- [ ] Fetches real crypto prices from Yahoo Finance
- [ ] Trains Logistic Regression on trailing 150-day window
- [ ] Generates BUY/SELL signals based on confidence threshold (0.70)
- [ ] Manages open positions with SL=10%, TP=15%
- [ ] Persists portfolio state to JSON between runs
- [ ] Appends every action to CSV trade log
- [ ] `.github/workflows/daily-trade.yml` triggers on schedule (daily)
- [ ] CI commits updated files back to repo
- [ ] Works with `--dry-run` flag for testing without state changes
- [ ] Summary output shows portfolio value, open positions, today's actions

## Technical Design

### Files to Create
| File | Purpose |
|------|---------|
| `scripts/live_paper_trader.py` | Daily trading script: fetch → train → predict → trade → persist |
| `paper_trading/portfolio_state.json` | Virtual portfolio state (cash, positions, metadata) |
| `paper_trading/trade_log.csv` | Append-only trade journal |
| `.github/workflows/daily-trade.yml` | GitHub Actions cron schedule |

### Files to Modify
| File | Change |
|------|--------|
| `.gitignore` | Ensure `paper_trading/` is tracked (not ignored) |

### Data Model (portfolio_state.json)
```json
{
  "initial_capital": 1000.0,
  "cash": 1000.0,
  "positions": {},
  "total_costs": 0.0,
  "total_trades": 0,
  "wins": 0,
  "losses": 0,
  "last_retrain_date": null,
  "days_since_retrain": 0,
  "feature_cols": null,
  "start_date": "2026-04-16",
  "last_run_date": null
}
```

### Daily Flow
```
1. Load portfolio_state.json
2. Fetch 200 days of crypto OHLCV from Yahoo Finance
3. Check if retrain needed (every 10 days)
   - If yes: build features for all 12 cryptos, train Logistic(C=0.15)
4. For each open position: check SL/TP exits
5. For each crypto without position: predict up/down probability
   - If up_prob > 0.70 and slots available: open long
   - If down_prob > 0.70 and slots available: open short
6. Save updated portfolio_state.json
7. Append actions to trade_log.csv
8. Print summary to stdout (captured by GitHub Actions log)
```

## Dependencies
- S13 (SmartLogistic + crypto config)
- Yahoo Finance (free, no API key)
- GitHub Actions (free tier: 2,000 min/month)

## Verification
```bash
# Local dry run (no state changes)
python scripts/live_paper_trader.py --dry-run

# Local real run
python scripts/live_paper_trader.py

# Check results
cat paper_trading/portfolio_state.json
cat paper_trading/trade_log.csv
```

## Notes
- Crypto trades 24/7 so daily close is always available
- Model pickle is NOT persisted between runs — retrained fresh each time from data (simpler, more robust)
- Position limit: max 8 open positions (same as time_machine)
- Cost model: 0.1% per trade (Binance taker fee)
- After 30 days: review trade_log.csv and portfolio_state.json to assess real-world viability
