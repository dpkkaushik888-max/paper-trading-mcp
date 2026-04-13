# S01: MVP Paper Trading Engine

**Status:** VERIFIED
**Branch:** `feature/s01-mvp-engine`
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Build a paper trading engine that evaluates technical indicator rules against real-time ETF prices, simulates trades with realistic costs, and tracks daily P&L. Expose as both CLI and MCP server so Cascade can drive trades.

## User Story
As a retail investor, I want to paper-trade ETFs using AI-evaluated technical rules so that I can measure the strategy's daily profit before risking real money.

## Design Decisions
Decisions from the Discuss phase:

1. **Asset class: ETFs only** — SPY, QQQ, IWM, DIA, XLF, XLK, GLD, TLT. Lower noise, tighter spreads, well-studied patterns. Stocks/crypto deferred.
2. **Starting capital: $1,000** — Target $5-10/day (0.5-1% daily return). Conservative position sizing.
3. **Price source: Yahoo Finance (yfinance)** — Free, no API key, covers all ETFs. Real-time quotes + historical OHLCV for indicator calculation.
4. **Indicators: pandas-ta** — Built-in calculation. No TradingView dependency. RSI, EMA, MACD, VWAP, Bollinger Bands, ATR.
5. **Rule format: machine-evaluable JSON** — Hybrid: `condition` field (evaluable expression) + `description` field (human-readable). Inspired by jackson-video-resources/claude-tradingview-mcp-trading `rules.json` but deterministic.
6. **Persistence: SQLite** — Single file, zero config. Tables: portfolio, positions, trades, daily_snapshots.
7. **Fees: FULL eToro-realistic cost model** — Every cost that would eat into real profit is simulated. Zero surprises. Researched from etoro.com/trading/fees/ and matchmybroker.com (April 2026):
   - **Commission:** $0 for ETFs (eToro is commission-free on ETFs, confirmed 2026)
   - **Market spread:** Simulated as bid-ask spread from Yahoo Finance data. Fallback: 0.03% for liquid ETFs (SPY/QQQ), 0.08% for less liquid (XLF/GLD)
   - **Slippage:** 0.02% per trade (market orders fill slightly worse than quoted)
   - **Currency conversion (EUR→USD):** 1.5% on deposit/withdrawal for non-USD accounts (eToro default for EU users). Applied to starting capital and when calculating EUR-equivalent P&L
   - **Overnight fee (CFD only):** If user enables leverage/short selling (CFD), overnight fee applies. For non-leveraged long ETF buys = $0 overnight fee. Tracked so user sees the difference.
   - **Withdrawal fee:** $5 per withdrawal from USD account ($0 from EUR account). Tracked in cumulative cost.
   - **Inactivity fee:** $10/month after 12 months no login. Not simulated (active trader won't hit this).
   - **Tax withholding (US dividends):** 30% withholding on US ETF dividends for non-US residents (W-8BEN reduces to 15%). Simulated at 30% default, configurable.
   - **Capital gains tax (Germany):** 26.375% (25% + 5.5% Soli) on realized gains. Shown separately — not deducted from paper P&L but displayed as "after-tax P&L".
   - **Cost summary per trade:** Every trade shows: gross P&L, spread cost, slippage cost, FX cost, tax estimate, **net P&L after ALL costs**
   - **Daily report:** Shows gross daily P&L vs net daily P&L (after all costs). The difference = "hidden cost drag"
8. **MCP + CLI dual interface** — Engine is a Python module. CLI wraps it. MCP server exposes it as tools.
9. **Risk management** — Max 10% of capital per position, max 2% daily loss → stop trading for the day.
10. **Future-proof: Broker abstraction** — Cost engine loads fee profiles from `broker_profiles/etoro.json`. MVP ships with eToro profile only, but adding Interactive Brokers / Trading 212 / Degiro later = just a new JSON file, zero code changes. No eToro-specific logic leaks outside `cost_engine.py`.
11. **Future-proof: Session isolation** — Every portfolio has a `session_id` (default: "default"). All tables are scoped by session_id. MVP uses one session, but future specs can run multiple strategies in parallel with isolated P&L tracking. No schema migration needed.

## Research
- **Reference repo:** `jackson-video-resources/claude-tradingview-mcp-trading` — Node.js bot connecting Claude + TradingView + BitGet. Uses `rules.json` with natural language conditions. Good UX pattern but non-deterministic (AI interprets rules each time). We use machine-evaluable conditions instead.
- **Realistic ETF day-trading returns:** 0.5-1% daily is ambitious but achievable with disciplined rules. Most retail day traders lose money — the whole point is to paper-trade first and measure before committing real capital.
- **Yahoo Finance limitations:** 15-min delayed quotes for some exchanges. For US ETFs, real-time data is available. Rate limits: ~2000 requests/hour (more than enough).
- **eToro fee research (April 2026, sources: etoro.com/trading/fees/, matchmybroker.com):**
  - ETFs: $0 commission (confirmed). No eToro spread markup — only natural market spread.
  - Stocks: $1-$2 per trade (NOT applicable to ETFs).
  - CFD on ETFs: 0.15% opening/closing spread charged from cash balance + overnight fees. Non-leveraged long buys are NOT CFDs on eToro.
  - Currency conversion: ~1.5% for EUR→USD (varies by Club tier: Diamond=0%, Platinum=50% discount).
  - Withdrawal: $5 from USD account, $0 from EUR/GBP account.
  - US dividend withholding: 30% for non-US (15% with W-8BEN treaty). eToro auto-applies.
  - German capital gains: 26.375% (Abgeltungsteuer 25% + Solidaritätszuschlag 5.5%). €1,000 Sparerpauschbetrag annual exemption.
  - No overnight fees for non-leveraged ETF positions (only CFDs incur overnight).

## Acceptance Criteria
- [ ] AC1: `python -m trading_engine.cli scan` fetches real-time prices for all watchlist ETFs and calculates indicators (RSI, EMA, MACD)
- [ ] AC2: `python -m trading_engine.cli scan` evaluates rules from `rules.json` and prints BUY/SELL/HOLD signals per symbol
- [ ] AC3: `python -m trading_engine.cli trade buy SPY 10` executes a paper buy (deducting cash + spread/slippage) and stores in SQLite
- [ ] AC4: `python -m trading_engine.cli portfolio` shows current cash, positions, unrealized P&L, total portfolio value
- [ ] AC5: `python -m trading_engine.cli report` shows today's realized P&L, win/loss count, win rate, cumulative P&L
- [ ] AC6: `python -m trading_engine.cli history` shows all trades with entry/exit price, P&L per trade, outcome
- [ ] AC7: MCP server starts and exposes tools: `scan_signals`, `place_trade`, `get_portfolio`, `daily_report`, `trade_history`, `get_quote`
- [ ] AC8: Risk guard: trade is rejected if position would exceed 10% of portfolio or daily loss exceeds 2%
- [ ] AC9: `rules.json` contains a working default ETF strategy (RSI + EMA + MACD based)
- [ ] AC10: Backtest command: `python -m trading_engine.cli backtest --days 30` runs strategy against historical data and reports simulated P&L
- [ ] AC11: Every trade and daily report shows FULL cost breakdown: gross P&L, spread, slippage, FX conversion, estimated tax, and **net P&L after all costs**. No hidden costs.
- [ ] AC12: `python -m trading_engine.cli costs` shows cumulative cost summary: total spread paid, total slippage, total FX fees, total estimated tax, total cost drag as % of profit

## Technical Design

### Files to Create/Modify

| File | Change |
|------|--------|
| `trading_engine/__init__.py` | Package init |
| `trading_engine/config.py` | Constants: watchlist, fees, risk limits, DB path |
| `trading_engine/models.py` | Dataclasses: Trade, Position, Portfolio, Signal, DailySnapshot |
| `trading_engine/price_engine.py` | Yahoo Finance wrapper: get_quote(), get_history(), calculate_indicators() |
| `trading_engine/rule_evaluator.py` | Load rules.json, evaluate conditions against indicator data, return signals |
| `trading_engine/portfolio.py` | SQLite portfolio: buy(), sell(), get_positions(), get_cash(), daily_snapshot() |
| `trading_engine/cost_engine.py` | Broker-agnostic cost calculator. Loads fee profile from `broker_profiles/*.json`. Ships with eToro profile. |
| `broker_profiles/etoro.json` | eToro fee profile: spread, slippage, FX, dividend tax, capital gains tax rates |
| `trading_engine/risk_manager.py` | Position size check, daily loss check, max exposure check |
| `trading_engine/backtest.py` | Historical simulation: iterate days, evaluate rules, simulate trades |
| `trading_engine/cli.py` | Click CLI: scan, trade, portfolio, report, history, backtest |
| `trading_engine/mcp_server.py` | MCP server exposing all tools |
| `rules.json` | Default ETF strategy |
| `requirements.txt` | Dependencies (already created) |
| `README.md` | Project documentation (already created) |

### Data Model Changes

**SQLite tables:**

```sql
-- Portfolio state
CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT 'default',
    broker_profile TEXT NOT NULL DEFAULT 'etoro',
    initial_capital REAL NOT NULL,
    cash REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id)
);

-- Open and closed positions
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    shares REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_cost REAL NOT NULL,  -- price * shares + fees
    exit_price REAL,
    exit_proceeds REAL,
    pnl REAL,
    status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    entry_reason TEXT,
    exit_reason TEXT
);

-- All executed trades (buy/sell events)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    position_id INTEGER REFERENCES positions(id),
    symbol TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('buy', 'sell')),
    shares REAL NOT NULL,
    price REAL NOT NULL,
    spread_cost REAL NOT NULL,
    slippage_cost REAL NOT NULL,
    fx_cost REAL NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL,
    gross_value REAL NOT NULL,     -- price * shares (before costs)
    net_value REAL NOT NULL,        -- after all costs
    timestamp TEXT NOT NULL,
    reason TEXT
);

-- Daily portfolio snapshots for P&L tracking
CREATE TABLE daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    date TEXT NOT NULL,
    UNIQUE(session_id, date),
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_value REAL NOT NULL,
    daily_pnl_gross REAL NOT NULL,
    daily_pnl_net REAL NOT NULL,     -- after spread + slippage + FX
    cumulative_pnl_gross REAL NOT NULL,
    cumulative_pnl_net REAL NOT NULL,
    total_spread_cost REAL NOT NULL,
    total_slippage_cost REAL NOT NULL,
    total_fx_cost REAL NOT NULL,
    estimated_tax REAL NOT NULL,     -- German capital gains on realized profits
    daily_pnl_after_tax REAL NOT NULL,  -- net P&L minus estimated tax
    cost_drag_pct REAL NOT NULL,     -- total costs as % of gross profit
    trades_today INTEGER NOT NULL,
    wins_today INTEGER NOT NULL,
    losses_today INTEGER NOT NULL
);
```

### API Changes
MCP tools (not REST — MCP protocol):

| Tool | Input | Output |
|------|-------|--------|
| `scan_signals` | `{watchlist?: string[]}` | Signals per symbol with indicator values |
| `place_trade` | `{symbol, action, shares?, reason?}` | Trade confirmation or rejection reason |
| `get_portfolio` | `{}` | Cash, positions, unrealized P&L, total value |
| `daily_report` | `{date?: string}` | P&L, win rate, streak, cumulative |
| `trade_history` | `{limit?: int, symbol?: string}` | List of trades with outcomes |
| `get_quote` | `{symbol}` | Price + all calculated indicators |
| `backtest` | `{days?: int, strategy?: string}` | Historical simulation results |
| `cost_summary` | `{}` | Cumulative costs: spread, slippage, FX, tax, drag % |

## Dependencies
- **External:** yfinance, pandas, pandas-ta, mcp
- **Other specs:** None — this is the foundation

## Verification
| AC | How to Verify |
|----|---------------|
| AC1 | Run `scan` command, verify prices + indicators print for SPY, QQQ, etc. |
| AC2 | Modify rules.json to force a signal, run `scan`, verify signal detected |
| AC3 | Run `trade buy SPY 10`, check SQLite has position + cash reduced |
| AC4 | Run `portfolio` after buying, verify cash + position + total matches |
| AC5 | Buy then sell at different price, run `report`, verify P&L correct |
| AC6 | Run `history`, verify trade list with entry/exit/P&L |
| AC7 | Start MCP server, verify tools are registered (list tools) |
| AC8 | Try to buy more than 10% of portfolio, verify rejection |
| AC9 | Read rules.json, verify it has valid entry/exit/risk rules |
| AC10 | Run `backtest --days 30`, verify output shows daily P&L series |
| AC11 | Execute a buy+sell, verify trade output shows gross P&L, spread, slippage, FX, tax estimate, net P&L |
| AC12 | Run `costs` after several trades, verify cumulative cost breakdown matches sum of individual trades |

## UAT
- [x] AC1: **PASS** — `scan` fetches live prices for all 8 ETFs (SPY=$679.46, QQQ=$611.07, etc.) and shows RSI, EMA, MACD, BB indicators
- [x] AC2: **PASS** — `scan` evaluates rules and shows BUY/SELL/HOLD per symbol. 6 BUY (SPY, QQQ, IWM, DIA, XLF, XLK), 2 HOLD (GLD, TLT)
- [x] AC3: **PASS** — `trade buy XLF 1` deducts $51.58 (including $0.81 costs) and creates position in SQLite
- [x] AC4: **PASS** — `portfolio` shows cash=$948.42, XLF position 1 share, total=$999.19
- [x] AC5: **PASS** — `report` after buy+sell shows daily P&L=-$1.62, 0W/1L, cumulative costs breakdown
- [x] AC6: **PASS** — `history` shows 2 trades (buy+sell XLF) with timestamps, prices, costs
- [x] AC7: **PASS** — MCP server lists 8 tools: scan_signals_tool, place_trade, get_portfolio, daily_report, trade_history, get_quote_tool, cost_summary, backtest_tool
- [x] AC8: **PASS** — `trade buy XLF 2` blocked: "Position too large: $101.54 = 10.2% of portfolio (max 10%)"
- [x] AC9: **PASS** — `rules.json` has 4 entry rules (RSI<35, price>EMA50, MACD>0, price>BB_lower), 3 exit rules, risk rules with 2% stop/3% TP
- [x] AC10: **PASS** — `backtest --days 30` runs and reports: 4 trades, 0W/2L, -$4.04 net P&L, 702.8% cost drag (reveals costs destroy small-account profits)
- [x] AC11: **PASS** — Every trade shows: gross value, spread cost, slippage cost, FX cost, total cost, net deducted. Sell shows gross P&L, entry costs, exit costs, net P&L, est. tax, after-tax P&L
- [x] AC12: **PASS** — `costs` shows cumulative: spread=$0.0812, slippage=$0.0203, FX=$1.5231, total=$1.6246. Matches sum of individual trade costs

## Notes
- **Market hours:** US market 09:30-16:00 ET. Trades outside hours will use last close price with a warning.
- **Yahoo Finance rate limits:** ~2000 req/hour. Scan command fetches all watchlist symbols in one batch call.
- **No short selling initially:** Long-only for MVP. Short selling can be added in S02.
- **Backtest is approximate:** Uses daily OHLC, not intraday ticks. Good enough for signal validation, not for precise P&L.
- **eToro migration path:** If paper trading proves profitable, user can manually replicate signals on eToro. Future spec could add eToro browser automation.
