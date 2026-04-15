# Backlog

Ideas and future work not yet scheduled into a milestone.

| # | Idea | Priority | Status | Notes |
|---|------|----------|--------|-------|
| 1 | Short selling support | P1 | ✅ DONE (S04) | Long + short with separate position tracking |
| 2 | Run multiple strategies in parallel, compare daily P&L | P2 | Planned (S09) | US vs India head-to-head |
| 3 | Web dashboard with Chart.js P&L graphs | P3 | Planned (S13) | Visual daily/weekly/monthly performance |
| 4 | eToro browser automation (Playwright) | P3 | — | Auto-replicate paper signals on eToro virtual portfolio |
| 5 | Telegram/email alerts on signals | P3 | Planned (S14) | Push notification when signal fires |
| 6 | Crypto support (BTC-USD, ETH-USD) | P3 | — | 24/7 trading, higher volatility |
| 7 | Options strategies (covered calls on ETFs) | P3 | — | Income generation |
| 8 | Machine learning signal scoring | P1 | ✅ DONE (S02) | LightGBM walk-forward with 60+ features |
| 9 | CSV trade export for tax reporting | P2 | — | Like reference repo's trades.csv |
| 10 | Scheduled cron execution (auto-scan + trade) | P2 | Planned (S11) | Unattended daily runs |
| 11 | Indian market integration (Zerodha) | P1 | ✅ DONE (S03) | NSE stocks, full statutory cost model |
| 12 | Market-aware ML models | P1 | ✅ DONE (S04) | Separate features + configs for US vs India |
| 13 | Time-machine backtest + learning | P1 | ✅ DONE (S05) | Day-by-day replay, confidence calibration, model persistence |
| 14 | Trailing stop-loss | P2 | Planned (S07) | Lock in profits as price moves favorably |
| 15 | Indian F&O for live short selling | P2 | Planned (S08) | SLBM or F&O margin for Zerodha shorts |
| 16 | Zerodha Kite Connect API integration | P2 | — | Live order placement via API |
| 17 | Mid-cap Indian stocks | P3 | — | Extend watchlist beyond Nifty 50 |
| 18 | Delivery volume as feature (NSE bhavcopy) | P2 | — | Actual delivery % is strong FII/DII proxy |
| 19 | Multi-timeframe features (weekly + daily) | P3 | — | Higher timeframe trend confirmation |
| 20 | **Fix model calibration** | **P1** | Planned (S06) | Model says 75% but wins 31-44%. Must fix before live. |
| 21 | Platt scaling / isotonic regression | P2 | — | Post-hoc probability calibration for LightGBM |
| 22 | Reduce feature set for time-machine | P2 | — | Drop SMA200/trend_slope that produce NaN in short slices |
| 23 | Walk-forward validation within training window | P2 | — | Train/val split inside each retrain cycle |
