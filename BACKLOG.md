# Backlog

Ideas and future work not yet scheduled into a milestone.

| # | Idea | Priority | Status | Notes |
|---|------|----------|--------|-------|
| 1 | Short selling support | P1 | ✅ DONE (S04) | Long + short with separate position tracking |
| 2 | Run multiple strategies in parallel, compare daily P&L | P1 | ✅ DONE (S11) | Candlestick+SR + ML Sniper multi-strategy engine |
| 3 | Web dashboard with Chart.js P&L graphs | P3 | Planned (S16) | Visual daily/weekly/monthly performance |
| 4 | eToro browser automation (Playwright) | P3 | — | Auto-replicate paper signals on eToro virtual portfolio |
| 5 | Telegram/email alerts on signals | P3 | Planned (S17) | Push notification when signal fires |
| 6 | Crypto support (BTC-USD, ETH-USD) | P1 | ✅ DONE (S09) | 12 crypto assets, best-performing market |
| 7 | Options strategies (covered calls on ETFs) | P3 | — | Income generation |
| 8 | Machine learning signal scoring | P1 | ✅ DONE (S02) | LightGBM walk-forward with 60+ features |
| 9 | CSV trade export for tax reporting | P2 | — | Like reference repo's trades.csv |
| 10 | Scheduled cron execution (auto-scan + trade) | P2 | Planned (S14) | Unattended daily runs |
| 11 | Indian market integration (Zerodha) | P1 | ✅ DONE (S03) | NSE stocks, full statutory cost model |
| 12 | Market-aware ML models | P1 | ✅ DONE (S04) | Separate features + configs for US vs India + Crypto |
| 13 | Time-machine backtest + learning | P1 | ✅ DONE (S05) | Day-by-day replay, confidence calibration, model persistence |
| 14 | Trailing stop-loss | P2 | DROPPED | Tested via Fabio insights — not validated for daily bars |
| 15 | Indian F&O for live short selling | P2 | Planned (S18) | SLBM or F&O margin for Zerodha shorts |
| 16 | Zerodha Kite Connect API integration | P2 | Planned (S15) | Live order placement via API |
| 17 | Mid-cap Indian stocks | P3 | — | Extend watchlist beyond Nifty 50 |
| 18 | Delivery volume as feature (NSE bhavcopy) | P2 | — | Actual delivery % is strong FII/DII proxy |
| 19 | Multi-timeframe features (weekly + daily) | P3 | — | Higher timeframe trend confirmation |
| 20 | Fix model calibration | P1 | ✅ DONE (S06) | Platt scaling reduced cal error from 0.45 to 0.125 (US) |
| 21 | Platt scaling / isotonic regression | P1 | ✅ DONE (S06) | Integrated into _SmartLGBM and all SmartClassifier wrappers |
| 22 | Reduce feature set for time-machine | P2 | ✅ DONE (S06) | SMA200 → SMA100, trend_slope_pct removed |
| 23 | Walk-forward validation within training window | P2 | — | Train/val split inside each retrain cycle |
| 24 | ML algorithm comparison | P1 | ✅ DONE (S13) | Logistic Regression beats LightGBM, XGBoost, RF, MLP |
| 25 | Autoresearch optimizer | P1 | ✅ DONE (S10) | 3 rounds: balanced → winrate → logistic C-tuning |
| 26 | Circuit breakers (portfolio-level) | P1 | ✅ DONE (S07) | 4-tier drawdown protection (HMM regime stripped) |
| 27 | Strategy accuracy experiments | P1 | ✅ DONE (S08) | 5-day target, VIX, sector, ensemble — all failed |
| 28 | India feature engineering v2 | P1 | — | India model still not profitable — needs new approach |
| 29 | Per-symbol calibration | P2 | — | Currently pooled; per-symbol could help India |
| 30 | Logistic Regression for US/India markets | P2 | — | Currently only crypto uses Logistic; test on other markets |
