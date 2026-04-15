# S02: ML Signal Generator with Walk-Forward Backtesting

**Status:** DONE
**Branch:** `feature/s02-ml-signals`
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Replace rule-based signals with a LightGBM ML model that learns from 60+ technical features, trained in a walk-forward manner to avoid lookahead bias. Includes feature selection, ensemble classification, and realistic backtest with cost-aware P&L tracking.

## User Story
As a retail investor, I want ML-driven buy/sell signals so that I can exploit patterns in historical data that simple rule-based indicators miss.

## Design Decisions
1. **LightGBM over XGBoost** — faster training, lower memory, comparable accuracy for tabular data
2. **Walk-forward backtest** — retrain every 20 days on trailing 200-day window to avoid lookahead bias
3. **Feature selection** — top 40 features by importance if total exceeds 40 (prevents overfitting)
4. **Confidence threshold** — only trade when model confidence exceeds threshold (default 65%)
5. **Cost model** — 0.1% per trade (approximation of spread + slippage)
6. **Long-only initially** — buy on bullish signal, sell on bearish/stop-loss/take-profit

## Research
- 60+ features from pandas_ta: RSI, IBS, SMA/EMA crossovers, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX, ATR, OBV, VWAP proxy, Keltner Channels, Mean Reversion Z-scores, candlestick patterns, Heikin-Ashi, momentum divergence, seasonality, efficiency ratio
- Chart pattern detection (head-and-shoulders, double top/bottom, triangles) added as rolling features

## Acceptance Criteria
- [x] `build_feature_matrix()` generates 60+ features from OHLCV data
- [x] `MLSignalGenerator` trains LightGBM in walk-forward fashion (retrain every N days)
- [x] Backtest produces trades with entry/exit prices, P&L, and cost tracking
- [x] Confidence threshold filters low-conviction signals
- [x] Stop-loss (3%) and take-profit (5%) exits are enforced
- [x] Feature selection reduces dimensionality when feature count > 40
- [x] Results include win rate, total P&L, return %, max drawdown

## Technical Design
### Files to Create/Modify
| File | Change |
|------|--------|
| `trading_engine/ml_model.py` | New: `build_feature_matrix()`, `_SmartLGBM`, `MLSignalGenerator` classes |
| `ml_backtest.py` | Walk-forward backtest runner for US ETFs |
| `benchmark.py` | Compare ML vs rule-based strategies |

### Data Model Changes
None — backtest results returned as dicts, not persisted to SQLite.

### API Changes
None — ML backtesting is offline/batch, not exposed via MCP.

## Dependencies
- lightgbm, pandas-ta, numpy
- S01 (price engine, cost engine)

## Verification
- Run `python ml_backtest.py` — produces positive returns on US ETFs with >50% win rate
- Compare ML vs buy-and-hold — ML should outperform on risk-adjusted basis

## UAT
- [x] Feature matrix: 60+ columns generated from SPY OHLCV
- [x] Walk-forward: model retrains every 20 days, no lookahead
- [x] US backtest: $10K/65% → +3-5% return, ~50% win rate
- [x] Stop-loss/take-profit: trades exit correctly on threshold breach
- [x] Feature selection: active when >40 features, top 40 by importance

## Notes
- Performance depends heavily on market regime — long-only model struggles in bear markets
- Chart pattern features add modest lift (~0.5-1% improvement)
