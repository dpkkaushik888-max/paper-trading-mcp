# S04: Market-Aware ML Model with Long + Short Trading

**Status:** DONE
**Branch:** `feature/s04-market-aware-long-short`
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Refactor the ML signal generator to support multiple markets (US, India) with dedicated feature sets, hyperparameters, and cross-asset references. Add short selling capability so the model profits from both up and down moves.

## User Story
As a multi-market trader, I want the ML model to automatically use the right features and tuning for each market, and I want to profit from both bullish and bearish signals.

## Design Decisions
1. **Market dispatch via `MARKET_CONFIGS` dict** — each market has its own train window, min train, retrain interval, confidence threshold, LightGBM hyperparams, cross-asset symbol, and cross-asset features.
2. **India-specific feature builder** — `build_feature_matrix_india()` with ~65 features tuned for NSE: momentum (5d/10d/20d), gap analysis, volume spikes, regime detection, FII/DII proxies, A/D line, MFI, smart money flow, month-end effects.
3. **Regime detection** — SMA50-based bull/bear filter, higher highs/higher lows, SMA20 vs SMA50 crossover. Helps model learn when to go long vs short.
4. **FII/DII proxy features** — Accumulation/Distribution line, Money Flow Index, A/D divergence (price up but A/D down = distribution), smart money flow (volume × price direction), volume on up-days ratio.
5. **SMA100 not SMA200 for India** — saves 100 warmup days. Indian market has shorter cycles than US.
6. **Cross-asset reference** — US uses SPY, India uses NIFTYBEES.NS. Cross-asset features are prefixed (`spy_*`, `nifty_*`) and joined to each stock's features.
7. **Long + short positions** — model opens longs when `up_prob > threshold`, shorts when `down_prob > threshold`. Each side has independent stop-loss, take-profit, and ML signal exits.
8. **Wider stops for India** — 5% SL / 8% TP (vs 3% / 5% US) because Indian stocks are more volatile.
9. **Max 8 concurrent positions** — combined long + short, to manage risk.
10. **5 years of data for India** — 1236 days gives the model bull + bear cycles to learn from (vs 414 days previously).

## Research
- Indian market had a 17% correction Sept 2024 → Mar 2025 (Nifty 26,200 → 21,700). Long-only model lost money; long+short model with regime detection turned profitable.
- FII/DII flows are the dominant driver of Indian market direction. Direct flow data isn't available via Yahoo Finance, but volume-based proxies (A/D line, MFI, smart money flow) capture institutional behavior.
- SMA200 burned too many warmup days with only 414 days of data. SMA100 is sufficient for Indian cycle detection and preserves training data.
- Month-end/start effects correspond to FII rebalancing patterns in Indian markets.

## Acceptance Criteria
- [x] `MARKET_CONFIGS` dict with separate configs for "us" and "india"
- [x] `build_feature_matrix_india()` generates ~65 India-specific features
- [x] `build_features_for_market()` dispatches to correct builder based on market param
- [x] `MLSignalGenerator(market="india")` uses India config, features, and hyperparams
- [x] `MLSignalGenerator(market="us")` uses US config (backward compatible)
- [x] `_SmartLGBM` accepts custom hyperparameters from market config
- [x] Long + short positions tracked separately (`long_positions`, `short_positions`)
- [x] Stop-loss exits for both long and short positions
- [x] Take-profit exits for both long and short positions
- [x] ML signal exits: longs close on bearish signal, shorts cover on bullish signal
- [x] Cross-asset features handled robustly (reindex with fill_value=0 for missing cols)
- [x] Results include long/short trade breakdown
- [x] India backtest uses 5 years of data (1236 days)
- [x] Regime detection features: `regime_bull`, `regime_trend_strength`, `regime_higher_highs`, `regime_higher_lows`, `regime_sma20_above_50`
- [x] FII/DII proxy features: `ad_trend`, `ad_divergence`, `mfi`, `mfi_overbought`, `mfi_oversold`, `smart_money_flow`, `vol_on_up_days`

## Technical Design
### Files to Create/Modify
| File | Change |
|------|--------|
| `trading_engine/ml_model.py` | Add `MARKET_CONFIGS`, `build_feature_matrix_india()`, `build_features_for_market()`. Refactor `_SmartLGBM` to accept params. Refactor `MLSignalGenerator` with market param, long+short logic. |
| `india_backtest.py` | Use `MLSignalGenerator(market="india")` instead of custom inline backtest. Fetch 5y data. Show long/short breakdown. |

### Data Model Changes
None.

### API Changes
- `MLSignalGenerator.__init__()` now accepts `market: str = "us"` parameter
- Return dict now includes `long_trades`, `short_trades` counts
- Trade dicts now include `side` ("long" or "short") and `action` includes "short" and "cover"

## Dependencies
- S02 (ML model base), S03 (India/Zerodha integration)

## Verification
- `MLSignalGenerator(market="us")` still produces positive returns on US ETFs
- `MLSignalGenerator(market="india")` produces positive returns at 65-70% confidence
- Both long and short trades appear in results

## UAT
- [x] US market (market="us"): $10K/70% → +$336 (+3.36%), 84 long / 8 short trades
- [x] India market (market="india") ₹1L/70% → +₹27,213 (+27.21%), 408 long / 266 short, 52.7% win rate, 7.33% max DD
- [x] India ₹5L/70% → +₹1,20,182 (+24.04%), 7.82% max DD
- [x] India ₹10L/70% → +₹2,38,406 (+23.84%), 7.89% max DD
- [x] Regime features present in India feature matrix
- [x] FII/DII proxy features present in India feature matrix
- [x] Cross-asset features (nifty_*) correctly joined, missing cols filled with 0
- [x] Stop-loss triggers for both long and short verified in trade logs

## Notes
- India at 55-60% confidence is still negative — model needs high confidence to filter noise
- 70% confidence is the sweet spot: fewer trades (340 closed over 5 years) but higher quality
- Short selling in Indian delivery market requires Securities Lending (SLBM) — backtest assumes this is available. For live trading, F&O would be needed for shorting.
- TATAMOTORS.NS is unavailable on Yahoo Finance — excluded from watchlist
