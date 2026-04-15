# S03: Indian Market Integration — Zerodha Broker

**Status:** DONE
**Branch:** `feature/s03-india-zerodha`
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Extend the trading engine to support the Indian market (NSE) via Zerodha as broker. Includes full Zerodha cost model (STT, stamp duty, GST, SEBI fees), Indian stock data via Yahoo Finance `.NS` suffix, and a dedicated Indian backtest script.

## User Story
As an Indian retail investor, I want to paper-trade NSE stocks using Zerodha's real cost structure so that I can evaluate ML strategies against Indian market dynamics before committing real capital.

## Design Decisions
1. **Yahoo Finance for NSE data** — symbols use `.NS` suffix (e.g., `RELIANCE.NS`). Free, 5 years history available. No separate Indian data vendor needed for MVP.
2. **Zerodha cost profile in JSON** — follows same broker_profiles pattern as eToro. Zero brokerage on equity delivery, full statutory charges modeled.
3. **Indian statutory charges** — STT (0.1% buy + 0.1% sell delivery), stamp duty (0.015% buy only), GST (18% on transaction charges), SEBI turnover fee (0.0001%), exchange transaction charges (0.00345%).
4. **Indian tax model** — STCG 20% (<1 year), LTCG 12.5% (>1 year, above ₹1.25L exemption). Budget 2024 rates.
5. **Large-cap watchlist** — 20 NSE stocks: RELIANCE, TCS, HDFC Bank, Infosys, ICICI Bank, HUL, SBI, Bharti Airtel, ITC, Kotak Bank, L&T, Axis Bank, Bajaj Finance, Maruti, Titan, Sun Pharma, HCL Tech, Wipro, Tata Motors, NIFTYBEES ETF.
6. **No FX cost** — trading in INR, no currency conversion needed.

## Research
- Zerodha charges (2024-25): Zero brokerage on equity delivery. ₹20 per order intraday/F&O.
- STT is the dominant cost (~0.1% each way on delivery). Total round-trip cost ~0.16%.
- eToro comparison: ~1.6% round-trip (mostly FX). Zerodha is 10x cheaper.
- Yahoo Finance `.NS` provides OHLCV for all NSE stocks, 5+ years history.
- TATAMOTORS.NS delisted from Yahoo Finance as of April 2026.

## Acceptance Criteria
- [x] `broker_profiles/zerodha.json` exists with all Indian statutory charges
- [x] `CostEngine("zerodha")` loads and calculates Indian costs correctly
- [x] Indian statutory charges calculated: STT, stamp duty, GST, SEBI, exchange fees
- [x] Indian tax calculation: STCG 20%, LTCG 12.5% with ₹1.25L exemption
- [x] `india_backtest.py` downloads NSE data and runs ML backtest
- [x] Cost breakdown shows spread, slippage, statutory charges separately
- [x] Zerodha spread estimates for large-cap stocks (0.02%) are applied

## Technical Design
### Files to Create/Modify
| File | Change |
|------|--------|
| `broker_profiles/zerodha.json` | New: full Zerodha fee profile with Indian statutory charges |
| `trading_engine/cost_engine.py` | Add `is_indian_market()`, `_calculate_indian_statutory()`, `_calculate_indian_tax()` |
| `india_backtest.py` | New: Indian market backtest script with NSE watchlist |

### Data Model Changes
None.

### API Changes
None — backtest is offline.

## Dependencies
- S01 (cost engine base), S02 (ML model)

## Verification
- `CostEngine("zerodha").calculate_trade_cost("RELIANCE.NS", 2500, 40, "buy", "INR")` returns correct cost with statutory charges
- `python india_backtest.py` downloads 19+ stocks and runs backtest

## UAT
- [x] Zerodha profile loads: zero brokerage, STT 0.1%, stamp 0.015%, GST 18%
- [x] Cost per ₹1L trade: ₹158.60 (0.159%) — dominated by STT
- [x] Data: 19 NSE stocks downloaded, 1236 days (5 years) each
- [x] Backtest runs end-to-end with Zerodha cost model applied

## Notes
- TATAMOTORS.NS returns 404 from Yahoo Finance — possibly delisted or symbol changed
- Yahoo Finance Indian data quality is good for large-caps but may have gaps for mid/small-caps
- DP charges (₹15.93 per sell transaction) not yet included in backtest cost model
