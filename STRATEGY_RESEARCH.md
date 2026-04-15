# Strategy Research — Proven ETF Trading Strategies

Research date: 2026-04-13
Sources: QuantifiedStrategies.com, Larry Connors' books, Enlightened Stock Trading, The Robust Trader, academic papers

## Executive Summary

Our current strategy (RSI(3) mean reversion + MACD/BB/EMA filters) is a custom blend that overfits to recent windows. Research shows **5 well-documented strategies** with published backtest results spanning 20+ years on the exact ETFs we trade. All are mean-reversion based — the academic evidence for momentum on daily ETF bars is weak, but mean reversion is robust.

---

## Strategy 1: Connors RSI(2) — "The Classic"

**Source:** Larry Connors, *Short Term Trading Strategies That Work* (2008)
**Backtest:** SPY 1993–2026, 9% CAGR, 91% win rate, 28% time invested

### Rules (Long Only)
| # | Rule | Indicator |
|---|------|-----------|
| 1 | Close > SMA(200) | Uptrend filter |
| 2 | RSI(2) < 10 | Extreme oversold |
| 3 | Buy at close | Entry |
| 4 | Exit when RSI(2) > 80 | Exit |
| 5 | Stop loss: 3% (optional, Bellelli's addition) | Risk |

### Short Rules (Mirror)
| # | Rule | Indicator |
|---|------|-----------|
| 1 | Close < SMA(200) | Downtrend filter |
| 2 | RSI(2) > 90 | Extreme overbought |
| 3 | Short at close | Entry |
| 4 | Cover when RSI(2) < 20 | Exit |

### Why it works
- RSI(2) is extremely sensitive — 2 down days push it near 0
- SMA(200) filter keeps you with the macro trend
- Only trades during panic/euphoria extremes

### What we need to add
- [x] RSI(2) indicator (new)
- [x] SMA(200), SMA(5) (new — currently only EMA)

---

## Strategy 2: Connors 3-Day High/Low Method

**Source:** Larry Connors, *High Probability ETF Trading* (2009)
**Backtest:** SPY avg gain 0.72% per trade, profit factor 2.93. Works on SPY, QQQ, IWM, DIA, GLD.

### Rules (Long Only)
| # | Rule | Indicator |
|---|------|-----------|
| 1 | Close > SMA(200) | Uptrend filter |
| 2 | Close < SMA(5) | Short-term weakness |
| 3 | Today's High < Yesterday's High | 3 consecutive lower highs |
| 4 | Today's Low < Yesterday's Low | 3 consecutive lower lows |
| 5 | Yesterday's H/L < Day-before's H/L | Pattern continues |
| 6 | Day-before's H/L < 3-days-ago's H/L | Full 3-day pattern |
| 7 | Buy at close | Entry |
| 8 | Exit when Close > SMA(5) | Exit |

### Why it works
- 3 consecutive lower highs AND lower lows = panic selling exhaustion
- SMA(200) ensures you're buying in a bull market dip
- SMA(5) exit = quick profit-taking at mean

### What we need to add
- [x] SMA(5), SMA(200)
- [x] Previous day High/Low tracking (prev_high_1, prev_low_1, prev_high_2, etc.)
- [x] Consecutive lower high/low count

---

## Strategy 3: Turnaround Tuesday (with IBS filter)

**Source:** QuantifiedStrategies.com, Enlightened Stock Trading
**Backtest:** SPY, 69% win rate, 7% CAGR, 0.46% avg gain per trade

### Rules (Best Variation)
| # | Rule | Indicator |
|---|------|-----------|
| 1 | Today is Monday | Day-of-week |
| 2 | Monday close < Friday close | Weekend gap down |
| 3 | IBS < 0.5 | Closed in lower half of range |
| 4 | Buy at Monday close | Entry |
| 5 | Exit when Close > Yesterday's High OR 4 days later | Exit |

### IBS Formula
```
IBS = (Close - Low) / (High - Low)
```
IBS < 0.2 = closed near the low (very bearish day → reversal likely)
IBS > 0.8 = closed near the high

### Why it works
- Weekend fear-driven selling creates Monday dip
- Institutional repositioning on Tuesday creates bounce
- IBS filter captures days where selling was aggressive (close near low)

### What we need to add
- [x] IBS indicator
- [x] Day-of-week in indicators
- [x] Previous day's high (for exit condition)
- [x] Days-held counter for "exit after N days"

---

## Strategy 4: IBS Mean Reversion (Pure)

**Source:** Pagonidis (NAAIM paper), QuantifiedStrategies
**Backtest:** SPY, works since 1993

### Rules (Long Only)
| # | Rule | Indicator |
|---|------|-----------|
| 1 | Close > SMA(200) | Uptrend filter |
| 2 | IBS < 0.2 | Closed near the low |
| 3 | Buy at close | Entry |
| 4 | Exit when IBS > 0.8 | Exit (closed near high) |

### Why it works
- Pure price-action based — no lagging oscillators
- When a stock closes near its daily low in an uptrend, next-day bounce is statistically likely
- Very capital efficient — trades infrequently

### What we need to add
- [x] IBS (same as Strategy 3)

---

## Strategy 5: RSI(2) + IBS Combo

**Source:** Combining Connors RSI(2) with IBS filter (Pagonidis research)
**Rationale:** IBS filter reduces number of trades while boosting per-trade returns

### Rules (Long Only)
| # | Rule | Indicator |
|---|------|-----------|
| 1 | Close > SMA(200) | Uptrend filter |
| 2 | RSI(2) < 10 | Extreme oversold |
| 3 | IBS < 0.3 | Confirmation: closed near low |
| 4 | Buy at close | Entry |
| 5 | Exit when RSI(2) > 80 AND IBS > 0.7 | Both confirm overbought |

---

## Implementation Priority

| # | Strategy | Difficulty | Expected Edge | Priority |
|---|----------|-----------|---------------|----------|
| 1 | Connors RSI(2) | Low | High (91% win rate proven) | **P1** |
| 4 | IBS Mean Reversion | Low | Medium | **P1** |
| 5 | RSI(2) + IBS Combo | Low | High (best of both) | **P1** |
| 3 | Turnaround Tuesday | Medium (day-of-week) | Medium (7% CAGR) | **P2** |
| 2 | 3-Day High/Low | Medium (multi-bar pattern) | Medium-High | **P2** |

## New Indicators Required

| Indicator | Formula | Used By |
|-----------|---------|---------|
| `rsi_2` | RSI(close, period=2) | S1, S5 |
| `sma_5` | SMA(close, period=5) | S1, S2 |
| `sma_200` | SMA(close, period=200) | S1, S2, S4, S5 |
| `ibs` | (Close - Low) / (High - Low) | S3, S4, S5 |
| `day_of_week` | 0=Mon, 1=Tue, ... 4=Fri | S3 |
| `prev_high_1` | Yesterday's high | S2, S3 |
| `prev_low_1` | Yesterday's low | S2 |
| `prev_high_2` | 2 days ago high | S2 |
| `prev_low_2` | 2 days ago low | S2 |
| `prev_high_3` | 3 days ago high | S2 |
| `prev_low_3` | 3 days ago low | S2 |
| `prev_close_1` | Yesterday's close | S3 |
| `consecutive_lower_highs` | Count of consecutive lower highs | S2 |
| `consecutive_lower_lows` | Count of consecutive lower lows | S2 |

## What NOT to Implement

- **Dual Momentum (Antonacci)** — Monthly rebalance, not daily. Different engine paradigm.
- **Pure Momentum/Breakout** — Academic evidence weak on daily ETF bars, better on weekly/monthly.
- **Martingale/Grid** — High risk, not suitable for small capital.
