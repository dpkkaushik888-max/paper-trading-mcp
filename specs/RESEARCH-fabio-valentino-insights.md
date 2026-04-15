# Research: Fabio Valentino Scalping Insights

**Source:** Words of Wisdom Podcast — Live NQ Scalping Session (Oct 15, 2024)
**Relevance:** Risk management, session management, statistical edge optimization

## Actionable Improvements for ML Engine

### 1. Trailing Profit Circuit Breaker (S07 Enhancement)
**Insight:** Fabio uses trailing profit protection — once up $20K+, he risks only a portion of profits.
**Implementation:** Add a `trailing_profit` tier to circuit breakers that activates when portfolio is up X% for the session. Instead of only tracking drawdown from peak, also track "profit at risk" — the max amount of today's profit we're willing to give back.
- **Feature:** `CIRCUIT_BREAKER_TIERS["profit_protection"]` — e.g., if day PnL > 3%, risk max 50% of day's profit
- **Priority:** P2 (S07 enhancement or S08)

### 2. Session Volatility / Expansion-Contraction Detection (S08 Feature)
**Insight:** "70% of the time the market is stationary. I want to make money also when it's stationary." He switches models based on expansion vs contraction days.
**Implementation:** Add features measuring intraday/recent session character:
- `session_range_ratio` = today's range / avg 20-day range (>1.5 = expansion, <0.5 = contraction)
- `vwap_std_dev_position` = where price is relative to VWAP bands
- `volume_profile_shape` = P-shape (accumulation top) vs b-shape (accumulation bottom) vs D-shape (balanced)
- Use these to adjust confidence threshold or model selection
- **Priority:** P2 (new spec S08)

### 3. Day-of-Week Filter (Quick Win)
**Insight:** "On Friday you always lose money — remove Friday. I have one day of vacation per week." He removed Wednesdays from crude oil. Each instrument has bad days.
**Implementation:** After accumulating enough trade history, compute per-day-of-week win rate. If a day's win rate is <30% over 50+ trades, skip trading that day.
- **Feature:** `day_of_week` as a categorical feature for LightGBM
- **Meta-rule:** Add to learning loop — track per-DOW performance
- **Priority:** P3 (S09 or backlog)

### 4. Time-of-Day Decay (Quick Win)
**Insight:** "After 7pm European time, your win rate is 20%. Remove this."
**Implementation:** For daily bars this doesn't apply directly, but the principle translates: if signals generated on certain calendar periods (month-end, FOMC weeks, earnings season) have low win rate, filter them.
- **Feature:** `is_fomc_week`, `is_earnings_season`, `is_quarter_end`
- **Priority:** P3 (backlog)

### 5. Win Rate vs R:R Balance
**Insight:** "You cannot have 1:20 R:R with 75% win rate." His sweet spot: 43-49% win rate with avg win = 1.7x avg loss. The key metric is *profit factor*, not win rate alone.
**Implementation:** Currently we use fixed stop-loss (3%) and take-profit (5%). This gives ~1.67 R:R. Consider:
- Dynamic SL/TP based on recent volatility (ATR-based)
- Asymmetric: tighter SL in low-vol regimes, wider in high-vol
- Track profit factor per regime in learning loop
- **Priority:** P2 (S07 or S08 enhancement)

### 6. Scale-In Position Management
**Insight:** Fabio starts small and adds as the trade works. "I'm not going to load everything while the market is collapsing."
**Implementation:** Instead of entering full position at once, consider:
- Enter 50% on initial signal
- Add 50% if price moves in favorable direction by 1%
- This naturally reduces average loss size
- **Priority:** P3 (backlog — complex to implement in daily bar framework)

### 7. Loss Streak Circuit Breaker
**Insight:** "Days that start with 3-4-5 stop losses end up losing $15-20K. I started cutting them at 8-9K."
**Implementation:** Track consecutive losses per session. If N consecutive losses in a row, halt trading for the day.
- Add `consecutive_loss_count` to circuit breaker state
- If 3+ consecutive losses, activate DANGER tier
- **Priority:** P2 (S07 enhancement)

## Key Statistical Principles
1. **Edge = asymmetric R:R × acceptable win rate** — not just one or the other
2. **Remove bad periods** — identify and exclude statistically losing conditions
3. **Build cushion, then risk profit** — protect principal, risk only gains
4. **Quick losses, slow wins** — avg loss must be smaller than avg win
5. **Track everything** — you can't improve what you don't measure
6. **Discretionary adapts, systematic decays** — HMM regime detection is our "discretionary" layer

## Backtesting Validation (2026-04-14)

**Method:** 5y US backtest (31 stocks, ~1200 days, 68 trades, 33 closed)
**Script:** `validate_fabio_insights.py`

### Results

| # | Insight | Verdict | Key Finding |
|---|---------|---------|-------------|
| 1 | Trailing profit protection | **NOT VALIDATED** | Delta: -$118.62. Protection never triggered (0 triggers). With only $89 peak profit on $10K, the profit cushion is too small for this to matter. |
| 3 | Day-of-week filter | **VALIDATED** | Monday = 0% WR (0/3 wins, -$146 PnL). Removing Monday improves PnL by +$146. Friday is actually the best day (75% WR, +$256 PnL) — opposite of Fabio's rule. |
| 5 | Win Rate vs R:R balance | **VALIDATED** | WR=48.5%, R:R=1.22x, PF=1.15. R:R is below Fabio's 1.7x target. Simulating tighter SL (2% vs 3%) improves PnL by +$424 — significant. Avg win ($57) barely exceeds avg loss ($47). |
| 7 | Consecutive loss breaker | **NOT VALIDATED** | Loss streaks exist (3 × 2-streak, 1 × 6-streak) but halting doesn't improve PnL — the daily framework means max 1 exit per symbol per day, so multi-loss days are rare. |

### Insights for Implementation

**Implement (validated):**
- **#3 Day-of-week feature:** Add `day_of_week` as a categorical feature to LightGBM. The model can learn which days are bad per market/regime. Simple, zero-risk addition.
- **#5 Tighter SL (ATR-based):** ~~Current 3% SL lets losses run too far.~~ Retroactive simulation showed +$424, but actual ATR-based implementation tested worse: ATR 1.5x/2.5x → -0.18%, ATR 2.0x/3.0x → -2.79% vs baseline +0.23%. The ML model's signals are calibrated around fixed 3%/5% thresholds. Code kept as opt-in `--dynamic-sl` but not recommended.

**Do not implement (not validated):**
- **#1 Trailing profit protection:** Makes sense for intraday scalping with large profit days, but our daily-bar system never accumulates enough intraday profit for this to help.
- **#7 Consecutive loss breaker:** In daily-bar framework, we rarely have multiple exits per day. This is an intraday scalping concept that doesn't translate to our timeframe.

**Cannot test yet (need code changes):**
- **#2 Expansion/contraction detection:** Needs new features
- **#4 Macro calendar features:** Needs external data
- **#6 Scale-in positions:** Needs position management rework

## Backlog Items for Future Specs
| # | Idea | Priority | Validated? | Inspired By |
|---|------|----------|-----------|-------------|
| 1 | Trailing profit circuit breaker | ~~P2~~ DROPPED | No | "I'm not giving back the 20K I made" |
| 2 | Expansion/contraction session type detector | P2 | Untested | "70% of time market is stationary" |
| 3 | Day-of-week feature + filter | **P1** | **Yes** | "Remove Fridays" (for us: Mondays) |
| 4 | Macro calendar features (FOMC, earnings) | P3 | Untested | "Time-of-day w/r is 20%" |
| 5 | ATR-based dynamic SL/TP | ~~P1~~ DROPPED | No (impl failed) | "R:R balance is everything" |
| 6 | Scale-in position management | P3 | Untested | "Load as we go up" |
| 7 | Consecutive loss circuit breaker | ~~P2~~ DROPPED | No | "3-5 stop losses = bad day" |
