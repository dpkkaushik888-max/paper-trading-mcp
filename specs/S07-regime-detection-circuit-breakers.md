# S07: HMM Regime Detection + Portfolio Circuit Breakers

**Status:** DRAFT
**Branch:** `feature/s07-regime-circuit-breakers`
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Add two critical missing components inspired by the "Regime Trader" YouTube system:
1. **HMM Regime Detection** — classify the market into states (crash/bear/neutral/bull/euphoria) using Hidden Markov Models, and feed the regime as a feature to the ML model so it adapts behavior to market conditions.
2. **Portfolio-Level Circuit Breakers** — tiered safety system that reduces position sizes or halts trading when portfolio drawdown exceeds thresholds, independent of per-trade SL/TP.

## User Story
As a trader, I want the system to detect the current market regime so the ML model can adjust its behavior, and I want portfolio-level circuit breakers so a streak of bad trades doesn't destroy my capital.

## Design Decisions
1. **HMM as feature, not strategy** — Unlike the YouTube system which uses HMM to drive allocation, we feed regime state as additional features to our existing LightGBM model. This lets the model learn regime-specific patterns itself.
2. **Forward algorithm only** — HMM predict uses only past data (no Viterbi on full sequence) to avoid look-ahead bias.
3. **Auto regime count** — Test 3-7 regimes, pick best via BIC. Label by mean return.
4. **Stability filter** — Regime must persist 3+ bars before acting on it. Flickering = uncertainty → reduce sizing.
5. **Circuit breakers are independent** — They override ML signals regardless of model confidence.
6. **SPY for US, NIFTYBEES for India** — HMM trains on the index, not individual stocks.

## Research
- YouTube "Regime Trader" system uses HMM with 5 regimes (crash/bear/neutral/bull/euphoria).
- `hmmlearn` library provides GaussianHMM with forward algorithm access.
- Our existing cross-asset feature infrastructure already passes SPY/NIFTY data through — regime features fit naturally alongside.

## Acceptance Criteria
- [ ] HMM regime detector trains on index data (SPY or NIFTYBEES) using only past data
- [ ] Forward algorithm used for prediction (no look-ahead bias)
- [ ] Auto-selects optimal regime count (3-7) via BIC
- [ ] Regimes labeled by mean return (crash → euphoria)
- [ ] Regime state + confidence fed as features to LightGBM
- [ ] Stability filter: regime persists 3+ bars before changing
- [ ] Portfolio circuit breaker: -2% daily drawdown → halve new position sizes
- [ ] Portfolio circuit breaker: -3% daily drawdown → no new positions
- [ ] Portfolio circuit breaker: -5% weekly drawdown → close all positions
- [ ] Portfolio circuit breaker: -8% from peak → full halt (skip remaining days in backtest)
- [ ] 6-month quick backtest shows no errors and circuit breakers activate correctly
- [ ] Full 5-year backtest runs successfully (deferred — run after quick validation)

## Technical Design

### Files to Create
| File | Purpose |
|------|---------|
| `trading_engine/regime_detector.py` | HMM regime detection engine |

### Files to Modify
| File | Change |
|------|--------|
| `trading_engine/time_machine.py` | Add regime features during feature build, add circuit breaker logic in `_process_day` |
| `trading_engine/ml_model.py` | Add regime feature names to MARKET_CONFIGS |
| `trading_engine/config.py` | Add circuit breaker threshold constants |
| `requirements.txt` | Add `hmmlearn` |
| `time_machine_run.py` | Add `--period` arg to support quick 6m backtests |

### Architecture

```
                    ┌─────────────┐
  SPY/NIFTY data ──►│ HMM Regime  │──► regime_state (0-4)
                    │ Detector    │──► regime_confidence (0-1)
                    │ (forward)   │──► regime_volatility
                    └─────────────┘──► regime_stability (bool)
                           │
                           ▼
                    ┌─────────────┐
  Stock features ──►│ LightGBM    │──► trade signal
  Cross-asset    ──►│ (existing)  │
  Rel. strength  ──►│             │
  + Regime feats ──►│             │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Circuit     │──► ALLOW / REDUCE / BLOCK / HALT
                    │ Breakers    │
                    └─────────────┘
```

### Circuit Breaker Tiers
| Tier | Trigger | Action |
|------|---------|--------|
| NORMAL | DD < 2% daily | Full position sizing |
| CAUTION | DD ≥ 2% daily | Halve new position sizes |
| DANGER | DD ≥ 3% daily | No new positions (exits only) |
| CRITICAL | DD ≥ 5% weekly | Close all positions immediately |
| HALT | DD ≥ 8% from peak | Stop trading entirely |

### Regime Features Added to ML
| Feature | Description |
|---------|-------------|
| `regime_state` | Integer 0-4 (crash to euphoria) |
| `regime_confidence` | HMM state probability (0-1) |
| `regime_volatility` | Mean volatility of current regime |
| `regime_is_stable` | 1 if regime persisted 3+ bars, else 0 |

## Dependencies
- `hmmlearn>=0.3.0` — Gaussian HMM implementation
- S06 — Model calibration (already implemented)

## Verification
1. **Quick test (6 months):** `python time_machine_run.py --market us --period 6m`
   - Verify regime features appear in model
   - Verify circuit breakers activate (if drawdown triggers occur)
   - Check regime distribution in output
2. **Full test (5 years):** `python time_machine_run.py --market us --period 5y`
   - Compare return/WR/DD vs pre-S07 baseline
   - Verify circuit breakers prevented worst drawdowns

## UAT
- [ ] Criterion 1 (HMM trains): pass/fail + evidence
- [ ] Criterion 2 (forward algorithm): pass/fail + evidence
- [ ] Criterion 3 (auto regime count): pass/fail + evidence
- [ ] Criterion 4 (regime labels): pass/fail + evidence
- [ ] Criterion 5 (features fed to LightGBM): pass/fail + evidence
- [ ] Criterion 6 (stability filter): pass/fail + evidence
- [ ] Criterion 7-10 (circuit breakers): pass/fail + evidence
- [ ] Criterion 11 (6m backtest): pass/fail + evidence
- [ ] Criterion 12 (5y backtest): deferred

## Notes
- HMM needs ~252 days (1 year) of data to train reliably. For 6-month backtest, use 1y of data before the test window for HMM training.
- Circuit breaker thresholds may need tuning after first backtest.
- India model uses NIFTYBEES.NS as regime index.
