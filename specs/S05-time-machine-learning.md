# S05: Time-Machine Backtest + Continuous Learning

**Status:** IMPLEMENTED
**Branch:** `feature/s05-time-machine-learning`
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Replace the batch walk-forward backtest with a day-by-day "time machine" replay that enforces strict temporal isolation and integrates a continuous learning loop with confidence calibration.

## User Story
As a trader, I want my backtest to simulate real-time trading exactly — processing one day at a time with only past data visible — so I can trust that backtested results will hold in live trading.

## Design Decisions
1. **Strict temporal isolation** — Features are computed per-symbol with `df[df.index <= current_day]`. No future data can leak through rolling windows or cross-asset joins.
2. **Learning loop** — After each closed trade, outcome (P&L, holding period, MAE/MFE) is stored. Model retrains with outcome-weighted samples — upweighting cases where the model was confident but wrong.
3. **Confidence calibration** — Tracks win rate per confidence bucket (e.g., "when I say 70%, am I right 70%?"). Reports calibration error and whether model is overconfident.
4. **Adaptive threshold** — If model is overconfident, the threshold auto-adjusts to the lowest bucket where win rate > 50%.
5. **SQLite persistence** — Trade journal + daily snapshots + model training snapshots all persisted for analysis.
6. **Model persistence** — LightGBM models saved to `models/` directory as pickle files after each retrain.
7. **Margin for shorts** — Short positions deduct `entry_price * shares` as margin from cash (fixing a cash accounting bug in the batch backtest).
8. **None-safe feature builders** — All SMA/indicator computations guard against `None` return when data is shorter than indicator period.

## Research
### Batch vs Time-Machine Comparison
| Metric | US Batch | US Time-Machine | India Batch | India Time-Machine |
|--------|----------|-----------------|-------------|-------------------|
| Return | +1.09% | -4.62% | +24.65% | -6.77% |
| Win rate | 44.2% | 30.9% | 52.4% | 43.9% |
| Trades | 89 | 189 | 674 | 581 |
| Max DD | 4.37% | 9.54% | 6.67% | 13.04% |
| Runtime | 9s | 126s | 51s | 883s |

### Key Insight: Massive Overconfidence
The model says 75% confidence but only wins 31-44% of the time. Calibration error = 0.39-0.45. This means the batch backtest's positive results benefited from subtle temporal leakage in feature computation (rolling windows computed on full history).

### What This Means for Live Trading
The time-machine results are the realistic baseline. The model needs significant improvement before going live:
- **US**: Not profitable at 70% threshold (-4.62%)
- **India**: Not profitable at 70% threshold (-6.77%)
- **Root cause**: Overconfidence — the model's probability estimates are poorly calibrated

## Acceptance Criteria
- [x] `TimeMachineBacktest` class processes one day at a time
- [x] Features computed with strict `df[df.index <= day]` slicing
- [x] No future data accessible during feature computation
- [x] Trade journal persists all trades to SQLite with outcome metrics
- [x] Learning loop computes confidence calibration per bucket
- [x] Adaptive threshold adjusts based on calibration
- [x] Model saved to disk after each retrain
- [x] Daily snapshots persisted with cash, positions, threshold
- [x] Short position margin correctly deducted from cash
- [x] Comparison runner shows batch vs time-machine side-by-side
- [x] Calibration report shows overconfidence clearly
- [x] Both US and India markets run without errors

## Technical Design
### Files Created
| File | Purpose |
|------|---------|
| `trading_engine/trade_journal.py` | SQLite trade journal + outcome tracking + daily snapshots |
| `trading_engine/learning_loop.py` | Confidence calibration + outcome weighting + adaptive thresholds |
| `trading_engine/time_machine.py` | Day-by-day replay engine with strict temporal isolation |
| `time_machine_run.py` | CLI runner comparing batch vs time-machine for US + India |

### Files Modified
| File | Change |
|------|--------|
| `trading_engine/ml_model.py` | Added `save()`/`load()` to `_SmartLGBM`. Added None guards for SMA/indicator returns in both US and India feature builders. |

### Data Model (SQLite: trade_journal.db)
- `journal_trades` — entry/exit dates, prices, confidence, P&L, MAE/MFE, holding days
- `model_snapshots` — training date, samples, feature count, calibration JSON
- `daily_snapshots_tm` — daily cash, position values, threshold, calibration

## Dependencies
- S02 (ML model), S03 (India market), S04 (long+short)

## Verification
```bash
# Run US comparison
python time_machine_run.py --market us

# Run India comparison
python time_machine_run.py --market india

# Run both with learning disabled
python time_machine_run.py --no-learning

# Verbose mode (day-by-day output)
python time_machine_run.py --market india --verbose
```

## UAT
- [x] US time-machine runs without errors — 189 trades, -4.62% return, 30.9% WR
- [x] India time-machine runs without errors — 581 trades, -6.77% return, 43.9% WR
- [x] Calibration report shows model is overconfident (error ~0.4)
- [x] Confidence buckets show win rate per threshold band
- [x] Model persisted to `models/india_latest.pkl`
- [x] Trade journal populated in `trade_journal.db`
- [x] Batch vs time-machine delta clearly shown in comparison table

## Notes
- Time-machine is ~15x slower than batch (883s vs 51s for India) because it rebuilds features per-symbol per-day
- The negative returns in time-machine are *honest* — this is what live trading would actually produce
- Next steps: improve feature engineering and model calibration to achieve positive returns in time-machine mode
- Consider: reduce feature set (fewer NaN-producing long-period indicators), improve training sample weighting, add walk-forward validation within the training window
