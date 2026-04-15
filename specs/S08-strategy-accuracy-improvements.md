# S08: Strategy Accuracy Improvements

**Status:** IMPLEMENTED (all changes tested, none improved baseline)
**Branch:** master (direct)
**Priority:** P1 (critical)

## Overview
Improve ML model accuracy via 5 changes: multi-day target, VIX feature, sector-relative returns, model ensemble, and per-sector models.

## User Story
As a trader, I want the ML model to make more accurate predictions so that the backtest win rate and return improve beyond the current +0.23% / 44% WR baseline.

## Design Decisions
- 5-day target chosen because SL/TP (3%/5%) typically takes multiple days — 1-day target mismatches actual hold period.
- VIX is free alpha — market regime indicator that changes how signals should be interpreted.
- Sector-relative return isolates stock-specific alpha from sector beta.
- Ensemble (3 seeds, majority vote) reduces overconfident single-model trades.
- Per-sector models last — highest complexity, implement only if earlier changes insufficient.

## Research
Current system: 35 US features, 1-day return target, single LightGBM, pooled 31 stocks.
Calibration at 80% threshold: error=0.39, overconfident by ~40%.
Best result: +1.69% at 70% threshold (10 trades) or +0.23% at 80% (68 trades).

## Acceptance Criteria
- [ ] Target changed from 1-day to 5-day return direction
- [ ] VIX level + VIX 5-day change added as features
- [ ] Sector-relative return (stock return - sector ETF return) added as feature
- [ ] Ensemble of 3 models with majority vote
- [ ] 5y US backtest run and compared against S07 baseline (+0.23%, 44% WR)
- [ ] Results documented in STATE.md

## Technical Design
### Files to Modify
| File | Change |
|------|--------|
| `trading_engine/ml_model.py` | 5-day target, VIX features, sector-relative features |
| `trading_engine/ml_model.py` | `_SmartLGBMEnsemble` class (3 seeds, majority vote) |
| `trading_engine/time_machine.py` | Pass VIX data, use ensemble model |
| `trading_engine/config.py` | SECTOR_MAP constant |
| `time_machine_run.py` | Download VIX data alongside stocks |

### Sector Mapping
```
SECTOR_MAP = {
    "XLF": ["JPM", "BAC", "GS"],
    "XLE": ["XOM", "CVX"],
    "XLK": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "AVGO", "NFLX", "CRM"],
    "XLV": ["UNH", "JNJ", "PFE"],
    "XLY": ["WMT", "HD", "KO"],
    "XLI": ["CAT", "BA"],
}
```

## Dependencies
None

## Verification
Run `time_machine_run.py --market us --period 5y` before and after changes. Compare return %, win rate, max DD, calibration error.

## Baseline (S07, pre-S08)
| Metric | Value |
|--------|-------|
| Return | +0.23% |
| Win Rate | 44.1% |
| Max DD | 4.21% |
| Cal Error | 0.390 |
| Trades | 68 |

## Results

| Variant | Return | Win Rate | Max DD | Trades | Cal Error |
|---------|--------|----------|--------|--------|-----------|  
| **S07 Baseline** | **+0.23%** | **44.1%** | **4.21%** | **68** | **0.390** |
| +5-day target +all | -62.65% | 37.7% | 8.06% | 220 | 0.455 |
| +VIX +sector only | -56.63% | 46.1% | 7.09% | 364 | 0.425 |
| +Ensemble only | -3.22% | 35.3% | 6.33% | 102 | 0.503 |

**Conclusion:** All changes degraded performance. Root cause: 300-day × 31-stock training set is too small for additional features (curse of dimensionality). Adding features gives the model more dimensions to find spurious patterns, increasing overconfident trades. The baseline single-model with ~35 features is already near the accuracy ceiling for this data volume.

**Code disposition:** VIX/sector/ensemble code kept in codebase but NOT wired into the active path. Time machine reverted to baseline single `_SmartLGBM`.

## Notes
- 5-day target caused model to see more "up" signals → massive overtrading
- VIX/sector features added noise dimensions → more overfit
- Ensemble averaged 3 overfit models → still overfit
- Real improvement path: more training data, not more features
- Per-sector models deferred (would make data-per-model even smaller)
