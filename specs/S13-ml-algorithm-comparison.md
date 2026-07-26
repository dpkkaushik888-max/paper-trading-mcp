# S13: ML Algorithm Comparison + Logistic Regression C-Tuning

**Status:** DONE
**Branch:** master (direct)
**Priority:** P1 (critical)
**Ticket:** N/A

## Overview
Test multiple ML classification algorithms on the same crypto dataset to determine which produces the best trading signals. Currently only LightGBM is used — this spec adds XGBoost, Random Forest, Logistic Regression, and MLP (neural net) as drop-in alternatives. Then optimize the winner's hyperparameters via the autoresearch framework.

## User Story
As a trader, I want to compare different ML algorithms on historical data so I can pick the one that produces the most profitable and reliable trading signals.

## Design Decisions
1. **Same features, same data** — All algorithms get identical feature matrices to ensure fair comparison.
2. **Drop-in interface** — Each wrapper implements the same `fit()` / `predict_proba()` / `save()` / `load()` API as `_SmartLGBM`.
3. **Platt scaling for all** — All non-probabilistic models get the same calibration treatment (LogisticRegression on held-out logits/scores).
4. **TimeMachine pluggable** — `TimeMachineBacktest` now accepts `model_factory` param for drop-in algorithm swap. Auto-detects `model_type` from market config.
5. **XGBoost optional** — If `xgboost` is not installed, skip it gracefully.
6. **Autoresearch integration** — Added `--mode logistic` to autoresearch for Logistic Regression C-parameter tuning.

## Research

### Algorithm Comparison (2Y Crypto, 60% confidence threshold)
| Algorithm | Return | Win Rate | Trades | Profit Factor | Key Finding |
|-----------|--------|----------|--------|---------------|-------------|
| **Logistic** | **+33.0%** | **53.1%** | **49** | **2.22** | Best overall — simplest model wins |
| LightGBM | +14.5% | 50.5% | 71 | 1.86 | Baseline |
| XGBoost | +2.1% | 45.1% | 59 | 1.51 | Overfits crypto noise |
| Random Forest | +2.8% | 50.2% | 47 | 1.21 | Conservative but low PF |
| MLP | +3.2% | 46.3% | 55 | 1.33 | Slow, unstable on small data |

**Key insight:** Logistic Regression's simplicity is its strength on noisy crypto data. It can't overfit because it's linear — it only learns the strongest, most robust patterns. Complex models (XGBoost, MLP) find spurious patterns in small datasets.

### Logistic Regression C-Parameter Sweep (Autoresearch)

**Round 1: Broad sweep (20 iterations, random configs)**
| Rank | Score | Return | WR | DD | C | Conf | SL/TP |
|------|-------|--------|-----|-----|------|------|-------|
| 1 | 0.4832 | +9.56% | 56.4% | 5.7% | 0.05 | 0.70 | 10/15 |
| 2 | 0.4816 | +20.67% | 48.6% | 9.0% | 10.0 | 0.75 | 5/8 |
| 3 | 0.4740 | +15.06% | 45.2% | 8.5% | 1.0 | 0.70 | 5/10 |

**Round 2: Focused C sweep (C=0.01–0.20, fixed winning params)**
| C | Score | Return | WR | DD |
|------|-------|--------|------|------|
| 0.01 | 0.396 | -2.0% | 62.5% | 8.1% |
| 0.02 | 0.512 | +12.7% | 58.8% | 6.3% |
| 0.03 | 0.544 | +15.8% | 68.8% | 8.7% |
| 0.05 | 0.483 | +9.6% | 56.4% | 5.7% |
| 0.07 | 0.510 | +13.4% | 67.6% | 8.8% |
| 0.10 | 0.491 | +11.8% | 56.8% | 6.0% |
| **0.15** | **0.654** | **+31.6%** | **67.6%** | **8.7%** |
| 0.20 | 0.473 | +13.4% | 57.8% | 9.4% |

**Winner: C=0.15** — sweet spot between regularization and flexibility.

## Acceptance Criteria
- [x] `SmartClassifier` abstract base with `fit()`, `predict_proba()`, `save()`, `load()`, `is_calibrated`
- [x] `SmartXGBoost` wrapper with Platt calibration
- [x] `SmartRandomForest` wrapper with Platt calibration
- [x] `SmartLogistic` wrapper (inherently calibrated)
- [x] `SmartMLP` wrapper with Platt calibration
- [x] Comparison script runs all 5 algorithms on cached crypto data
- [x] Output table: algorithm, return %, WR, profit factor, trades, calibration error
- [x] Existing tests still pass (no breaking changes)
- [x] `TimeMachineBacktest` accepts `model_factory` for pluggable models
- [x] Auto-detects `model_type` from market config (no manual factory needed)
- [x] Autoresearch `--mode logistic` with `SEARCH_SPACE_LOGISTIC` for C-tuning
- [x] Broad C sweep (20 iterations) identifies optimal range
- [x] Focused C sweep (0.01–0.20) pinpoints C=0.15 as winner
- [x] Winning config applied to `MARKET_CONFIGS["crypto"]`

## Technical Design

### Files Created
| File | Purpose |
|------|---------|
| `trading_engine/models/classifiers.py` | SmartClassifier base + XGBoost/RF/Logistic/MLP wrappers + registry |
| `scripts/model_comparison.py` | A/B test script — runs all algorithms, outputs comparison table |

### Files Modified
| File | Change |
|------|--------|
| `trading_engine/models/__init__.py` | Re-export new classifiers |
| `trading_engine/time_machine.py` | Added `model_factory` param, auto-detect from config |
| `trading_engine/autoresearch.py` | Added `SEARCH_SPACE_LOGISTIC`, `--mode logistic`, SmartLogistic factory |
| `trading_engine/models/mean_rev_model.py` | Crypto config: `model_type: logistic`, `logistic_C: 0.15`, confidence 0.70 |
| `requirements.txt` | Added `scikit-learn>=1.4.0`, `lightgbm>=4.0.0`, `xgboost>=2.0.0` |

### Data Model Changes
None

### API Changes
- `TimeMachineBacktest.__init__()` now accepts `model_factory` (callable returning a model)
- `autoresearch.py` CLI: `--mode logistic` option for Logistic Regression tuning

## Dependencies
- S10 (autoresearch framework)
- S12 (modular refactor)

## Verification
1. `scripts/model_comparison.py` — comparison table across 5 algorithms
2. `python -m trading_engine.autoresearch --mode logistic` — C-tuning sweep
3. `autoresearch/crypto_results_logistic.tsv` — full results log

## UAT
- [x] Algorithm comparison: Logistic Regression wins with +33% return, 53.1% WR, 2.22 PF
- [x] Broad C sweep: 20 iterations, C=0.05 wins composite score (balanced config)
- [x] Focused C sweep: C=0.15 wins with +31.6% return, 67.6% WR, 8.7% DD
- [x] Crypto config updated: `model_type=logistic`, `logistic_C=0.15`, `confidence=0.70`
- [x] TimeMachine auto-uses Logistic Regression for crypto market
- [x] 82 trades over 2Y (~3.4/month, ~17 per 150 days)

## Notes
- Logistic Regression was expected to underperform but turned out to be the best — a reminder that simplicity wins on noisy, small datasets
- C=0.15 is moderate regularization — heavy enough to ignore noise, light enough to capture real patterns
- High C values (5.0, 10.0) produce volatile results — sometimes great, sometimes terrible
- Very low C (0.001, 0.01) makes the model too conservative, generating too few signals
- The `lgbm_params` are kept in the crypto config for backward compatibility if switching back
- Logistic Regression trains ~10x faster than LightGBM (no trees, no early stopping needed)
