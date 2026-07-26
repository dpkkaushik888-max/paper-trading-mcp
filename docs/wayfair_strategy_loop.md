# Wayfair (W) Strategy Search — Loop Journal

**Goal:** find a strategy that makes W *robustly* profitable in `scripts/sim_stocks.py` —
beating the baseline REGIME config across a **majority of walk-forward folds AND out-of-sample**.

**Discipline (hard-won):** single-window gains are noise. ATR-exits + conf 0.68 looked like
+2.22% but was an overfit fold-2 fluke; it was a 2-2 wash vs baseline across folds and lost OOS.
A win must clear: ≥3/4 folds positive AND OOS (`--all-days`) ≥ baseline.

## Baseline to beat (REGIME mode, conf 0.65, fixed exits)
| Window | Result |
|--------|--------|
| 4-fold walk-forward | +0.17 / +2.19 / +0.15 / −0.21%  → **3/4 folds positive** |
| OOS (last 7d, 1-min) | **+1.00%** (4W/0L) |

## Idea backlog (try one per iteration)
1. Long-only (drop shorts) — W may lose on the short side
2. Time-of-day filter (trade only high-edge hours)
3. Higher-timeframe trend filter (project's proven edge: trade only with trend) [[trend-timed-btc-beats-hodl]]
4. TP:SL ratio / MIN_HOLD tuning (let winners run)
5. LOGISTIC_C model regularization sweep
6. bull-long mode evaluation

## Iterations
<!-- each iteration appended below: idea, change, fold table, OOS, verdict -->

### Iteration 1 — Long-only? (diagnostic, no code) — ❌ REJECTED
Split W trades by side (NO-FILTER, 5m window):
| Side | n | Win% | Gross sum | Avg/trade |
|------|---|------|-----------|-----------|
| long | 31 | 55% | +6.68% | +0.215% |
| short | 37 | 59% | +7.94% | +0.215% |

No asymmetry — shorts are slightly *better*. Dropping a side won't help.
**Key learning:** avg gross move/trade (0.215%) ≈ round-trip cost (0.2%). Both sides
have positive gross edge that costs nearly erase. The fix must make **moves bigger per
trade**, not cut a side. → pursue idea #3 (HTF trend filter) next.
