# Project State

**Last updated:** 2026-04-13
**Current milestone:** M1: MVP Paper Trading Engine
**Active spec:** S01 — MVP Paper Trading Engine (IMPLEMENTED — UAT PASSED)

## Completed Specs
| Spec | Title | Date Completed |
|------|-------|----------------|
| S01 | MVP Paper Trading Engine | 2026-04-13 |

## In Progress
| Spec | Title | Status | Notes |
|------|-------|--------|-------|

## Next Actions
- Commit S01
- Consider increasing capital or lowering position limit — backtest shows $1K + 10% limit + FX fees = unprofitable

## Blockers
- None

## Key Finding from UAT
**FX conversion fee (1.5%) dominates costs on small accounts.** Round-trip on $50 position costs $1.62 (3.2%). Backtest: 702% cost drag. Options to address in future specs:
1. Open USD account on eToro (eliminates FX fee)
2. Use broker with lower FX fees (Trading 212, Degiro)
3. Increase capital to make costs proportionally smaller
