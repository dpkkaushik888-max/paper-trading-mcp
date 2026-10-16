# S34: Shadow Variants and Data Quality

**Status:** IN REVIEW
**Branch:** feat/shadow-variants
**Priority:** P2 (observation only; the frozen S18 forward test is untouched)
**Depends on:** S18 (paper-forward), S32 (selector series), S33 (backfill)

## Overview
The S18 90-day forward test is frozen (config.py:1-5), so candidate improvements cannot be
tried on the live book. S34 runs four **shadow** books alongside it on the same daily bars.
Each shadow is a full paper book with its own cash, positions and closed trades, stored as a
journal benchmark. Shadows never feed back into the live book. Separately, S34 adds a
stale-bar warning and corrects an off-by-one in the S33 backfill.

## The four variants (exact rules)
All variants use the live Connors swing rules (UNIVERSE, POS_SIZE 15 %, MAX_CONCURRENT 6,
RSI(2) exit, SMA trend filter) except for the override listed.

| Series | Override |
|---|---|
| `shadow_atr_stop` | Exit "SL" when close <= entry_price − 3.0 × ATR(14, Wilder) measured at entry; the % stop widens to 25 % and is an emergency backstop only |
| `shadow_btc200` | New entries only when BTC close > BTC SMA200. If the SMA is unknown, the gate is closed. Exits are unchanged |
| `shadow_rsi15` | Entry RSI threshold 15 instead of 10 |
| `shadow_combo` | All three together: RSI 15, 3 × ATR stop with a 25 % backstop, BTC SMA200 gate |

## Start semantics
- Each shadow starts at **10,000** on its first run day. It has no backfill and no history before that day.
- `shadow_*_value == 0.0` means **not started**. The dashboard ignores those days.
- If a shadow's step raises, the run logs the exception, carries the last positive value forward, and records `DaySnapshot.shadow_errors[name]` (short exception text). The persisted book is deep-copied on load, so a failed step leaves the stored book unchanged. The dashboard flags the row "error on last run".

## Explicitly NOT changed
- UNIVERSE, RSI 10, the 7 % stop, MAX_CONCURRENT 6, POS_SIZE 15 %, sizing, and the live signal/exit path.
- No MATIC drop. No TRX guard.
- The live journal fields stay byte-identical, enforced by `tests/fixtures/live_path_golden.json`, which was captured on master. The new keys (`shadow_*_value`, `stale_symbols`, `shadow_errors`) are outside the golden signature.

## Data quality: stale-bar warning
`STALE_BAR_DAYS = 2`. A symbol whose latest usable bar is older than that relative to the run
day is listed in `DaySnapshot.stale_symbols` and logged as a warning. This is a warning only:
trading behaviour does not change.

## Backfill fix
`run_daily(as_of=D)` now truncates bars to `index < D`, which is what the live 00:30 UTC run
sees. The same fix applies to `scripts/backfill_trend_timed.py` and `scripts/backfill_selector.py`.
Journal rows 2026-06-23..06-29 were written under the old `<=` rule and are deliberately not
rewritten. See the S33 note dated 2026-10-16.

## Acceptance Criteria
- [x] Four shadow series are journalled daily, start at 10,000, and use 0.0 for not started.
- [x] A shadow failure never affects the live book. Errors are recorded and shown.
- [x] The live-path golden test passes.
- [x] The dashboard shows a shadow table vs Connors, utilisation KPIs, and an empty state.
- [x] Stale-bar warning; `as_of` uses strict `<`.

## Graduation
A shadow never promotes itself. Graduation needs three steps:
1. An explicit owner decision.
2. A new spec that changes the live rules.
3. A **restart of the 90-day clock** per config.py. The frozen test's results are not carried over to a changed strategy.

## Pool registry (C1 of S35)
The four shadows now live in the strategy-pool registry (`trading_engine/paper/pool/registry.py`) as
`connors`-family lines with unchanged ids and journal keys (`shadow_*_value`, `benchmarks[shadow_*]`,
`shadow_errors`). They keep live data handling (`data_guard=False`) and carry a baseline waiver note
("full-history numbers in this spec"). `paper/shadow.py` is a compatibility shim over `pool/book.step`;
a parity test pins the new book to the pre-C1 step. Daily rows gain optional `context`, `lines`
(incl. the read-only `live_s18`) and `pool_errors`. Full design: `specs/S35-strategy-pool.md` (C4).

### C2 — families, full-history baselines, registration gate (2026-10-16)
New pure-pandas families (`pool/families/{breakout,xsec_momentum,basket_trend}.py`) plus two connors lines.
Baselines: `state/pool_baselines/<id>.json` from `scripts/pool_backtest.py --line all --venue com`
(binance.com daily bars 2020-01-01..2026-10-15, MATIC excluded, real `book.step`); S18 reference
`state/pool_baselines/_s18_reference_com.json` (CAGR 8.6 %, MaxDD 24.8 %, Sharpe 0.415, 587 trades).
Gate (design §7): active iff beats the S18 reference in >= 4/7 calendar years (2020–2026, chained from
one full run) or the per-bucket evidence supports the primary hypothesis; otherwise `candidate` (not stepped).

| line | family | status | years beating S18 | gate |
|---|---|---|---|---|
| breakout_s21 | breakout | active | 5/7 | pass (years) |
| xsec_mom_top3 | xsec_momentum | active | 4/7 | pass (years) |
| basket_sma50_band | basket_trend | active | 5/7 | pass (years; below_both MaxDD also < bh_basket-eq) |
| connors_exit_sma10 | connors | active | 5/7 | pass (years) |
| connors_guarded | connors | candidate | 0/7 | fail: identical to S18 on .com (no MATIC/TRX artefact in .com data) |
| shadow_* (4) | connors | active (S34 waiver) | 6/7, 3/7, 4/7, 5/7 | baselines recorded; waiver kept |

Day context buckets frozen: `BTC_VOL20_TERCILES = (0.4330, 0.6134)`; basket_ret20 {<-10%, -10..0, 0..+10, >+10%}.
`.us` parity of the reference vs the live journal (2026-04-20..2026-10-15): 30/31 closed trades.
`state/pool_context_history.json`: context for every journal day from `.us` bars (`scripts/pool_context_backfill.py`).
