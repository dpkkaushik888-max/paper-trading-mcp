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
Gate (design §7): active iff beats the S18 reference in >= 4/7 calendar years or the evidence supports the
primary hypothesis; otherwise `candidate` (not stepped). Per-year figures are **chained** (`years_method:
"chained"`): one continuous 2020-01-01.. run sliced by calendar year, each year starting from the previous
year's end value and open positions, so e.g. 2022 differs from a fresh-start 2022 run.
Hypothesis evidence: `reduces_dd_in(below200)` (BTC close < SMA200, i.e. `btc_trend` in {below200_above50,
below_both}, 1026 bar days on .com) compares the line's conditional MaxDD (path compounded from the masked
days' daily returns) with the same statistic of the S18 reference run and, for the basket family, of the live
`bh_basket` buy-and-hold path (`benchmarks.init_bh_basket`, not rebalanced); the line must be lower than every
comparator. `earns_when_live_idle`: mean pnl of trades entered when the reference held nothing going into the
bar (its previous-bar `n_open == 0`). `control` (owner ruling S35 §14.1/§14.2): exempt from the years rule,
passes only through its `baseline_waiver`.

| line | family | status | years beating S18 | gate |
|---|---|---|---|---|
| breakout_s21 | breakout | active | 5/7 | pass (years) |
| xsec_mom_top3 | xsec_momentum | active | 5/7 | pass (years; data guard now applied before ranking) |
| basket_sma50_band | basket_trend | active | 5/7 | pass (years); below200 cond. MaxDD 0.323 < S18 0.392 and < bh_basket 0.971 |
| connors_exit_sma10 | connors | active | 5/7 | pass (years) |
| connors_guarded | connors | active (control) | 0/7 | pass via control waiver: identical to S18 on clean .com history by construction; measures the data-guard effect on .us |
| shadow_btc200 | connors | active (S34 waiver) | 3/7 | hypothesis only: below200 cond. MaxDD 0.159 < S18 0.392 |
| shadow_atr_stop / rsi15 / combo | connors | active (S34 waiver) | 6/7, 4/7, 5/7 | pass (years); waiver kept |

Active lines: 9 (MAX_ACTIVE 10). Each baseline also carries `live_window_us`: the same line on `.us` bars
2026-04-20..last cached bar (return, MaxDD, trades, win rate, mean pnl, per-bucket trade stats); symbols with no
`.us` bars or flagged stale/gap200 are listed (`flagged_symbols`; `skipped_symbols` when the line's data guard
is on; MATIC is frozen on .us from 2025-01-10).
Provenance: `generated_from = {sha: git rev-parse HEAD, dirty}`; baselines are regenerated as the last step
before the commit that adds them, so `generated_from.sha` is the **parent** of the committing commit and
`dirty` is true. `git_sha` (short) is kept for compatibility.

Day context buckets frozen: `BTC_VOL20_TERCILES = (0.4330, 0.6134)`; basket_ret20 {<-10%, -10..0, 0..+10, >+10%}.
`.us` parity of the reference vs the live journal (2026-04-20..2026-10-15): 30/31 closed trades.
`state/pool_context_history.json`: context for every journal day from `.us` bars (`scripts/pool_context_backfill.py`).

### C3 scoring (S35 §4/§5; `paper/pool/scoring.py`, `paper/pool/verdicts.py`)
Read-only over `state/journal.json`, `state/pool_baselines/*.json` and `state/pool_context_history.json`;
nothing changes a line's status or a journal row. `run_daily` regenerates `state/pool_report.json` after the
day is appended (non-dry runs only; skipped-bar days keep the previous report); a scoring exception is logged
into that day's `pool_errors["_scoring"]` and never fails the live run. CLI: `scripts/pool_report.py --print`.

Sources. Live: value path = `portfolio_value` of non-skipped days, trades = `journal.closed_trades`. Pool
line: value path = `days[].lines[<id>].value` (S34 shadows also `shadow_*_value`), trades =
`benchmarks[<id>].closed_trades` (each with `entry_ctx`/`exit_ctx`/`mae_pct`/`mfe_pct`). A pool line counts
from `registered_on`; S34 shadows from their first journal day with a value > 0.

Methods (pure numpy, seed = sha256(line id | last journal date | purpose), so a day's report is reproducible):
- Trade stats: n, distinct entry days, win rate, mean/median `pnl_pct`, profit factor (None + `no_losses`),
  expectancy, mean MAE/MFE, days active, current / max drawdown of the value path, mean exposure.
- Trade-mean CI: cluster bootstrap over entry dates (all same-day trades travel together), B = 5000, 90 % percentile.
- Win probability: Beta(a0 + wins, b0 + losses) with a0 = k p, b0 = k (1 - p), k = 10, p = baseline
  `full.win_rate` (Beta(1,1) without a baseline). Mean analytic; 90 % credible interval from 20 000 seeded
  `rng.beta` draws (no scipy). `disagrees` when the interval excludes p.
- Evidence tier: < 8 trades or < 5 entry days = insufficient (no CI); 8-14 indicative; >= 15 usable.
- Paired vs live: common window = days both have a value > 0, neither skipped; d_t = r_line - r_live on that
  window. Stationary block bootstrap (mean block 5 days, B = 5000, vectorised (B, n) index array) of mean d:
  90 % CI and p = 2 min(P*(mean <= 0), P*(mean >= 0)); sign test on non-zero days; half-window means;
  cumulative return and MaxDD difference. Holm across ACTIVE lines with a paired p at alpha = 0.10.
  (Hypothesis-specific restrictions of §4 — cond days, live-idle days — are not in C3.)
- Condition table (exploratory, uncorrected, never drives a verdict): per dimension btc_trend /
  btc_vol20_bucket / basket_ret20_bucket / dow and bucket: n, entry days, win rate, mean, tier, cluster CI
  (B = 2000, non-insufficient cells), posterior mean with the baseline bucket win rate as prior when the
  bucket has backtest n >= 30, else the line's overall rate. Bucket = trade `entry_ctx`, else that day's
  journal `context`, else context history, else "unknown".

Verdicts (recommendations only; the owner rules monthly; text says "cand" / "recommend"), in order:
LIVE (live_s18); NOT-STEPPED (status candidate); CONTROL (hypothesis control, with `guard_effect` = control
minus live return over the common window); HARD-KILL (max DD > 25 % or >= 10 consecutive losing trades);
RETIRE-cand (>= 60 days, >= 15 trades, Holm p_adj < 0.10 with mean d < 0); DORMANT (< 5 signals in the
trailing 90 journal days, evaluated only once the line has 90 LineDay rows there); PROMOTE-cand (>= 90 days,
>= 25 trades, CI lower bound > 0 or Holm-significant positive, mean d > 0 in both halves, baseline
`years_beating_s18` >= 4; baseline-waived and control lines never promote); KEEP with rule INCONCLUSIVE-180
and flag "inconclusive 180d; owner rules monthly" at >= 180 days (§14.3); else KEEP. Report:
`schema_version` 1, sorted keys, floats rounded to 6 digits so the daily diff is readable.
