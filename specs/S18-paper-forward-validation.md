# S18: Paper-Forward Validation of S20 Config

**Status:** FAIL — terminated day 29/90 (2026-05-27)
**Branch:** `feature/s18-paper-forward`
**Priority:** P1 (gate before real capital)
**Depends on:** S20 (VERIFIED)
**Verdict:** Active strategy produced 0 trades vs 2 passive benchmarks both
positive. Regime mismatch (broad crypto downtrend) — not a bug. See UAT below.

## Overview

Run the frozen S20 rule set live (Connors rules, 20 crypto, cap=6) against
real-time Binance daily bars for 90 days. No real orders — paper only. Measure
live performance vs S20 backtest (+13.39% CAGR, Sharpe 1.06, 70.5% WR) to
confirm the edge survives out-of-sample in real regimes.

## User Story

As the trader, I want a 90-day paper-forward test of the S20 strategy on live
market data so I can measure real-world slippage, signal frequency, and edge
persistence **before committing any real capital**.

## Design Decisions

All decisions below are locked before day 1 to prevent mid-test tampering.

### D1 — State persistence: journal committed to repo
Each cron run reads `state/journal.json`, updates it, and commits it back.
Pros: full audit trail in git history, zero external dependencies, trivial
debugging. Cons: 1 commit/day noise. **Accepted — the audit trail outweighs
the noise for a 90-day trial.**

### D2 — Reporting: GitHub Pages static dashboard
Auto-deployed on each journal commit. Public URL, mobile-accessible, no auth.
Built from `state/journal.json` into `docs/index.html` with Chart.js (CDN).
**Accepted because repo is public; no sensitive data in journal.**

### D3 — Alerting: pinned GitHub Issue, edited daily
Single issue titled "S18 Paper-Forward — Progress", body rewritten each run.
User subscribes once → gets GitHub mobile push notifications on each edit.
No SendGrid/email setup. **Accepted.**

### D4 — Fill convention: same-bar close (matches backtest)
Entries/exits fill at `today_close * (1 ± slip_bps)`. This is optimistic by
~5-15 bps vs reality (cron runs 00:30 UTC, ~30 min after bar close). The
gap is exactly what S18 measures — log the modeled fill vs the actual
next-bar-open price and compute realized slippage per symbol.

### D5 — No real orders ever
Paper trader has **no Binance API credentials** — cannot physically place
orders. Fetches public kline data via unauthenticated REST endpoint. This is
an architectural guarantee, not a config flag.

### D6 — Market data source: Binance public klines
`https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=300`
No authentication. Rate limit is 1200 req/min — we do 20 req/day, negligible.
Fallback: `https://api.binance.us/...` if geoblocked.

### D7 — Symbol universe frozen at S20 (20 coins)
No additions, no removals, no ranking changes for the entire 90 days.
Drift from this = parameter tuning on the forward test = invalidates results.

### D8 — Configuration frozen from S20
- 20 symbols (BTC, ETH, SOL, AVAX, LINK, MATIC, DOGE, XRP, ADA, DOT, ATOM,
  NEAR, LTC, TRX, BCH, APT, UNI, ARB, OP, SUI)
- MAX_CONCURRENT = 6
- POS_SIZE_PCT = 0.15
- COST_PCT = 0.0020 (20 bps/side)
- SLIPPAGE_BPS = 0.0005 (5 bps)
- SL_SLIPPAGE_BPS = 0.0010 (10 bps)
- Connors rules unchanged (RSI<10, Close>SMA200, Close<SMA5, ADX>=20)
- Exit: Close>SMA5 | 10d max | -7% SL
- Starting capital: $10,000

### D9 — Failure handling: skip day, do not retry
If the cron fails (API down, Actions outage), the journal skips that day.
No retroactive catch-up runs — we'd be trading on stale prices. Log gap
in journal and continue next day. If >5 days are skipped, halt and
investigate.

### D10 — Early-termination triggers
Stop the test early and diagnose if ANY of:
- Drawdown > 15% from starting capital ($8,500 floor)
- 10 consecutive losing trades
- Zero trades for 30 consecutive days (rules aren't firing)
- Any exception/crash that can't be fixed same-day

### D11 — 90-day duration, evaluated weekly
Don't check daily. Once/week review, full verdict at day 90.
Daily peeking = emotional tampering.

## Research

- **Prior cron infra (S14):** `.github/workflows/daily-trade.yml` existed for
  the retired ML strategy. File renamed to `.disabled`. Provides template for
  schedule syntax, commit-back pattern, and timeout-minutes config.
- **Binance kline API:** free, unauthenticated, reliable. Returns
  `[open_time, open, high, low, close, volume, close_time, ...]` per bar.
- **GitHub Pages deploy:** `actions/deploy-pages@v4` with artifact from
  `docs/` dir. Requires repo Settings → Pages → Source: "GitHub Actions".
- **Connors rule function (S17/S20):**
  `trading_engine/strategies/connors_swing.py` — pure stateless functions
  `long_entry()` and `long_exit()` reusable as-is, zero refactor needed.
- **Backtest baseline (S20):** 44 trades / 326 days = ~1 trade / 7.4 days
  expected live. 90-day forward should produce ~12 trades. If live produces
  < 4 or > 25, something is wrong.

## Acceptance Criteria

### Infrastructure (days 0-1)
- [ ] `trading_engine/paper/` package created with journal.py, signals.py, run_daily.py
- [ ] `scripts/build_dashboard.py` generates `docs/index.html` from journal
- [ ] `.github/workflows/paper-forward.yml` runs on cron `30 0 * * *`
- [ ] `.github/workflows/deploy-pages.yml` rebuilds dashboard on journal commits
- [ ] Pinned issue #1 created titled "S18 Paper-Forward — Progress"
- [ ] Day-0 dry run produces valid journal.json (zero trades expected)
- [ ] GitHub Pages deployed and accessible at public URL

### Daily behavior (days 1-90)
- [ ] Cron runs at 00:30 UTC daily; success rate ≥ 95% over 90 days
- [ ] Each run: fetches 20 symbols, evaluates exits, evaluates entries, marks
      portfolio, commits journal, updates dashboard, edits pinned issue
- [ ] Journal contains: date, per-symbol close prices, indicator snapshot,
      decisions made (enter/exit/hold/block), realized fills, portfolio value
- [ ] Dashboard shows: KPIs, equity curve, open positions, closed trades,
      backtest-vs-live comparison
- [ ] No real orders placed (verified by absence of Binance API credentials)

### Day-90 verdict
- [ ] Compute live CAGR, Sharpe, MaxDD, WR, profit factor, trade count
- [ ] Compute per-symbol realized slippage (modeled vs actual fill)
- [ ] Compare live vs S20 backtest in side-by-side table
- [ ] **Decision gate:**
  - **PASS (deploy real capital €500):** live CAGR ≥ +6%, WR ≥ 55%,
    no early-termination trigger hit, slippage within 2× model
  - **MIXED (extend 30 days):** live CAGR 0-6%, WR 50-55%, edge unclear
  - **FAIL (abandon live, iterate):** live CAGR < 0%, WR < 50%, or any
    early-termination trigger hit

## Technical Design

### Files to create
| File | Role |
|------|------|
| `trading_engine/paper/__init__.py` | Package marker |
| `trading_engine/paper/journal.py` | JSON state: load, save, add_entry, add_exit, mark_portfolio |
| `trading_engine/paper/signals.py` | Thin wrapper: fetch_bars() + reuse connors_swing.long_entry/long_exit |
| `trading_engine/paper/run_daily.py` | Main cron entry: orchestrates fetch→eval→journal→report |
| `scripts/build_dashboard.py` | Read journal.json → write docs/index.html + docs/data.json |
| `scripts/update_issue.py` | gh CLI wrapper: edit pinned issue body with daily summary |
| `.github/workflows/paper-forward.yml` | Daily cron @ 00:30 UTC |
| `.github/workflows/deploy-pages.yml` | Deploy docs/ on journal commit |
| `docs/index.html` | Static dashboard template (Chart.js from CDN) |
| `state/journal.json` | Initialized empty on day 0 |

### Journal schema (`state/journal.json`)
```json
{
  "version": 1,
  "start_date": "2026-04-22",
  "starting_capital": 10000.0,
  "config": {
    "max_concurrent": 6,
    "pos_size_pct": 0.15,
    "universe": ["BTCUSDT", "ETHUSDT", ...],
    "rules": "connors_swing_v1"
  },
  "days": [
    {
      "date": "2026-04-22",
      "portfolio_value": 10000.0,
      "cash": 10000.0,
      "closes": {"BTCUSDT": 67234.5, ...},
      "indicators_snapshot": {"BTCUSDT": {"rsi2": 45.1, "sma5": 67100, "sma200": 62000, "adx": 18.3}, ...},
      "decisions": [
        {"symbol": "ETHUSDT", "action": "hold", "reason": "adx<20"}
      ],
      "fills": []
    }
  ],
  "open_positions": [
    {"symbol": "BCHUSDT", "entry_date": "2026-04-25", "entry_price": 380.20, "shares": 3.95, "entry_cost": 3.00}
  ],
  "closed_trades": [
    {"symbol": "ETHUSDT", "entry_date": "2026-04-23", "exit_date": "2026-04-27", "entry_price": 3200.50, "exit_price": 3340.00, "shares": 0.469, "pnl": 64.21, "pnl_pct": 0.0436, "hold_days": 4, "reason": "signal"}
  ]
}
```

### Workflow: `paper-forward.yml` (sketch)
```yaml
name: S18 Paper Forward
on:
  schedule: [{cron: '30 0 * * *'}]
  workflow_dispatch:
permissions:
  contents: write
  issues: write
  pages: write
  id-token: write
jobs:
  paper:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.13', cache: pip}
      - run: pip install -r requirements.txt
      - run: python -m trading_engine.paper.run_daily
      - run: python scripts/build_dashboard.py
      - run: python scripts/update_issue.py
        env: {GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}}
      - run: |
          git config user.name "S18 Paper Bot"
          git config user.email "bot@paper-forward.local"
          git add state/journal.json docs/
          git diff --cached --quiet || git commit -m "paper-fwd $(date -u +%Y-%m-%d): $(jq -r '.days[-1].portfolio_value' state/journal.json)"
          git push
      - uses: actions/upload-pages-artifact@v3
        with: {path: docs/}
      - uses: actions/deploy-pages@v4
```

### Dashboard content (`docs/index.html`)
- Header: "S18 Paper-Forward — Day N of 90"
- KPI strip: Portfolio, CAGR (annualized), MaxDD, WR, PF, Trades
- Equity curve (Chart.js line, daily portfolio values)
- Open positions table (symbol, entry date, entry price, current price, unrealized P&L)
- Closed trades table (with per-symbol color coding)
- Backtest-vs-live comparison panel

### Dependencies
- `requests` (Binance API) — already in requirements.txt
- `pandas` — already present
- Existing `trading_engine.strategies.connors_swing` — reused unchanged
- Existing cost/sizing constants from `scripts.sim_swing_backtest` — extracted to a shared constants module

## Verification

### Infrastructure verification (day 0)
- `python -m trading_engine.paper.run_daily --dry-run` runs locally against
  Binance, writes a test journal, produces a valid dashboard HTML
- Manual workflow_dispatch trigger on GitHub succeeds
- Pinned issue created and editable via `gh issue edit`
- Dashboard accessible at `https://dpkkaushik888-max.github.io/paper-trading-mcp/`

### Weekly check-ins (days 7, 14, 30, 60)
- Cron success rate > 95% (check Actions tab)
- Journal schema consistent (no missing fields)
- Dashboard renders cleanly on mobile
- No early-termination trigger breached

### Day 90 final verification
- Run `scripts/s18_final_report.py` → produces markdown + CSV summary
- Walk through acceptance criteria with user (UAT phase)
- Decide: PASS / MIXED / FAIL per gate above

## Dependencies

- **GitHub Pages enabled:** Settings → Pages → Source = "GitHub Actions"
  (manual step, one-time)
- **Repo must stay public:** GitHub Pages on free plan requires public
- **Binance public API reachable from GitHub Actions runners** (verified
  in infra setup)

## UAT — closed FAIL on day 29/90 (2026-05-27)

Test terminated early. D10 early-termination trigger
(`MAX_NO_SIGNAL_DAYS = 30`) fired automatically starting day 30 (2026-05-20)
and every cron run since exited with code 2. Disabled the workflow on
2026-05-27 to stop polluting the Actions tab.

### Live results (days 1-29: 2026-04-21 → 2026-05-19)
| Portfolio | Final value | Return | Trades |
|-----------|------------:|------:|------:|
| **Connors (S20 active)** | **$10,000.00** | **+0.00%** | **0** |
| BH_BTC (passive) | $10,133.36 | +1.33% | 1 entry |
| BH_BASKET (passive) | $10,264.93 | +2.65% | 20 entries |

### Acceptance criteria walkthrough
- [x] Infrastructure (days 0-1) — all built and ran for 29 days, success rate 100% before halt
- [~] Daily behavior (days 1-90) — ran 29/90 then auto-halted; 580 decisions logged, all `no_entry_signal`
- [ ] **Day-90 verdict (FAIL)** — early-termination D10 triggered; addendum gate
      (Connors > BH_BTC on both CAGR and Sharpe) also fails since Connors == 0%

### Root-cause analysis (2026-05-27 backtest replay)

Replayed Connors `long_entry` against the same 20-symbol universe over the
identical 36-day window using fresh Binance data. Result: **1 backtest
signal** (TRX 2026-04-24) vs **0 live** — functionally identical given
RSI(2) tick-level sensitivity across binance.com vs binance.us closes.

Filter pass-rate audit over 684 symbol-days:

| Filter | Pass rate | Verdict |
|---|---:|---|
| F1: Close > SMA(200) | **8.0%** | 🔴 Binding constraint — 17/20 symbols never closed above their 200-day SMA |
| F2: RSI(2) < 10 | 13.3% | Normal sparsity |
| F3: Close < SMA(5) | 51.2% | Healthy |
| F4: ADX(14) ≥ 20 | 55.3% | Healthy |

**Diagnosis:** The 36-day window was a broad crypto downtrend. S20 is by
design a "buy oversold pullbacks **in uptrends**" strategy — it correctly
refused to trade because there were no uptrends to trade. Dropping F1
would have produced 56 signals over the same window, but those signals
are buying dips in a downtrend — exactly what blows up retail mean-revert
traders.

### Lessons preserved for S21
1. **Single-regime strategies have multi-month idle stretches.** S20 alone
   is unsuitable as a personal-compounder primary engine.
2. **The honest-cost + pre-committed methodology held up.** The "+13.39%
   backtest CAGR" was an upper bound (data-snooped across 3 holdout passes);
   the actual live regime delivered 0 trades. Backtest replay confirmed
   live code was not buggy — the regime simply did not match.
3. **Passive BH_BTC beat the active strategy** on the same window. Active
   management must demonstrably beat doing nothing or it's pure overhead.
4. **Next strategy must cover non-uptrend regimes** (range, downtrend) —
   see S21.

### Bugs fixed alongside closure
- `binance.com` swapped from primary to fallback in `paper/config.py`
  (geoblocks GH Actions runners with 451; .us was the de-facto source).
- Cron workflow renamed to `.disabled` (was failing daily on the D10 halt).
- Initial "benchmarks broken" alarm in audit was a diagnostic-script error
  — `bh_btc_value` / `bh_basket_value` were always present on per-day
  records. No fix required.

## Addendum (2026-04-21): Passive Benchmarks

Added same-day as test launch (before any trade fires — no data snooping).

Two phantom portfolios tracked alongside Connors:
- **BH_BTC:** $10,000 → 100% BTC at the first-day snapshot close, held to day 90.
  One-time 20 bps entry cost. No rebalance, no exit.
- **BH_BASKET:** $10,000 → equal-weight ($500) across all 20 universe coins at
  the first-day snapshot closes. One-time 20 bps entry cost per symbol. No
  rebalance, no exit.

Both benchmarks pay the same 20 bps/side cost as Connors for entry realism.
They exit at simulated day-90 close (one-time exit cost at verdict) or if a
halt triggers.

### Why
Answer the "did I beat doing nothing" question at day 90. Without this,
"+8% CAGR" on Connors means nothing — it could be the whole market.

### Journal schema bump (v1 → v2)
`benchmarks` key added; v1 journals auto-migrate on load.

### Day-90 success gate tightening
- **PASS** now also requires: **Connors CAGR > BH_BTC CAGR** AND
  **Connors Sharpe > BH_BTC Sharpe**. Beating buy-and-hold BTC is the minimum
  bar for active management to be worth the effort.
- **FAIL** if Connors underperforms BOTH benchmarks — the active work adds no value.

## Notes & Risks

- **Data snooping:** S17→S19→S20 used the same holdout 3 times. S18 is the
  FIRST truly out-of-sample test. The reported backtest +13.4% is an upper
  bound. Realistic live expectation: +6-9%.
- **Regime risk:** 90 days may not cover a full market cycle. If the whole
  90 days is a strong uptrend, we learn only that the strategy works in
  uptrends — repeat S17.1 rolling-window test afterwards for regime coverage.
- **Binance geoblocking:** if api.binance.com blocks the Actions runner IP
  range, fall back to api.binance.us (same schema). Unlikely but possible.
- **GitHub Actions reliability:** ~99% uptime typical. Expect 1-3 skipped
  days over 90. D9 handles this.
- **Real capital gate:** Even a PASS at day 90 does NOT mean go live with
  all capital. Start with €500 for 30 more days of real-money validation
  before scaling to €1k-10k.
