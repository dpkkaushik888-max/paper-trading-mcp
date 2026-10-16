"""S18 daily paper-trade cron entry point.

Flow:
  1. Load journal
  2. Fetch daily bars for every symbol in the frozen universe
  3. For each open position: evaluate Connors exit → close if triggered
  4. For each non-open symbol: evaluate Connors entry → enter if triggered
     (subject to max_concurrent cap and available cash)
  5. Mark portfolio at today's close
  6. Append day snapshot to journal
  7. Check early-termination triggers
  8. Save journal

No Binance credentials. Cannot place real orders.

Usage:
    python -m trading_engine.paper.run_daily                # live mode
    python -m trading_engine.paper.run_daily --dry-run      # no journal write
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_engine.strategies.connors_swing import (
    precompute_indicators,
    long_entry,
    long_exit,
)

from .config import (
    UNIVERSE,
    STARTING_CAPITAL,
    POS_SIZE_PCT,
    MAX_CONCURRENT,
    COST_PCT,
    SLIPPAGE_BPS,
    SL_SLIPPAGE_BPS,
    TEST_DAYS,
    MAX_DRAWDOWN_PCT,
    MAX_CONSECUTIVE_LOSSES,
    MAX_SKIPPED_DAYS,
)
from .data import fetch_universe
from .journal import (
    Journal, OpenPosition, ClosedTrade, Decision, DaySnapshot, Fill,
)
from .benchmarks import (
    init_bh_btc, init_bh_basket, mark_benchmark, BH_BTC_SYMBOL,
    serialize as bench_serialize, deserialize as bench_deserialize,
)
from .trend_timed import (
    TrendTimer, step as trend_step, value_of as trend_value,
    serialize as trend_serialize, deserialize as trend_deserialize,
)
from .selector_track import (
    step as selector_step, serialize as selector_serialize,
    deserialize as selector_deserialize,
)
from .shadow import SHADOW_VALUE_KEYS
from .pool import book as pool_book
from .pool.book import step as pool_step
from .pool.context import DayContext, build_context
from .pool.families import make_strategy
from .pool.live_adapter import live_line_day
from .pool.registry import LIVE_ID, active_lines

POOL_LINES = active_lines()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s18.run_daily")


def carry_forward_benchmarks(journal) -> dict[str, float]:
    """Previous day's benchmark values, for days with no market data."""
    keys = ("bh_btc_value", "bh_basket_value", "trend_timed_value", "selector_value")
    out = {k: journal.last_positive_value(k) for k in keys}
    # Shadows: carry the last positive value; stay 0.0 (not started) before their first run.
    out.update({k: journal.last_positive_value(k, default=0.0) for k in SHADOW_VALUE_KEYS})
    log.warning("No market data today; carrying benchmarks forward: %s", out)
    return out


def _today_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _latest_bar(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series]:
    """Return (date, row) for the most recent closed bar."""
    return df.index[-1], df.iloc[-1]


# A healthy symbol's latest closed bar at the 00:30 UTC run is yesterday's (age 1).
STALE_BAR_DAYS = 2


def _bar_age_days(bar_date: pd.Timestamp, today: str) -> int:
    """Whole days between a bar's open date and the run date."""
    return (pd.Timestamp(today).date() - pd.Timestamp(bar_date).date()).days


def _stale_symbols(latest_dates: dict[str, pd.Timestamp], today: str) -> list[str]:
    """Symbols whose latest bar is older than STALE_BAR_DAYS (e.g. a delisted / frozen
    pair such as MATICUSDT on binance.us). Logs a WARNING per symbol; no behaviour change."""
    stale = []
    for sym in sorted(latest_dates):
        age = _bar_age_days(latest_dates[sym], today)
        if age > STALE_BAR_DAYS:
            log.warning("STALE DATA: %s latest bar %s is %d days old (as_of %s)",
                        sym, pd.Timestamp(latest_dates[sym]).date(), age, today)
            stale.append(sym)
    return stale


def run(dry_run: bool = False, as_of: str | None = None) -> int:
    """Execute one daily cron cycle. Returns 0 on success, non-zero on halt.

    ``as_of`` (YYYY-MM-DD) replays a historical day: it is used as "today" and
    each symbol's bars are truncated to ``index < as_of`` so the exact same
    logic produces what the live cron would have on that date. Bars are labelled
    by open time and the live 00:30 UTC run drops the still-forming bar, so on day
    D the live run only ever sees bars opened before D. Used for backfill after a
    cron outage (S33); ``None`` = live mode (today's date, full bars).
    """
    log.info("=" * 70)
    log.info("S18 Paper-Forward — daily run (dry_run=%s, as_of=%s)", dry_run, as_of or "live")
    log.info("=" * 70)

    journal = Journal.load()
    today = as_of or _today_utc_date()

    log.info("Day %d of %d | cash=$%.2f | open=%d | closed_trades=%d",
             journal.days_elapsed() + 1, TEST_DAYS, journal.cash,
             len(journal.open_positions), len(journal.closed_trades))

    # ── Fetch market data ────────────────────────────────────────────────
    try:
        bars_by_sym = fetch_universe(UNIVERSE)
    except Exception as e:  # noqa: BLE001
        log.error("Market data fetch failed entirely: %s", e)
        snap = DaySnapshot(
            date=today, portfolio_value=journal.last_portfolio_value(),
            cash=journal.cash, n_open=len(journal.open_positions),
            closes={}, decisions=[], fills=[], skipped=True,
            # Carry benchmark series forward — never record 0 (breaks Sharpe/max DD).
            **carry_forward_benchmarks(journal),
        )
        journal.append_day(snap)
        if not dry_run:
            journal.save()
        skipped_count = sum(1 for d in journal.days if d.get("skipped"))
        if skipped_count > MAX_SKIPPED_DAYS:
            log.error("HALT: %d skipped days exceeds MAX_SKIPPED_DAYS=%d",
                      skipped_count, MAX_SKIPPED_DAYS)
            return 2
        return 0

    if as_of is not None:
        # Backfill replay (S33): keep only bars opened strictly before the as-of
        # date. Bar D is still forming at the live 00:30 UTC run on day D
        # (data.py drops it), so `<=` would leak one day of future data.
        cutoff = pd.Timestamp(as_of, tz="UTC")
        bars_by_sym = {s: df[df.index < cutoff] for s, df in bars_by_sym.items()}
        bars_by_sym = {s: df for s, df in bars_by_sym.items() if not df.empty}

    log.info("Fetched bars for %d/%d symbols", len(bars_by_sym), len(UNIVERSE))

    # ── Pre-compute indicators + today snapshot ─────────────────────────
    snapshot: dict[str, tuple[float, pd.Series]] = {}
    todays_closes: dict[str, float] = {}
    latest_dates: dict[str, pd.Timestamp] = {}
    for sym, df in bars_by_sym.items():
        ind = precompute_indicators(df)
        date, row = _latest_bar(df)
        latest_dates[sym] = date
        close_today = float(row["Close"])
        ind_row = ind.loc[date]
        snapshot[sym] = (close_today, ind_row)
        todays_closes[sym] = close_today
    stale_symbols = _stale_symbols(latest_dates, today)

    # ── 1. Evaluate exits on open positions ─────────────────────────────
    decisions: list[Decision] = []
    fills: list[Fill] = []

    for pos in list(journal.open_positions):
        if pos.symbol not in snapshot:
            log.warning("No bars for open position %s — skipping exit check", pos.symbol)
            decisions.append(Decision(pos.symbol, "hold", "no_data"))
            continue
        close_today, ind_row = snapshot[pos.symbol]
        entry_date = pd.Timestamp(pos.entry_date)
        today_ts = pd.Timestamp(today)
        reason = long_exit(close_today, ind_row, pos.entry_price, entry_date, today_ts)
        if reason is None:
            decisions.append(Decision(pos.symbol, "hold", "no_exit_signal"))
            continue

        exit_slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if reason == "SL" else 0.0)
        fill_price = close_today * (1.0 - exit_slip)
        exit_cost = fill_price * pos.shares * COST_PCT
        proceeds = fill_price * pos.shares - exit_cost
        pnl = proceeds - (pos.entry_price * pos.shares + pos.entry_cost)
        pnl_pct = pnl / (pos.entry_price * pos.shares)

        trade = ClosedTrade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=today,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=(today_ts - entry_date).days,
            reason=reason,
        )
        journal.close_position(pos.symbol, trade, proceeds)
        decisions.append(Decision(pos.symbol, "exit", reason))
        fills.append(Fill(pos.symbol, "SELL", close_today, fill_price, pos.shares, exit_cost))
        log.info("EXIT %s (%s): $%.2f→$%.2f, P&L $%.2f (%.2f%%)",
                 pos.symbol, reason, pos.entry_price, fill_price, pnl, pnl_pct * 100)

    # ── 2. Evaluate entries on non-open symbols ─────────────────────────
    open_syms = journal.open_symbols()
    candidates: list[tuple[str, float, pd.Series]] = []
    for sym, (close_today, ind_row) in snapshot.items():
        if sym in open_syms:
            continue
        if long_entry(close_today, ind_row, use_adx_filter=True):
            candidates.append((sym, close_today, ind_row))
        else:
            decisions.append(Decision(sym, "hold", "no_entry_signal"))

    capacity = MAX_CONCURRENT - len(journal.open_positions)
    if candidates:
        # Deterministic ordering: alphabetical (same as backtest).
        candidates.sort(key=lambda c: c[0])
        log.info("Entry candidates: %s (capacity=%d)",
                 [c[0] for c in candidates], capacity)

    for sym, close_today, _ in candidates[:capacity] if capacity > 0 else []:
        entry_fill = close_today * (1.0 + SLIPPAGE_BPS)
        pos_value = STARTING_CAPITAL * POS_SIZE_PCT
        if pos_value > journal.cash:
            decisions.append(Decision(sym, "blocked", "insufficient_cash"))
            continue
        shares = pos_value / entry_fill
        entry_cost = entry_fill * shares * COST_PCT
        debit = entry_fill * shares + entry_cost
        if debit > journal.cash:
            decisions.append(Decision(sym, "blocked", "debit>cash"))
            continue

        pos = OpenPosition(
            symbol=sym, entry_date=today, entry_price=entry_fill,
            shares=shares, entry_cost=entry_cost,
        )
        journal.add_position(pos, debit)
        decisions.append(Decision(sym, "enter", "connors_signal"))
        fills.append(Fill(sym, "BUY", close_today, entry_fill, shares, entry_cost))
        log.info("ENTER %s: %.6f shares @ $%.2f (debit $%.2f)",
                 sym, shares, entry_fill, debit)

    # Blocked by cap
    blocked = candidates[capacity:] if capacity > 0 else candidates
    for sym, _, _ in blocked:
        decisions.append(Decision(sym, "blocked", "max_concurrent"))

    # ── 3. Mark portfolio at today's close ──────────────────────────────
    portfolio = journal.cash
    for pos in journal.open_positions:
        if pos.symbol in todays_closes:
            portfolio += todays_closes[pos.symbol] * pos.shares
        else:
            portfolio += pos.entry_price * pos.shares  # stale fallback

    # ── 3b. Initialize benchmarks on first day, then mark ───────────────
    bh_btc_data = journal.get_benchmark("bh_btc")
    bh_basket_data = journal.get_benchmark("bh_basket")
    if not bh_btc_data.get("initialized"):
        try:
            bh = init_bh_btc(todays_closes, today)
            journal.set_benchmark("bh_btc", bench_serialize(bh))
            log.info("Initialized BH_BTC at $%.2f (%.6f BTC)",
                     bh.positions[0].entry_price, bh.positions[0].shares)
        except ValueError as e:
            log.warning("BH_BTC init deferred: %s", e)
    if not bh_basket_data.get("initialized"):
        bh = init_bh_basket(todays_closes, today)
        journal.set_benchmark("bh_basket", bench_serialize(bh))
        log.info("Initialized BH_BASKET with %d coins", len(bh.positions))

    bh_btc = bench_deserialize(journal.get_benchmark("bh_btc"))
    bh_basket = bench_deserialize(journal.get_benchmark("bh_basket"))
    bh_btc_value = mark_benchmark(bh_btc, todays_closes)
    bh_basket_value = mark_benchmark(bh_basket, todays_closes)
    # Persist last-known closes so a later missing close uses them (not entry price).
    journal.set_benchmark("bh_btc", bench_serialize(bh_btc))
    journal.set_benchmark("bh_basket", bench_serialize(bh_basket))

    # ── 3c. Trend-timed BTC tracker (S29): invested in bull, cash in bear ─
    timer = trend_deserialize(journal.get_benchmark("trend_timed_btc"))
    selector = selector_deserialize(journal.get_benchmark("regime_selector"))
    # Connors' daily return drives the selector's bear leg.
    prev_pf = journal.last_portfolio_value()
    connors_daily_ret = (portfolio / prev_pf - 1.0) if prev_pf > 0 else 0.0
    btc_df = bars_by_sym.get(BH_BTC_SYMBOL)
    btc_close, sma = None, None
    if btc_df is not None and len(btc_df) >= 1:
        btc_close = float(btc_df["Close"].iloc[-1])
        sma = (float(btc_df["Close"].rolling(timer.ma_window).mean().iloc[-1])
               if len(btc_df) >= timer.ma_window else None)
        trend_timed_value = trend_step(timer, btc_close, sma, today)
        journal.set_benchmark("trend_timed_btc", trend_serialize(timer))
        # ── 3d. Regime selector (S32): bull → BTC (S30), bear → Connors (S18) ─
        selector_value = selector_step(selector, btc_close, sma, connors_daily_ret, today)
        journal.set_benchmark("regime_selector", selector_serialize(selector))
        log.info("Trend-timed: %s $%.2f (%d sw) | Selector: %s $%.2f (%d sw)",
                 "INVESTED" if timer.invested else "CASH", trend_timed_value, timer.n_switches,
                 selector.mode.upper(), selector_value, selector.n_switches)
    else:
        # BTC bars missing on a partial fetch: never mark trend-timed at ~0.
        trend_timed_value = journal.last_positive_value("trend_timed_value")
        log.warning("BTC close missing; carrying trend_timed_value forward: $%.2f", trend_timed_value)
        selector_value = selector.value  # persisted state, starts at capital; cannot reach 0

    # ── 3e. Shadow variants (paper-only): own books, never touch live cash/positions ─
    live_args = ([d.__dict__ for d in decisions], [f.__dict__ for f in fills], portfolio,
                 journal.cash, len(journal.open_positions),
                 {s: snapshot[s][1].get("rsi_2") for s in snapshot})
    shadow_values, shadow_errors, context, lines, pool_errors = _run_pool(
        journal, bars_by_sym, stale_symbols, btc_close, sma, today, live_args)

    snap = DaySnapshot(
        date=today,
        portfolio_value=portfolio,
        cash=journal.cash,
        n_open=len(journal.open_positions),
        closes=todays_closes,
        decisions=[d.__dict__ for d in decisions],
        fills=[f.__dict__ for f in fills],
        skipped=False,
        bh_btc_value=bh_btc_value,
        bh_basket_value=bh_basket_value,
        trend_timed_value=trend_timed_value,
        selector_value=selector_value,
        stale_symbols=stale_symbols,
        **shadow_values,
        shadow_errors=shadow_errors,
        context=context,
        lines=lines,
        pool_errors=pool_errors,
    )
    journal.append_day(snap)

    log.info("Portfolio: $%.2f (%+.2f%%) | BH_BTC: $%.2f (%+.2f%%) | BH_BASKET: $%.2f (%+.2f%%)",
             portfolio, (portfolio / journal.starting_capital - 1) * 100,
             bh_btc_value, (bh_btc_value / journal.starting_capital - 1) * 100,
             bh_basket_value, (bh_basket_value / journal.starting_capital - 1) * 100)

    # ── 4. Early-termination checks ─────────────────────────────────────
    halt_reason = _check_halt(journal)
    if halt_reason and not dry_run:
        log.error("HALT: %s", halt_reason)
        journal.save()
        return 2

    if not dry_run:
        journal.save()
        log.info("Journal saved.")
    else:
        log.info("DRY RUN — journal NOT saved.")

    return 0


def _run_pool(journal, bars_by_sym, stale_symbols, btc_close, btc_sma, today, live_args
              ) -> tuple[dict[str, float], dict[str, str], dict, dict, dict[str, str]]:
    """Step every active pool line in its own book. A failing line is logged, its value carried
    forward and its error recorded, so a line can never break or alter the live run."""
    t0 = time.perf_counter()
    errors: dict[str, str] = {}
    try:
        ctx = build_context(bars_by_sym, today, stale_symbols, BH_BTC_SYMBOL, btc_close, btc_sma)
    except Exception as e:  # noqa: BLE001
        log.exception("Pool context failed; lines run with an unknown context")
        errors["_context"] = f"{type(e).__name__}: {e}"[:200]
        ctx = DayContext(btc_close=btc_close, btc_sma200=btc_sma)
    lines: dict[str, dict] = {}
    try:
        lines[LIVE_ID] = live_line_day(*live_args)   # read-only view of today's live decisions
    except Exception as e:  # noqa: BLE001
        log.exception("Live LineDay failed")
        errors[LIVE_ID] = f"{type(e).__name__}: {e}"[:200]
    snapshots: dict[tuple, dict] = {}   # indicators once per (family, indicator key)
    prev_lines = journal.days[-1].get("lines", {}) if journal.days else {}
    for spec in POOL_LINES:
        try:
            strat = make_strategy(spec)
            key = (spec.family, getattr(strat, "indicator_key", spec.id))
            if key not in snapshots:
                snap = {}
                for sym, df in bars_by_sym.items():
                    ind = strat.indicators(df)
                    snap[sym] = (float(df["Close"].iloc[-1]), ind.loc[df.index[-1]])
                snapshots[key] = snap
            # Copies of rows: a line can never mutate another line's inputs.
            snap = {s: (c, row.copy()) for s, (c, row) in snapshots[key].items()}
            book = pool_book.deserialize(journal.get_benchmark(spec.id), spec.id)
            lines[spec.id] = pool_step(book, spec, strat, snap, ctx, today)
            journal.set_benchmark(spec.id, pool_book.serialize(book))
        except Exception as e:  # noqa: BLE001
            log.exception("Pool line %s failed; carrying value forward", spec.id)
            errors[spec.id] = f"{type(e).__name__}: {e}"[:200]
            prev = (prev_lines.get(spec.id) or {}).get("value") or journal.last_positive_value(
                f"{spec.id}_value", default=0.0)
            lines[spec.id] = {"value": prev, "error": True}
    log.info("Pool: %d lines in %.2fs: %s", len(POOL_LINES), time.perf_counter() - t0,
             ", ".join(f"{k}=${v['value']:.2f}" for k, v in lines.items()))
    # S34 compatibility: shadow_*_value / shadow_errors for the declared shadow keys only.
    shadow_values = {f"{k}_value": v["value"] for k, v in lines.items() if f"{k}_value" in SHADOW_VALUE_KEYS}
    shadow_errors = {k: v for k, v in errors.items() if f"{k}_value" in SHADOW_VALUE_KEYS}
    return shadow_values, shadow_errors, ctx.to_dict(), lines, errors


def _check_halt(journal: Journal) -> str | None:
    dd = journal.max_drawdown()
    if dd > MAX_DRAWDOWN_PCT:
        return f"Drawdown {dd * 100:.2f}% exceeds {MAX_DRAWDOWN_PCT * 100:.0f}%"
    losses = journal.consecutive_losses()
    if losses >= MAX_CONSECUTIVE_LOSSES:
        return f"{losses} consecutive losing trades"
    # No-signal halt REMOVED (2026-06-22): "no trades during a downtrend" is
    # correct Connors behavior, not failure. The 30-day idle trigger produced a
    # false FAIL on day 29 while the live run was healthy and later profitable.
    # Drawdown and loss-streak halts remain — those are real failure modes.
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S18 Paper-Forward daily cron")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run end-to-end but do not write journal")
    args = parser.parse_args(argv)
    return run(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
