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
    MAX_NO_SIGNAL_DAYS,
    MAX_SKIPPED_DAYS,
)
from .data import fetch_universe
from .journal import (
    Journal, OpenPosition, ClosedTrade, Decision, DaySnapshot, Fill,
)
from .benchmarks import (
    init_bh_btc, init_bh_basket, mark_benchmark,
    serialize as bench_serialize, deserialize as bench_deserialize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s18.run_daily")


def _today_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _latest_bar(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series]:
    """Return (date, row) for the most recent closed bar."""
    return df.index[-1], df.iloc[-1]


def run(dry_run: bool = False) -> int:
    """Execute one daily cron cycle. Returns 0 on success, non-zero on halt."""
    log.info("=" * 70)
    log.info("S18 Paper-Forward — daily run (dry_run=%s)", dry_run)
    log.info("=" * 70)

    journal = Journal.load()
    today = _today_utc_date()

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

    log.info("Fetched bars for %d/%d symbols", len(bars_by_sym), len(UNIVERSE))

    # ── Pre-compute indicators + today snapshot ─────────────────────────
    snapshot: dict[str, tuple[float, pd.Series]] = {}
    todays_closes: dict[str, float] = {}
    for sym, df in bars_by_sym.items():
        ind = precompute_indicators(df)
        date, row = _latest_bar(df)
        close_today = float(row["Close"])
        ind_row = ind.loc[date]
        snapshot[sym] = (close_today, ind_row)
        todays_closes[sym] = close_today

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


def _check_halt(journal: Journal) -> str | None:
    dd = journal.max_drawdown()
    if dd > MAX_DRAWDOWN_PCT:
        return f"Drawdown {dd * 100:.2f}% exceeds {MAX_DRAWDOWN_PCT * 100:.0f}%"
    losses = journal.consecutive_losses()
    if losses >= MAX_CONSECUTIVE_LOSSES:
        return f"{losses} consecutive losing trades"
    no_sig = journal.days_since_last_trade()
    if no_sig >= MAX_NO_SIGNAL_DAYS:
        return f"{no_sig} days without any trade activity"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S18 Paper-Forward daily cron")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run end-to-end but do not write journal")
    args = parser.parse_args(argv)
    return run(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
