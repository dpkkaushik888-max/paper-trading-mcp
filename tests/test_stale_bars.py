"""Staleness warning: frozen-bar symbols are recorded in the DaySnapshot; no behaviour change."""
import logging

from trading_engine.paper import run_daily
from trading_engine.paper.journal import DaySnapshot, Journal
from tests.paper_fixtures import synthetic_bars, replay_live, live_signature


def _frozen(bars, sym, freeze_at):
    bars = dict(bars)
    bars[sym] = bars[sym].iloc[:freeze_at]
    return bars


def test_frozen_symbol_flagged_and_logged(caplog):
    bars = _frozen(synthetic_bars(), "DOGEUSDT", 300)
    with caplog.at_level(logging.WARNING, logger="s18.run_daily"):
        j = replay_live(run_daily, bars, n_days=3)
    assert all(d["stale_symbols"] == ["DOGEUSDT"] for d in j.days)
    assert any("STALE DATA: DOGEUSDT" in r.getMessage() for r in caplog.records)


def test_fresh_bars_not_flagged():
    j = replay_live(run_daily, synthetic_bars(), n_days=3)
    assert all(d["stale_symbols"] == [] for d in j.days)


def test_age_boundary():
    import pandas as pd
    d = {"A": pd.Timestamp("2026-10-14", tz="UTC"), "B": pd.Timestamp("2026-10-13", tz="UTC")}
    assert run_daily._stale_symbols(d, "2026-10-16") == ["B"]  # age 2 ok, age 3 stale


def test_staleness_does_not_change_trading():
    bars = _frozen(synthetic_bars(), "DOGEUSDT", 260)
    a = live_signature(replay_live(run_daily, bars))
    # Same input data, staleness detection disabled -> identical live series.
    orig = run_daily._stale_symbols
    run_daily._stale_symbols = lambda *_: []
    try:
        b = live_signature(replay_live(run_daily, bars))
    finally:
        run_daily._stale_symbols = orig
    assert a == b


def test_old_journal_rows_load_without_field():
    snap = DaySnapshot(date="2026-01-01", portfolio_value=1.0, cash=1.0, n_open=0,
                       closes={}, decisions=[], fills=[])
    assert snap.stale_symbols == []
    j = Journal({"version": 4, "days": [{"date": "2026-01-01", "portfolio_value": 1.0}],
                 "starting_capital": 1.0, "cash": 1.0, "open_positions": [], "closed_trades": []})
    assert j.days[0].get("stale_symbols", []) == []
