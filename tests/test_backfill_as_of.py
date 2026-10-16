"""Backfill (as_of=D) must see exactly the bars the live 00:30 UTC run on D saw: bars opened before D."""
from trading_engine.paper import run_daily
from tests.paper_fixtures import synthetic_bars


def _run_as_of(bars, as_of):
    j = run_daily.Journal.init_fresh()
    orig = (run_daily.Journal.load, run_daily.fetch_universe)
    run_daily.Journal.load = classmethod(lambda cls, *a, **k: j)
    run_daily.fetch_universe = lambda syms: dict(bars)  # full history incl. bar D and later
    try:
        assert run_daily.run(dry_run=True, as_of=as_of) == 0
    finally:
        run_daily.Journal.load, run_daily.fetch_universe = orig
    return j


def test_backfill_excludes_bar_d():
    bars = synthetic_bars()
    idx = bars["BTCUSDT"].index
    d, prev = idx[250], idx[249]
    j = _run_as_of(bars, d.date().isoformat())
    day = j.days[-1]
    assert day["date"] == d.date().isoformat()
    for sym, df in bars.items():
        assert day["closes"][sym] == float(df.loc[prev, "Close"])
        assert day["closes"][sym] != float(df.loc[d, "Close"])


def test_backfill_matches_live_run_on_same_day():
    from tests.paper_fixtures import replay_live
    bars = synthetic_bars()
    d = bars["BTCUSDT"].index[-1]
    live = replay_live(run_daily, bars, n_days=1)
    back = _run_as_of(bars, d.date().isoformat())
    assert live.days[-1]["closes"] == back.days[-1]["closes"]
    assert live.days[-1]["decisions"] == back.days[-1]["decisions"]
