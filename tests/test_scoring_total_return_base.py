"""Pool-line total return / drawdown are measured from the book's starting capital, not the first marked value."""
import pytest

from trading_engine.paper.pool import scoring as sc
from trading_engine.paper.pool.registry import LIVE_ID, LINES

FIRST = 9982.896447


def _journal(spec, cap=None):
    days = [{"date": f"2026-10-{10 + i:02d}", "portfolio_value": 10000.0 + 10 * i, "cash": 10000.0,
             "lines": {spec.id: {"value": v}}} for i, v in enumerate([FIRST, FIRST, FIRST + 5])]
    bench = {"closed_trades": []}
    if cap is not None:
        bench["starting_capital"] = cap
    return {"days": days, "benchmarks": {spec.id: bench}, "closed_trades": []}


@pytest.mark.parametrize("cap", [10000.0, None])   # persisted, and fallback to the config default
def test_pool_line_return_and_dd_from_starting_capital(cap):
    spec = LINES[0]
    jd = _journal(spec, cap)
    s, _ = sc._score_line(spec, jd, ".", {}, {}, sc._path(jd["days"], LIVE_ID, None), None, 50)
    last = FIRST + 5
    assert s.total_return == pytest.approx(last / 10000.0 - 1.0)
    assert s.max_dd == pytest.approx(1.0 - FIRST / 10000.0)   # 0.0017104, positive fraction
    assert s.max_dd == pytest.approx(0.0017104, abs=1e-7)


def test_live_line_total_return_unchanged():
    spec = LINES[0]
    jd = _journal(spec, 10000.0)
    live = sc._path(jd["days"], LIVE_ID, None)
    s, _ = sc._score_line(None, jd, ".", {}, {}, live, None, 50)
    assert s.total_return == pytest.approx(10020.0 / 10000.0 - 1.0)   # live: last / first marked value
