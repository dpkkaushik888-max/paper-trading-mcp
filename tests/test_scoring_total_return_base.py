"""Pool-line total return / drawdown are measured from the book's starting capital, not the first marked value."""
import pytest

from trading_engine.paper.pool import scoring as sc
from trading_engine.paper.pool.registry import LIVE_ID, LINES

FIRST = 9982.896447
DATES = ["2026-10-10", "2026-10-11", "2026-10-12"]


def _journal(spec, cap=None, init=DATES[0], live_vals=(10000.0, 10010.0, 10020.0), pool_vals=(FIRST, FIRST, FIRST + 5)):
    days = [{"date": d, "portfolio_value": lv, "cash": 10000.0, "lines": {spec.id: {"value": v}}}
            for d, lv, v in zip(DATES, live_vals, pool_vals)]
    bench = {"closed_trades": [], "init_date": init}
    if cap is not None:
        bench["starting_capital"] = cap
    return {"days": days, "benchmarks": {spec.id: bench}, "closed_trades": []}


def _score(spec, jd):
    return sc._score_line(spec, jd, ".", {}, {}, sc._path(jd["days"], LIVE_ID, None), None, 50)[0]


@pytest.mark.parametrize("cap,base", [(20000.0, 20000.0), (None, sc.STARTING_CAPITAL)])
def test_pool_line_return_and_dd_from_starting_capital(cap, base):
    spec = LINES[0]
    s = _score(spec, _journal(spec, cap))
    assert s.total_return == pytest.approx((FIRST + 5) / base - 1.0)
    assert s.max_dd == pytest.approx(1.0 - FIRST / base)


def test_pool_line_default_base_dd_value():
    spec = LINES[0]
    s = _score(spec, _journal(spec, None))
    assert s.max_dd == pytest.approx(0.0017104, abs=1e-7)


def test_live_line_total_return_unchanged():
    spec = LINES[0]
    jd = _journal(spec, 10000.0, live_vals=(9990.0, 10010.0, 10020.0))
    live = sc._path(jd["days"], LIVE_ID, None)
    s, _ = sc._score_line(None, jd, ".", {}, {}, live, None, 50)
    assert s.total_return == pytest.approx(10020.0 / 9990.0 - 1.0)   # live: last / first marked value


def test_init_date_mismatch_keeps_first_marked_value_base():
    spec = LINES[0]
    s = _score(spec, _journal(spec, 10000.0, init="2026-10-01"))
    assert s.total_return == pytest.approx((FIRST + 5) / FIRST - 1.0)


def test_paired_includes_day1_return_from_starting_capital():
    spec = LINES[0]
    jd = _journal(spec, 10000.0, live_vals=(10000.0, 10000.0, 10000.0), pool_vals=(9980.0, 9980.0, 9980.0))
    # live had a prior day; pool's day-1 entry cost (-0.2 %) must be in the comparison
    jd["days"].insert(0, {"date": "2026-10-09", "portfolio_value": 10000.0, "cash": 10000.0, "lines": {}})
    s = _score(spec, jd)
    assert s.paired is not None and s.paired["n_days"] == 4   # 10-09 (cap anchor) + 3 path days
    assert s.paired["start"] == "2026-10-09"
    assert s.paired["cum_ret_diff"] == pytest.approx(-0.002)
