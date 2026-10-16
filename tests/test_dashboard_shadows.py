"""Dashboard: time-weighted invested share, days fully in cash, shadow variants table."""
import importlib.util
import json
import math
from pathlib import Path

from trading_engine.paper import run_daily
from trading_engine.paper.journal import Journal
from tests.paper_fixtures import replay_live, synthetic_bars

_p = Path(__file__).parent.parent / "scripts" / "build_dashboard.py"
_spec = importlib.util.spec_from_file_location("build_dashboard", _p)
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


def _journal(days):
    j = Journal.init_fresh()
    j.days.extend(days)
    return j


def test_utilisation_metrics():
    j = _journal([
        {"date": "a", "portfolio_value": 100.0, "cash": 100.0, "n_open": 0},
        {"date": "b", "portfolio_value": 100.0, "cash": 50.0, "n_open": 2},
        {"date": "c", "portfolio_value": 100.0, "cash": 0.0, "n_open": 0, "skipped": True},
    ])
    u = bd.compute_utilisation(j)
    assert u == {"invested_share_avg": 0.25, "days_fully_in_cash": 1}
    assert bd.compute_utilisation(_journal([])) == {"invested_share_avg": None, "days_fully_in_cash": 0}


def test_no_shadow_rows_for_pre_shadow_journal():
    j = _journal([{"date": "a", "portfolio_value": 100.0, "cash": 100.0, "n_open": 0,
                   "shadow_rsi15_value": 0.0, "closes": {}}])
    assert bd.compute_shadow_rows(j) == []
    data = bd.build_data_json(j)
    assert data["shadows"] == []
    json.dumps(data, allow_nan=False)


def test_shadow_rows_same_window_and_finite():
    j = replay_live(run_daily, synthetic_bars(), n_days=60)
    rows = bd.compute_shadow_rows(j)
    assert [r["name"] for r in rows] == ["shadow_atr_stop", "shadow_btc200", "shadow_rsi15", "shadow_combo"]
    for r in rows:
        assert r["start_date"] == j.days[0]["date"] and r["days"] == 60
        for k, v in r.items():
            assert v is not None, k
            if isinstance(v, float):
                assert math.isfinite(v), k
    # Connors over the same window equals the live series' own return since start.
    assert rows[0]["connors_return_pct"] == (j.days[-1]["portfolio_value"] / j.starting_capital - 1) * 100
    json.dumps(bd.build_data_json(j), allow_nan=False)


def test_single_day_shadow_renders_dashes_not_nan():
    j = _journal([{"date": "a", "portfolio_value": 100.0, "cash": 100.0, "n_open": 0,
                   "shadow_combo_value": 10_000.0}])
    (r,) = bd.compute_shadow_rows(j)
    assert r["sharpe"] is None and r["value"] == 10_000.0
    json.dumps(r, allow_nan=False)


def test_index_html_has_empty_state_and_labels():
    html = bd.INDEX_HTML
    assert "no shadow data yet; starts at next daily run" in html
    assert "Avg invested share (time-weighted)" in html and "Days fully in cash" in html
