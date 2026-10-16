"""Benchmarks must never record/propagate 0.0 for a missing-data day."""
import importlib.util
import pathlib

from trading_engine.paper.benchmarks import init_bh_btc, mark_benchmark
from trading_engine.paper.journal import Journal
from trading_engine.paper.run_daily import carry_forward_benchmarks

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mark_benchmark_missing_close_not_zero():
    b = init_bh_btc({"BTCUSDT": 100.0}, "2026-01-01")
    assert mark_benchmark(b, {}) > 0


def _journal(tmp_path, days):
    return Journal({"version": 4, "start_date": "2026-01-01", "starting_capital": 10000.0,
                             "cash": 10000.0, "open_positions": [], "closed_trades": [], "days": days,
                             "benchmarks": {}})


def test_carry_forward_on_skipped_day(tmp_path):
    days = [{"date": "d1", "portfolio_value": 1.0, "bh_btc_value": 10100.0, "bh_basket_value": 9900.0,
             "trend_timed_value": 10050.0, "selector_value": 10020.0}]
    j = _journal(tmp_path, days)
    out = carry_forward_benchmarks(j)
    assert out == {"bh_btc_value": 10100.0, "bh_basket_value": 9900.0,
                   "trend_timed_value": 10050.0, "selector_value": 10020.0}


def test_metrics_ignore_non_positive():
    bd = _load("build_dashboard", "scripts/build_dashboard.py")

    class J:
        starting_capital = 10000.0
        days = [{"bh_btc_value": v} for v in (10000.0, 10100.0, 0.0, 10200.0, 10300.0)]

    m = bd.compute_benchmark_metrics(J, "bh_btc_value")
    assert m["max_dd_pct"] < 1.0 and m["sharpe"] > 0


def test_repair_script_idempotent():
    rp = _load("repair", "scripts/repair_benchmark_zero_days.py")
    days = [{"date": f"d{i}", "bh_btc_value": v} for i, v in enumerate((10.0, 0.0, 12.0))]
    assert len(rp.repair_days(days)) == 1
    assert days[1]["bh_btc_value"] == 10.0
    assert rp.repair_days(days) == []
