"""S35 C2 backtest script, registration gate and candidate lines in _run_pool."""
import importlib.util
import json
from pathlib import Path

import pytest

from trading_engine.paper import run_daily
from trading_engine.paper.pool.gate import gate_result, years_beating
from tests.paper_fixtures import GOLDEN_PATH, live_signature, replay_live, synthetic_bars

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location("pool_backtest", ROOT / "scripts" / "pool_backtest.py")
pbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pbt)

KEYS = {"line", "venue", "period", "python", "git_sha", "full", "years", "oos_2023", "years_beating_s18",
        "n_years", "buckets", "trades", "gate", "evidence", "hypothesis"}


def test_backtest_writes_baseline_json(tmp_path):
    for s, df in synthetic_bars().items():
        (tmp_path / "bars" / "com").mkdir(parents=True, exist_ok=True)
        df.tz_localize(None).to_csv(tmp_path / "bars" / "com" / f"{s}.csv", index_label="date")
    out = tmp_path / "out"
    rc = pbt.main(["--line", "breakout_s21", "--bars", str(tmp_path / "bars"), "--out", str(out), "--start", "2025-01-01"])
    assert rc == 0
    d = json.loads((out / "breakout_s21.json").read_text())
    assert KEYS <= set(d) and d["venue"] == "com"
    assert {"ret", "cagr", "sharpe", "maxdd", "trades", "win_rate", "mean_pnl_pct"} <= set(d["full"])
    assert set(d["buckets"]) == {"btc_trend", "btc_vol20_bucket", "basket_ret20_bucket", "dow"}
    assert "trade_list" not in d and "closed_trades" not in d
    ref = json.loads((out / "_s18_reference_com.json").read_text())
    assert ref["line"] == "connors_live_params" and ref["years_beating_s18"] == 0


def test_years_beating_and_gate():
    ref = {str(y): {"ret": 0.0} for y in range(2020, 2027)}
    line = {str(y): {"ret": 0.01 if y < 2024 else -0.01} for y in range(2020, 2027)}
    assert years_beating(line, ref) == (4, 7)
    assert gate_result("beats_live", 4, 7, {})["status"] == "active"
    assert gate_result("beats_live", 3, 7, {})["status"] == "candidate"
    assert gate_result("beats_live", 3, 7, {"idle_n": 5, "idle_mean": 0.1})["status"] == "candidate"
    assert gate_result("earns_when_live_idle", 1, 7, {"idle_n": 5, "idle_mean": 0.01})["status"] == "active"
    assert gate_result("earns_when_live_idle", 1, 7, {"idle_n": 5, "idle_mean": -0.01})["status"] == "candidate"
    assert gate_result("reduces_dd_in", 0, 7, {"line_maxdd": 0.2, "bench_maxdd": 0.3})["status"] == "active"
    assert gate_result("reduces_dd_in", 0, 7, {"line_maxdd": 0.3, "bench_maxdd": 0.3})["status"] == "candidate"


def test_committed_baselines_match_registry():
    from trading_engine.paper.pool.registry import LINES
    for l in LINES:
        d = json.loads((ROOT / l.baseline).read_text())
        assert d["line"] == l.id and d["venue"] == "com" and (ROOT / l.baseline).stat().st_size < 60_000
        if not l.id.startswith("shadow_"):   # S34 shadows keep their waiver
            assert d["gate"]["status"] == l.status, l.id


def test_candidate_not_stepped_and_live_unchanged():
    j = replay_live(run_daily, synthetic_bars())
    assert live_signature(j) == json.loads(GOLDEN_PATH.read_text())
    day = j.days[-1]
    assert "connors_guarded" not in day["lines"] and j.get_benchmark("connors_guarded") in ({}, None)
    for lid in ("breakout_s21", "xsec_mom_top3", "basket_sma50_band", "connors_exit_sma10"):
        assert lid in day["lines"] and lid not in day["pool_errors"], (lid, day["pool_errors"])
    assert day["context"]["btc_vol20_bucket"] in ("low", "mid", "high")
