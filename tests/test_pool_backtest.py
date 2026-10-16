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
        "n_years", "buckets", "trades", "gate", "evidence", "hypothesis", "generated_from", "years_method",
        "live_window_us"}


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
    assert d["years_method"] == "chained" and set(d["generated_from"]) == {"sha", "dirty"}
    assert d["live_window_us"] is None   # no .us bars in this fixture
    assert {"line_maxdd", "comparators", "idle_n"} <= set(d["evidence"])
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
    dd = lambda lm, **c: {"line_maxdd": lm, "comparators": c}  # noqa: E731
    assert gate_result("reduces_dd_in", 0, 7, dd(0.2, s18_reference=0.3))["status"] == "active"
    assert gate_result("reduces_dd_in", 0, 7, dd(0.3, s18_reference=0.3))["status"] == "candidate"
    # must beat EVERY comparator
    assert gate_result("reduces_dd_in", 0, 7, dd(0.2, s18_reference=0.3, bh_basket=0.25))["status"] == "active"
    assert gate_result("reduces_dd_in", 0, 7, dd(0.2, s18_reference=0.3, bh_basket=0.15))["status"] == "candidate"
    assert gate_result("reduces_dd_in", 0, 7, dd(0.2))["status"] == "candidate"


def test_control_gate_passes_only_via_waiver():
    assert gate_result("control", 0, 7, {}, waiver="control: x")["status"] == "active"
    assert "waiver" in gate_result("control", 0, 7, {}, waiver="control: x")["by"]
    assert gate_result("control", 7, 7, {})["status"] == "candidate"   # years alone never activate a control
    assert gate_result("beats_live", 0, 7, {}, waiver="ignored for non-control")["status"] == "candidate"


def test_bh_basket_path_is_buy_and_hold():
    import pandas as pd
    days = list(pd.date_range("2026-01-01", periods=3))
    bars = {"AUSDT": pd.DataFrame({"Close": [10.0, 20.0, 20.0]}, index=days),
            "BUSDT": pd.DataFrame({"Close": [10.0, 10.0, 5.0]}, index=days)}
    v = pbt.bh_basket_path(bars, days)
    per = pbt.STARTING_CAPITAL / len(pbt.UNIVERSE) * (1 - pbt.COST_PCT)
    cash = pbt.STARTING_CAPITAL - 2 * pbt.STARTING_CAPITAL / len(pbt.UNIVERSE)
    assert v.tolist() == pytest.approx([2 * per + cash, 3 * per + cash, 2.5 * per + cash])   # not rebalanced


def test_committed_baselines_match_registry():
    from trading_engine.paper.pool.registry import LINES
    for l in LINES:
        d = json.loads((ROOT / l.baseline).read_text())
        assert d["line"] == l.id and d["venue"] == "com" and (ROOT / l.baseline).stat().st_size < 60_000
        assert {"passed", "by", "status"} <= set(d["gate"]), l.id   # gate result recorded for every line
        assert d["years_method"] == "chained" and d["generated_from"]["sha"], l.id
        w = d["live_window_us"]
        assert w["venue"] == "us" and w["period"][0] == "2026-04-20" and "skipped_symbols" in w, l.id
        assert {"ret", "maxdd", "trades", "win_rate", "mean_pnl_pct"} <= set(w["full"]) and w["buckets"], l.id
        if l.hypothesis.kind == "control":   # passes the gate via its waiver
            assert d["gate"]["status"] == l.status == "active" and l.baseline_waiver in d["gate"]["by"], l.id
        elif l.baseline_waiver:   # S34 shadows: status from the waiver, not the gate
            assert l.status == "active", l.id
        else:
            assert d["gate"]["status"] == l.status, l.id
        if l.hypothesis.kind == "reduces_dd_in":
            assert d["gate"]["passed"] and d["evidence"]["cond"] == l.hypothesis.cond, l.id
            assert all(d["evidence"]["line_maxdd"] < v for v in d["evidence"]["comparators"].values()), l.id
    assert set(json.loads((ROOT / "state/pool_baselines/basket_sma50_band.json").read_text())["evidence"]
               ["comparators"]) == {"s18_reference", "bh_basket"}


def test_candidate_not_stepped_and_live_unchanged(monkeypatch):
    from dataclasses import replace
    from trading_engine.paper.pool.registry import LINES, active_lines
    cand = replace(LINES[-1], id="cand_probe", status="candidate")
    monkeypatch.setattr(run_daily, "POOL_LINES", active_lines(LINES + (cand,)))
    j = replay_live(run_daily, synthetic_bars())
    assert live_signature(j) == json.loads(GOLDEN_PATH.read_text())
    day = j.days[-1]
    assert "cand_probe" not in day["lines"] and j.get_benchmark("cand_probe") in ({}, None)
    for lid in ("breakout_s21", "xsec_mom_top3", "basket_sma50_band", "connors_exit_sma10", "connors_guarded"):
        assert lid in day["lines"] and lid not in day["pool_errors"], (lid, day["pool_errors"])
    assert day["context"]["btc_vol20_bucket"] in ("low", "mid", "high")
