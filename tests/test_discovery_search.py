"""S25 — walk-forward search: gating, trial budget, determinism (offline)."""

import numpy as np
import pandas as pd

from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.discovery.primitives import Condition, build_features
from trading_engine.discovery.search import (
    CandidateResult, WFGateConfig, gate_candidate, run_candidate_window,
    search_walk_forward, select_survivors, buy_hold, cagr_of,
)


def _bars(n=400, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


def _data():
    d = {"BTCUSDT": _bars(seed=1, drift=0.0015), "ETHUSDT": _bars(seed=2, drift=0.001)}
    feats = {s: build_features(df) for s, df in d.items()}
    return d, feats


# A frequently-trading mean-reversion dip-buyer (loose entry → many trades).
def _active_spec(cid="cand_active"):
    return CandidateSpec(
        id=cid, name=cid,
        entry=[Condition("rsi_14", "<", 45)],
        exit=ExitSpec(sl_pct=0.08, max_hold_days=5,
                      exit_conditions=[Condition("rsi_14", ">", 55)]),
    )


# A never-entering candidate (impossible threshold) → zero trades.
def _dead_spec(cid="cand_dead"):
    return CandidateSpec(
        id=cid, name=cid,
        entry=[Condition("rsi_14", "<", -1)],   # RSI ∈ [0,100], never true
        exit=ExitSpec(sl_pct=0.07, max_hold_days=10),
    )


def test_run_candidate_window_smoke():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[210], data["BTCUSDT"].index[-1]
    res, days = run_candidate_window(_active_spec(), data, feats, start, end)
    assert days > 0
    assert res.final_value > 0


def test_dead_candidate_fails_trade_gate():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    r = gate_candidate(_dead_spec(), data, feats, start, end)
    assert r.n_trades == 0
    assert r.gates["G4_trades"] is False
    assert r.passed is False
    assert "G4_trades" in r.reject_reason


def test_search_logs_all_candidates():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    cands = [_active_spec("a"), _dead_spec("b"), _active_spec("c")]
    sr = search_walk_forward(cands, data, feats, start, end, trial_budget=10)
    assert sr.n_proposed == 3
    assert len(sr.all_candidates) == 3            # nothing silently dropped
    assert {c.candidate.id for c in sr.all_candidates} == {"a", "b", "c"}


def test_determinism_same_inputs_same_result():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    cands = [_active_spec("a"), _active_spec("b")]
    s1 = search_walk_forward(cands, data, feats, start, end, trial_budget=5)
    s2 = search_walk_forward(cands, data, feats, start, end, trial_budget=5)
    assert s1.to_dict() == s2.to_dict()


# ── select_survivors mechanics (pure, deterministic) ────────────────────────
def _mk_result(cid, sharpe, cagr, passed=True):
    spec = _active_spec(cid)
    return CandidateResult(
        candidate=spec, return_pct=0.0, cagr=cagr, sharpe=sharpe, max_dd=0.1,
        n_trades=20, bh_cagr=0.0, gates={}, passed=passed, rank_score=sharpe,
    )


def test_trial_budget_caps_survivors():
    results = [_mk_result(f"c{i}", sharpe=float(i), cagr=0.1) for i in range(10)]
    survivors = select_survivors(results, trial_budget=3)
    assert len(survivors) == 3
    # highest sharpe first
    assert [s.candidate.id for s in survivors] == ["c9", "c8", "c7"]
    assert all(s.reached_holdout for s in survivors)
    # the rest are logged as budget-rejected, not silently dropped
    rejected = [r for r in results if not r.reached_holdout]
    assert all(r.reject_reason == "trial_budget_exceeded" for r in rejected)


def test_failed_candidates_never_reach_holdout():
    results = [_mk_result("good", 2.0, 0.2, passed=True),
               _mk_result("bad", 5.0, 0.9, passed=False)]
    survivors = select_survivors(results, trial_budget=10)
    assert [s.candidate.id for s in survivors] == ["good"]   # bad excluded despite high sharpe


def test_survivor_ordering_tiebreak_by_id():
    results = [_mk_result("z", 1.0, 0.1), _mk_result("a", 1.0, 0.1)]
    survivors = select_survivors(results, trial_budget=10)
    assert [s.candidate.id for s in survivors] == ["a", "z"]   # equal scores → id asc


# ── benchmark helpers ────────────────────────────────────────────────────────
def test_buy_hold_uses_btc_and_reports_metrics():
    data, _ = _data()
    start, end = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    m = buy_hold(data, start, end)
    assert set(m) == {"cagr", "sharpe", "total_return"}


def test_cagr_of_doubling_one_year():
    assert cagr_of(20_000.0, 365) > 0.9
