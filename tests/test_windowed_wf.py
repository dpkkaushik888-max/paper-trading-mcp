"""S28 — multi-window (full-cycle) walk-forward: splitting, consistency gating."""

import numpy as np
import pandas as pd

from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.discovery.primitives import Condition, build_features
from trading_engine.discovery.search import (
    WindowedWFConfig, window_bounds, gate_candidate_windowed,
    search_walk_forward_windows, _window_pass,
)


def _bars(n=480, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


def _data():
    d = {"BTCUSDT": _bars(seed=1, drift=0.0012), "ETHUSDT": _bars(seed=2, drift=0.001)}
    feats = {s: build_features(df) for s, df in d.items()}
    return d, feats


def _spec(cid="cand"):
    return CandidateSpec(cid, cid, [Condition("rsi_14", "<", 45)],
                         ExitSpec(sl_pct=0.08, max_hold_days=5,
                                  exit_conditions=[Condition("rsi_14", ">", 55)]))


# ── window splitting ────────────────────────────────────────────────────────
def test_window_bounds_partitions():
    data, _ = _data()
    s, e = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    b = window_bounds(data, s, e, 4)
    assert len(b) == 4
    assert b[0][0] == s and b[-1][1] == e
    for i in range(3):
        assert b[i][1] < b[i + 1][0]            # contiguous, non-overlapping


def test_window_bounds_degenerate():
    data, _ = _data()
    s, e = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    assert window_bounds(data, s, e, 1) == [(s, e)]


# ── per-window pass logic ───────────────────────────────────────────────────
def test_window_pass_modes():
    bh = {"cagr": 0.30, "sharpe": 0.80}
    # risk_adjusted: beat BH Sharpe AND profitable — losing on CAGR-but-higher-Sharpe still passes
    assert _window_pass(0.9, 0.05, bh, "risk_adjusted") is True
    assert _window_pass(0.9, -0.01, bh, "risk_adjusted") is False   # not profitable
    assert _window_pass(0.5, 0.40, bh, "risk_adjusted") is False    # Sharpe below BH
    # raw_cagr: just beat BH CAGR
    assert _window_pass(0.1, 0.31, bh, "raw_cagr") is True
    assert _window_pass(2.0, 0.20, bh, "raw_cagr") is False


# ── consistency gate ────────────────────────────────────────────────────────
def test_gate_windowed_produces_per_window_breakdown():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    r = gate_candidate_windowed(_spec(), data, feats, s, e,
                                cfg=WindowedWFConfig(n_windows=4))
    assert r.windows is not None and len(r.windows) == 4
    assert "G1_consistency" in r.gates
    # rank_score == number of windows passed
    assert r.rank_score == sum(1 for w in r.windows if w["pass"])


def test_min_windows_pass_binds():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    easy = gate_candidate_windowed(_spec(), data, feats, s, e,
                                   cfg=WindowedWFConfig(n_windows=4, min_windows_pass=0,
                                                        min_total_trades=0, max_dd=1.0))
    hard = gate_candidate_windowed(_spec(), data, feats, s, e,
                                   cfg=WindowedWFConfig(n_windows=4, min_windows_pass=5,
                                                        min_total_trades=0, max_dd=1.0))
    assert easy.gates["G1_consistency"] is True       # 0 required → always passes
    assert hard.gates["G1_consistency"] is False      # 5 of 4 impossible → never passes


def test_worst_window_dd_ceiling():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    r = gate_candidate_windowed(_spec(), data, feats, s, e,
                                cfg=WindowedWFConfig(max_dd=0.0001, min_windows_pass=0,
                                                     min_total_trades=0))
    assert r.gates["G3_worst_dd"] is False            # any DD > 0.01% trips it


# ── search wrapper + determinism ────────────────────────────────────────────
def test_search_windows_logs_all_and_is_deterministic():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    cands = [_spec("a"), _spec("b"), _spec("c")]
    cfg = WindowedWFConfig(n_windows=3, min_windows_pass=2)
    s1 = search_walk_forward_windows(cands, data, feats, s, e, trial_budget=5, cfg=cfg)
    s2 = search_walk_forward_windows(cands, data, feats, s, e, trial_budget=5, cfg=cfg)
    assert len(s1.all_candidates) == 3
    assert s1.to_dict() == s2.to_dict()
    assert len(s1.survivors) <= 5
