"""S25 — locked-holdout judge: deflated Sharpe + sub-period robustness."""

import math

import numpy as np
import pandas as pd
import pytest

from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.discovery.primitives import Condition, build_features
from trading_engine.discovery.gates import (
    norm_ppf, expected_max_sharpe, deflated_sharpe_threshold,
    subperiod_alphas, evaluate_holdout, _subperiod_bounds,
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


def _spec(cid="cand"):
    return CandidateSpec(
        id=cid, name=cid,
        entry=[Condition("rsi_14", "<", 45)],
        exit=ExitSpec(sl_pct=0.08, max_hold_days=5,
                      exit_conditions=[Condition("rsi_14", ">", 55)]),
    )


# ── norm_ppf (Acklam) ────────────────────────────────────────────────────────
def test_norm_ppf_known_quantiles():
    assert abs(norm_ppf(0.5)) < 1e-6
    assert abs(norm_ppf(0.975) - 1.959963985) < 1e-4
    assert abs(norm_ppf(0.025) + 1.959963985) < 1e-4
    # tail (exercises the plow/phigh branches)
    assert abs(norm_ppf(0.999) - 3.090232306) < 1e-3


def test_norm_ppf_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            norm_ppf(bad)


# ── deflated-Sharpe machinery ────────────────────────────────────────────────
def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_expected_max_sharpe_monotonic_in_trials():
    vals = [expected_max_sharpe(n, 1.0) for n in (2, 5, 10, 50, 100)]
    assert all(b > a for a, b in zip(vals, vals[1:]))   # strictly increasing


def test_deflated_threshold_equals_base_for_one_trial():
    assert deflated_sharpe_threshold(1.0, 1, 365) == 1.0


def test_deflated_threshold_rises_with_trials():
    t1 = deflated_sharpe_threshold(1.0, 1, 365)
    t10 = deflated_sharpe_threshold(1.0, 10, 365)
    t50 = deflated_sharpe_threshold(1.0, 50, 365)
    assert t1 < t10 < t50
    assert t10 > 1.0   # the bar is genuinely raised


def test_deflated_threshold_tighter_with_more_observations():
    # more holdout data → less estimation noise → smaller haircut
    few = deflated_sharpe_threshold(1.0, 10, n_obs=180)
    many = deflated_sharpe_threshold(1.0, 10, n_obs=1000)
    assert few > many


# ── sub-period robustness ────────────────────────────────────────────────────
def test_subperiod_bounds_partitions_window():
    data, _ = _data()
    start, end = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    bounds = _subperiod_bounds(data, start, end, 3)
    assert len(bounds) == 3
    assert bounds[0][0] == start
    assert bounds[-1][1] == end
    # contiguous, non-overlapping, increasing
    assert bounds[0][1] < bounds[1][0] <= bounds[1][1] < bounds[2][0]


def test_subperiod_alphas_returns_one_per_slice():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    alphas = subperiod_alphas(_spec(), data, feats, start, end, n=3)
    assert len(alphas) == 3
    assert all(isinstance(a, float) for a in alphas)


# ── evaluate_holdout end-to-end ──────────────────────────────────────────────
def test_evaluate_holdout_structure_and_determinism():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    v1 = evaluate_holdout(_spec(), data, feats, start, end, n_trials=5)
    v2 = evaluate_holdout(_spec(), data, feats, start, end, n_trials=5)
    assert v1.to_dict() == v2.to_dict()                     # determinism
    assert v1.candidate_id == "cand"
    assert len(v1.subperiod_alphas) == 3
    assert isinstance(v1.promoted, bool)


def test_more_trials_make_promotion_strictly_harder():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    lo = evaluate_holdout(_spec(), data, feats, start, end, n_trials=1)
    hi = evaluate_holdout(_spec(), data, feats, start, end, n_trials=100)
    assert hi.deflated_threshold > lo.deflated_threshold


def test_promotion_requires_all_three_conditions():
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    v = evaluate_holdout(_spec(), data, feats, start, end, n_trials=5)
    assert v.promoted == (v.sharpe_pass and v.robustness_pass and v.alpha_pass)


def test_dead_candidate_not_promoted():
    # never trades → flat → loses to buy-and-hold → must be rejected
    data, feats = _data()
    start, end = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    dead = CandidateSpec(id="dead", name="dead",
                         entry=[Condition("rsi_14", "<", -1)],
                         exit=ExitSpec(sl_pct=0.07, max_hold_days=10))
    v = evaluate_holdout(dead, data, feats, start, end, n_trials=5)
    assert v.n_trades == 0
    assert v.promoted is False
