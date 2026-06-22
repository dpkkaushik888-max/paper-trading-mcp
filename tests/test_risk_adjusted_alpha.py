"""S26 — risk-adjusted alpha gate: Sharpe-based G2/G9, profitability floor, modes."""

import numpy as np
import pandas as pd

from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.discovery.primitives import Condition, build_features
from trading_engine.discovery.search import WFGateConfig, gate_candidate
from trading_engine.discovery.gates import evaluate_holdout, subperiod_alphas


def _bars(n=400, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


def _data(btc_drift=0.0015):
    d = {"BTCUSDT": _bars(seed=1, drift=btc_drift), "ETHUSDT": _bars(seed=2, drift=0.001)}
    feats = {s: build_features(df) for s, df in d.items()}
    return d, feats


def _spec(cid="cand"):
    return CandidateSpec(
        id=cid, name=cid,
        entry=[Condition("rsi_14", "<", 45)],
        exit=ExitSpec(sl_pct=0.08, max_hold_days=5,
                      exit_conditions=[Condition("rsi_14", ">", 55)]),
    )


# ── default mode is unchanged (raw_cagr) ────────────────────────────────────
def test_default_mode_is_raw_cagr():
    assert WFGateConfig().alpha_mode == "raw_cagr"


def test_gate_candidate_records_mode_in_gates():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    raw = gate_candidate(_spec(), data, feats, s, e, gate_cfg=WFGateConfig())
    radj = gate_candidate(_spec(), data, feats, s, e,
                          gate_cfg=WFGateConfig(alpha_mode="risk_adjusted"))
    # both produce a G2 verdict; they may differ because the benchmark differs
    assert "G2_alpha" in raw.gates and "G2_alpha" in radj.gates


# ── risk-adjusted G2: Sharpe-beat with profitability floor ──────────────────
def test_risk_adjusted_g2_matches_sharpe_rule():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    r = gate_candidate(_spec(), data, feats, s, e,
                       gate_cfg=WFGateConfig(alpha_mode="risk_adjusted"))
    # G2 in risk-adjusted mode == (candidate Sharpe ≥ BH Sharpe) AND (CAGR ≥ 0)
    from trading_engine.discovery.search import buy_hold
    bh = buy_hold(data, s, e)
    expected = (r.sharpe >= bh["sharpe"]) and (r.cagr >= 0.0)
    assert r.gates["G2_alpha"] == expected


def test_unprofitable_candidate_fails_risk_adjusted_floor():
    # a high-drift BTC benchmark + a weak candidate: even if Sharpe noise favors it,
    # the CAGR≥0 floor must bind when the candidate loses money.
    data, feats = _data(btc_drift=0.003)
    s, e = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    r = gate_candidate(_spec(), data, feats, s, e,
                       gate_cfg=WFGateConfig(alpha_mode="risk_adjusted"))
    if r.cagr < 0:
        assert r.gates["G2_alpha"] is False


# ── holdout: alpha_mode threads through + sub-period uses Sharpe diff ────────
def test_evaluate_holdout_carries_mode_and_bh_sharpe():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    v = evaluate_holdout(_spec(), data, feats, s, e, n_trials=5, alpha_mode="risk_adjusted")
    assert v.alpha_mode == "risk_adjusted"
    assert isinstance(v.bh_sharpe, float)
    # risk-adjusted alpha_pass == Sharpe-beat AND profitable
    assert v.alpha_pass == ((v.sharpe >= v.bh_sharpe) and (v.cagr >= 0.0))


def test_subperiod_alphas_mode_differs():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    raw = subperiod_alphas(_spec(), data, feats, s, e, n=3, alpha_mode="raw_cagr")
    radj = subperiod_alphas(_spec(), data, feats, s, e, n=3, alpha_mode="risk_adjusted")
    assert len(raw) == len(radj) == 3
    assert raw != radj          # CAGR-diff vs Sharpe-diff are different scales


def test_determinism_risk_adjusted():
    data, feats = _data()
    s, e = data["BTCUSDT"].index[60], data["BTCUSDT"].index[-1]
    v1 = evaluate_holdout(_spec(), data, feats, s, e, n_trials=5, alpha_mode="risk_adjusted")
    v2 = evaluate_holdout(_spec(), data, feats, s, e, n_trials=5, alpha_mode="risk_adjusted")
    assert v1.to_dict() == v2.to_dict()
