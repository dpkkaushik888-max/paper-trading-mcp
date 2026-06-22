"""S23 Stage 5 — gates harness helpers (deterministic, no network)."""

import numpy as np
import pandas as pd

import scripts.sim_crypto_leaf as gh
from trading_engine.strategies.connors_strategy import ConnorsSwingStrategy, default_config as a_cfg


def _bars(n=400, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


class FakeResult:
    def __init__(self, final_value):
        self.final_value = final_value


def test_cagr_of_one_year_doubling():
    r = FakeResult(20_000.0)  # 2x on 10k
    assert gh.cagr_of(r, 365) > 0.9  # ~+100% CAGR over 1 year


def test_cagr_of_flat():
    assert abs(gh.cagr_of(FakeResult(10_000.0), 365)) < 1e-6


def test_bh_btc_metrics():
    data = {"BTCUSDT": _bars(seed=1, drift=0.002)}
    start, end = data["BTCUSDT"].index[0], data["BTCUSDT"].index[-1]
    m = gh.bh_btc(data, start, end)
    assert set(m) == {"cagr", "sharpe", "total_return"}
    assert isinstance(m["sharpe"], float)


def test_bh_btc_missing_btc_returns_zero():
    data = {"ETHUSDT": _bars(seed=2)}
    m = gh.bh_btc(data, data["ETHUSDT"].index[0], data["ETHUSDT"].index[-1])
    assert m["cagr"] == 0.0


def test_split_proportions():
    data = {"BTCUSDT": _bars(n=1000, seed=3)}
    start, train_end, wf_end, ho_end = gh._split(data)
    dates = data["BTCUSDT"].index
    assert start == dates[0] and ho_end == dates[-1]
    assert train_end == dates[600]     # 60%
    assert wf_end == dates[800]        # 80%


def test_run_leaf_smoke():
    data = {"BTCUSDT": _bars(seed=4), "ETHUSDT": _bars(seed=5)}
    ind = {s: gh.build_indicators(df) for s, df in data.items()}
    r, days = gh.run_leaf([ConnorsSwingStrategy(a_cfg())], data, ind,
                          data["BTCUSDT"].index[210], data["BTCUSDT"].index[-1])
    assert days > 0
    assert r.final_value > 0
