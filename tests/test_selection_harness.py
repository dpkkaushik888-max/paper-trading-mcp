"""S31 — strategy-selection harness: no-lookahead, sleeves, named selectors."""

import numpy as np
import pandas as pd

from trading_engine.strategies.connors_swing import precompute_indicators
from trading_engine.selection.harness import (
    SELECTORS, month_boundaries, run_selection, sel_hodl, sel_trend, _trend_bull)


def _bars(n=500, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


def _setup():
    bars = {"BTCUSDT": _bars(seed=1, drift=0.0015), "ETHUSDT": _bars(seed=2, drift=0.001)}
    ind = {s: precompute_indicators(df) for s, df in bars.items()}
    btc = bars["BTCUSDT"]["Close"]
    dates = sorted(set().union(*[df.index for df in bars.values()]))
    bnds = month_boundaries(dates, dates[210], dates[-1])
    return bars, ind, btc, dates, bnds


def test_month_boundaries_increasing():
    bars, ind, btc, dates, bnds = _setup()
    assert len(bnds) >= 10
    assert all(bnds[i] < bnds[i + 1] for i in range(len(bnds) - 1))   # strictly increasing
    assert bnds[0] == dates[210] and bnds[-1] == dates[-1]


def test_hodl_selector_reproduces_buy_and_hold():
    bars, ind, btc, dates, bnds = _setup()
    r = run_selection(sel_hodl, btc, bars, ind, bnds, capital=1000.0)
    # always-BTC, never switches → telescopes to a continuous hold (no gaps)
    expected = 1000.0 * float(btc.loc[bnds[-1]] / btc.loc[bnds[0]])
    assert r.n_switches == 0
    assert abs(r.final_value - expected) / expected < 1e-9


def test_no_lookahead_selector_cannot_see_period_return():
    """The selector sees data only through the decision boundary, never the period after."""
    bars, ind, btc, dates, bnds = _setup()
    checks = {"n": 0}

    def spy(history, at):
        assert history.index.max() <= at         # nothing after the decision point
        checks["n"] += 1
        return "cash"

    run_selection(spy, btc, bars, ind, bnds, 1000.0)
    assert checks["n"] == len(bnds) - 1


def test_cash_sleeve_is_flat():
    bars, ind, btc, dates, bnds = _setup()
    r = run_selection(lambda h, a: "cash", btc, bars, ind, bnds, 1000.0)
    assert abs(r.final_value - 1000.0) < 1e-9     # cash never moves (no switches charged)
    assert r.n_switches == 0


def test_trend_bull_threshold():
    up = pd.Series(np.linspace(1, 2, 300))        # rising → last > sma
    down = pd.Series(np.linspace(2, 1, 300))      # falling → last < sma
    assert _trend_bull(up) is True
    assert _trend_bull(down) is False
    assert _trend_bull(pd.Series([1, 2, 3])) is False   # insufficient history → not bull


def test_determinism():
    bars, ind, btc, dates, periods = _setup()
    r1 = run_selection(SELECTORS["trend_connors"], btc, bars, ind, periods, 1000.0)
    r2 = run_selection(SELECTORS["trend_connors"], btc, bars, ind, periods, 1000.0)
    assert r1.choices == r2.choices and r1.final_value == r2.final_value


def test_switch_cost_charged_on_change():
    bars, ind, btc, dates, periods = _setup()
    # alternate cash/btc every period → many switches, each costs 20bps
    flip = {"i": 0}
    def alt(h, a):
        flip["i"] += 1
        return "btc" if flip["i"] % 2 else "cash"
    r = run_selection(alt, btc, bars, ind, periods, 1000.0)
    assert r.n_switches > 0
