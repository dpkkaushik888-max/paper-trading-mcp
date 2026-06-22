"""S25 — primitive grammar: build_features + Condition evaluation."""

import numpy as np
import pandas as pd
import pytest

from trading_engine.discovery.primitives import (
    Condition, FEATURE_COLUMNS, build_features,
)


def _bars(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


def test_build_features_has_all_columns():
    feats = build_features(_bars())
    for col in FEATURE_COLUMNS:
        assert col in feats.columns


def test_condition_numeric_rhs():
    row = pd.Series({"rsi_2": 5.0})
    assert Condition("rsi_2", "<", 10).evaluate(row) is True
    assert Condition("rsi_2", ">", 10).evaluate(row) is False


def test_condition_column_rhs_with_multiplier():
    row = pd.Series({"volume": 2000.0, "vol_sma_20": 1000.0})
    assert Condition("volume", ">", "vol_sma_20", rhs_mult=1.5).evaluate(row) is True   # 2000 > 1500
    row2 = pd.Series({"volume": 1200.0, "vol_sma_20": 1000.0})
    assert Condition("volume", ">", "vol_sma_20", rhs_mult=1.5).evaluate(row2) is False  # 1200 < 1500


def test_condition_close_substitution():
    row = pd.Series({"sma_200": 90.0})
    assert Condition("close", ">", "sma_200").evaluate(row, close=100.0) is True


def test_condition_nan_is_false():
    row = pd.Series({"sma_200": float("nan")})
    assert Condition("close", ">", "sma_200").evaluate(row, close=100.0) is False


def test_condition_all_ops():
    row = pd.Series({"x": 5.0})
    row = pd.Series({"adx_14": 5.0})
    assert Condition("adx_14", ">=", 5).evaluate(row) is True
    assert Condition("adx_14", "<=", 5).evaluate(row) is True
    assert Condition("adx_14", ">", 5).evaluate(row) is False


def test_condition_rejects_bad_op_and_columns():
    with pytest.raises(ValueError):
        Condition("rsi_2", "==", 10)
    with pytest.raises(ValueError):
        Condition("not_a_col", ">", 1)
    with pytest.raises(ValueError):
        Condition("rsi_2", ">", "not_a_col")


def test_condition_round_trip():
    c = Condition("volume", ">", "vol_sma_20", rhs_mult=1.5)
    assert Condition.from_dict(c.to_dict()) == c
