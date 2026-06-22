"""S27 — pattern/sequence grammar primitives: correctness, no-NaN, no-lookahead."""

import numpy as np
import pandas as pd

from trading_engine.discovery.primitives import (
    CANDLESTICK_COLUMNS, FEATURE_COLUMNS, Condition, build_features, _streak,
)
from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.discovery.generate import propose_candidates
from trading_engine.discovery.search import run_candidate_window


def _bars(n=320, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(0.0008, 0.025, n))
    return pd.DataFrame({"Open": price * (1 + rng.normal(0, 0.003, n)),
                         "High": price * 1.015, "Low": price * 0.985,
                         "Close": price, "Volume": rng.uniform(1e6, 3e6, n)}, index=idx)


# ── new columns present + grammar-valid ─────────────────────────────────────
def test_new_columns_in_features_and_grammar():
    feats = build_features(_bars())
    for col in ("roc_5", "roc_10", "up_streak", "down_streak", "prior_high_55",
                "prior_low_55", "sma20_slope", *CANDLESTICK_COLUMNS):
        assert col in feats.columns
        assert col in FEATURE_COLUMNS          # agent/grammar may reference it
        Condition(col, ">", 0)                 # validates against FEATURE_COLUMNS


def test_candlestick_columns_are_ternary_no_nan():
    feats = build_features(_bars())
    for col in CANDLESTICK_COLUMNS:
        vals = set(feats[col].unique())
        assert vals.issubset({-1, 0, 1})
        assert not feats[col].isna().any()     # 0-filled, never NaN


def test_streak_helper_counts_runs():
    s = pd.Series([1, 2, 3, 2, 4, 5, 6])       # up,up,down,up,up,up
    up = _streak(s.diff() > 0)
    assert list(up) == [0, 1, 2, 0, 1, 2, 3]


def test_streaks_nonnegative_ints():
    feats = build_features(_bars())
    assert (feats["up_streak"] >= 0).all()
    assert (feats["down_streak"] >= 0).all()
    # an up day and a down day are mutually exclusive
    assert ((feats["up_streak"] > 0) & (feats["down_streak"] > 0)).sum() == 0


# ── no lookahead: a row depends only on data up to and including that day ────
def test_no_lookahead_prefix_stability():
    df = _bars(n=300)
    full = build_features(df)
    cutoff = df.index[250]
    prefix = build_features(df.loc[:cutoff])
    # roc/streak/donchian/slope at the cutoff must match whether or not future bars exist
    for col in ("roc_5", "roc_10", "up_streak", "down_streak",
                "prior_high_55", "prior_low_55", "sma20_slope", *CANDLESTICK_COLUMNS):
        a, b = full.loc[cutoff, col], prefix.loc[cutoff, col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"{col} changed when future bars were added (lookahead)"


# ── a pattern candidate compiles and backtests through the leaf ─────────────
def test_pattern_candidate_backtests():
    df = _bars(n=300, seed=3)
    data = {"BTCUSDT": df}
    feats = {"BTCUSDT": build_features(df)}
    spec = CandidateSpec(
        id="p", name="p",
        entry=[Condition("bullish_engulfing", ">", 0)],
        exit=ExitSpec(sl_pct=0.07, max_hold_days=10,
                      exit_conditions=[Condition("shooting_star", "<", 0)]),
    )
    res, days = run_candidate_window(spec, data, feats,
                                     df.index[60], df.index[-1])
    assert days > 0 and res.final_value > 0


def test_generator_proposes_pattern_template_and_is_deterministic():
    a = propose_candidates(seed=0, n=12)
    b = propose_candidates(seed=0, n=12)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]   # determinism
    assert a[2].id == "tmpl_engulfing"                            # pattern template present
    # at least one invented candidate references a new-family column
    new_cols = {"bullish_engulfing", "hammer", "morning_star", "up_streak",
                "down_streak", "prior_high_55", "roc_5", "sma20_slope"}
    used = {c.lhs for cand in a for c in cand.entry}
    assert used & new_cols
