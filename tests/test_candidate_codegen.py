"""S25 — candidate codegen reproduces the hand-written strategies exactly."""

import numpy as np
import pandas as pd

from trading_engine.discovery.primitives import Condition, build_features
from trading_engine.discovery.candidate import CandidateSpec, ExitSpec
from trading_engine.engine.indicators import build_indicators
from trading_engine.strategies.connors_strategy import ConnorsSwingStrategy, default_config as a_cfg
from trading_engine.strategies.breakout_continuation import BreakoutContinuationStrategy, default_config as b_cfg


def _bars(n=320, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.cumprod(1 + rng.normal(0.0008, 0.025, n))
    return pd.DataFrame({"Open": price, "High": price * 1.015, "Low": price * 0.985,
                         "Close": price, "Volume": rng.uniform(1e6, 3e6, n)}, index=idx)


def _connors_spec():
    return CandidateSpec(
        id="cand_connors", name="gen_connors",
        entry=[Condition("close", ">", "sma_200"),
               Condition("rsi_2", "<", 10),
               Condition("close", "<", "sma_5"),
               Condition("adx_14", ">=", 20)],
        exit=ExitSpec(sl_pct=0.07, max_hold_days=10,
                      exit_conditions=[Condition("close", ">", "sma_5")]),
    )


def _breakout_spec():
    return CandidateSpec(
        id="cand_breakout", name="gen_breakout",
        entry=[Condition("close", ">", "prior_high_20"),
               Condition("volume", ">", "vol_sma_20", rhs_mult=1.5),
               Condition("adx_14", ">=", 25),
               Condition("close", ">", "sma_50")],
        exit=ExitSpec(sl_pct=0.08, max_hold_days=15,
                      exit_conditions=[Condition("close", "<", "sma_10")]),
    )


def test_generated_connors_matches_handwritten_entries():
    df = _bars()
    gen = _connors_spec().to_strategy()
    hand = ConnorsSwingStrategy(a_cfg())
    feats = build_features(df)
    ind = build_indicators(df)
    mismatches = 0
    for day in df.index[210:]:                      # after SMA(200) warms up
        close = float(df.loc[day, "Close"])
        if gen.entry(close, feats.loc[day]) != hand.entry(close, ind.loc[day]):
            mismatches += 1
    assert mismatches == 0


def test_generated_breakout_matches_handwritten_entries():
    df = _bars(seed=11)
    gen = _breakout_spec().to_strategy()
    hand = BreakoutContinuationStrategy(b_cfg())
    feats = build_features(df)
    ind = build_indicators(df)
    mismatches = sum(
        gen.entry(float(df.loc[d, "Close"]), feats.loc[d]) != hand.entry(float(df.loc[d, "Close"]), ind.loc[d])
        for d in df.index[60:]
    )
    assert mismatches == 0


def test_exit_reasons():
    df = _bars()
    gen = _connors_spec().to_strategy()
    feats = build_features(df)
    day = df.index[250]
    row = feats.loc[day]
    # SL: 10% loss < -7%
    assert gen.exit_reason(90.0, row, 100.0, day, day) == "SL"
    # MAX_HOLD after 10 days when no SL / rule exit
    far = day + pd.Timedelta(days=10)
    # pick a row where close < sma_5 so RULE_EXIT (close>sma_5) does NOT fire
    assert gen.exit_reason(100.0, pd.Series({"sma_5": 200.0}), 100.0, day, far) == "MAX_HOLD"
    # RULE_EXIT: close > sma_5
    assert gen.exit_reason(100.0, pd.Series({"sma_5": 90.0}), 100.0, day, day) == "RULE_EXIT"


def test_take_profit():
    spec = _connors_spec()
    spec.exit.tp_pct = 0.05
    gen = spec.to_strategy()
    assert gen.exit_reason(106.0, pd.Series({"sma_5": 200.0}), 100.0,
                           pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")) == "TP"


def test_spec_round_trip():
    spec = _breakout_spec()
    assert CandidateSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
