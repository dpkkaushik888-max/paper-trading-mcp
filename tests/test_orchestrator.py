"""S23 Stage 2 — StrategyOrchestrator selection, caps, conflict, regime gating."""

import pandas as pd
import pytest

from trading_engine.engine.orchestrator import (
    Order, PortfolioView, StrategyOrchestrator, DEFAULT_REGIME_MAP,
)
from trading_engine.regime.regime_filter import RegimeState
from trading_engine.strategies.connors_strategy import ConnorsSwingStrategy, default_config as a_cfg
from trading_engine.strategies.breakout_continuation import BreakoutContinuationStrategy, default_config as b_cfg
from trading_engine.strategies.range_meanrev import RangeMeanRevStrategy, default_config as c_cfg

DAY = pd.Timestamp("2026-06-22")


def connors_fire_row():
    # close=100 fires Connors: close>sma200, rsi2<10, close<sma5, adx>=20
    return pd.Series(dict(sma_200=90.0, sma_5=101.0, rsi_2=5.0, adx_14=25.0,
                          prior_high_20=999.0, vol_sma_20=1.0, volume=1.0, sma_50=999.0,
                          sma_10=0.0, bb_width=1.0))


def breakout_fire_row():
    return pd.Series(dict(sma_200=999.0, sma_5=0.0, rsi_2=50.0, adx_14=30.0,
                          prior_high_20=100.0, vol_sma_20=1000.0, volume=2000.0, sma_50=95.0,
                          sma_10=90.0, bb_width=1.0))


def stack():
    return StrategyOrchestrator(
        strategies=[ConnorsSwingStrategy(a_cfg()),
                    BreakoutContinuationStrategy(b_cfg()),
                    RangeMeanRevStrategy(c_cfg())],
        regime_filter=None, regime_gating=False,
        global_max_concurrent=8, per_strategy_cap=4, base_pos_size_pct=0.12,
    )


def test_entries_fire_when_rule_met():
    orch = stack()
    snap = {"AAA": (100.0, connors_fire_row()), "BBB": (100.0, connors_fire_row())}
    view = PortfolioView(cash=10000.0, equity=10000.0, positions={})
    _, orders = orch.step(view, snap, None, DAY)
    enters = [o for o in orders if o.action == "enter"]
    assert {o.symbol for o in enters} == {"AAA", "BBB"}
    assert all(o.strategy == "A_connors" and o.size_pct == pytest.approx(0.12) for o in enters)


def test_global_cap_respected():
    orch = stack()
    orch.global_max = 2
    snap = {s: (100.0, connors_fire_row()) for s in ["AAA", "BBB", "CCC", "DDD"]}
    view = PortfolioView(10000.0, 10000.0, {})
    _, orders = orch.step(view, snap, None, DAY)
    assert sum(o.action == "enter" for o in orders) == 2


def test_per_strategy_cap_respected():
    orch = stack()
    orch.per_strategy_cap = 1
    snap = {s: (100.0, connors_fire_row()) for s in ["AAA", "BBB", "CCC"]}
    view = PortfolioView(10000.0, 10000.0, {})
    _, orders = orch.step(view, snap, None, DAY)
    assert sum(o.action == "enter" for o in orders) == 1  # only 1 Connors slot


def test_conflict_symbol_already_held_skipped():
    orch = stack()
    snap = {"AAA": (100.0, connors_fire_row())}
    view = PortfolioView(10000.0, 10000.0,
                         {"AAA": {"strategy": "A_connors", "entry_price": 90.0,
                                  "entry_date": DAY, "shares": 1.0}})
    _, orders = orch.step(view, snap, None, DAY)
    assert not any(o.action == "enter" for o in orders)


def test_priority_a_beats_b_on_same_symbol():
    # symbol fires BOTH connors and breakout; A should claim it (priority 0)
    orch = stack()
    r = connors_fire_row()
    # also satisfy breakout on same row
    r["prior_high_20"] = 99.0; r["vol_sma_20"] = 1.0; r["volume"] = 100.0
    r["adx_14"] = 30.0; r["sma_50"] = 95.0
    snap = {"AAA": (100.0, r)}
    view = PortfolioView(10000.0, 10000.0, {})
    _, orders = orch.step(view, snap, None, DAY)
    enters = [o for o in orders if o.action == "enter"]
    assert len(enters) == 1 and enters[0].strategy == "A_connors"


def test_exit_order_emitted():
    orch = stack()
    # breakout position, close 99 below sma_10 100 → MA_BREAK
    r = breakout_fire_row()
    r["sma_10"] = 100.0
    snap = {"AAA": (99.0, r)}
    view = PortfolioView(10000.0, 10000.0,
                         {"AAA": {"strategy": "B_breakout", "entry_price": 100.0,
                                  "entry_date": DAY, "shares": 1.0}})
    _, orders = orch.step(view, snap, None, DAY)
    exits = [o for o in orders if o.action == "exit"]
    assert len(exits) == 1 and exits[0].reason == "MA_BREAK"


def test_regime_gating_off_by_default_all_weight_one():
    orch = stack()
    _, policy = orch.regime_policy(None, DAY)
    assert all(w == 1.0 for w in policy.weights.values())
    assert all(policy.allow_new.values())


def test_weight_advisor_overrides_when_valid():
    # Agent (advisor) zeroes out breakout → breakout blocked; others normal.
    def advisor(rr, names):
        return {"A_connors": 1.0, "B_breakout": 0.0, "C_range": 1.0}

    orch = stack()
    orch.weight_advisor = advisor
    _, policy = orch.regime_policy(None, DAY)
    assert policy.weights["B_breakout"] == 0.0
    assert policy.allow_new["B_breakout"] is False
    assert policy.allow_new["A_connors"] is True


def test_weight_advisor_rejected_on_bad_keys_falls_back():
    def advisor(rr, names):
        return {"A_connors": 1.0}  # missing keys → invalid

    orch = stack()
    orch.weight_advisor = advisor
    _, policy = orch.regime_policy(None, DAY)
    assert all(w == 1.0 for w in policy.weights.values())  # deterministic fallback


def test_weight_advisor_rejected_on_out_of_range():
    def advisor(rr, names):
        return {"A_connors": 5.0, "B_breakout": 1.0, "C_range": 1.0}  # 5 > max 2

    orch = stack()
    orch.weight_advisor = advisor
    _, policy = orch.regime_policy(None, DAY)
    assert all(w == 1.0 for w in policy.weights.values())


def test_weight_advisor_none_returns_deterministic():
    def advisor(rr, names):
        return None  # agent unavailable / declined

    orch = stack()
    orch.weight_advisor = advisor
    _, policy = orch.regime_policy(None, DAY)
    assert all(w == 1.0 for w in policy.weights.values())


def test_regime_gating_on_bear_blocks_connors_and_breakout():
    orch = stack()
    orch.regime_gating = True

    class FakeRegime:
        def evaluate(self, btc_df, current_day=None):
            from trading_engine.regime.regime_filter import RegimeResult
            return RegimeResult(state=RegimeState.BEAR, confidence=0.9, signals=[])

    orch.regime_filter = FakeRegime()
    btc = pd.DataFrame({"Close": [1, 2, 3]})
    _, policy = orch.regime_policy(btc, DAY)
    assert policy.allow_new["A_connors"] is False
    assert policy.allow_new["B_breakout"] is False
    assert policy.allow_new["C_range"] is True  # range fades still allowed in bear
