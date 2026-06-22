"""S32 — regime-selector tracker: bull→BTC, bear→Connors, hysteresis, switch cost."""

from trading_engine.paper.config import COST_PCT, STARTING_CAPITAL
from trading_engine.paper.selector_track import (
    RegimeSelector, step, serialize, deserialize)


def test_starts_in_bear_tracking_connors():
    s = RegimeSelector()
    # BTC below band → bear leg → compounds the Connors return
    v = step(s, btc_close=100.0, btc_sma=110.0, connors_daily_return=0.05, date="d1")
    assert s.mode == "bear"
    assert abs(v - STARTING_CAPITAL * 1.05) < 1e-9       # tracked Connors +5%


def test_bear_leg_mirrors_connors_returns():
    s = RegimeSelector()
    for r in (0.02, -0.01, 0.03):
        step(s, 100.0, 110.0, r, "d")
    assert abs(s.value - STARTING_CAPITAL * 1.02 * 0.99 * 1.03) < 1e-6
    assert s.n_switches == 0                             # never left the bear leg


def test_switches_to_btc_in_bull():
    s = RegimeSelector()
    step(s, 100.0, 110.0, 0.0, "d1")                     # bear
    v = step(s, 120.0, 100.0, 0.0, "d2")                 # 120 > 100*1.02 → bull, buy BTC
    assert s.mode == "bull" and s.shares > 0
    assert s.n_switches == 1
    assert abs(v - s.value) < 1e-9


def test_bull_leg_tracks_btc_not_connors():
    s = RegimeSelector()
    step(s, 120.0, 100.0, 0.0, "d1")                     # enter BTC at 120
    v_up = step(s, 150.0, 100.0, 0.99, "d2")             # BTC +25%; Connors return ignored in bull
    assert v_up > STARTING_CAPITAL                        # tracked BTC, not the +99% connors
    assert abs(v_up - s.shares * 150.0) < 1e-9


def test_hysteresis_holds_btc_through_minor_dip():
    s = RegimeSelector()
    step(s, 120.0, 100.0, 0.0, "d1")                     # bull
    step(s, 99.0, 100.0, 0.0, "d2")                      # 1% below SMA — inside band → stay BTC
    assert s.mode == "bull"
    step(s, 97.0, 100.0, 0.0, "d3")                      # >2% below → back to Connors
    assert s.mode == "bear" and s.n_switches == 2


def test_none_sma_is_bear():
    s = RegimeSelector()
    step(s, 100.0, None, 0.01, "d1")
    assert s.mode == "bear"


def test_round_trip_serialization():
    s = RegimeSelector()
    step(s, 120.0, 100.0, 0.0, "d1")
    assert deserialize(serialize(s)) == s
    assert deserialize({}) == RegimeSelector()
