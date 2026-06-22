"""S29/S30 — trend-timed BTC tracker: bull→invested, bear→cash, hysteresis band, costs."""

from trading_engine.paper.config import COST_PCT, STARTING_CAPITAL
from trading_engine.paper.trend_timed import (
    TrendTimer, step, value_of, serialize, deserialize, DEFAULT_BAND,
)


def test_starts_in_cash_and_initializes():
    t = TrendTimer()
    # below SMA on day 1 → stays cash
    v = step(t, close=100.0, sma=110.0, date="2026-01-01")
    assert t.initialized and not t.invested
    assert t.shares == 0.0
    assert abs(v - STARTING_CAPITAL) < 1e-9


def test_bull_goes_fully_invested_with_cost():
    t = TrendTimer()
    v = step(t, close=100.0, sma=90.0, date="2026-01-02")   # well above band → bull
    assert t.invested and t.cash == 0.0
    assert t.shares == STARTING_CAPITAL * (1 - COST_PCT) / 100.0
    assert t.n_switches == 1
    assert abs(v - STARTING_CAPITAL * (1 - COST_PCT)) < 1e-9   # paid entry cost


# ── S30 hysteresis band ──────────────────────────────────────────────────────
def test_band_blocks_entry_just_above_sma():
    t = TrendTimer(band_pct=0.02)
    # close 1% above SMA — inside the +2% band → must NOT enter
    step(t, close=101.0, sma=100.0, date="d1")
    assert not t.invested
    # close 3% above SMA — clears the band → enters
    step(t, close=103.0, sma=100.0, date="d2")
    assert t.invested


def test_band_holds_through_minor_dip_below_sma():
    t = TrendTimer(band_pct=0.02)
    step(t, close=110.0, sma=100.0, date="d1")          # enter
    assert t.invested
    step(t, close=99.0, sma=100.0, date="d2")           # 1% below SMA — inside band → HOLD
    assert t.invested
    step(t, close=97.0, sma=100.0, date="d3")           # >2% below → exit
    assert not t.invested


def test_band_suppresses_whipsaw_switch_count():
    """Oscillating within the band must not rack up switches."""
    t = TrendTimer(band_pct=0.02)
    step(t, 103.0, 100.0, "d1")    # enter (1 switch)
    for i, px in enumerate([99.5, 100.5, 99.0, 101.0]):   # all within ±2% of SMA 100
        step(t, px, 100.0, f"x{i}")
    assert t.invested and t.n_switches == 1   # held throughout, no churn


def test_stays_invested_through_bull_no_double_switch():
    t = TrendTimer()
    step(t, 100.0, 90.0, "2026-01-02")        # enter
    shares = t.shares
    step(t, 120.0, 95.0, "2026-01-03")        # still bull — hold, no new switch
    step(t, 130.0, 100.0, "2026-01-04")       # still bull
    assert t.invested and t.shares == shares  # untouched
    assert t.n_switches == 1                   # only the initial entry


def test_bear_exits_to_cash_with_cost():
    t = TrendTimer()
    step(t, 100.0, 90.0, "2026-01-02")        # enter at 100
    step(t, 80.0, 90.0, "2026-01-05")         # close < sma → exit
    assert not t.invested and t.shares == 0.0
    assert t.n_switches == 2
    # value marked at the exit close, after exit cost
    assert t.cash > 0


def test_value_tracks_btc_while_invested():
    t = TrendTimer()
    step(t, 100.0, 90.0, "2026-01-02")        # invested
    v_up = value_of(t, 150.0)                  # BTC rallies 50%
    assert v_up > STARTING_CAPITAL             # gains captured
    v_dn = value_of(t, 50.0)
    assert v_dn < STARTING_CAPITAL


def test_none_sma_treated_as_not_bull():
    t = TrendTimer()
    v = step(t, 100.0, None, "2026-01-01")     # insufficient history
    assert not t.invested
    assert abs(v - STARTING_CAPITAL) < 1e-9


def test_round_trip_serialization():
    t = TrendTimer()
    step(t, 100.0, 90.0, "2026-01-02")
    t2 = deserialize(serialize(t))
    assert t2 == t


def test_deserialize_empty_is_fresh():
    assert deserialize({}) == TrendTimer()


def test_full_cycle_protects_capital():
    """Enter a bull, ride it up, exit on the break — net positive, then cash holds flat."""
    t = TrendTimer()
    step(t, 100.0, 95.0, "d1")   # enter
    step(t, 200.0, 150.0, "d2")  # ride to 200 (bull)
    v_exit = step(t, 180.0, 190.0, "d3")  # close<sma → exit near top
    assert not t.invested
    # locked in ~80% gain (minus 2 costs), now in cash
    assert v_exit > 1.5 * STARTING_CAPITAL
    v_later = step(t, 50.0, 190.0, "d4")  # BTC craters but we're in cash
    assert abs(v_later - v_exit) < 1e-9   # cash unaffected by the crash
