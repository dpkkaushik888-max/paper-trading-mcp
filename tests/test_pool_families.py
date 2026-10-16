"""S35 C2 families (pure pandas), bucket functions and registry status."""
import pandas as pd
import pytest

from trading_engine.paper.pool import book as pool_book
from trading_engine.paper.pool.context import (BASKET_RET20_LABELS, BTC_VOL20_TERCILES, DayContext,
                                               ret20_bucket, vol20_bucket)
from trading_engine.paper.pool.families import make_strategy
from trading_engine.paper.pool.families import breakout as bo
from trading_engine.paper.pool.families.basket_trend import BasketTrendStrategy
from trading_engine.paper.pool.families.connors import ConnorsStrategy
from trading_engine.paper.pool.families.xsec_momentum import XsecMomentumStrategy
from trading_engine.paper.pool.registry import LINES, BookParams, Hypothesis, LineSpec, active_lines, validate

SPECS = {l.id: l for l in LINES}


def _pos(sym="AUSDT", price=100.0, date="2026-01-01"):
    return dict(symbol=sym, entry_date=date, entry_price=price, shares=10.0, entry_cost=2.0)


def test_breakout_constants_match_s21_file():
    from trading_engine.strategies import breakout_continuation as s21
    assert (bo.SL_PCT, bo.MAX_HOLD_DAYS) == (s21.SL_PCT, s21.MAX_HOLD_DAYS)


BO_ROW = dict(prior_high_20=100.0, vol_sma_20=10.0, volume=16.0, adx_14=25.0, sma_50=90.0, sma_10=95.0)


@pytest.mark.parametrize("change,fires", [({}, True), ({"volume": 15.0}, False), ({"adx_14": 24.9}, False),
                                          ({"sma_50": 101.0}, False), ({"prior_high_20": 101.0}, False),
                                          ({"adx_14": float("nan")}, False)])
def test_breakout_entry(change, fires):
    sig = bo.BreakoutStrategy({}).entry("A", 101.0, {**BO_ROW, **change}, DayContext())
    assert (sig is not None) == fires
    if fires:
        assert sig.strength == pytest.approx(0.01)


@pytest.mark.parametrize("close,today,reason", [(92.0, "2026-01-02", "SL"), (94.0, "2026-01-02", "MA_BREAK"),
                                                (96.0, "2026-01-16", "MAX_HOLD"), (96.0, "2026-01-15", None)])
def test_breakout_exit(close, today, reason):
    assert bo.BreakoutStrategy({}).exit(_pos(), close, BO_ROW, DayContext(), today) == reason


def test_breakout_indicators_prior_high_excludes_today():
    df = pd.DataFrame({"Close": range(1, 31), "High": range(2, 32), "Low": range(0, 30), "Volume": [1.0] * 30},
                      index=pd.date_range("2026-01-01", periods=30), dtype=float)
    ind = bo.BreakoutStrategy({}).indicators(df)
    assert ind["prior_high_20"].iloc[-1] == 29.0 and ind["sma_10"].iloc[-1] == pytest.approx(25.5)


def _mom_snap():
    r = lambda ret, sma=50.0: {"sma_100": sma, "ret_28": ret}  # noqa: E731
    return {"AUSDT": (100.0, r(0.5)), "BUSDT": (100.0, r(0.4)), "CUSDT": (100.0, r(0.3)),
            "DUSDT": (100.0, r(0.2)), "EUSDT": (100.0, r(0.9, sma=150.0))}   # E below SMA100: ineligible


def test_xsec_monday_only_rotation():
    spec = SPECS["xsec_mom_top3"]
    strat = make_strategy(spec)
    b = pool_book.Book(name=spec.id, initialized=True, cash=9_000.0, open_positions=[_pos("DUSDT")])
    tue = pool_book.step(b, spec, strat, _mom_snap(), DayContext(dow="Tuesday"), "2026-01-06")
    assert tue["entries"] == tue["exits"] == 0 and [p["symbol"] for p in b.open_positions] == ["DUSDT"]
    mon = pool_book.step(b, spec, strat, _mom_snap(), DayContext(dow="Monday"), "2026-01-12")
    assert [t["reason"] for t in b.closed_trades] == ["ROTATE"] and mon["entries"] == 3
    assert sorted(p["symbol"] for p in b.open_positions) == ["AUSDT", "BUSDT", "CUSDT"]
    assert b.open_positions[0]["entry_ctx"]["rank_today"] == 0 and b.open_positions[0]["symbol"] == "AUSDT"


def test_xsec_exit_below_sma100_on_monday():
    s = XsecMomentumStrategy({})
    snap = _mom_snap()
    s.prepare(snap, DayContext(dow="Monday"), "2026-01-12")
    assert s.exit(_pos("AUSDT"), 40.0, snap["AUSDT"][1], None, "2026-01-12") == "ROTATE"
    assert s.exit(_pos("AUSDT"), 100.0, snap["AUSDT"][1], None, "2026-01-12") is None


def test_basket_band_needs_two_closes():
    c = [100.0] * 60 + [97.0, 99.5, 97.0, 97.0]
    df = pd.DataFrame({"Close": c}, index=pd.date_range("2026-01-01", periods=len(c)))
    ind = BasketTrendStrategy({"band": 0.98}).indicators(df)
    assert ind["band_exit"].tolist()[-4:] == [False, False, False, True]
    s = BasketTrendStrategy({})
    assert s.exit(_pos(), 97.0, ind.iloc[-1], None, "2026-03-05") == "TREND_EXIT"
    assert s.entry("A", 101.0, {"sma_50": 100.0}, None).strength == pytest.approx(0.01)
    assert s.entry("A", 100.0, {"sma_50": 100.0}, None) is None


def test_connors_exit_sma10_and_guard():
    row = pd.Series({"sma_200": 50.0, "sma_5": 99.0, "sma_10": 105.0, "rsi_2": 50.0, "adx_14": 30.0})
    live, sma10 = ConnorsStrategy({}), make_strategy(SPECS["connors_exit_sma10"])
    assert live.exit(_pos(), 100.0, row, None, "2026-01-02") == "MR_EXIT"
    assert sma10.exit(_pos(), 100.0, row, None, "2026-01-02") is None
    assert sma10.exit(_pos(), 106.0, row, None, "2026-01-02") == "MR_EXIT"
    assert row["sma_5"] == 99.0   # input row untouched
    assert sma10.exit(_pos(), 90.0, row, None, "2026-01-02") == "SL"   # stop unchanged (7 %)
    g = SPECS["connors_guarded"]
    assert g.book.data_guard and dict(g.params) == {} and SPECS["connors_exit_sma10"].book.data_guard
    b = pool_book.Book(name=g.id, initialized=True, cash=10_000.0)
    snap = {"TRXUSDT": (100.0, pd.Series({"sma_200": 50.0, "sma_5": 110.0, "rsi_2": 5.0, "adx_14": 30.0}))}
    day = pool_book.step(b, g, make_strategy(g), snap, DayContext(data_flags={"stale": [], "gap200": ["TRXUSDT"]}),
                         "2026-01-02")
    assert day["blocked"]["data"] == 1 and b.open_positions == []


@pytest.mark.parametrize("v,b", [(None, None), (BTC_VOL20_TERCILES[0] - 1e-9, "low"), (BTC_VOL20_TERCILES[0], "mid"),
                                 (BTC_VOL20_TERCILES[1] - 1e-9, "mid"), (BTC_VOL20_TERCILES[1], "high")])
def test_vol20_bucket_boundaries(v, b):
    assert vol20_bucket(v) == b


@pytest.mark.parametrize("r,i", [(-0.1000001, 0), (-0.10, 1), (-1e-9, 1), (0.0, 2), (0.10, 2), (0.1000001, 3)])
def test_ret20_bucket_boundaries(r, i):
    assert ret20_bucket(r) == BASKET_RET20_LABELS[i] and ret20_bucket(None) is None


def test_registry_c2_lines_and_candidate_status():
    assert {"breakout_s21", "xsec_mom_top3", "basket_sma50_band", "connors_guarded", "connors_exit_sma10"} <= set(SPECS)
    assert SPECS["xsec_mom_top3"].book == BookParams(pos_size_pct=0.30, max_concurrent=3, catastrophe_sl=0.25)
    assert SPECS["basket_sma50_band"].hypothesis == Hypothesis("reduces_dd_in", "below_both")
    assert SPECS["connors_guarded"].status == "candidate"
    assert "connors_guarded" not in {l.id for l in active_lines()}
    c = LineSpec("cand", "connors", {}, registered_on="2026-10-16", status="candidate")   # no baseline needed
    validate((c,))
    assert active_lines((c,)) == ()
