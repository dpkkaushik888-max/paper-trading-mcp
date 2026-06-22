"""S23 Stage 2 — strategy wrappers A/B/C entry & exit rules."""

import pandas as pd
import pytest

from trading_engine.strategies.connors_strategy import ConnorsSwingStrategy, default_config as a_cfg
from trading_engine.strategies.breakout_continuation import (
    BreakoutContinuationStrategy, default_config as b_cfg, SL_PCT as B_SL, MAX_HOLD_DAYS as B_HOLD,
)
from trading_engine.strategies.range_meanrev import (
    RangeMeanRevStrategy, default_config as c_cfg, SL_PCT as C_SL, MAX_HOLD_DAYS as C_HOLD,
)

DAY = pd.Timestamp("2026-06-22")


def row(**kw):
    return pd.Series(kw)


class TestBreakout:
    def setup_method(self):
        self.s = BreakoutContinuationStrategy(b_cfg())

    def test_entry_fires_on_new_high_with_volume(self):
        r = row(prior_high_20=100.0, vol_sma_20=1000.0, volume=2000.0, adx_14=30.0, sma_50=95.0)
        assert self.s.entry(101.0, r) is True

    def test_entry_blocked_without_volume(self):
        r = row(prior_high_20=100.0, vol_sma_20=1000.0, volume=1200.0, adx_14=30.0, sma_50=95.0)
        assert self.s.entry(101.0, r) is False  # 1200 < 1.5*1000

    def test_entry_blocked_weak_adx(self):
        r = row(prior_high_20=100.0, vol_sma_20=1000.0, volume=2000.0, adx_14=20.0, sma_50=95.0)
        assert self.s.entry(101.0, r) is False

    def test_entry_blocked_on_nan(self):
        r = row(prior_high_20=float("nan"), vol_sma_20=1000.0, volume=2000.0, adx_14=30.0, sma_50=95.0)
        assert self.s.entry(101.0, r) is False

    def test_exit_stop_loss(self):
        r = row(sma_10=100.0)
        assert self.s.exit_reason(91.0, r, 100.0, DAY, DAY) == "SL"  # -9% < -8%

    def test_exit_ma_break(self):
        r = row(sma_10=100.0)
        assert self.s.exit_reason(99.0, r, 100.0, DAY, DAY) == "MA_BREAK"

    def test_exit_max_hold(self):
        r = row(sma_10=90.0)  # above MA, no SL
        assert self.s.exit_reason(100.0, r, 100.0, DAY, DAY + pd.Timedelta(days=B_HOLD)) == "MAX_HOLD"

    def test_no_exit(self):
        r = row(sma_10=90.0)
        assert self.s.exit_reason(100.0, r, 100.0, DAY, DAY + pd.Timedelta(days=3)) is None


class TestRange:
    def setup_method(self):
        self.s = RangeMeanRevStrategy(c_cfg())

    def test_entry_fires_deep_oversold_low_trend_tight_range(self):
        r = row(rsi_2=3.0, adx_14=12.0, bb_width=0.05)
        assert self.s.entry(50.0, r) is True

    def test_entry_blocked_high_adx(self):
        r = row(rsi_2=3.0, adx_14=25.0, bb_width=0.05)
        assert self.s.entry(50.0, r) is False

    def test_entry_blocked_wide_bands(self):
        r = row(rsi_2=3.0, adx_14=12.0, bb_width=0.20)
        assert self.s.entry(50.0, r) is False

    def test_exit_mr_complete(self):
        r = row(rsi_2=75.0)
        assert self.s.exit_reason(52.0, r, 50.0, DAY, DAY) == "MR_EXIT"

    def test_exit_stop_loss(self):
        r = row(rsi_2=40.0)
        assert self.s.exit_reason(47.0, r, 50.0, DAY, DAY) == "SL"  # -6% < -5%

    def test_exit_max_hold(self):
        r = row(rsi_2=40.0)
        assert self.s.exit_reason(50.0, r, 50.0, DAY, DAY + pd.Timedelta(days=C_HOLD)) == "MAX_HOLD"


class TestConnorsWrapper:
    def test_delegates_to_connors_swing(self):
        import trading_engine.strategies.connors_swing as cs
        s = ConnorsSwingStrategy(a_cfg())
        # entry true case: close>sma200, rsi2<10, close<sma5, adx>=20
        r = row(sma_200=90.0, sma_5=101.0, rsi_2=5.0, adx_14=25.0)
        assert s.entry(100.0, r) == cs.long_entry(100.0, r, use_adx_filter=True) is True
        assert s.sl_pct == cs.SL_PCT


def test_priorities_are_a_b_c():
    assert ConnorsSwingStrategy(a_cfg()).priority == 0
    assert BreakoutContinuationStrategy(b_cfg()).priority == 1
    assert RangeMeanRevStrategy(c_cfg()).priority == 2
