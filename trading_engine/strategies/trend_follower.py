"""Strategy 3: Trend Follower — EMA crossover with volume and ADX confirmation.

Trades when EMA 8 crosses EMA 20 with confirming volume spike and trend strength.
Uses trailing stop (2x ATR) for exits. Medium frequency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal


class TrendFollowerStrategy(BaseStrategy):
    """EMA crossover + volume + ADX trend confirmation."""

    def __init__(
        self,
        config: StrategyConfig,
        ema_fast: int = 8,
        ema_slow: int = 20,
        adx_threshold: float = 25.0,
        volume_spike: float = 1.5,
        atr_period: int = 14,
        atr_sl_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        min_bars: int = 40,
    ):
        super().__init__(config)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_threshold = adx_threshold
        self.volume_spike = volume_spike
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.min_bars = min_bars

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_day: pd.Timestamp,
        model=None,
        feature_cols: list | None = None,
        cross_asset_data: dict | None = None,
    ) -> Optional[StrategySignal]:
        if not self.can_open():
            return None

        temporal_df = df[df.index <= current_day]
        if len(temporal_df) < self.min_bars:
            return None

        close = temporal_df["Close"]
        high = temporal_df["High"]
        low = temporal_df["Low"]
        volume = temporal_df["Volume"]

        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

        cross_up = (ema_fast.iloc[-1] > ema_slow.iloc[-1] and
                    ema_fast.iloc[-2] <= ema_slow.iloc[-2])
        cross_down = (ema_fast.iloc[-1] < ema_slow.iloc[-1] and
                      ema_fast.iloc[-2] >= ema_slow.iloc[-2])

        if not cross_up and not cross_down:
            return None

        adx = self._compute_adx(high, low, close)
        if adx < self.adx_threshold:
            return None

        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        if avg_volume > 0 and current_volume < avg_volume * self.volume_spike:
            return None

        atr_pct = self._compute_atr_pct(high, low, close)
        sl = max(0.02, min(0.08, self.atr_sl_mult * atr_pct))
        tp = max(0.03, min(0.12, self.atr_tp_mult * atr_pct))

        if cross_up:
            confidence = self._compute_confidence(adx, current_volume, avg_volume)
            return StrategySignal(
                direction="long",
                confidence=confidence,
                strategy=self.name,
                reason=f"EMA {self.ema_fast}/{self.ema_slow} bullish cross"
                       f" (ADX={adx:.0f}, vol={current_volume/avg_volume:.1f}x)",
                sl_pct=sl,
                tp_pct=tp,
                size_pct=self.config.max_position_pct,
                trailing_stop=True,
            )
        elif cross_down:
            confidence = self._compute_confidence(adx, current_volume, avg_volume)
            return StrategySignal(
                direction="short",
                confidence=confidence,
                strategy=self.name,
                reason=f"EMA {self.ema_fast}/{self.ema_slow} bearish cross"
                       f" (ADX={adx:.0f}, vol={current_volume/avg_volume:.1f}x)",
                sl_pct=sl,
                tp_pct=tp,
                size_pct=self.config.max_position_pct,
                trailing_stop=True,
            )

        return None

    def _compute_adx(self, high: pd.Series, low: pd.Series,
                     close: pd.Series, period: int = 14) -> float:
        """Compute Average Directional Index (ADX)."""
        if len(close) < period + 1:
            return 0.0

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() /
                         atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() /
                          atr.replace(0, np.nan))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()

        val = adx.iloc[-1]
        return float(val) if pd.notna(val) else 0.0

    def _compute_atr_pct(self, high: pd.Series, low: pd.Series,
                         close: pd.Series) -> float:
        """Compute ATR as percentage of price."""
        if len(close) < self.atr_period + 1:
            return 0.03

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.tail(self.atr_period).mean()
        price = close.iloc[-1]
        return float(atr / price) if price > 0 else 0.03

    def _compute_confidence(self, adx: float, current_vol: float,
                            avg_vol: float) -> float:
        """Compute confidence from ADX strength and volume spike.

        Base: 0.60
        +0.05 per 5 ADX points above 25
        +0.05 per 0.5x volume above 1.5x
        Cap: 0.85
        """
        conf = 0.60
        conf += min(0.10, max(0, (adx - 25)) / 5 * 0.05)
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            conf += min(0.10, max(0, (vol_ratio - 1.5)) / 0.5 * 0.05)
        return min(0.85, conf)
