"""Strategy 2: Candlestick + Support/Resistance — Conservative daily signals.

Trades when a recognized candlestick pattern forms at a support/resistance level.
Higher frequency than ML Sniper, with tighter stops for conservative gains.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from trading_engine.chart_patterns import (
    find_support_resistance,
    fibonacci_levels,
    get_candlestick_signal,
)
from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal


class CandlestickSRStrategy(BaseStrategy):
    """Candlestick patterns at support/resistance levels."""

    def __init__(
        self,
        config: StrategyConfig,
        sr_proximity_pct: float = 0.02,
        sl_pct: float = 0.03,
        tp_pct: float = 0.05,
        min_bars: int = 60,
        sr_window: int = 10,
    ):
        super().__init__(config)
        self.sr_proximity_pct = sr_proximity_pct
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.min_bars = min_bars
        self.sr_window = sr_window

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
        open_ = temporal_df["Open"]
        price = float(close.iloc[-1])

        candle = get_candlestick_signal(open_, high, low, close)
        if candle["direction"] is None:
            return None

        sr = find_support_resistance(
            high, low, close, window=self.sr_window
        )

        near_support = False
        near_resistance = False
        sr_strength = 0

        ns = sr.get("nearest_support")
        if ns:
            dist_to_support = (price - ns["price"]) / price
            if dist_to_support <= self.sr_proximity_pct:
                near_support = True
                sr_strength = max(sr_strength, ns["strength"])

        nr = sr.get("nearest_resistance")
        if nr:
            dist_to_resistance = (nr["price"] - price) / price
            if dist_to_resistance <= self.sr_proximity_pct:
                near_resistance = True
                sr_strength = max(sr_strength, nr["strength"])

        near_fib = self._check_fibonacci_proximity(high, low, price)

        at_level = near_support or near_resistance or near_fib

        if candle["direction"] == "bullish" and (near_support or near_fib):
            confidence = self._compute_confidence(
                candle["strength"], sr_strength, near_fib
            )
            return StrategySignal(
                direction="long",
                confidence=confidence,
                strategy=self.name,
                reason=f"Candlestick {candle['pattern']} at support"
                       f" (SR str={sr_strength}, fib={near_fib})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
                size_pct=self.config.max_position_pct,
            )

        elif candle["direction"] == "bearish" and (near_resistance or near_fib):
            confidence = self._compute_confidence(
                candle["strength"], sr_strength, near_fib
            )
            return StrategySignal(
                direction="short",
                confidence=confidence,
                strategy=self.name,
                reason=f"Candlestick {candle['pattern']} at resistance"
                       f" (SR str={sr_strength}, fib={near_fib})",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
                size_pct=self.config.max_position_pct,
            )

        return None

    def _check_fibonacci_proximity(
        self, high: pd.Series, low: pd.Series, price: float,
    ) -> bool:
        """Check if price is near a Fibonacci retracement level."""
        if len(high) < 50:
            return False

        fib = fibonacci_levels(high, low, lookback=50)
        if not fib:
            return False

        fib_levels = [
            fib.get("fib_236"), fib.get("fib_382"),
            fib.get("fib_500"), fib.get("fib_618"),
        ]

        for level in fib_levels:
            if level is not None:
                dist = abs(price - level) / price
                if dist <= self.sr_proximity_pct:
                    return True
        return False

    def _compute_confidence(
        self, candle_strength: int, sr_strength: int, near_fib: bool,
    ) -> float:
        """Compute composite confidence from pattern + level strength.

        Base: 0.55
        +0.05 per additional candlestick pattern on same bar
        +0.05 per S/R touch count above 1
        +0.10 if near Fibonacci level
        Cap: 0.90
        """
        conf = 0.55
        conf += min(0.10, (candle_strength - 1) * 0.05)
        conf += min(0.10, max(0, sr_strength - 1) * 0.05)
        if near_fib:
            conf += 0.10
        return min(0.90, conf)
