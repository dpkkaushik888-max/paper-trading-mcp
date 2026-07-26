"""Simple rule-based regime filter for crypto markets.

Classifies the market into BULL / NEUTRAL / BEAR using:
  1. BTC price vs SMA(50) — from price data (no API needed)
  2. Fear & Greed Index — from alternative.me free API (optional)

No ML, no complex models. Just three signals combined with majority vote.
Designed for crypto but extensible to other markets.

Usage:
    from trading_engine.regime import RegimeFilter
    rf = RegimeFilter()
    state = rf.evaluate(btc_df)  # returns RegimeState.BULL / NEUTRAL / BEAR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeState(str, Enum):
    """Market regime classification."""
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"


@dataclass
class RegimeSignal:
    """Individual regime signal with source and value."""
    name: str
    value: float
    signal: RegimeState
    description: str


@dataclass
class RegimeResult:
    """Composite regime classification result."""
    state: RegimeState
    confidence: float
    signals: list[RegimeSignal]
    timestamp: Optional[str] = None

    @property
    def allows_new_longs(self) -> bool:
        return self.state != RegimeState.BEAR

    @property
    def allows_new_shorts(self) -> bool:
        return self.state != RegimeState.BULL

    @property
    def is_caution(self) -> bool:
        return self.state == RegimeState.NEUTRAL or self.confidence < 0.6


class RegimeFilter:
    """Rule-based regime filter — no ML, just price + sentiment signals.

    Signals:
      1. btc_vs_sma50: BTC above SMA(50) → BULL, below → BEAR
      2. btc_momentum: 20d return > 0 → BULL, < -10% → BEAR
      3. fear_greed: > 60 → BULL, < 25 → BEAR (optional, needs API call)

    Voting: majority of available signals determines regime.
    """

    def __init__(
        self,
        sma_period: int = 50,
        momentum_period: int = 20,
        momentum_bear_threshold: float = -0.10,
        fear_greed_bull: int = 60,
        fear_greed_bear: int = 25,
    ):
        self.sma_period = sma_period
        self.momentum_period = momentum_period
        self.momentum_bear_threshold = momentum_bear_threshold
        self.fear_greed_bull = fear_greed_bull
        self.fear_greed_bear = fear_greed_bear
        self._cached_fg: Optional[dict] = None
        self._fg_cache_date: Optional[str] = None

    def evaluate(
        self,
        btc_df: pd.DataFrame,
        fear_greed_value: Optional[int] = None,
        current_day: Optional[pd.Timestamp] = None,
    ) -> RegimeResult:
        """Evaluate current market regime from BTC price data.

        Args:
            btc_df: BTC-USD OHLCV DataFrame (temporal, up to current day)
            fear_greed_value: Optional pre-fetched Fear & Greed index (0-100)
            current_day: Day to evaluate (defaults to last row in btc_df)

        Returns:
            RegimeResult with state, confidence, and individual signals.
        """
        if btc_df is None or len(btc_df) < self.sma_period + 5:
            return RegimeResult(
                state=RegimeState.NEUTRAL,
                confidence=0.0,
                signals=[],
                timestamp=str(current_day) if current_day else None,
            )

        close = btc_df["Close"]
        if current_day is not None:
            close = close[close.index <= current_day]

        if len(close) < self.sma_period:
            return RegimeResult(
                state=RegimeState.NEUTRAL,
                confidence=0.0,
                signals=[],
                timestamp=str(current_day) if current_day else None,
            )

        signals = []

        # Signal 1: BTC vs SMA(50)
        sma = close.rolling(self.sma_period).mean()
        current_price = float(close.iloc[-1])
        current_sma = float(sma.iloc[-1])
        if not np.isnan(current_sma) and current_sma > 0:
            pct_above = (current_price - current_sma) / current_sma
            if pct_above > 0.02:
                sig = RegimeState.BULL
            elif pct_above < -0.02:
                sig = RegimeState.BEAR
            else:
                sig = RegimeState.NEUTRAL
            signals.append(RegimeSignal(
                name="btc_vs_sma50",
                value=round(pct_above, 4),
                signal=sig,
                description=f"BTC {pct_above:+.1%} vs SMA({self.sma_period})",
            ))

        # Signal 2: BTC momentum (20d return)
        if len(close) > self.momentum_period:
            momentum = float(close.iloc[-1] / close.iloc[-self.momentum_period] - 1)
            if momentum > 0.05:
                sig = RegimeState.BULL
            elif momentum < self.momentum_bear_threshold:
                sig = RegimeState.BEAR
            else:
                sig = RegimeState.NEUTRAL
            signals.append(RegimeSignal(
                name="btc_momentum_20d",
                value=round(momentum, 4),
                signal=sig,
                description=f"BTC 20d return: {momentum:+.1%}",
            ))

        # Signal 3: Fear & Greed (if provided)
        if fear_greed_value is not None:
            fg = int(fear_greed_value)
            if fg >= self.fear_greed_bull:
                sig = RegimeState.BULL
            elif fg <= self.fear_greed_bear:
                sig = RegimeState.BEAR
            else:
                sig = RegimeState.NEUTRAL
            signals.append(RegimeSignal(
                name="fear_greed",
                value=fg,
                signal=sig,
                description=f"Fear & Greed: {fg}/100",
            ))

        # Majority vote
        if not signals:
            return RegimeResult(
                state=RegimeState.NEUTRAL, confidence=0.0, signals=[],
                timestamp=str(current_day) if current_day else None,
            )

        bull_count = sum(1 for s in signals if s.signal == RegimeState.BULL)
        bear_count = sum(1 for s in signals if s.signal == RegimeState.BEAR)
        total = len(signals)

        if bull_count > bear_count:
            state = RegimeState.BULL
            confidence = bull_count / total
        elif bear_count > bull_count:
            state = RegimeState.BEAR
            confidence = bear_count / total
        else:
            state = RegimeState.NEUTRAL
            confidence = 0.5

        day_str = str(current_day)[:10] if current_day else str(close.index[-1])[:10]

        return RegimeResult(
            state=state,
            confidence=round(confidence, 2),
            signals=signals,
            timestamp=day_str,
        )

    @staticmethod
    def fetch_fear_greed() -> Optional[int]:
        """Fetch current Fear & Greed Index from alternative.me (free API).

        Returns:
            Integer 0-100 or None if API fails.
        """
        try:
            import urllib.request
            import json
            url = "https://api.alternative.me/fng/?limit=1"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            return int(data["data"][0]["value"])
        except Exception as e:
            logger.warning(f"Fear & Greed API failed: {e}")
            return None
