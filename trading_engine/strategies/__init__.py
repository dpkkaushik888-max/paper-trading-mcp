"""Strategy abstraction layer for multi-strategy trading engine.

Each strategy implements BaseStrategy and returns StrategySignal objects.
The time machine queries all registered strategies per symbol per day
and executes the best signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class StrategySignal:
    """Signal produced by a strategy for a specific symbol on a specific day."""
    direction: str          # "long" | "short" | "none"
    confidence: float       # 0.0 - 1.0
    strategy: str           # strategy name for tracking
    reason: str             # human-readable reason
    sl_pct: float           # stop loss percentage
    tp_pct: float           # take profit percentage
    size_pct: float         # position size as fraction of strategy's capital budget
    trailing_stop: bool = False  # whether to use trailing stop instead of fixed SL


@dataclass
class StrategyConfig:
    """Configuration for a strategy's capital allocation and limits."""
    name: str
    capital_pct: float      # fraction of total capital allocated to this strategy
    max_position_pct: float # max single position as fraction of strategy's budget
    max_concurrent: int     # max simultaneous positions for this strategy
    enabled: bool = True


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.open_positions: int = 0

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_day: pd.Timestamp,
        model=None,
        feature_cols: list | None = None,
        cross_asset_data: dict | None = None,
    ) -> Optional[StrategySignal]:
        """Evaluate whether to trade this symbol today.

        Args:
            symbol: ticker symbol
            df: OHLCV DataFrame for this symbol (temporal, up to current_day)
            current_day: the day being evaluated
            model: ML model (only used by ML strategies)
            feature_cols: feature column names (only used by ML strategies)
            cross_asset_data: dict of cross-asset DataFrames if needed

        Returns:
            StrategySignal if a trade should be taken, None otherwise.
        """
        ...

    def can_open(self) -> bool:
        """Check if this strategy can open more positions."""
        return self.open_positions < self.config.max_concurrent


DEFAULT_STRATEGY_CONFIGS = {
    "ml_sniper": StrategyConfig(
        name="ml_sniper",
        capital_pct=0.40,
        max_position_pct=0.10,
        max_concurrent=3,
    ),
    "candlestick_sr": StrategyConfig(
        name="candlestick_sr",
        capital_pct=0.35,
        max_position_pct=0.05,
        max_concurrent=3,
    ),
    "trend_follower": StrategyConfig(
        name="trend_follower",
        capital_pct=0.25,
        max_position_pct=0.05,
        max_concurrent=2,
    ),
}
