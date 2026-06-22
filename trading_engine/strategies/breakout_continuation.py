"""Strategy B — Breakout Continuation (S21 D2 / S23).

Locked rules (lifted verbatim from the validated scripts/sim_s21_window.py):
  Long entry (all true):
    1. Close > prior 20-day high (closing basis, prior bar)
    2. Volume > 1.5 × SMA(Volume, 20)
    3. ADX(14) >= 25
    4. Close > SMA(50)
  Long exit (first true):
    1. Close < SMA(10)          → MA_BREAK
    2. Max hold 15 trading days → MAX_HOLD
    3. -8% from entry           → SL
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal
from trading_engine.engine.indicators import build_indicators

SL_PCT = 0.08
MAX_HOLD_DAYS = 15


class BreakoutContinuationStrategy(BaseStrategy):
    priority = 1  # B is second in A > B > C
    reason = "breakout_continuation"

    def entry(self, close: float, r: pd.Series) -> bool:
        for k in ("prior_high_20", "vol_sma_20", "adx_14", "sma_50"):
            if pd.isna(r.get(k)):
                return False
        return bool(
            close > r["prior_high_20"]
            and r["volume"] > 1.5 * r["vol_sma_20"]
            and r["adx_14"] >= 25
            and close > r["sma_50"]
        )

    def exit_reason(
        self, close: float, r: pd.Series,
        entry_price: float, entry_date: pd.Timestamp, today: pd.Timestamp,
    ) -> Optional[str]:
        if (close - entry_price) / entry_price <= -SL_PCT:
            return "SL"
        if not pd.isna(r.get("sma_10")) and close < r["sma_10"]:
            return "MA_BREAK"
        if (today - entry_date).days >= MAX_HOLD_DAYS:
            return "MAX_HOLD"
        return None

    @property
    def sl_pct(self) -> float:
        return SL_PCT

    def evaluate(
        self, symbol, df, current_day, model=None,
        feature_cols=None, cross_asset_data=None,
    ) -> Optional[StrategySignal]:
        ind = build_indicators(df)
        if current_day not in ind.index:
            return None
        close = float(df.loc[current_day, "Close"])
        if not self.entry(close, ind.loc[current_day]):
            return None
        return StrategySignal(
            direction="long", confidence=1.0, strategy=self.name,
            reason=self.reason, sl_pct=SL_PCT, tp_pct=0.0,
            size_pct=self.config.max_position_pct,
        )


def default_config() -> StrategyConfig:
    return StrategyConfig(name="B_breakout", capital_pct=1.0,
                          max_position_pct=0.12, max_concurrent=4)
