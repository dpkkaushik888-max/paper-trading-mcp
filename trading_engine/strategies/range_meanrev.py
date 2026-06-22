"""Strategy C — Range Mean Reversion (S21 D3 / S23).

Locked rules (lifted verbatim from the validated scripts/sim_s21_window.py):
  Long entry (all true):
    1. RSI(2) < 5                 — deeper oversold than Connors (avoids overlap)
    2. ADX(14) < 18               — low-trend regime (anti-Connors filter)
    3. bb_width < 0.10            — tight Bollinger range
    (no SMA(200) trend filter)
  Long exit (first true):
    1. RSI(2) > 70               → MR_EXIT
    2. Max hold 7 trading days   → MAX_HOLD
    3. -5% from entry            → SL
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal
from trading_engine.engine.indicators import build_indicators

SL_PCT = 0.05
MAX_HOLD_DAYS = 7


class RangeMeanRevStrategy(BaseStrategy):
    priority = 2  # C is last in A > B > C
    reason = "range_mean_reversion"

    def entry(self, close: float, r: pd.Series) -> bool:
        for k in ("rsi_2", "adx_14", "bb_width"):
            if pd.isna(r.get(k)):
                return False
        return bool(r["rsi_2"] < 5 and r["adx_14"] < 18 and r["bb_width"] < 0.10)

    def exit_reason(
        self, close: float, r: pd.Series,
        entry_price: float, entry_date: pd.Timestamp, today: pd.Timestamp,
    ) -> Optional[str]:
        if (close - entry_price) / entry_price <= -SL_PCT:
            return "SL"
        if not pd.isna(r.get("rsi_2")) and r["rsi_2"] > 70:
            return "MR_EXIT"
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
    return StrategyConfig(name="C_range", capital_pct=1.0,
                          max_position_pct=0.12, max_concurrent=4)
