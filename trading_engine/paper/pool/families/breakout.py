"""Breakout family: locked S21 rules (trading_engine/strategies/breakout_continuation.py is authoritative)."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from ..book import Signal
from ._ind import adx_wilder, num

# Mirrored from breakout_continuation.py (importing it pulls the pandas-ta engine); a test asserts equality.
SL_PCT = 0.08
MAX_HOLD_DAYS = 15
VOL_MULT = 1.5
ADX_MIN = 25


class BreakoutStrategy:
    indicator_key = "breakout"

    def __init__(self, params: Mapping):
        self.params = params

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        return pd.DataFrame({"prior_high_20": c.shift(1).rolling(20).max(),
                             "vol_sma_20": df["Volume"].rolling(20).mean(), "volume": df["Volume"],
                             "adx_14": adx_wilder(df, 14), "sma_50": c.rolling(50).mean(),
                             "sma_10": c.rolling(10).mean()}, index=df.index)

    def gate(self, ctx) -> bool:
        return True

    def entry(self, sym, close, r, ctx) -> Optional[Signal]:
        if any(num(r.get(k)) is None for k in ("prior_high_20", "vol_sma_20", "adx_14", "sma_50", "volume")):
            return None
        if (close > r["prior_high_20"] and r["volume"] > VOL_MULT * r["vol_sma_20"]
                and r["adx_14"] >= ADX_MIN and close > r["sma_50"]):
            return Signal(strength=close / r["prior_high_20"] - 1.0)
        return None

    def exit(self, pos, close, r, ctx, today) -> Optional[str]:
        # Same order as BreakoutContinuationStrategy.exit_reason: SL, MA_BREAK, MAX_HOLD.
        if (close - pos["entry_price"]) / pos["entry_price"] <= -SL_PCT:
            return "SL"
        if num(r.get("sma_10")) is not None and close < r["sma_10"]:
            return "MA_BREAK"
        if (pd.Timestamp(today) - pd.Timestamp(pos["entry_date"])).days >= MAX_HOLD_DAYS:
            return "MAX_HOLD"
        return None

    def rank_key(self, sym, sig) -> tuple:
        return (sym,)
