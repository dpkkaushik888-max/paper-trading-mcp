"""Basket trend: hold each coin while close > SMA50; exit after 2 consecutive closes below SMA50 * band."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from ..book import Signal
from ._ind import num


class BasketTrendStrategy:
    indicator_key = "basket_trend"

    def __init__(self, params: Mapping):
        self.band = float(params.get("band", 0.98))

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        sma = c.rolling(50).mean()
        below = c < sma * self.band
        return pd.DataFrame({"sma_50": sma, "band_exit": below & below.shift(1, fill_value=False)}, index=df.index)

    def gate(self, ctx) -> bool:
        return True

    def entry(self, sym, close, r, ctx) -> Optional[Signal]:
        sma = num(r.get("sma_50"))
        return Signal(strength=close / sma - 1.0) if sma is not None and close > sma else None

    def exit(self, pos, close, r, ctx, today) -> Optional[str]:
        return "TREND_EXIT" if bool(r.get("band_exit")) else None

    def rank_key(self, sym, sig) -> tuple:
        return (sym,)
