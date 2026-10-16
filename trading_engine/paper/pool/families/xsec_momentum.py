"""Cross-sectional momentum: on Monday bars hold the top-N coins above SMA100 by 28-day return; ROTATE out."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from ..book import Signal
from ._ind import bar_dow, num


class XsecMomentumStrategy:
    indicator_key = "xsec_momentum"

    def __init__(self, params: Mapping, data_guard: bool = False):
        self.top_n = int(params.get("top_n", 3))
        self.lookback = int(params.get("lookback", 28))
        self.sma_len = int(params.get("sma_len", 100))
        self.data_guard = data_guard   # line's BookParams.data_guard: drop stale/gap200 symbols before ranking
        self._monday, self._top = False, set()

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        return pd.DataFrame({"sma_100": c.rolling(self.sma_len).mean(),
                             "ret_28": c / c.shift(self.lookback) - 1.0}, index=df.index)

    def _eligible(self, close, r) -> bool:
        return num(r.get("sma_100")) is not None and num(r.get("ret_28")) is not None and close > r["sma_100"]

    def prepare(self, snapshot, ctx, today) -> None:
        """Cross-sectional rank for the day (called by book.step before exits)."""
        self._monday = bar_dow(ctx, today) == "Monday"
        flags = (getattr(ctx, "data_flags", None) or {}) if self.data_guard else {}
        flagged = set(flags.get("stale", [])) | set(flags.get("gap200", []))
        ranked = sorted((-float(r["ret_28"]), s) for s, (c, r) in snapshot.items()
                        if s not in flagged and self._eligible(c, r))
        self._top = {s for _, s in ranked[:self.top_n]} if self._monday else set()

    def gate(self, ctx) -> bool:
        return True

    def entry(self, sym, close, r, ctx) -> Optional[Signal]:
        if self._monday and sym in self._top:
            return Signal(strength=float(r["ret_28"]))
        return None

    def exit(self, pos, close, r, ctx, today) -> Optional[str]:
        if self._monday and (pos["symbol"] not in self._top or not self._eligible(close, r)):
            return "ROTATE"
        return None

    def rank_key(self, sym, sig) -> tuple:
        return (-sig.strength, sym)
