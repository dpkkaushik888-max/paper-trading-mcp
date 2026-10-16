"""Connors family: live connors_swing rules with ConnorsOverrides and the optional BTC>SMA200 gate."""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping, Optional

import pandas as pd

from trading_engine.strategies.connors_swing import (
    ConnorsOverrides, atr_wilder, long_entry, long_exit, precompute_indicators,
)

from ..book import Signal

ATR_COL = "atr_wilder"   # ATR(14) Wilder at the bar, stored on entry for the ATR stop
_OV_FIELDS = {f.name for f in fields(ConnorsOverrides)}


def _f(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


class ConnorsStrategy:
    indicator_key = "connors"   # indicators do not depend on params: one computation per run

    def __init__(self, params: Mapping):
        self.overrides = ConnorsOverrides(**{k: v for k, v in params.items() if k in _OV_FIELDS})
        self.btc_sma_gate = bool(params.get("btc_sma_gate", False))
        # Pool-only override: mean-reversion exit on close > SMA(n) instead of SMA5 (connors_swing.py untouched).
        self.mr_exit_sma = params.get("mr_exit_sma")

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = precompute_indicators(df).copy()
        ind[ATR_COL] = atr_wilder(df)
        ind["sma_10"] = df["Close"].rolling(10).mean()
        return ind

    def gate(self, ctx) -> bool:
        if not self.btc_sma_gate:
            return True
        # Closed when BTC close or SMA200 is unknown (as S34).
        return ctx.btc_close is not None and ctx.btc_sma200 is not None and ctx.btc_close > ctx.btc_sma200

    def entry(self, sym, close, ind_row, ctx) -> Optional[Signal]:
        if not long_entry(close, ind_row, use_adx_filter=True, overrides=self.overrides):
            return None
        return Signal(strength=_f(ind_row.get("rsi_2")), meta={"entry_atr": _f(ind_row.get(ATR_COL))})

    def exit(self, pos: dict, close, ind_row, ctx, today) -> Optional[str]:
        if self.mr_exit_sma:
            ind_row = ind_row.copy()
            ind_row["sma_5"] = ind_row[f"sma_{int(self.mr_exit_sma)}"]
        return long_exit(close, ind_row, pos["entry_price"], pd.Timestamp(pos["entry_date"]),
                         pd.Timestamp(today), overrides=self.overrides, entry_atr=pos.get("entry_atr"))

    def rank_key(self, sym, sig) -> tuple:
        return (sym,)   # alphabetical = live parity
