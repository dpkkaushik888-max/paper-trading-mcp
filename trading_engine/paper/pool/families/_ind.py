"""Pure-pandas indicator helpers (Wilder RMA as in the deep-dive connors_lib); no pandas-ta (S35 §2, B7)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, min_periods=n).mean()


def adx_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    atr = rma(tr, n)
    up, dn = h - h.shift(), l.shift() - l
    dmp = 100 * rma(((up > dn) & (up > 0)) * up, n) / atr
    dmn = 100 * rma(((dn > up) & (dn > 0)) * dn, n) / atr
    return rma(100 * (dmp - dmn).abs() / (dmp + dmn), n)


def num(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


def bar_dow(ctx, today: str) -> str:
    return getattr(ctx, "dow", None) or pd.Timestamp(today).day_name()
