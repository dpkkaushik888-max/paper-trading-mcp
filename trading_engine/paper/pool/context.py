"""Day context shared by all lines (S35 §3). Buckets frozen in C2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

GAP200_BARS = 200
_DAY_NS = 86_400_000_000_000
TRADING_DAYS_YEAR = 365   # crypto trades every day
CONTEXT_BARS = 300        # = config.KLINE_LIMIT: the window live run_daily sees
# Frozen 2026-10-16 (S35 C2), never rolling: terciles (1/3, 2/3 quantiles) of BTC btc_vol20 = annualised
# (sqrt 365) std of 20 daily log returns, binance.com BTCUSDT daily closes 2020-01-01..2025-12-31 (2192 values).
BTC_VOL20_TERCILES = (0.4330, 0.6134)
BASKET_RET20_EDGES = (-0.10, 0.0, 0.10)
BASKET_RET20_LABELS = ("<-10%", "-10..0", "0..+10", ">+10%")


def vol20_bucket(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    lo, hi = BTC_VOL20_TERCILES
    return "low" if v < lo else "mid" if v < hi else "high"


def ret20_bucket(r: Optional[float]) -> Optional[str]:
    """<-10% | -10..0 | 0..+10 (inclusive of 0 and +10 %) | >+10%."""
    if r is None:
        return None
    a, b, c = BASKET_RET20_EDGES
    return BASKET_RET20_LABELS[0 if r < a else 1 if r < b else 2 if r <= c else 3]


@dataclass
class DayContext:
    bar_date: Optional[str] = None
    dow: Optional[str] = None
    btc_trend: str = "unknown"   # above_both | above200_below50 | below200_above50 | below_both | unknown
    btc_vol20: Optional[float] = None
    btc_vol20_bucket: Optional[str] = None        # C2
    basket_ret20: Optional[float] = None
    basket_ret20_bucket: Optional[str] = None     # C2
    data_flags: dict = field(default_factory=lambda: {"stale": [], "gap200": []})
    btc_close: Optional[float] = None
    btc_sma200: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def contiguous_bars(df: pd.DataFrame) -> int:
    """Number of consecutive daily bars ending at the latest bar (a gap = a missing calendar day)."""
    if df is None or len(df) == 0:
        return 0
    days = pd.DatetimeIndex(df.index).normalize()
    breaks = np.flatnonzero(np.diff(days.asi8) != _DAY_NS)
    return len(days) - (int(breaks[-1]) + 1 if len(breaks) else 0)


def btc_trend(close: Optional[float], sma50: Optional[float], sma200: Optional[float]) -> str:
    if close is None or sma50 is None or sma200 is None:
        return "unknown"
    a50, a200 = close > sma50, close > sma200
    return {(True, True): "above_both", (False, True): "above200_below50",
            (True, False): "below200_above50", (False, False): "below_both"}[(a50, a200)]


def build_context(bars_by_sym: dict[str, pd.DataFrame], today: str, stale: list[str],
                  btc_symbol: str, btc_close: Optional[float], btc_sma200: Optional[float]) -> DayContext:
    """``btc_close``/``btc_sma200`` are the run's values (same as the BTC gate and trend tracker)."""
    bar = pd.Timestamp(today).date() - timedelta(days=1)   # journal day D trades bar D-1
    gap200 = sorted(s for s, df in bars_by_sym.items() if contiguous_bars(df) < GAP200_BARS)
    ctx = DayContext(bar_date=bar.isoformat(), dow=bar.strftime("%A"),
                     data_flags={"stale": sorted(stale), "gap200": gap200},
                     btc_close=btc_close, btc_sma200=btc_sma200)
    btc = bars_by_sym.get(btc_symbol)
    if btc is not None and len(btc) >= 50:
        sma50 = float(btc["Close"].iloc[-50:].mean())
        ctx.btc_trend = btc_trend(btc_close, sma50, btc_sma200)
    if btc is not None and len(btc) >= 21:
        lr = np.log(btc["Close"].iloc[-21:].astype(float)).diff().dropna()
        ctx.btc_vol20 = float(lr.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR))
    rets = [float(df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1.0)
            for s, df in bars_by_sym.items()
            if s not in ctx.data_flags["stale"] and s not in gap200 and len(df) >= 21]
    ctx.basket_ret20 = float(np.mean(rets)) if rets else None
    ctx.btc_vol20_bucket = vol20_bucket(ctx.btc_vol20)
    ctx.basket_ret20_bucket = ret20_bucket(ctx.basket_ret20)
    return ctx


def context_for_bar(bars: dict[str, pd.DataFrame], bar: pd.Timestamp, btc_symbol: str) -> DayContext:
    """Context for journal day bar+1 from full cached bars, as live would see it (last CONTEXT_BARS bars;
    stale = no bar on ``bar``). Used by the backtest and the context backfill."""
    vis, stale = {}, []
    for s, df in bars.items():
        w = df.iloc[max(0, df.index.searchsorted(bar, side="right") - CONTEXT_BARS):df.index.searchsorted(bar, side="right")]
        if len(w):
            vis[s] = w
            if w.index[-1] < bar:
                stale.append(s)
    btc = vis.get(btc_symbol)
    btc_close = float(btc["Close"].iloc[-1]) if btc is not None and len(btc) else None
    btc_sma = float(btc["Close"].iloc[-200:].mean()) if btc is not None and len(btc) >= 200 else None
    return build_context(vis, (bar + timedelta(days=1)).date().isoformat(), stale, btc_symbol, btc_close, btc_sma)
