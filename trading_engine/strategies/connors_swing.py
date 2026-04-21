"""Connors-style mean-reversion swing signals (S17).

Pure, stateless functions — no training, no self-tuning, no state.
Given (OHLCV history, date) the output is deterministic.

Canonical rule set (Connors "Short Term Trading Strategies That Work", 2009):
    Long entry:
        1. Close > SMA(Close, 200)   — established uptrend
        2. RSI(2)  < 10              — deeply oversold on 2-day RSI
        3. Close < SMA(Close, 5)     — pullback in progress
        (optional) ADX(14) >= 20     — avoid low-trend noise

    Long exit:
        1. Close > SMA(Close, 5)     — mean-reversion complete
        2. Max hold = 10 days
        3. Hard stop at −7% from entry
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


# ── Fixed parameters (no tuning allowed — see spec S17) ───────────────────
RSI_LENGTH = 2
RSI_THRESHOLD = 10

SMA_TREND_LEN = 200
SMA_PULLBACK_LEN = 5
SMA_EXIT_LEN = 5

ADX_LENGTH = 14
ADX_THRESHOLD = 20

MAX_HOLD_DAYS = 10
SL_PCT = 0.07


def precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute all Connors indicators once per symbol to avoid recomputation
    on every signal evaluation.

    Returns a DataFrame with columns:
        sma_200, sma_5, rsi_2, adx_14
    indexed identically to ``df``.
    """
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["sma_200"] = ta.sma(close, length=SMA_TREND_LEN)
    out["sma_5"] = ta.sma(close, length=SMA_PULLBACK_LEN)
    out["rsi_2"] = ta.rsi(close, length=RSI_LENGTH)

    adx = ta.adx(df["High"], df["Low"], close, length=ADX_LENGTH)
    if adx is not None and not adx.empty:
        adx_col = next((c for c in adx.columns if c.startswith("ADX")), None)
        out["adx_14"] = adx[adx_col] if adx_col else float("nan")
    else:
        out["adx_14"] = float("nan")

    return out


def long_entry(
    close_today: float,
    ind_row: pd.Series,
    use_adx_filter: bool = True,
) -> bool:
    """Return True if long entry conditions met.

    ``ind_row`` is a single row from ``precompute_indicators(...)`` containing
    sma_200, sma_5, rsi_2, adx_14.
    """
    if pd.isna(close_today):
        return False
    if pd.isna(ind_row.get("sma_200")) or pd.isna(ind_row.get("sma_5")) or pd.isna(ind_row.get("rsi_2")):
        return False

    trend_up = close_today > ind_row["sma_200"]
    oversold = ind_row["rsi_2"] < RSI_THRESHOLD
    pullback = close_today < ind_row["sma_5"]
    passed = trend_up and oversold and pullback

    if use_adx_filter and passed:
        adx_val = ind_row.get("adx_14")
        if pd.isna(adx_val):
            return False  # No ADX data → be conservative, skip.
        passed = passed and (adx_val >= ADX_THRESHOLD)

    return bool(passed)


def long_exit(
    close_today: float,
    ind_row: pd.Series,
    entry_price: float,
    entry_date: pd.Timestamp,
    today: pd.Timestamp,
) -> str | None:
    """Return exit reason or None.

    Reasons: "MR_EXIT" (mean reversion complete), "MAX_HOLD", "SL".
    """
    if pd.isna(close_today):
        return None

    pnl_pct = (close_today - entry_price) / entry_price
    if pnl_pct <= -SL_PCT:
        return "SL"

    if not pd.isna(ind_row.get("sma_5")) and close_today > ind_row["sma_5"]:
        return "MR_EXIT"

    if (today - entry_date).days >= MAX_HOLD_DAYS:
        return "MAX_HOLD"

    return None
