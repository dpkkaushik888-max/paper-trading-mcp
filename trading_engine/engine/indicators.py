"""Superset indicator frame for the L3 strategy stack (S23).

One pass per symbol computes every column the three strategies read (Connors
A + breakout B + range-MR C). Lifted verbatim from the validated
``scripts/sim_s21_window.py`` build_indicators so the orchestrator reproduces
that regression oracle exactly.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from trading_engine.strategies.connors_swing import precompute_indicators as _connors_indicators


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with all columns A/B/C strategies need, indexed like df.

    Columns: sma_200, sma_5, rsi_2, adx_14 (Connors) + prior_high_20, vol_sma_20,
    volume, sma_50, sma_10 (breakout) + bb_width (range mean-reversion).
    """
    out = _connors_indicators(df)  # sma_200, sma_5, rsi_2, adx_14
    close = df["Close"]
    # Strategy B (breakout continuation) inputs
    out["prior_high_20"] = close.shift(1).rolling(20).max()
    out["vol_sma_20"] = df["Volume"].rolling(20).mean()
    out["volume"] = df["Volume"]
    out["sma_50"] = ta.sma(close, length=50)
    out["sma_10"] = ta.sma(close, length=10)
    # Strategy C (range mean-reversion) inputs — Bollinger width on 20 / 2σ
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["bb_width"] = (4.0 * std) / mid  # (upper - lower) / mid = 4σ/mid
    return out
