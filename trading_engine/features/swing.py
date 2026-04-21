"""Curated 12-feature set for daily-bar swing trading (S16).

Deliberately smaller than `build_bar_features()` in price_engine to reduce
overfitting risk on the low trade counts typical of swing timeframes
(30–80 trades per year per symbol).

Features are chosen for orthogonal signal across four axes:
    trend (3), momentum (3), volatility (2), volume (2), microstructure (2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


SWING_FEATURE_COLS = [
    # Trend
    "close_vs_sma_50",
    "close_vs_sma_200",
    "adx_14",
    # Momentum
    "rsi_14",
    "return_5d",
    "return_20d",
    # Volatility
    "atr_pct_14",
    "bb_pct_20",
    # Volume
    "volume_ratio_20",
    "obv_slope_10",
    # Microstructure
    "funding_rate",
    "dist_from_20d_high",
]


def build_swing_features(
    df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the S16 swing feature matrix from daily OHLCV bars.

    Parameters
    ----------
    df : pd.DataFrame
        Daily bars with columns Open, High, Low, Close, Volume, indexed by date.
    funding_df : pd.DataFrame | None
        Optional funding-rate series (indexed by timestamp, column ``funding_rate``).
        If None, the ``funding_rate`` feature is zero-filled.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by the same dates as ``df``. Early rows will have
        NaNs where rolling windows aren't yet filled — caller is responsible for
        ``.dropna()`` before training.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    ret_1d = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # ── Trend ─────────────────────────────────────────────────────────────
    sma_50 = ta.sma(close, length=50)
    sma_200 = ta.sma(close, length=200)
    feat["close_vs_sma_50"] = (close - sma_50) / sma_50
    feat["close_vs_sma_200"] = (close - sma_200) / sma_200

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and not adx.empty:
        # pandas_ta returns columns ADX_14, DMP_14, DMN_14
        adx_col = next((c for c in adx.columns if c.startswith("ADX")), None)
        feat["adx_14"] = adx[adx_col] if adx_col else np.nan
    else:
        feat["adx_14"] = np.nan

    # ── Momentum ─────────────────────────────────────────────────────────
    feat["rsi_14"] = ta.rsi(close, length=14)
    feat["return_5d"] = close.pct_change(5)
    feat["return_20d"] = close.pct_change(20)

    # ── Volatility ────────────────────────────────────────────────────────
    atr = ta.atr(high, low, close, length=14)
    feat["atr_pct_14"] = atr / close

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        cols = bb.columns  # BBL, BBM, BBU, BBB, BBP
        lower = bb[cols[0]]
        upper = bb[cols[2]]
        feat["bb_pct_20"] = (close - lower) / (upper - lower).replace(0, np.nan)
    else:
        feat["bb_pct_20"] = np.nan

    # ── Volume ────────────────────────────────────────────────────────────
    avg_vol_20 = volume.rolling(20).mean()
    feat["volume_ratio_20"] = volume / avg_vol_20.replace(0, np.nan)

    obv = ta.obv(close, volume)
    if obv is not None and not obv.empty:
        obv_slope = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan,
            raw=False,
        )
        # Normalize by 10-day volume average so cross-symbol scale is comparable.
        vol_norm = volume.rolling(10).mean().replace(0, np.nan)
        feat["obv_slope_10"] = obv_slope / vol_norm
    else:
        feat["obv_slope_10"] = np.nan

    # ── Microstructure ───────────────────────────────────────────────────
    feat["dist_from_20d_high"] = (close - close.rolling(20).max()) / close

    # Funding rate via merge_asof (already sorted, forward-filled from last observation).
    if funding_df is not None and not funding_df.empty:
        fr_sorted = funding_df.sort_index()
        merged = pd.merge_asof(
            feat.reset_index().sort_values(feat.index.name or "index"),
            fr_sorted.reset_index(),
            left_on=feat.index.name or "index",
            right_on=fr_sorted.index.name or "index",
            direction="backward",
        )
        merged = merged.set_index(feat.index.name or "index")
        feat["funding_rate"] = merged["funding_rate"].reindex(feat.index).fillna(0.0)
    else:
        feat["funding_rate"] = 0.0

    return feat[SWING_FEATURE_COLS]
