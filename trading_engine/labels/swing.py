"""Label generation for swing-timeframe classification (S16).

Primary label: binary direction of the ``horizon``-day forward return.
    y = 1 if close[t + horizon] > close[t] else 0
"""

from __future__ import annotations

import pandas as pd


def forward_direction_label(
    df: pd.DataFrame,
    horizon_days: int = 5,
    close_col: str = "Close",
) -> pd.Series:
    """Return a binary Series indicating whether the ``horizon``-day forward
    return is positive.

    The last ``horizon_days`` rows will be NaN (target is unknown) — caller must
    drop them before training.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV frame, indexed by date.
    horizon_days : int, default 5
        Forward-return horizon.
    close_col : str, default "Close"
        Column name for close price.
    """
    close = df[close_col]
    future_close = close.shift(-horizon_days)
    fwd_ret = (future_close / close) - 1.0
    label = (fwd_ret > 0).astype("Int64")
    # Where fwd_ret is NaN (last `horizon_days` rows), label should also be NaN.
    label = label.where(fwd_ret.notna(), other=pd.NA)
    label.name = "target_dir"
    return label


def forward_return(
    df: pd.DataFrame,
    horizon_days: int = 5,
    close_col: str = "Close",
) -> pd.Series:
    """Raw forward return (useful for evaluation / magnitude labels)."""
    close = df[close_col]
    future_close = close.shift(-horizon_days)
    ret = (future_close / close) - 1.0
    ret.name = "fwd_ret"
    return ret
