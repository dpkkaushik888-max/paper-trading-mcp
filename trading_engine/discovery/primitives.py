"""Indicator primitive library + condition grammar (S25 D2).

`build_features(df)` computes every primitive a candidate may reference.
A `Condition` is a single bounded comparison (lhs op rhs); a candidate's entry is
an AND of conditions, its exit an OR. The grammar is deliberately small and
deterministic — the agent can only emit conditions over these named columns, so
every generated strategy is backtestable and free of hallucination in its hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pandas as pd
import pandas_ta as ta

# Columns build_features produces (the agent may reference only these, plus "close").
FEATURE_COLUMNS = [
    "close", "rsi_2", "rsi_14", "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_mid",
    "bb_width", "adx_14", "atr_14", "volume", "vol_sma_20", "prior_high_20",
    "prior_low_20", "roc_20",
]

VALID_OPS = (">", "<", ">=", "<=")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all primitives, indexed like df. NaN where lookback insufficient."""
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["rsi_2"] = ta.rsi(close, length=2)
    out["rsi_14"] = ta.rsi(close, length=14)
    for n in (5, 10, 20, 50, 200):
        out[f"sma_{n}"] = ta.sma(close, length=n)
    out["ema_12"] = ta.ema(close, length=12)
    out["ema_26"] = ta.ema(close, length=26)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        cols = list(macd.columns)
        out["macd"] = macd[next((c for c in cols if c.startswith("MACD_")), cols[0])]
        out["macd_signal"] = macd[next((c for c in cols if c.startswith("MACDs_")), cols[-1])]
    else:
        out["macd"] = out["macd_signal"] = float("nan")

    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2 * sd
    out["bb_lower"] = mid - 2 * sd
    out["bb_width"] = (4 * sd) / mid

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and not adx.empty:
        col = next((c for c in adx.columns if c.startswith("ADX")), None)
        out["adx_14"] = adx[col] if col else float("nan")
    else:
        out["adx_14"] = float("nan")

    out["atr_14"] = ta.atr(high, low, close, length=14)
    out["volume"] = vol
    out["vol_sma_20"] = vol.rolling(20).mean()
    out["prior_high_20"] = close.shift(1).rolling(20).max()
    out["prior_low_20"] = close.shift(1).rolling(20).min()
    out["roc_20"] = close.pct_change(20)
    return out


@dataclass(frozen=True)
class Condition:
    """A single bounded comparison: lhs op (rhs [* rhs_mult]).

    lhs: a FEATURE_COLUMNS name (or "close").
    rhs: a constant (float) OR a feature-column name; rhs_mult scales a column rhs.
    """
    lhs: str
    op: str
    rhs: Union[float, str]
    rhs_mult: float = 1.0

    def __post_init__(self):
        if self.op not in VALID_OPS:
            raise ValueError(f"bad op: {self.op}")
        if self.lhs not in FEATURE_COLUMNS:
            raise ValueError(f"unknown lhs: {self.lhs}")
        if isinstance(self.rhs, str) and self.rhs not in FEATURE_COLUMNS:
            raise ValueError(f"unknown rhs column: {self.rhs}")

    def evaluate(self, row: pd.Series, close: float | None = None) -> bool:
        lval = close if (self.lhs == "close" and close is not None) else row.get(self.lhs)
        if lval is None or pd.isna(lval):
            return False
        if isinstance(self.rhs, str):
            rcol = row.get(self.rhs)
            if rcol is None or pd.isna(rcol):
                return False
            rval = float(rcol) * self.rhs_mult
        else:
            rval = float(self.rhs)
        lval = float(lval)
        if self.op == ">":
            return lval > rval
        if self.op == "<":
            return lval < rval
        if self.op == ">=":
            return lval >= rval
        return lval <= rval  # "<="

    def to_dict(self) -> dict:
        return {"lhs": self.lhs, "op": self.op, "rhs": self.rhs, "rhs_mult": self.rhs_mult}

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(lhs=d["lhs"], op=d["op"], rhs=d["rhs"], rhs_mult=d.get("rhs_mult", 1.0))
