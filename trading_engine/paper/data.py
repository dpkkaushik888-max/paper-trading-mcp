"""Binance public kline fetch for S18 paper-forward.

No API key required. Uses public /api/v3/klines endpoint.
Returns pandas DataFrame with columns: Open, High, Low, Close, Volume
indexed by UTC date (daily bars).
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from .config import (
    BINANCE_KLINES_URL,
    BINANCE_FALLBACK_URL,
    KLINE_INTERVAL,
    KLINE_LIMIT,
)

log = logging.getLogger(__name__)


def fetch_klines(symbol: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    """Fetch daily OHLCV bars from Binance public API.

    Tries binance.com first, falls back to binance.us on geo-block.
    Returns DataFrame indexed by UTC date with Open/High/Low/Close/Volume.
    Excludes the currently-forming bar (only closed bars).
    """
    for url in (BINANCE_KLINES_URL, BINANCE_FALLBACK_URL):
        try:
            resp = requests.get(
                url,
                params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": limit},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                log.warning("Empty kline response for %s from %s", symbol, url)
                continue
            df = _klines_to_df(raw)
            # Drop currently-forming bar (close_time in the future).
            now_ms = pd.Timestamp.utcnow().value // 1_000_000
            df = df[df["_close_ms"] <= now_ms].drop(columns=["_close_ms"])
            return df
        except requests.RequestException as e:
            log.warning("Binance fetch failed for %s at %s: %s", symbol, url, e)
    raise RuntimeError(f"All Binance endpoints failed for {symbol}")


def _klines_to_df(raw: list) -> pd.DataFrame:
    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = df[c].astype(float)
    df["_close_ms"] = df["close_time"].astype("int64")
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume", "_close_ms"]]


def fetch_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch bars for every symbol. Logs failures but does not raise unless ALL fail."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = fetch_klines(sym)
        except Exception as e:  # noqa: BLE001
            log.error("Skipping %s: %s", sym, e)
    if not out:
        raise RuntimeError("Failed to fetch ANY symbol from Binance")
    return out
