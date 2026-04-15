"""Yahoo Finance price engine with technical indicator calculation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import HISTORY_DAYS, INDICATOR_PERIODS, WATCHLIST


def get_quote(symbol: str) -> dict:
    """Get real-time quote for a symbol."""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    try:
        price = info.last_price
        prev_close = info.previous_close
        change = price - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
    except Exception:
        hist = ticker.history(period="1d")
        if hist.empty:
            return {"symbol": symbol, "error": "No data available"}
        price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Open"].iloc[0])
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

    return {
        "symbol": symbol,
        "price": round(price, 4),
        "previous_close": round(prev_close, 4) if prev_close else None,
        "change": round(change, 4),
        "change_pct": round(change_pct, 2),
        "timestamp": datetime.now().isoformat(),
    }


def get_history(
    symbol: str,
    days: int = HISTORY_DAYS,
    interval: str = "1d",
) -> pd.DataFrame:
    """Get historical OHLCV data."""
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def calculate_indicators(df: pd.DataFrame) -> dict:
    """Calculate all technical indicators on a price DataFrame.

    Returns dict with latest indicator values.
    """
    import pandas_ta as ta

    if df.empty or len(df) < 50:
        return {"error": "Insufficient data for indicator calculation"}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    indicators = {}

    indicators["price"] = round(float(close.iloc[-1]), 4)

    rsi_14 = ta.rsi(close, length=INDICATOR_PERIODS["rsi_14"])
    if rsi_14 is not None and not rsi_14.empty:
        indicators["rsi_14"] = round(float(rsi_14.iloc[-1]), 2)

    rsi_3 = ta.rsi(close, length=INDICATOR_PERIODS["rsi_3"])
    if rsi_3 is not None and not rsi_3.empty:
        indicators["rsi_3"] = round(float(rsi_3.iloc[-1]), 2)

    rsi_2 = ta.rsi(close, length=INDICATOR_PERIODS["rsi_2"])
    if rsi_2 is not None and not rsi_2.empty:
        indicators["rsi_2"] = round(float(rsi_2.iloc[-1]), 2)

    for period_key in ["sma_5", "sma_200"]:
        length = INDICATOR_PERIODS[period_key]
        sma = ta.sma(close, length=length)
        if sma is not None and not sma.empty:
            indicators[period_key] = round(float(sma.iloc[-1]), 4)

    for period_key in ["ema_8", "ema_20", "ema_50", "ema_200"]:
        length = INDICATOR_PERIODS[period_key]
        ema = ta.ema(close, length=length)
        if ema is not None and not ema.empty:
            indicators[period_key] = round(float(ema.iloc[-1]), 4)

    macd_result = ta.macd(
        close,
        fast=INDICATOR_PERIODS["macd_fast"],
        slow=INDICATOR_PERIODS["macd_slow"],
        signal=INDICATOR_PERIODS["macd_signal"],
    )
    if macd_result is not None and not macd_result.empty:
        cols = macd_result.columns
        indicators["macd"] = round(float(macd_result[cols[0]].iloc[-1]), 4)
        indicators["macd_histogram"] = round(float(macd_result[cols[1]].iloc[-1]), 4)
        indicators["macd_signal"] = round(float(macd_result[cols[2]].iloc[-1]), 4)

    bb = ta.bbands(
        close,
        length=INDICATOR_PERIODS["bb_period"],
        std=INDICATOR_PERIODS["bb_std"],
    )
    if bb is not None and not bb.empty:
        cols = bb.columns
        indicators["bb_lower"] = round(float(bb[cols[0]].iloc[-1]), 4)
        indicators["bb_mid"] = round(float(bb[cols[1]].iloc[-1]), 4)
        indicators["bb_upper"] = round(float(bb[cols[2]].iloc[-1]), 4)

    atr = ta.atr(high, low, close, length=INDICATOR_PERIODS["atr_period"])
    if atr is not None and not atr.empty:
        indicators["atr"] = round(float(atr.iloc[-1]), 4)

    if volume is not None and not volume.empty:
        avg_vol = volume.rolling(20).mean()
        if not avg_vol.empty:
            indicators["volume"] = int(volume.iloc[-1])
            indicators["avg_volume_20"] = int(avg_vol.iloc[-1])

    h_val = float(high.iloc[-1])
    l_val = float(low.iloc[-1])
    if h_val > l_val:
        indicators["ibs"] = round((float(close.iloc[-1]) - l_val) / (h_val - l_val), 4)
    else:
        indicators["ibs"] = 0.5

    indicators["day_of_week"] = int(df.index[-1].weekday())

    if len(close) >= 2:
        indicators["prev_close_1"] = round(float(close.iloc[-2]), 4)
    if len(high) >= 2:
        indicators["prev_high_1"] = round(float(high.iloc[-2]), 4)
    if len(low) >= 2:
        indicators["prev_low_1"] = round(float(low.iloc[-2]), 4)
    if len(high) >= 3:
        indicators["prev_high_2"] = round(float(high.iloc[-3]), 4)
    if len(low) >= 3:
        indicators["prev_low_2"] = round(float(low.iloc[-3]), 4)
    if len(high) >= 4:
        indicators["prev_high_3"] = round(float(high.iloc[-4]), 4)
    if len(low) >= 4:
        indicators["prev_low_3"] = round(float(low.iloc[-4]), 4)

    n_lower_highs = 0
    n_lower_lows = 0
    for i in range(1, min(5, len(high))):
        if float(high.iloc[-i]) < float(high.iloc[-i - 1]):
            n_lower_highs += 1
        else:
            break
    for i in range(1, min(5, len(low))):
        if float(low.iloc[-i]) < float(low.iloc[-i - 1]):
            n_lower_lows += 1
        else:
            break
    indicators["consecutive_lower_highs"] = n_lower_highs
    indicators["consecutive_lower_lows"] = n_lower_lows

    return indicators


def scan_watchlist(symbols: Optional[list[str]] = None) -> list[dict]:
    """Fetch quotes and indicators for all watchlist symbols."""
    symbols = symbols or WATCHLIST
    results = []

    for symbol in symbols:
        quote = get_quote(symbol)
        if "error" in quote:
            results.append(quote)
            continue

        df = get_history(symbol)
        indicators = calculate_indicators(df)

        results.append({
            **quote,
            "indicators": indicators,
        })

    return results
