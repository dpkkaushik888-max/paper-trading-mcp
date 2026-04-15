"""Chart Pattern Recognition — Support/Resistance, Classic Patterns, Fibonacci.

Detects:
1. Support & Resistance levels (pivot-based)
2. Breakout / Breakdown signals
3. Double Top / Double Bottom
4. Head & Shoulders / Inverse H&S
5. Fibonacci retracement levels
6. Trendline (linear regression channel)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series,
                      window: int = 10) -> dict:
    """Find pivot highs and pivot lows using rolling window.

    A pivot high: highest high in window on both sides.
    A pivot low: lowest low in window on both sides.
    """
    pivot_highs = []
    pivot_lows = []

    h = high.values
    l = low.values

    for i in range(window, len(h) - window):
        if h[i] == max(h[i - window:i + window + 1]):
            pivot_highs.append((i, float(h[i])))
        if l[i] == min(l[i - window:i + window + 1]):
            pivot_lows.append((i, float(l[i])))

    return {"pivot_highs": pivot_highs, "pivot_lows": pivot_lows}


def find_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series,
                            window: int = 10, tolerance: float = 0.02) -> dict:
    """Find support and resistance levels by clustering pivot points.

    Returns levels sorted by strength (how many times tested).
    """
    pivots = find_pivot_points(high, low, close, window)

    resistance_levels = _cluster_levels(
        [p[1] for p in pivots["pivot_highs"]], tolerance
    )
    support_levels = _cluster_levels(
        [p[1] for p in pivots["pivot_lows"]], tolerance
    )

    current_price = float(close.iloc[-1])

    nearest_resistance = None
    nearest_support = None

    for level, count in sorted(resistance_levels, key=lambda x: x[0]):
        if level > current_price * 1.001:
            nearest_resistance = {"price": level, "strength": count}
            break

    for level, count in sorted(support_levels, key=lambda x: -x[0]):
        if level < current_price * 0.999:
            nearest_support = {"price": level, "strength": count}
            break

    return {
        "resistance_levels": resistance_levels[:5],
        "support_levels": support_levels[:5],
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
    }


def _cluster_levels(prices: list[float], tolerance: float) -> list[tuple[float, int]]:
    """Cluster nearby price levels and count touches."""
    if not prices:
        return []

    prices = sorted(prices)
    clusters = []
    current_cluster = [prices[0]]

    for p in prices[1:]:
        if abs(p - current_cluster[-1]) / current_cluster[-1] <= tolerance:
            current_cluster.append(p)
        else:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
            current_cluster = [p]
    clusters.append((np.mean(current_cluster), len(current_cluster)))

    clusters.sort(key=lambda x: -x[1])
    return [(round(p, 2), c) for p, c in clusters]


def detect_double_top(high: pd.Series, close: pd.Series,
                      window: int = 10, tolerance: float = 0.02) -> dict:
    """Detect double top pattern (bearish reversal).

    Two peaks at roughly the same level with a valley between.
    """
    pivots = find_pivot_points(high, pd.Series(np.zeros(len(high))), close, window)
    peaks = pivots["pivot_highs"]

    if len(peaks) < 2:
        return {"detected": False}

    p1_idx, p1_price = peaks[-2]
    p2_idx, p2_price = peaks[-1]

    price_diff = abs(p1_price - p2_price) / p1_price

    if price_diff <= tolerance and (p2_idx - p1_idx) >= window:
        valley = float(close.iloc[p1_idx:p2_idx + 1].min())
        neckline = valley
        current = float(close.iloc[-1])

        return {
            "detected": True,
            "peak1": {"idx": p1_idx, "price": round(p1_price, 2)},
            "peak2": {"idx": p2_idx, "price": round(p2_price, 2)},
            "neckline": round(neckline, 2),
            "confirmed": current < neckline,
            "signal": "bearish" if current < neckline else "potential_bearish",
        }

    return {"detected": False}


def detect_double_bottom(low: pd.Series, close: pd.Series,
                         window: int = 10, tolerance: float = 0.02) -> dict:
    """Detect double bottom pattern (bullish reversal).

    Two troughs at roughly the same level with a peak between.
    """
    pivots = find_pivot_points(pd.Series(np.zeros(len(low))), low, close, window)
    troughs = pivots["pivot_lows"]

    if len(troughs) < 2:
        return {"detected": False}

    t1_idx, t1_price = troughs[-2]
    t2_idx, t2_price = troughs[-1]

    price_diff = abs(t1_price - t2_price) / t1_price

    if price_diff <= tolerance and (t2_idx - t1_idx) >= window:
        peak = float(close.iloc[t1_idx:t2_idx + 1].max())
        neckline = peak
        current = float(close.iloc[-1])

        return {
            "detected": True,
            "trough1": {"idx": t1_idx, "price": round(t1_price, 2)},
            "trough2": {"idx": t2_idx, "price": round(t2_price, 2)},
            "neckline": round(neckline, 2),
            "confirmed": current > neckline,
            "signal": "bullish" if current > neckline else "potential_bullish",
        }

    return {"detected": False}


def detect_head_shoulders(high: pd.Series, low: pd.Series, close: pd.Series,
                          window: int = 10, tolerance: float = 0.03) -> dict:
    """Detect head & shoulders (bearish) or inverse H&S (bullish).

    H&S: three peaks, middle one highest.
    Inv H&S: three troughs, middle one lowest.
    """
    pivots = find_pivot_points(high, low, close, window)

    result = {"hs_detected": False, "inv_hs_detected": False}

    peaks = pivots["pivot_highs"]
    if len(peaks) >= 3:
        ls, h, rs = peaks[-3], peaks[-2], peaks[-1]
        if (h[1] > ls[1] and h[1] > rs[1] and
                abs(ls[1] - rs[1]) / ls[1] <= tolerance):
            neckline = min(
                float(close.iloc[ls[0]:h[0] + 1].min()),
                float(close.iloc[h[0]:rs[0] + 1].min()),
            )
            current = float(close.iloc[-1])
            result["hs_detected"] = True
            result["hs_neckline"] = round(neckline, 2)
            result["hs_confirmed"] = current < neckline
            result["hs_signal"] = "bearish" if current < neckline else "potential_bearish"

    troughs = pivots["pivot_lows"]
    if len(troughs) >= 3:
        ls, h, rs = troughs[-3], troughs[-2], troughs[-1]
        if (h[1] < ls[1] and h[1] < rs[1] and
                abs(ls[1] - rs[1]) / ls[1] <= tolerance):
            neckline = max(
                float(close.iloc[ls[0]:h[0] + 1].max()),
                float(close.iloc[h[0]:rs[0] + 1].max()),
            )
            current = float(close.iloc[-1])
            result["inv_hs_detected"] = True
            result["inv_hs_neckline"] = round(neckline, 2)
            result["inv_hs_confirmed"] = current > neckline
            result["inv_hs_signal"] = "bullish" if current > neckline else "potential_bullish"

    return result


def fibonacci_levels(high: pd.Series, low: pd.Series,
                     lookback: int = 50) -> dict:
    """Calculate Fibonacci retracement levels from recent swing high/low.

    Key levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    """
    recent_high = float(high.iloc[-lookback:].max())
    recent_low = float(low.iloc[-lookback:].min())
    diff = recent_high - recent_low

    if diff <= 0:
        return {}

    levels = {
        "fib_0": round(recent_low, 2),
        "fib_236": round(recent_low + diff * 0.236, 2),
        "fib_382": round(recent_low + diff * 0.382, 2),
        "fib_500": round(recent_low + diff * 0.500, 2),
        "fib_618": round(recent_low + diff * 0.618, 2),
        "fib_786": round(recent_low + diff * 0.786, 2),
        "fib_1000": round(recent_high, 2),
    }

    current = float(high.iloc[-1] + low.iloc[-1]) / 2.0

    nearest_fib_support = None
    nearest_fib_resistance = None
    for name, level in sorted(levels.items(), key=lambda x: x[1]):
        if level < current * 0.999 and (nearest_fib_support is None or level > nearest_fib_support[1]):
            nearest_fib_support = (name, level)
        if level > current * 1.001 and nearest_fib_resistance is None:
            nearest_fib_resistance = (name, level)

    levels["nearest_fib_support"] = nearest_fib_support
    levels["nearest_fib_resistance"] = nearest_fib_resistance

    return levels


def trendline_channel(close: pd.Series, lookback: int = 20) -> dict:
    """Calculate linear regression channel (trendline).

    Returns slope direction, channel width, and position within channel.
    """
    y = close.iloc[-lookback:].values
    x = np.arange(len(y))

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted

    upper = predicted + 2 * np.std(residuals)
    lower = predicted - 2 * np.std(residuals)

    current = y[-1]
    channel_width = (upper[-1] - lower[-1]) / predicted[-1]
    position_in_channel = (current - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5

    return {
        "slope": round(float(slope), 4),
        "slope_pct": round(float(slope / predicted[-1] * 100), 4),
        "channel_upper": round(float(upper[-1]), 2),
        "channel_lower": round(float(lower[-1]), 2),
        "channel_mid": round(float(predicted[-1]), 2),
        "channel_width_pct": round(float(channel_width * 100), 2),
        "position_in_channel": round(float(position_in_channel), 4),
        "trend": "up" if slope > 0 else "down",
        "near_upper": position_in_channel > 0.8,
        "near_lower": position_in_channel < 0.2,
    }


def calculate_chart_features(df: pd.DataFrame) -> dict:
    """Calculate all chart pattern features for a single DataFrame.

    Returns dict of numeric features suitable for ML model input.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    features = {}

    sr = find_support_resistance(high, low, close, window=10)
    price = float(close.iloc[-1])

    nr = sr.get("nearest_resistance")
    ns = sr.get("nearest_support")
    if nr:
        features["dist_to_resistance_pct"] = round((nr["price"] - price) / price, 4)
        features["resistance_strength"] = nr["strength"]
    else:
        features["dist_to_resistance_pct"] = 0.1
        features["resistance_strength"] = 0

    if ns:
        features["dist_to_support_pct"] = round((price - ns["price"]) / price, 4)
        features["support_strength"] = ns["strength"]
    else:
        features["dist_to_support_pct"] = 0.1
        features["support_strength"] = 0

    dt = detect_double_top(high, close)
    features["double_top"] = 1 if dt.get("confirmed") else 0
    features["double_top_potential"] = 1 if dt.get("detected") else 0

    db = detect_double_bottom(low, close)
    features["double_bottom"] = 1 if db.get("confirmed") else 0
    features["double_bottom_potential"] = 1 if db.get("detected") else 0

    hs = detect_head_shoulders(high, low, close)
    features["head_shoulders"] = 1 if hs.get("hs_confirmed") else 0
    features["inv_head_shoulders"] = 1 if hs.get("inv_hs_confirmed") else 0

    fib = fibonacci_levels(high, low)
    if fib.get("nearest_fib_support"):
        features["dist_to_fib_support_pct"] = round(
            (price - fib["nearest_fib_support"][1]) / price, 4
        )
    else:
        features["dist_to_fib_support_pct"] = 0.1
    if fib.get("nearest_fib_resistance"):
        features["dist_to_fib_resistance_pct"] = round(
            (fib["nearest_fib_resistance"][1] - price) / price, 4
        )
    else:
        features["dist_to_fib_resistance_pct"] = 0.1

    trend = trendline_channel(close, lookback=20)
    features["trend_slope_pct"] = trend["slope_pct"]
    features["channel_position"] = trend["position_in_channel"]
    features["channel_width_pct"] = trend["channel_width_pct"]
    features["near_channel_upper"] = 1 if trend["near_upper"] else 0
    features["near_channel_lower"] = 1 if trend["near_lower"] else 0

    return features


def detect_candlestick_patterns(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
) -> pd.DataFrame:
    """Detect 10 candlestick patterns from OHLC data.

    Returns DataFrame with columns per pattern (1=bullish, -1=bearish, 0=none).
    Patterns detected:
      Bullish: hammer, bullish_engulfing, morning_star, dragonfly_doji, piercing_line
      Bearish: hanging_man, bearish_engulfing, evening_star, shooting_star, dark_cloud_cover
    """
    body = close - open_
    body_abs = body.abs()
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low
    candle_range = high - low
    avg_body = body_abs.rolling(20).mean()

    result = pd.DataFrame(index=open_.index)

    # — Hammer (bullish): small body at top, long lower shadow, at bottom of move —
    small_body = body_abs < candle_range * 0.35
    long_lower = lower_shadow > body_abs * 2.0
    short_upper = upper_shadow < body_abs * 0.5
    prior_down = close.shift(1) < close.shift(3)
    result["hammer"] = ((small_body & long_lower & short_upper & prior_down)
                        .astype(int))

    # — Hanging Man (bearish): same shape as hammer but at top of move —
    prior_up = close.shift(1) > close.shift(3)
    result["hanging_man"] = -((small_body & long_lower & short_upper & prior_up)
                              .astype(int))

    # — Bullish Engulfing: bearish candle followed by larger bullish candle —
    prev_bearish = body.shift(1) < 0
    curr_bullish = body > 0
    engulfs_body = (close > open_.shift(1)) & (open_ < close.shift(1))
    result["bullish_engulfing"] = ((prev_bearish & curr_bullish & engulfs_body
                                    & prior_down)
                                   .astype(int))

    # — Bearish Engulfing: bullish candle followed by larger bearish candle —
    prev_bullish = body.shift(1) > 0
    curr_bearish = body < 0
    engulfs_body_bear = (open_ > close.shift(1)) & (close < open_.shift(1))
    result["bearish_engulfing"] = -((prev_bullish & curr_bearish
                                     & engulfs_body_bear & prior_up)
                                    .astype(int))

    # — Morning Star (bullish): bearish candle, small body (any), bullish candle —
    first_bearish = body.shift(2) < 0
    first_big = body_abs.shift(2) > avg_body.shift(2) * 0.8
    mid_small = body_abs.shift(1) < avg_body.shift(1) * 0.5
    third_bullish = body > 0
    third_big = body_abs > avg_body * 0.8
    third_closes_into = close > (open_.shift(2) + close.shift(2)) / 2
    result["morning_star"] = ((first_bearish & first_big & mid_small
                               & third_bullish & third_big & third_closes_into)
                              .astype(int))

    # — Evening Star (bearish): bullish candle, small body, bearish candle —
    first_bullish = body.shift(2) > 0
    first_big_bull = body_abs.shift(2) > avg_body.shift(2) * 0.8
    third_bearish = body < 0
    third_big_bear = body_abs > avg_body * 0.8
    third_closes_into_bear = close < (open_.shift(2) + close.shift(2)) / 2
    result["evening_star"] = -((first_bullish & first_big_bull & mid_small
                                & third_bearish & third_big_bear
                                & third_closes_into_bear)
                               .astype(int))

    # — Dragonfly Doji (bullish): open ≈ close ≈ high, long lower shadow —
    doji_body = body_abs < candle_range * 0.10
    doji_long_lower = lower_shadow > candle_range * 0.60
    doji_short_upper = upper_shadow < candle_range * 0.10
    result["dragonfly_doji"] = ((doji_body & doji_long_lower & doji_short_upper
                                 & prior_down)
                                .astype(int))

    # — Shooting Star (bearish): small body at bottom, long upper shadow —
    long_upper = upper_shadow > body_abs * 2.0
    short_lower = lower_shadow < body_abs * 0.5
    result["shooting_star"] = -((small_body & long_upper & short_lower
                                 & prior_up)
                                .astype(int))

    # — Piercing Line (bullish): bearish candle, then bullish opens below and closes >50% —
    opens_below = open_ < low.shift(1)
    closes_above_mid = close > (open_.shift(1) + close.shift(1)) / 2
    doesnt_fully_engulf = close < open_.shift(1)
    result["piercing_line"] = ((prev_bearish & curr_bullish & opens_below
                                & closes_above_mid & doesnt_fully_engulf)
                               .astype(int))

    # — Dark Cloud Cover (bearish): bullish candle, then bearish opens above and closes <50% —
    opens_above = open_ > high.shift(1)
    closes_below_mid = close < (open_.shift(1) + close.shift(1)) / 2
    doesnt_fully_engulf_bear = close > open_.shift(1)
    result["dark_cloud_cover"] = -((prev_bullish & curr_bearish & opens_above
                                    & closes_below_mid
                                    & doesnt_fully_engulf_bear)
                                   .astype(int))

    return result.fillna(0).astype(int)


def get_candlestick_signal(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
) -> dict:
    """Get the candlestick signal for the most recent bar.

    Returns:
        {"pattern": str|None, "direction": "bullish"|"bearish"|None, "strength": int}
        strength: number of confirming patterns on same bar.
    """
    patterns = detect_candlestick_patterns(open_, high, low, close)
    if patterns.empty:
        return {"pattern": None, "direction": None, "strength": 0}

    last = patterns.iloc[-1]
    bullish = [col for col in patterns.columns if last[col] > 0]
    bearish = [col for col in patterns.columns if last[col] < 0]

    if bullish:
        return {
            "pattern": bullish[0],
            "direction": "bullish",
            "strength": len(bullish),
            "patterns": bullish,
        }
    elif bearish:
        return {
            "pattern": bearish[0],
            "direction": "bearish",
            "strength": len(bearish),
            "patterns": bearish,
        }

    return {"pattern": None, "direction": None, "strength": 0}


def build_chart_features_series(df: pd.DataFrame, min_bars: int = 60) -> pd.DataFrame:
    """Build chart pattern features for every bar in DataFrame.

    Returns DataFrame with same index as input, NaN for first min_bars rows.
    """
    result_rows = []
    for i in range(len(df)):
        if i < min_bars:
            result_rows.append({})
            continue
        try:
            subset = df.iloc[:i + 1]
            features = calculate_chart_features(subset)
            result_rows.append(features)
        except Exception:
            result_rows.append({})

    result = pd.DataFrame(result_rows, index=df.index)
    return result
