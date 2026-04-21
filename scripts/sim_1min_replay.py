#!/usr/bin/env python3
"""Time-Machine 1-Minute Replay with Regime Filter.

Strict temporal isolation: at each 1-min bar, the model can ONLY see
data up to that bar. No future leakage. Model trained on data before target day.

Regime filter: reads hourly EMA cross, RSI, return to classify the
current regime as BULL / BEAR / NEUTRAL.  Longs only allowed in BULL,
shorts only in BEAR.  NEUTRAL = no new entries (existing positions still managed).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_1min_replay.py
    PYTHONPATH=. .venv/bin/python scripts/sim_1min_replay.py --verbose
    PYTHONPATH=. .venv/bin/python scripts/sim_1min_replay.py --date 2026-04-18
    PYTHONPATH=. .venv/bin/python scripts/sim_1min_replay.py --all-days
    PYTHONPATH=. .venv/bin/python scripts/sim_1min_replay.py --no-regime-filter
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests as _requests
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_engine.models.classifiers import SmartLogistic, SmartXGBoost
from trading_engine.price_engine import build_mtf_features

from scripts.sim_journal import SimJournal

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
CAPITAL = 1000.0
CONFIDENCE_LONG = 0.70
CONFIDENCE_SHORT = 0.72
LOGISTIC_C = 0.15
SL_PCT = 0.015
TP_PCT = 0.030
TRAIL_ACTIVATE = 0.010
TRAIL_OFFSET = 0.005
MAX_POS_PCT = 0.20
# HONEST-SIM: costs reflect realistic retail-crypto frictions.
#   COST_PCT is the per-side taker fee (was 0.001 = 10 bps; realistic = 20 bps).
#   SLIPPAGE_BPS is additional per-fill slippage on market orders.
#   SL_SLIPPAGE_BPS is extra slippage when a stop-loss triggers
#     (stop orders execute through the book in fast markets).
# Round-trip economic cost = 2*(COST_PCT + SLIPPAGE_BPS) ≈ 50 bps baseline.
# Use --optimistic-costs to revert to the legacy 10 bps model.
COST_PCT = 0.0020
SLIPPAGE_BPS = 0.0005
SL_SLIPPAGE_BPS = 0.0010
MAX_POSITIONS = 5
TRAIN_WINDOW = 2000  # 5-min bars; expanded to 20000 on --binance (1-min)
MIN_HOLD_BARS = 15
MIN_ATR_PCT = 0.0005

# ── Short model risk params (tighter: bear moves are fast) ────────────────
SL_SHORT = 0.010
TP_SHORT = 0.020
TRAIL_ACTIVATE_SHORT = 0.0075
TRAIL_OFFSET_SHORT = 0.004
CONFIDENCE_SHORT_MODEL = 0.72
MIN_HOLD_BARS_SHORT = 10
USE_DUAL_MODEL = False         # --adaptive-v2 enables dual long+short models
SHORT_MODEL_TYPE = "xgboost"   # XGBoost default; was logistic (30d A/B: XGB +1.79% vs Logistic -1.42%)

# ── Ratchet Stop-Loss / Progressive TP ──────────────────────────────────────
RATCHET_STEP = 0.01        # every 1% move triggers a ratchet level
RATCHET_SL_LOCK = 0.005    # lock SL 0.5% below the ratchet level
RATCHET_TP_EXTEND = 0.015  # extend TP by 1.5% at each ratchet level
ML_PROFIT_THRESHOLD = 0.005 # if unrealized gain > 0.5%, use stricter ML exit
ML_PROFIT_CONFIDENCE = 0.85 # need 85% model confidence to close a winner
USE_RATCHET = False         # ratchet off by default; legacy trail won A/B test

TARGET_DATE = "2026-04-15"
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]

BINANCE_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
}

REGIME_RSI_BULL = 55
REGIME_RSI_BEAR = 45
REGIME_EMA_THRESHOLD = 0.002


# ── Data fetching ───────────────────────────────────────────────────────────

def _fetch_with_retry(sym: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    """Fetch Yahoo data with retry + backoff for rate limits."""
    for attempt in range(retries):
        try:
            df = yf.Ticker(sym).history(period=period, interval=interval)
            if df is not None and not df.empty:
                df.index = df.index.tz_localize(None) if df.index.tz else df.index
                return df
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def fetch_1min_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 1-min bars -- Yahoo allows max 7 days of 1m data."""
    data = {}
    for sym in symbols:
        df = _fetch_with_retry(sym, "7d", "1m")
        if not df.empty:
            data[sym] = df
    return data


def fetch_5min_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 5-min bars (up to 60 days) -- used for training."""
    data = {}
    for sym in symbols:
        df = _fetch_with_retry(sym, "59d", "5m")
        if not df.empty:
            data[sym] = df
    return data


def fetch_1h_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 1h bars for MTF features."""
    data = {}
    for sym in symbols:
        df = _fetch_with_retry(sym, "60d", "1h")
        if not df.empty:
            data[sym] = df
    return data


# ── Binance data fetching ──────────────────────────────────────────────────

def _binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch Binance klines with pagination (1000 bars per request)."""
    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        resp = _requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=10)
        data = resp.json()
        if not data or isinstance(data, dict):
            break
        all_bars.extend(data)
        cursor = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.05)
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote", "ignore",
    ])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df.index.name = "Datetime"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_binance_1m(symbols: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
    """Fetch 1-min bars from Binance (free, up to years of history)."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    data = {}
    for sym in symbols:
        bsym = BINANCE_MAP.get(sym, sym)
        print(f"    {sym} ({bsym})...", end="", flush=True)
        df = _binance_klines(bsym, "1m", start_ms, end_ms)
        if not df.empty:
            data[sym] = df
            print(f" {len(df):,} bars")
        else:
            print(" NO DATA")
    return data


def fetch_binance_1h(symbols: list[str], days: int = 90) -> dict[str, pd.DataFrame]:
    """Fetch 1h bars from Binance for MTF features + regime."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    data = {}
    for sym in symbols:
        bsym = BINANCE_MAP.get(sym, sym)
        df = _binance_klines(bsym, "1h", start_ms, end_ms)
        if not df.empty:
            data[sym] = df
    return data


def fetch_binance_funding_rates(symbols: list[str], days: int = 90) -> dict[str, pd.DataFrame]:
    """Fetch historical funding rates from Binance Futures (8-hourly)."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    data = {}
    for sym in symbols:
        bsym = BINANCE_MAP.get(sym, sym)
        all_rates = []
        cursor = start_ms
        while cursor < end_ms:
            try:
                resp = _requests.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": bsym, "startTime": cursor, "endTime": end_ms, "limit": 1000},
                    timeout=10,
                )
                rates = resp.json()
                if not rates or isinstance(rates, dict):
                    break
                all_rates.extend(rates)
                cursor = rates[-1]["fundingTime"] + 1
                if len(rates) < 1000:
                    break
                time.sleep(0.05)
            except Exception:
                break
        if all_rates:
            df = pd.DataFrame(all_rates)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df.index = pd.to_datetime(df["fundingTime"], unit="ms")
            df.index.name = "Datetime"
            data[sym] = df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    return data


def get_funding_rate_at(funding_data: dict[str, pd.DataFrame], sym: str, bar_time) -> float:
    """Look up the latest funding rate at or before bar_time."""
    if sym not in funding_data or funding_data[sym].empty:
        return 0.0
    df = funding_data[sym]
    before = df[df.index <= bar_time]
    if before.empty:
        return 0.0
    return float(before.iloc[-1]["funding_rate"])


# ── Regime filter ───────────────────────────────────────────────────────────

def compute_regime(hourly_df: pd.DataFrame, bar_time: pd.Timestamp) -> str:
    """Classify current market regime from hourly data visible before bar_time.

    Uses 3 independent hourly signals and a voting system:
      - EMA8/EMA21 cross direction
      - RSI(14) level
      - 24-hour return sign

    Returns 'BULL', 'BEAR', or 'NEUTRAL'.
    """
    if hourly_df is None or hourly_df.empty:
        return "NEUTRAL"

    h_before = hourly_df[hourly_df.index < bar_time]
    if len(h_before) < 30:
        return "NEUTRAL"

    h_close = h_before["Close"]

    bull_votes = 0
    bear_votes = 0

    # Signal 1: EMA8 vs EMA21 cross
    ema_8 = ta.ema(h_close, length=8)
    ema_21 = ta.ema(h_close, length=21)
    if ema_8 is not None and ema_21 is not None and not ema_8.empty:
        ema_val = ema_21.iloc[-1]
        if ema_val != 0:
            cross = (ema_8.iloc[-1] - ema_val) / abs(ema_val)
            if cross > REGIME_EMA_THRESHOLD:
                bull_votes += 1
            elif cross < -REGIME_EMA_THRESHOLD:
                bear_votes += 1

    # Signal 2: RSI(14) level
    rsi_h = ta.rsi(h_close, length=14)
    if rsi_h is not None and not rsi_h.empty:
        rsi_val = float(rsi_h.iloc[-1])
        if rsi_val > REGIME_RSI_BULL:
            bull_votes += 1
        elif rsi_val < REGIME_RSI_BEAR:
            bear_votes += 1

    # Signal 3: 24-hour return
    if len(h_close) >= 24:
        ret_24h = float(h_close.iloc[-1] / h_close.iloc[-24] - 1)
        if ret_24h > 0.005:
            bull_votes += 1
        elif ret_24h < -0.005:
            bear_votes += 1

    # Voting: need 2/3 agreement for a directional call
    if bull_votes >= 2 and bear_votes == 0:
        return "BULL"
    elif bear_votes >= 2 and bull_votes == 0:
        return "BEAR"
    return "NEUTRAL"


# ── Feature builder for short-timeframe bars ────────────────────────────────

def build_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from any OHLCV bars (1m, 5m, 15m)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    ret_1 = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    feat["rsi_7"] = ta.rsi(close, length=7)
    feat["rsi_14"] = ta.rsi(close, length=14)
    feat["ibs"] = (close - low) / (high - low).replace(0, np.nan)

    for p in [5, 20, 50]:
        sma = ta.sma(close, length=p)
        if sma is not None:
            feat[f"close_vs_sma_{p}"] = (close - sma) / sma

    ema_8 = ta.ema(close, length=8)
    ema_21 = ta.ema(close, length=21)
    if ema_8 is not None and ema_21 is not None:
        feat["ema_cross"] = (ema_8 - ema_21) / ema_21.replace(0, np.nan)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        feat["macd_hist_norm"] = macd.iloc[:, 1] / close

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        cols = bb.columns
        feat["bb_pct"] = (close - bb[cols[0]]) / (bb[cols[2]] - bb[cols[0]]).replace(0, np.nan)

    feat["atr_pct"] = ta.atr(high, low, close, length=14) / close
    feat["return_1b"] = ret_1
    feat["return_5b"] = close.pct_change(5)
    feat["return_10b"] = close.pct_change(10)
    feat["return_20b"] = close.pct_change(20)

    feat["vol_5b"] = ret_1.rolling(5).std()
    feat["vol_20b"] = ret_1.rolling(20).std()
    feat["vol_ratio"] = feat["vol_5b"] / feat["vol_20b"].replace(0, np.nan)

    if volume is not None and not volume.empty:
        avg_vol = volume.rolling(20).mean()
        feat["volume_ratio"] = volume / avg_vol.replace(0, np.nan)

    feat["high_low_range"] = (high - low) / close
    feat["dist_20b_high"] = (close - close.rolling(20).max()) / close
    feat["dist_20b_low"] = (close - close.rolling(20).min()) / close

    feat["hour"] = df.index.hour
    feat["minute"] = df.index.minute

    feat["target"] = close.shift(-1).pct_change(-1) * -1
    feat["target_dir"] = (feat["target"] > 0).astype(int)

    return feat


def build_short_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix with bear-specific indicators for the short model.

    Extends build_bar_features() with 9 additional features tuned to detect
    panic selling, exhaustion, and liquidation cascades.
    """
    feat = build_bar_features(df)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    # 1. Volume spike (3-bar) — panic selling proxy
    avg_vol_3 = volume.rolling(3).mean()
    feat["volume_spike_3b"] = volume / avg_vol_3.replace(0, np.nan)

    # 2. Volume spike (10-bar) — sustained selling
    avg_vol_10 = volume.rolling(10).mean()
    feat["volume_spike_10b"] = volume / avg_vol_10.replace(0, np.nan)

    # 3. RSI divergence — price up but RSI down = bearish exhaustion
    rsi_14 = ta.rsi(close, length=14)
    if rsi_14 is not None and not rsi_14.empty:
        price_slope = close.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
        )
        rsi_slope = rsi_14.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
        )
        feat["rsi_divergence"] = (price_slope > 0).astype(float) * (rsi_slope < 0).astype(float)

    # 4. Funding rate extreme — over-leveraged longs (populated later from external data)
    if "funding_rate" in feat.columns:
        feat["funding_rate_extreme"] = (feat["funding_rate"] > 0.0005).astype(float)
    else:
        feat["funding_rate_extreme"] = 0.0

    # 5. Distance from local high — already falling
    feat["dist_from_local_high"] = (close - close.rolling(20).max()) / close

    # 6. Bearish engulfing candle
    body = close - open_
    prev_body = body.shift(1)
    feat["bearish_engulfing"] = ((body < 0) & (prev_body > 0) & (abs(body) > abs(prev_body))).astype(float)

    # 7. Red candle streak — consecutive red candles
    is_red = (close < open_).astype(int)
    streak = is_red.copy()
    for i in range(1, 6):
        streak = streak + is_red.shift(i).fillna(0)
    feat["red_candle_streak"] = streak

    # 8. High-low range spike — volatility expansion
    hlr = (high - low) / close
    avg_hlr = hlr.rolling(20).mean()
    feat["hlr_spike"] = hlr / avg_hlr.replace(0, np.nan)

    # 9. OBV slope — on-balance volume trend (negative = distribution)
    obv = ta.obv(close, volume)
    if obv is not None and not obv.empty:
        obv_slope = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
        )
        feat["obv_slope"] = obv_slope / (volume.rolling(10).mean().replace(0, np.nan))

    feat["target_dir_short"] = (feat.get("target", pd.Series(dtype=float)) < 0).astype(int)

    return feat


# ── Single-day replay ───────────────────────────────────────────────────────

def run_day(
    target_date: str,
    data_1m: dict[str, pd.DataFrame],
    data_5m: dict[str, pd.DataFrame],
    data_1h: dict[str, pd.DataFrame],
    regime_filter: bool = True,
    long_only: bool = False,
    verbose: bool = False,
    funding_data: dict[str, pd.DataFrame] | None = None,
    journal: SimJournal | None = None,
) -> dict:
    """Run a single-day 1-min replay. Returns result dict."""
    target_start = pd.Timestamp(f"{target_date} 00:00:00")
    target_end = pd.Timestamp(f"{target_date} 23:59:59")
    if long_only and USE_DUAL_MODEL:
        mode_label = "ADAPTIVE-V2"
    elif long_only:
        mode_label = "ADAPTIVE"
    elif regime_filter:
        mode_label = "REGIME"
    else:
        mode_label = "NO-FILTER"

    print(f"\n{'='*90}")
    print(f"  1-MIN REPLAY -- {target_date} [{mode_label}]")
    print(f"  Capital: ${CAPITAL:,.0f} | Conf: L={CONFIDENCE_LONG:.0%}/S={CONFIDENCE_SHORT:.0%} | SL: {SL_PCT:.1%} | TP: {TP_PCT:.1%} | Trail: {TRAIL_ACTIVATE:.1%}/{TRAIL_OFFSET:.1%}")
    print(f"  Model: Logistic(C={LOGISTIC_C}) | ATR gate: {MIN_ATR_PCT:.1%} | Symbols: {len(SYMBOLS)}")
    if long_only and USE_DUAL_MODEL:
        short_ml = "XGBoost" if SHORT_MODEL_TYPE == "xgboost" else "Logistic"
        print(f"  ADAPTIVE-V2: BULL->long, BEAR->short ({short_ml}), NEUTRAL->skip")
        print(f"  Short params: SL={SL_SHORT:.1%} | TP={TP_SHORT:.1%} | Trail={TRAIL_ACTIVATE_SHORT:.2%}/{TRAIL_OFFSET_SHORT:.2%} | Conf={CONFIDENCE_SHORT_MODEL:.0%}")
    elif long_only:
        print(f"  ADAPTIVE: BULL->long only, BEAR->skip, NEUTRAL->skip")
    elif regime_filter:
        print(f"  Regime gate: BULL->long only, BEAR->short only, NEUTRAL->skip")
    print(f"{'='*90}")

    # Determine bar_label
    sample_sym = next(iter(data_1m))
    sample_df = data_1m[sample_sym]
    has_1m_on_day = any(target_start <= t <= target_end for t in sample_df.index)
    bar_label = "1min" if has_1m_on_day else "5min"

    # Train model on data BEFORE target date
    print(f"\n  Training long model on data before {target_date}...")
    t0 = time.time()

    feature_cols = None
    train_X_list, train_y_list = [], []
    short_feature_cols = None
    short_X_list, short_y_list = [], []

    for sym, df in data_5m.items():
        temporal = df[df.index < target_start]
        if len(temporal) < 100:
            continue
        feat = build_bar_features(temporal)

        if data_1h and sym in data_1h:
            h_df = data_1h[sym]
            h_before = h_df[h_df.index < target_start]
            if len(h_before) >= 50:
                mtf = build_mtf_features(
                    pd.DataFrame({"Close": temporal["Close"]}), h_before
                )
                for k, v in mtf.items():
                    feat.loc[feat.index[-1], k] = v

        # Add funding rate feature to training data (efficient merge)
        if funding_data and sym in funding_data:
            fr_df = funding_data[sym].sort_index()
            feat = feat.sort_index()
            merged = pd.merge_asof(
                feat.reset_index(), fr_df.reset_index(),
                left_on="Datetime", right_on="Datetime", direction="backward"
            ).set_index("Datetime")
            feat["funding_rate"] = merged["funding_rate"].fillna(0.0)

        if feature_cols is None:
            exclude = {"target", "target_dir"}
            feature_cols = [c for c in feat.columns if c not in exclude]

        valid = feat.dropna(subset=["target_dir"]).tail(TRAIN_WINDOW)
        if len(valid) < 50:
            continue
        train_X_list.append(valid.reindex(columns=feature_cols, fill_value=0).fillna(0))
        train_y_list.append(valid["target_dir"])

        # Build short model training data (bear-specific features)
        if USE_DUAL_MODEL:
            sfeat = build_short_features(temporal)
            if funding_data and sym in funding_data:
                sfeat["funding_rate"] = feat.get("funding_rate", 0.0)
            if short_feature_cols is None:
                sexclude = {"target", "target_dir", "target_dir_short"}
                short_feature_cols = [c for c in sfeat.columns if c not in sexclude]
            svalid = sfeat.dropna(subset=["target_dir_short"]).tail(TRAIN_WINDOW)
            if len(svalid) >= 50:
                short_X_list.append(svalid.reindex(columns=short_feature_cols, fill_value=0).fillna(0))
                short_y_list.append(svalid["target_dir_short"])

    if not train_X_list:
        print("  ERROR: Not enough training data.")
        return {"error": "no training data"}

    X_train = pd.concat(train_X_list)
    y_train = pd.concat(train_y_list)
    model = SmartLogistic(params={"C": LOGISTIC_C, "class_weight": "balanced"})
    model.fit(X_train, y_train)
    print(f"  Long model: {len(X_train)} samples, {len(feature_cols)} features, "
          f"{time.time() - t0:.1f}s")

    model_short = None
    if USE_DUAL_MODEL and short_X_list:
        t1 = time.time()
        X_short = pd.concat(short_X_list)
        y_short = pd.concat(short_y_list)
        if SHORT_MODEL_TYPE == "xgboost":
            model_short = SmartXGBoost(params={
                "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 30,
                "scale_pos_weight": 1.0, "reg_alpha": 0.3, "reg_lambda": 2.0,
            })
        else:
            model_short = SmartLogistic(params={"C": LOGISTIC_C, "class_weight": "balanced"})
        model_short.fit(X_short, y_short)
        short_label = "XGBoost" if SHORT_MODEL_TYPE == "xgboost" else "Logistic"
        print(f"  Short model ({short_label}): {len(X_short)} samples, {len(short_feature_cols)} features, "
              f"{time.time() - t1:.1f}s")
    elif USE_DUAL_MODEL:
        print(f"  Short model: NOT ENOUGH DATA — shorts disabled for this day")

    # Replay target date bar-by-bar (try 1m first, then 5m)
    sim_bars_per_sym = {}
    for sym, df in data_1m.items():
        day_bars = df[(df.index >= target_start) & (df.index <= target_end)]
        if not day_bars.empty:
            sim_bars_per_sym[sym] = day_bars

    if not sim_bars_per_sym and data_5m:
        for sym, df in data_5m.items():
            day_bars = df[(df.index >= target_start) & (df.index <= target_end)]
            if not day_bars.empty:
                sim_bars_per_sym[sym] = day_bars
        if sim_bars_per_sym:
            bar_label = "5min"

    if not sim_bars_per_sym:
        print(f"  No bars on {target_date}, skipping.")
        return {"error": "no sim data"}
    actual_date = target_date

    all_sim_times = sorted(set().union(*(df.index for df in sim_bars_per_sym.values())))
    print(f"  Replaying {len(all_sim_times)} bars on {actual_date} ({bar_label})...")

    # Simulation state
    cash = CAPITAL
    positions = {}
    trades = []
    bar_log = []
    total_costs = 0.0
    wins = 0
    losses = 0
    entries = 0
    exits = 0

    peak = CAPITAL
    max_dd = 0.0

    hour_summaries = {}
    regime_blocks = 0

    # Use BTC hourly as regime reference (most liquid)
    regime_hourly = data_1h.get("BTC-USD")

    for bar_idx, bar_time in enumerate(all_sim_times):
        bar_trades = []
        bar_pnl = 0.0

        for sym, day_df in sim_bars_per_sym.items():
            if bar_time not in day_df.index:
                continue

            price = float(day_df.loc[bar_time, "Close"])
            if price <= 0:
                continue

            # Strict temporal: only see data up to this bar
            # Use 1m if available on this day, otherwise fall back to 5m
            full_1m = data_1m.get(sym)
            if full_1m is not None:
                temporal = full_1m[full_1m.index <= bar_time]
            if full_1m is None or len(temporal) < 30:
                temporal = data_5m[sym][data_5m[sym].index <= bar_time] if sym in data_5m else pd.DataFrame()
            if len(temporal) < 30:
                continue

            feat = build_bar_features(temporal)

            # MTF: hourly features up to this bar
            if data_1h and sym in data_1h:
                h_before = data_1h[sym][data_1h[sym].index < bar_time]
                if len(h_before) >= 50:
                    mtf = build_mtf_features(
                        pd.DataFrame({"Close": temporal["Close"]}), h_before
                    )
                    for k, v in mtf.items():
                        feat.loc[bar_time, k] = v

            # Funding rate feature (sentiment proxy)
            if funding_data and bar_time in feat.index:
                fr = get_funding_rate_at(funding_data, sym, bar_time)
                feat.loc[bar_time, "funding_rate"] = fr

            if bar_time not in feat.index:
                continue

            # Long model prediction (always available)
            row = feat.loc[bar_time]
            row_feats = row.reindex(feature_cols, fill_value=0).fillna(0)
            X_pred = row_feats.values.reshape(1, -1)
            proba = model.predict_proba(X_pred)[0]
            up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            down_prob = 1.0 - up_prob

            # Short model prediction — only compute in BEAR regime
            short_down_prob = 0.0
            if model_short is not None and short_feature_cols is not None:
                cur_regime = compute_regime(regime_hourly, bar_time) if regime_filter else "NEUTRAL"
                if cur_regime == "BEAR":
                    # Build bear features incrementally from existing feat + temporal
                    srow_data = row.copy()
                    close_s = temporal["Close"]
                    open_s = temporal["Open"]
                    high_s = temporal["High"]
                    low_s = temporal["Low"]
                    vol_s = temporal["Volume"]
                    # 1. Volume spike
                    vol_mean = vol_s.iloc[-20:].mean() if len(vol_s) >= 20 else vol_s.mean()
                    srow_data["volume_spike"] = float(vol_s.iloc[-1] / vol_mean) if vol_mean > 0 else 1.0
                    # 2. RSI divergence (simplified: compare last 10-bar price vs RSI direction)
                    rsi_val = srow_data.get("rsi_14", 50.0)
                    if len(close_s) >= 10:
                        p_chg = float(close_s.iloc[-1] - close_s.iloc[-10])
                        srow_data["rsi_divergence"] = 1.0 if (p_chg > 0 and rsi_val < 50) else 0.0
                    else:
                        srow_data["rsi_divergence"] = 0.0
                    # 3. Funding rate extreme
                    fr_val = srow_data.get("funding_rate", 0.0)
                    srow_data["funding_rate_extreme"] = 1.0 if fr_val > 0.0005 else 0.0
                    # 4. Distance from local high
                    local_high = float(close_s.iloc[-20:].max()) if len(close_s) >= 20 else float(close_s.max())
                    cur_price = float(close_s.iloc[-1])
                    srow_data["dist_from_local_high"] = (cur_price - local_high) / cur_price if cur_price > 0 else 0.0
                    # 5. Bearish engulfing
                    if len(close_s) >= 2 and len(open_s) >= 2:
                        body = float(close_s.iloc[-1] - open_s.iloc[-1])
                        prev_body = float(close_s.iloc[-2] - open_s.iloc[-2])
                        srow_data["bearish_engulfing"] = 1.0 if (body < 0 and prev_body > 0 and abs(body) > abs(prev_body)) else 0.0
                    else:
                        srow_data["bearish_engulfing"] = 0.0
                    # 6. Red candle streak
                    reds = 0
                    for k in range(1, min(7, len(close_s) + 1)):
                        if float(close_s.iloc[-k]) < float(open_s.iloc[-k]):
                            reds += 1
                        else:
                            break
                    srow_data["red_candle_streak"] = float(reds)
                    # 7. HLR spike
                    hlr = float((high_s.iloc[-1] - low_s.iloc[-1]) / close_s.iloc[-1]) if cur_price > 0 else 0.0
                    avg_hlr = float((high_s.iloc[-20:] - low_s.iloc[-20:]).div(close_s.iloc[-20:]).mean()) if len(close_s) >= 20 else hlr
                    srow_data["hlr_spike"] = hlr / avg_hlr if avg_hlr > 0 else 1.0
                    # 8. OBV slope (simplified: compare OBV over last 10 bars)
                    if len(close_s) >= 10 and len(vol_s) >= 10:
                        obv_end = 0.0
                        obv_start = 0.0
                        for k in range(max(0, len(close_s) - 10), len(close_s)):
                            sign = 1.0 if close_s.iloc[k] > close_s.iloc[k - 1] else (-1.0 if close_s.iloc[k] < close_s.iloc[k - 1] else 0.0) if k > 0 else 0.0
                            if k < len(close_s) - 5:
                                obv_start += sign * vol_s.iloc[k]
                            obv_end += sign * vol_s.iloc[k]
                        vol_avg = float(vol_s.iloc[-10:].mean())
                        srow_data["obv_slope"] = (obv_end - obv_start) / vol_avg if vol_avg > 0 else 0.0
                    else:
                        srow_data["obv_slope"] = 0.0

                    srow_feats = srow_data.reindex(short_feature_cols, fill_value=0).fillna(0)
                    X_short_pred = srow_feats.values.reshape(1, -1)
                    sproba = model_short.predict_proba(X_short_pred)[0]
                    short_down_prob = float(sproba[1]) if len(sproba) > 1 else float(sproba[0])

            # Check exits
            if sym in positions:
                pos = positions[sym]
                entry = pos["entry_price"]
                if pos["side"] == "long":
                    pnl_pct = (price - entry) / entry
                    if USE_RATCHET and pnl_pct > ML_PROFIT_THRESHOLD:
                        ml_exit = down_prob > ML_PROFIT_CONFIDENCE
                    else:
                        ml_exit = down_prob > CONFIDENCE_LONG
                else:
                    pnl_pct = (entry - price) / entry
                    if USE_RATCHET and pnl_pct > ML_PROFIT_THRESHOLD:
                        ml_exit = up_prob > ML_PROFIT_CONFIDENCE
                    else:
                        ml_exit = up_prob > CONFIDENCE_SHORT

                held_bars = bar_idx - pos.get("bar_idx", 0)
                is_short_pos = pos["side"] == "short"
                min_hold = MIN_HOLD_BARS_SHORT if is_short_pos else MIN_HOLD_BARS
                can_ml_exit = held_bars >= min_hold

                # Select risk params based on position side
                pos_sl = SL_SHORT if is_short_pos else SL_PCT
                pos_tp = TP_SHORT if is_short_pos else TP_PCT
                pos_trail_act = TRAIL_ACTIVATE_SHORT if is_short_pos else TRAIL_ACTIVATE
                pos_trail_off = TRAIL_OFFSET_SHORT if is_short_pos else TRAIL_OFFSET

                # Track peak for trailing stop (used by both modes)
                if pnl_pct > pos.get("peak_pnl", 0):
                    pos["peak_pnl"] = pnl_pct

                if USE_RATCHET:
                    # Hybrid ratchet: trailing stop + progressive SL/TP
                    ratchet_level = max(0, int(pnl_pct / RATCHET_STEP))
                    prev_level = pos.get("ratchet_level", 0)
                    if ratchet_level > prev_level:
                        pos["ratchet_level"] = ratchet_level
                    cur_level = pos.get("ratchet_level", 0)

                    if cur_level >= 1:
                        dynamic_sl = cur_level * RATCHET_STEP - RATCHET_SL_LOCK
                        sl_exit = pnl_pct <= dynamic_sl
                    else:
                        sl_exit = pnl_pct <= -pos_sl
                    dynamic_tp = pos_tp + cur_level * RATCHET_TP_EXTEND
                    tp_exit = pnl_pct >= dynamic_tp

                    # Trailing stop as safety net (catches gains before ratchet level 1)
                    trail_exit = (
                        cur_level == 0
                        and pos.get("peak_pnl", 0) >= pos_trail_act
                        and pnl_pct <= pos["peak_pnl"] - pos_trail_off
                    )
                else:
                    # Legacy trailing stop
                    trail_exit = (
                        pos.get("peak_pnl", 0) >= pos_trail_act
                        and pnl_pct <= pos["peak_pnl"] - pos_trail_off
                    )
                    sl_exit = pnl_pct <= -pos_sl
                    tp_exit = pnl_pct >= pos_tp

                if sl_exit or tp_exit or (ml_exit and can_ml_exit) or trail_exit:
                    shares = pos["shares"]
                    # HONEST-SIM: slippage on fills. Stops slip harder than limits.
                    # Long close fills at bid (below mid); short cover fills at ask.
                    exit_slip = SLIPPAGE_BPS + (SL_SLIPPAGE_BPS if sl_exit else 0.0)
                    if pos["side"] == "long":
                        fill_price = price * (1.0 - exit_slip)
                    else:
                        fill_price = price * (1.0 + exit_slip)
                    cost = fill_price * shares * COST_PCT
                    if pos["side"] == "long":
                        gross = (fill_price - entry) * shares
                        cash += fill_price * shares - cost
                    else:
                        gross = (entry - fill_price) * shares
                        cash += entry * shares + gross - cost
                    net = gross - pos["cost"] - cost
                    total_costs += cost
                    bar_pnl += net
                    # Use fill_price as the recorded exit price for journal/reports.
                    price = fill_price

                    if USE_RATCHET and pos.get("ratchet_level", 0) >= 1:
                        reason = ("RT" if tp_exit
                                  else "RS" if sl_exit
                                  else "ML")
                    else:
                        reason = ("TP" if tp_exit
                                  else "SL" if sl_exit
                                  else "TR" if trail_exit else "ML")
                    if net > 0:
                        wins += 1
                    else:
                        losses += 1
                    exits += 1

                    held_b = bar_idx - pos.get("bar_idx", 0)
                    bar_trades.append({
                        "time": str(bar_time)[11:19], "sym": sym, "action": "EXIT",
                        "side": pos["side"], "entry": entry, "exit": price,
                        "pnl": net, "pnl_pct": pnl_pct * 100, "reason": reason,
                        "held": held_b,
                    })
                    if journal:
                        journal.record_trade(
                            mode=mode_label, date=actual_date, symbol=sym,
                            side=pos["side"], entry_time=pos["entry_time_full"],
                            entry_price=entry, entry_confidence=pos.get("entry_conf", 0),
                            exit_time=str(bar_time), exit_price=price,
                            exit_reason=reason, pnl=net, pnl_pct=pnl_pct * 100,
                            hold_bars=held_b, regime=pos.get("entry_regime", ""),
                            funding_rate=pos.get("entry_funding", 0.0),
                            atr_pct=pos.get("entry_atr", 0.0),
                            up_prob=up_prob, down_prob=down_prob,
                        )
                    del positions[sym]

            # Check entries
            elif len(positions) < MAX_POSITIONS:
                direction = None
                conf = 0.0

                # Dual-model entry: use short model in BEAR, long model in BULL
                if USE_DUAL_MODEL and long_only and regime_filter:
                    regime = compute_regime(regime_hourly, bar_time)
                    if regime == "BULL" and up_prob > CONFIDENCE_LONG:
                        direction = "long"
                        conf = up_prob
                    elif regime == "BEAR" and model_short is not None and short_down_prob > CONFIDENCE_SHORT_MODEL:
                        direction = "short"
                        conf = short_down_prob
                    elif regime == "NEUTRAL":
                        regime_blocks += 1
                    else:
                        regime_blocks += 1
                else:
                    if up_prob > CONFIDENCE_LONG:
                        direction = "long"
                        conf = up_prob
                    elif down_prob > CONFIDENCE_SHORT and not long_only:
                        direction = "short"
                        conf = down_prob

                # Volatility filter: skip low-ATR (choppy) conditions
                if direction:
                    atr_val = row.get("atr_pct", 0.0)
                    if atr_val < MIN_ATR_PCT:
                        direction = None

                # Regime gate: filter against-trend entries (non-dual-model path)
                if direction and regime_filter and not (USE_DUAL_MODEL and long_only):
                    regime = compute_regime(regime_hourly, bar_time)
                    if direction == "long" and regime == "BEAR":
                        regime_blocks += 1
                        direction = None
                    elif direction == "short" and regime == "BULL":
                        regime_blocks += 1
                        direction = None
                    elif regime == "NEUTRAL":
                        regime_blocks += 1
                        direction = None

                if direction:
                    # HONEST-SIM: entry slips the worse side of the spread.
                    # Long buys the ask (above mid); short sells the bid (below mid).
                    if direction == "long":
                        entry_fill = price * (1.0 + SLIPPAGE_BPS)
                    else:
                        entry_fill = price * (1.0 - SLIPPAGE_BPS)
                    max_val = cash * MAX_POS_PCT
                    shares = max_val / entry_fill
                    if shares * entry_fill < 0.50:
                        continue
                    cost = entry_fill * shares * COST_PCT
                    debit = entry_fill * shares + cost
                    if debit > cash:
                        continue
                    cash -= debit
                    total_costs += cost
                    entry_regime = compute_regime(regime_hourly, bar_time) if regime_filter else ""
                    entry_funding = get_funding_rate_at(funding_data or {}, sym, bar_time)
                    positions[sym] = {
                        "side": direction, "shares": shares,
                        "entry_price": entry_fill, "cost": cost,
                        "entry_time": str(bar_time)[11:19],
                        "entry_time_full": str(bar_time),
                        "entry_conf": conf,
                        "entry_regime": entry_regime,
                        "entry_funding": entry_funding,
                        "entry_atr": float(row.get("atr_pct", 0.0)),
                        "bar_idx": bar_idx,
                    }
                    entries += 1
                    bar_trades.append({
                        "time": str(bar_time)[11:19], "sym": sym, "action": "ENTER",
                        "side": direction, "price": entry_fill, "conf": conf,
                        "size": shares * entry_fill,
                    })

        # Portfolio value
        port_val = cash
        for s, p in positions.items():
            if s in sim_bars_per_sym and bar_time in sim_bars_per_sym[s].index:
                cp = float(sim_bars_per_sym[s].loc[bar_time, "Close"])
            else:
                cp = p["entry_price"]
            if p["side"] == "long":
                port_val += p["shares"] * cp
            else:
                port_val += p["entry_price"] * p["shares"] + (p["entry_price"] - cp) * p["shares"]

        if port_val > peak:
            peak = port_val
        dd = (peak - port_val) / peak * 100
        if dd > max_dd:
            max_dd = dd

        pnl_from_start = port_val - CAPITAL
        ret_pct = pnl_from_start / CAPITAL * 100

        bar_log.append({
            "time": str(bar_time)[11:19],
            "value": port_val,
            "pnl": pnl_from_start,
            "ret_pct": ret_pct,
            "positions": len(positions),
            "dd_pct": dd,
        })

        # Hourly summary
        hour_key = str(bar_time)[11:13]
        if hour_key not in hour_summaries:
            hour_summaries[hour_key] = {
                "start_val": port_val, "end_val": port_val,
                "trades_in": 0, "trades_out": 0,
            }
        hour_summaries[hour_key]["end_val"] = port_val
        hour_summaries[hour_key]["trades_in"] += sum(1 for t in bar_trades if t["action"] == "ENTER")
        hour_summaries[hour_key]["trades_out"] += sum(1 for t in bar_trades if t["action"] == "EXIT")

        # Verbose output for trades
        if bar_trades and verbose:
            for t in bar_trades:
                if t["action"] == "ENTER":
                    arrow = "B" if t["side"] == "long" else "S"
                    print(f"  {t['time']} [{arrow}] ENTER {t['side']:>5} {t['sym']:<10} "
                          f"@ ${t['price']:>10.4f}  conf={t['conf']:.3f}  "
                          f"size=${t['size']:.2f}")
                else:
                    marker = "W" if t["pnl"] > 0 else "L"
                    print(f"  {t['time']} [{marker}] EXIT  {t['side']:>5} {t['sym']:<10} "
                          f"${t['entry']:>10.4f} -> ${t['exit']:>10.4f}  "
                          f"pnl=${t['pnl']:>+8.4f} ({t['pnl_pct']:>+5.2f}%) "
                          f"[{t['reason']}]  held={t['held']}bars")

        # Progress every 60 bars
        if bar_idx > 0 and bar_idx % 60 == 0 and not verbose:
            regime_now = compute_regime(regime_hourly, bar_time) if regime_filter else "-"
            print(f"  {str(bar_time)[11:16]} | Val: ${port_val:>8.2f} | "
                  f"PnL: ${pnl_from_start:>+7.2f} ({ret_pct:>+5.2f}%) | "
                  f"Pos: {len(positions)} | DD: {dd:.2f}% | "
                  f"W/L: {wins}W/{losses}L | Regime: {regime_now}")

        trades.extend(bar_trades)

    # ── Final summary ───────────────────────────────────────────────────────
    final_val = bar_log[-1]["value"] if bar_log else CAPITAL
    total_pnl = final_val - CAPITAL
    total_ret = total_pnl / CAPITAL * 100

    wr_str = f"{wins/(wins+losses)*100:.1f}%" if (wins + losses) > 0 else "N/A"
    wr_num = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n  -----------------------------------------------------------------------")
    print(f"  RESULT -- {actual_date} [{mode_label}] | "
          f"Ret: {total_ret:+.4f}% | W/L: {wins}W/{losses}L | "
          f"DD: {max_dd:.4f}% | Blocked: {regime_blocks}")
    print(f"  -----------------------------------------------------------------------")

    print(f"\n  {'METRIC':<28} {'VALUE':>14}")
    print(f"  {'-'*44}")
    print(f"  {'Initial Capital':<28} ${CAPITAL:>13,.2f}")
    print(f"  {'Final Value':<28} ${final_val:>13,.4f}")
    print(f"  {'Net P&L':<28} ${total_pnl:>+13,.4f}")
    print(f"  {'Return %':<28} {total_ret:>+13.4f}%")
    print(f"  {'Entries':<28} {entries:>14}")
    print(f"  {'Exits (closed)':<28} {exits:>14}")
    print(f"  {'Wins / Losses':<28} {f'{wins}W / {losses}L':>14}")
    print(f"  {'Win Rate':<28} {wr_str:>14}")
    print(f"  {'Regime Blocks':<28} {regime_blocks:>14}")
    print(f"  {'Open Positions':<28} {len(positions):>14}")
    print(f"  {'Max Drawdown':<28} {max_dd:>13.4f}%")
    print(f"  {'Total Costs':<28} ${total_costs:>13,.6f}")

    # Hourly P&L breakdown
    print(f"\n  HOURLY P&L BREAKDOWN:")
    print(f"  {'Hour':<6} {'End Value':>10} {'Hour PnL':>10} {'Entries':>8} {'Exits':>7}")
    print(f"  {'-'*43}")
    for hour in sorted(hour_summaries.keys()):
        h = hour_summaries[hour]
        h_pnl = h["end_val"] - h["start_val"]
        print(f"  {hour}:00  ${h['end_val']:>9.2f} ${h_pnl:>+9.4f} "
              f"{h['trades_in']:>8} {h['trades_out']:>7}")

    # All closed trades
    exit_trades = [t for t in trades if t["action"] == "EXIT"]
    if exit_trades:
        print(f"\n  ALL CLOSED TRADES:")
        print(f"  {'Time':<10} {'Side':>5} {'Symbol':<10} {'Entry':>10} {'Exit':>10} "
              f"{'PnL':>10} {'PnL%':>7} {'Reason':>6} {'Held':>6}")
        print(f"  {'-'*80}")
        for t in exit_trades:
            marker = "W" if t["pnl"] > 0 else "L"
            print(f"  {t['time']:<10} {t['side']:>5} {t['sym']:<10} "
                  f"${t['entry']:>9.2f} ${t['exit']:>9.2f} "
                  f"${t['pnl']:>+9.4f} {t['pnl_pct']:>+6.2f}% "
                  f"{t['reason']:>6} {t['held']:>5}b [{marker}]")

        avg_pnl = np.mean([t["pnl"] for t in exit_trades])
        avg_win = np.mean([t["pnl"] for t in exit_trades if t["pnl"] > 0]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in exit_trades if t["pnl"] <= 0]) if losses else 0
        print(f"\n  Avg PnL:  ${avg_pnl:+.4f} | Avg Win: ${avg_win:+.4f} | Avg Loss: ${avg_loss:+.4f}")
        if avg_loss != 0 and losses > 0:
            print(f"  Profit Factor: {abs(avg_win * wins) / abs(avg_loss * losses):.2f}")

    # Open positions
    if positions:
        print(f"\n  OPEN POSITIONS (end of day):")
        for sym, pos in positions.items():
            if sym in sim_bars_per_sym:
                last_bar = all_sim_times[-1]
                if last_bar in sim_bars_per_sym[sym].index:
                    cp = float(sim_bars_per_sym[sym].loc[last_bar, "Close"])
                else:
                    cp = pos["entry_price"]
            else:
                cp = pos["entry_price"]
            if pos["side"] == "long":
                unrealized = (cp - pos["entry_price"]) * pos["shares"]
            else:
                unrealized = (pos["entry_price"] - cp) * pos["shares"]
            print(f"    {pos['side']:>5} {sym:<10} entry=${pos['entry_price']:.4f} "
                  f"now=${cp:.4f} unrealized=${unrealized:+.4f}")

    # Record daily summary in journal
    if journal:
        exit_trades_j = [t for t in trades if t["action"] == "EXIT"]
        avg_win_j = float(np.mean([t["pnl"] for t in exit_trades_j if t["pnl"] > 0])) if wins else 0.0
        avg_loss_j = float(np.mean([t["pnl"] for t in exit_trades_j if t["pnl"] <= 0])) if losses else 0.0
        pf_j = abs(avg_win_j * wins) / abs(avg_loss_j * losses) if (avg_loss_j != 0 and losses > 0) else 0.0
        journal.record_daily(
            mode=mode_label, date=actual_date, return_pct=total_ret,
            wins=wins, losses=losses, entries=entries, exits=exits,
            regime_blocks=regime_blocks, max_dd=max_dd,
            final_value=final_val, avg_win_pnl=avg_win_j,
            avg_loss_pnl=avg_loss_j, profit_factor=pf_j,
        )
        journal.flush_predictions()

    return {
        "date": actual_date,
        "mode": mode_label,
        "return_pct": total_ret,
        "pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": wr_num,
        "max_dd": max_dd,
        "entries": entries,
        "exits": exits,
        "regime_blocks": regime_blocks,
        "open_pos": len(positions),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="1-min Time-Machine Replay")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD")
    parser.add_argument("--all-days", action="store_true", help="Run all available days")
    parser.add_argument("--backtest-30d", action="store_true", help="Backtest on 5m bars, excluding last 7 days")
    parser.add_argument("--no-regime-filter", action="store_true", help="Disable regime gate")
    parser.add_argument("--binance", action="store_true", help="[DEPRECATED — default] Use Binance API for 1-min data")
    parser.add_argument("--yahoo", action="store_true", help="Use Yahoo Finance (7 days 1-min, intraday data quality poor)")
    parser.add_argument("--days", type=int, default=60, help="Days of history for Binance (default: 60)")
    parser.add_argument("--no-funding", action="store_true", help="Disable funding rate feature (baseline mode)")
    parser.add_argument("--conf-long", type=float, default=None, help="Override CONFIDENCE_LONG threshold")
    parser.add_argument("--ratchet", action="store_true", help="Enable ratchet SL/TP (experimental, off by default)")
    parser.add_argument("--adaptive-v2", action="store_true", help="Enable dual model: long in BULL + short in BEAR")
    parser.add_argument("--short-logistic", action="store_true", help="Use Logistic for short model (default: XGBoost)")
    parser.add_argument("--optimistic-costs", action="store_true",
                        help="Revert to legacy 10bps cost, no slippage (matches pre-2026-04-20 sim)")
    args = parser.parse_args()

    # HONEST-SIM: allow user to revert to the old optimistic cost model for comparison.
    if args.optimistic_costs:
        global COST_PCT, SLIPPAGE_BPS, SL_SLIPPAGE_BPS
        COST_PCT = 0.001
        SLIPPAGE_BPS = 0.0
        SL_SLIPPAGE_BPS = 0.0
        print(f"  ** OPTIMISTIC COSTS: 10 bps per side, zero slippage (LEGACY MODE) **")
    else:
        print(f"  ** HONEST COSTS: {COST_PCT*10000:.0f}bps/side + {SLIPPAGE_BPS*10000:.0f}bps slip "
              f"+ {SL_SLIPPAGE_BPS*10000:.0f}bps extra on stops **")

    if args.conf_long is not None:
        global CONFIDENCE_LONG
        CONFIDENCE_LONG = args.conf_long
        print(f"  ** CONFIDENCE_LONG overridden to {CONFIDENCE_LONG:.2f} **")

    if args.ratchet:
        global USE_RATCHET
        USE_RATCHET = True
        print(f"  ** Ratchet ENABLED — hybrid progressive SL/TP **")

    if args.adaptive_v2:
        global USE_DUAL_MODEL
        USE_DUAL_MODEL = True
        print(f"  ** ADAPTIVE-V2 ENABLED — dual model: long(BULL) + short(BEAR) **")
        print(f"     Short params: SL={SL_SHORT:.1%} TP={TP_SHORT:.1%} Trail={TRAIL_ACTIVATE_SHORT:.2%}/{TRAIL_OFFSET_SHORT:.2%} Conf={CONFIDENCE_SHORT_MODEL:.0%}")

    if args.short_logistic:
        global SHORT_MODEL_TYPE
        SHORT_MODEL_TYPE = "logistic"
        print(f"  ** Short model: Logistic (override — default is XGBoost) **")

    use_regime = not args.no_regime_filter

    print("=" * 90)
    print("  TIME-MACHINE 1-MIN REPLAY WITH REGIME FILTER")
    print("=" * 90)

    # HONEST-SIM: Binance is the default data source. Yahoo is opt-in via --yahoo
    # because Yahoo intraday bars have gaps, post-hoc corrections, and don't
    # match the prices you'd actually trade against live.
    # When Binance is used, training and prediction both run on 1-min bars
    # (no timeframe mismatch), and TRAIN_WINDOW is expanded to match.
    use_binance = not args.yahoo  # default: Binance; --yahoo to opt-in to legacy

    if use_binance:
        global TRAIN_WINDOW
        if TRAIN_WINDOW < 20000:
            TRAIN_WINDOW = 20000  # ~2 weeks of 1-min bars
            print(f"  ** TRAIN_WINDOW expanded to {TRAIN_WINDOW} (1-min Binance data) **")
        fetch_days = args.days
        print(f"\n  [1/3] Fetching 1-min bars from Binance ({fetch_days} days)...")
        t0 = time.time()
        data_1m = fetch_binance_1m(SYMBOLS, days=fetch_days)
        data_5m = data_1m  # train + predict on same 1-min timeframe
        total_bars = max(len(df) for df in data_1m.values()) if data_1m else 0
        print(f"         {len(data_1m)} symbols, {total_bars:,} bars, {time.time() - t0:.0f}s")

        print(f"\n  [2/3] Fetching 1h bars from Binance (90 days)...")
        t0 = time.time()
        data_1h = fetch_binance_1h(SYMBOLS, days=90)
        print(f"         {len(data_1h)} symbols, {time.time() - t0:.0f}s")

        if not args.no_funding:
            print(f"\n  [3/3] Fetching funding rates from Binance (90 days)...")
            t0 = time.time()
            funding_data = fetch_binance_funding_rates(SYMBOLS, days=90)
            fr_count = sum(len(df) for df in funding_data.values()) if funding_data else 0
            print(f"         {len(funding_data)} symbols, {fr_count} rates, {time.time() - t0:.0f}s")
        else:
            funding_data = {}
            print(f"\n  [3/3] Funding rates: DISABLED (baseline mode)")
    else:
        print("\n  ** YAHOO MODE: intraday data is lagged, gappy, and does not match")
        print("     the prices you could actually trade against. Results are not live-faithful. **")
        print("\n  [1/3] Fetching 1-min bars (7 days)...")
        t0 = time.time()
        data_1m = fetch_1min_data(SYMBOLS)
        if not data_1m:
            data_1m = fetch_5min_data(SYMBOLS)
        print(f"         {len(data_1m)} symbols, "
              f"~{max(len(df) for df in data_1m.values()) if data_1m else 0} bars, "
              f"{time.time() - t0:.0f}s")

        print("\n  [2/3] Fetching 5-min bars (59 days, training)...")
        t0 = time.time()
        data_5m = fetch_5min_data(SYMBOLS)
        print(f"         {len(data_5m)} symbols, {time.time() - t0:.0f}s")

        print("\n  [3/3] Fetching 1h bars (60 days, MTF + regime)...")
        t0 = time.time()
        data_1h = fetch_1h_data(SYMBOLS)
        print(f"         {len(data_1h)} symbols, {time.time() - t0:.0f}s")
        funding_data = {}

    # Determine available dates
    all_dates_set = set()
    for df in data_1m.values():
        for d in df.index.normalize().unique():
            all_dates_set.add(str(d.date()))
    dates_1m = sorted(all_dates_set)

    if args.backtest_30d:
        all_5m_dates = set()
        for df in data_5m.values():
            for d in df.index.normalize().unique():
                all_5m_dates.add(str(d.date()))
        exclude = set(dates_1m)
        available_dates = sorted(all_5m_dates - exclude)
    else:
        available_dates = dates_1m

    if args.all_days or args.backtest_30d:
        dates_to_run = available_dates
    elif args.date:
        dates_to_run = [args.date]
    else:
        dates_to_run = available_dates

    print(f"\n  Available dates: {', '.join(available_dates)}")
    print(f"  Running: {', '.join(dates_to_run)}")
    print(f"  Regime filter: {'ON' if use_regime else 'OFF'}")

    # Initialize journal
    run_id = f"backtest_{args.days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    journal = SimJournal(run_id=run_id)
    journal.clear_run()
    print(f"  Journal: {journal.db_path} (run: {run_id})")

    # Store funding rates in journal
    if funding_data:
        for sym, df in funding_data.items():
            rates = [(str(ts), float(r)) for ts, r in zip(df.index, df["funding_rate"])]
            journal.store_funding_rates(sym, rates)

    results = []
    for d in dates_to_run:
        # Primary: ADAPTIVE (long-only in BULL, skip BEAR/NEUTRAL)
        r_adapt = run_day(d, data_1m, data_5m, data_1h,
                          regime_filter=True, long_only=True, verbose=args.verbose,
                          funding_data=funding_data, journal=journal)
        if r_adapt and "error" not in r_adapt:
            results.append(r_adapt)

        # Comparison: REGIME (long+short with regime gate)
        if use_regime:
            r_regime = run_day(d, data_1m, data_5m, data_1h,
                               regime_filter=True, long_only=False, verbose=False,
                               funding_data=funding_data, journal=journal)
            if r_regime and "error" not in r_regime:
                results.append(r_regime)

    # ── Multi-day comparison table ──────────────────────────────────────────
    primary_mode = "ADAPTIVE-V2" if USE_DUAL_MODEL else "ADAPTIVE"
    if len(results) > 1:
        print(f"\n\n{'='*90}")
        print(f"  MULTI-DAY COMPARISON: {primary_mode} vs REGIME")
        print(f"{'='*90}")
        print(f"\n  {'Date':<12} {'Mode':<14} {'Return':>8} {'W/L':>8} {'WR':>6} "
              f"{'MaxDD':>7} {'Entries':>8} {'Blocked':>8}")
        print(f"  {'-'*78}")

        totals = {}
        for mode in [primary_mode, "REGIME"]:
            totals[mode] = {"ret": 0.0, "wins": 0, "losses": 0}

        for r in sorted(results, key=lambda x: (x["date"], x["mode"])):
            wl = f"{r['wins']}W/{r['losses']}L"
            wr = f"{r['win_rate']:.0f}%" if r['exits'] > 0 else "N/A"
            print(f"  {r['date']:<12} {r['mode']:<14} {r['return_pct']:>+7.3f}% "
                  f"{wl:>8} {wr:>6} {r['max_dd']:>6.3f}% "
                  f"{r['entries']:>8} {r['regime_blocks']:>8}")

            if r["mode"] in totals:
                totals[r["mode"]]["ret"] += r["return_pct"]
                totals[r["mode"]]["wins"] += r["wins"]
                totals[r["mode"]]["losses"] += r["losses"]

        n_days = len(dates_to_run)
        for mode in [primary_mode, "REGIME"]:
            t = totals[mode]
            print(f"\n  {'TOTALS':<12} {mode:<14} {t['ret']:>+7.3f}% "
                  f"{t['wins']}W/{t['losses']}L")

        delta = totals[primary_mode]["ret"] - totals["REGIME"]["ret"]
        print(f"\n  {primary_mode} vs REGIME: {delta:+.3f}% advantage over {n_days} days")

    # ── Analysis Report from Journal ──────────────────────────────────────
    journal.print_report(primary_mode)
    if use_regime:
        journal.print_report("REGIME")
    journal.close()

    print(f"\n{'='*90}")
    print(f"  DONE — Results saved to {journal.db_path}")
    print(f"  Run ID: {run_id}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
