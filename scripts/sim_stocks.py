#!/usr/bin/env python3
"""Time-Machine 1-Minute Replay — Volatile Stocks Edition.

Targeting high-volatility stocks (MARA, MSTR, COIN, W, TSLA).
Includes BULL-LONG mode: in BULL regime, lower confidence for longs
to generate more long entries and capture bull momentum.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_stocks.py --all-days
    PYTHONPATH=. .venv/bin/python scripts/sim_stocks.py --all-days --bull-long
    PYTHONPATH=. .venv/bin/python scripts/sim_stocks.py --no-regime-filter
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
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_engine.models.classifiers import SmartLogistic
from trading_engine.price_engine import build_mtf_features

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
CAPITAL = 1000.0
CONFIDENCE = 0.65
BULL_LONG_CONFIDENCE = 0.55
LOGISTIC_C = 0.15
SL_PCT = 0.015
TP_PCT = 0.025
TRAIL_ACTIVATE = 0.010
TRAIL_OFFSET = 0.005
MAX_POS_PCT = 0.20
COST_PCT = 0.001
MAX_POSITIONS = 5
TRAIN_WINDOW = 2000
MIN_HOLD_BARS = 15
MIN_ATR_PCT = 0.0005

# ── ATR-scaled exits (--atr-exits) ────────────────────────────────────────────
# Exit thresholds = multiples of the symbol's ATR% at entry, so they adapt to
# each ticker's volatility instead of using fixed % tuned for high-vol names.
# Floors/caps keep thresholds sane: TP must clear round-trip cost; SL bounded.
ATR_SL_MULT = 3.0
ATR_TP_MULT = 5.0
ATR_TRAIL_ACTIVATE_MULT = 2.0
ATR_TRAIL_OFFSET_MULT = 1.0
SL_FLOOR, SL_CAP = 0.004, 0.020
TP_FLOOR, TP_CAP = 0.006, 0.035  # TP_FLOOR > 2x round-trip cost (0.2%)
TRAIL_ACTIVATE_FLOOR = 0.004
TRAIL_OFFSET_FLOOR = 0.002


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_exit_levels(atr_pct: float, use_atr: bool) -> dict:
    """Per-position exit thresholds. ATR-scaled when use_atr, else fixed globals."""
    if not use_atr or atr_pct <= 0:
        return {"sl": SL_PCT, "tp": TP_PCT,
                "trail_act": TRAIL_ACTIVATE, "trail_off": TRAIL_OFFSET}
    return {
        "sl": _clamp(atr_pct * ATR_SL_MULT, SL_FLOOR, SL_CAP),
        "tp": _clamp(atr_pct * ATR_TP_MULT, TP_FLOOR, TP_CAP),
        "trail_act": max(atr_pct * ATR_TRAIL_ACTIVATE_MULT, TRAIL_ACTIVATE_FLOOR),
        "trail_off": max(atr_pct * ATR_TRAIL_OFFSET_MULT, TRAIL_OFFSET_FLOOR),
    }

TARGET_DATE = "2026-04-15"
SYMBOLS = ["MARA", "MSTR", "COIN"]

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


# ── Single-day replay ───────────────────────────────────────────────────────

def run_day(
    target_date: str,
    data_1m: dict[str, pd.DataFrame],
    data_5m: dict[str, pd.DataFrame],
    data_1h: dict[str, pd.DataFrame],
    regime_filter: bool = True,
    bull_long: bool = False,
    verbose: bool = False,
    atr_exits: bool = False,
) -> dict:
    """Run a single-day 1-min replay. Returns result dict."""
    target_start = pd.Timestamp(f"{target_date} 00:00:00")
    target_end = pd.Timestamp(f"{target_date} 23:59:59")
    mode_label = "BULL-LONG" if bull_long else ("REGIME" if regime_filter else "NO-FILTER")

    print(f"\n{'='*90}")
    print(f"  1-MIN REPLAY -- {target_date} [{mode_label}]")
    if atr_exits:
        print(f"  Capital: ${CAPITAL:,.0f} | Conf: {CONFIDENCE:.0%} | Exits: ATR-scaled "
              f"(SL {ATR_SL_MULT:g}x/TP {ATR_TP_MULT:g}x, trail {ATR_TRAIL_ACTIVATE_MULT:g}x/{ATR_TRAIL_OFFSET_MULT:g}x ATR)")
    else:
        print(f"  Capital: ${CAPITAL:,.0f} | Conf: {CONFIDENCE:.0%} | SL: {SL_PCT:.1%} | TP: {TP_PCT:.1%} | Trail: {TRAIL_ACTIVATE:.1%}/{TRAIL_OFFSET:.1%}")
    print(f"  Model: Logistic(C={LOGISTIC_C}) | ATR gate: {MIN_ATR_PCT:.1%} | Symbols: {len(SYMBOLS)}")
    if bull_long:
        print(f"  Mode: BULL-LONG (longs at {BULL_LONG_CONFIDENCE:.0%} in BULL, normal regime gate otherwise)")
    elif regime_filter:
        print(f"  Regime gate: BULL->long only, BEAR->short only, NEUTRAL->skip")
    print(f"{'='*90}")

    # Determine bar_label
    sample_sym = next(iter(data_1m))
    sample_df = data_1m[sample_sym]
    has_1m_on_day = any(target_start <= t <= target_end for t in sample_df.index)
    bar_label = "1min" if has_1m_on_day else "5min"

    # Train model on data BEFORE target date
    print(f"\n  Training model on data before {target_date}...")
    t0 = time.time()

    feature_cols = None
    train_X_list, train_y_list = [], []

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

        if feature_cols is None:
            exclude = {"target", "target_dir"}
            feature_cols = [c for c in feat.columns if c not in exclude]

        valid = feat.dropna(subset=["target_dir"]).tail(TRAIN_WINDOW)
        if len(valid) < 50:
            continue
        train_X_list.append(valid.reindex(columns=feature_cols, fill_value=0).fillna(0))
        train_y_list.append(valid["target_dir"])

    if not train_X_list:
        print("  ERROR: Not enough training data.")
        return {"error": "no training data"}

    X_train = pd.concat(train_X_list)
    y_train = pd.concat(train_y_list)
    model = SmartLogistic(params={"C": LOGISTIC_C, "class_weight": "balanced"})
    model.fit(X_train, y_train)
    print(f"  Trained on {len(X_train)} samples, {len(feature_cols)} features, "
          f"{time.time() - t0:.1f}s")

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

    # Use first symbol's hourly as regime reference
    first_with_hourly = next((s for s in SYMBOLS if s in data_1h), SYMBOLS[0])
    regime_hourly = data_1h.get(first_with_hourly)

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

            if bar_time not in feat.index:
                continue

            row = feat.loc[bar_time]
            row_feats = row.reindex(feature_cols, fill_value=0).fillna(0)
            X_pred = row_feats.values.reshape(1, -1)
            proba = model.predict_proba(X_pred)[0]
            up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            down_prob = 1.0 - up_prob

            # Check exits
            if sym in positions:
                pos = positions[sym]
                entry = pos["entry_price"]
                if pos["side"] == "long":
                    pnl_pct = (price - entry) / entry
                    ml_exit = down_prob > CONFIDENCE
                else:
                    pnl_pct = (entry - price) / entry
                    ml_exit = up_prob > CONFIDENCE

                # Trailing stop: track peak unrealized gain
                if pnl_pct > pos.get("peak_pnl", 0):
                    pos["peak_pnl"] = pnl_pct
                lvl_sl = pos.get("sl", SL_PCT)
                lvl_tp = pos.get("tp", TP_PCT)
                lvl_trail_act = pos.get("trail_act", TRAIL_ACTIVATE)
                lvl_trail_off = pos.get("trail_off", TRAIL_OFFSET)
                trail_exit = (
                    pos.get("peak_pnl", 0) >= lvl_trail_act
                    and pnl_pct <= pos["peak_pnl"] - lvl_trail_off
                )

                held_bars = bar_idx - pos.get("bar_idx", 0)
                can_ml_exit = held_bars >= MIN_HOLD_BARS

                if pnl_pct <= -lvl_sl or pnl_pct >= lvl_tp or (ml_exit and can_ml_exit) or trail_exit:
                    shares = pos["shares"]
                    cost = price * shares * COST_PCT
                    if pos["side"] == "long":
                        gross = (price - entry) * shares
                        cash += price * shares - cost
                    else:
                        gross = (entry - price) * shares
                        cash += entry * shares + gross - cost
                    net = gross - pos["cost"] - cost
                    total_costs += cost
                    bar_pnl += net

                    reason = ("TP" if pnl_pct >= lvl_tp
                              else "SL" if pnl_pct <= -lvl_sl
                              else "TR" if trail_exit else "ML")
                    if net > 0:
                        wins += 1
                    else:
                        losses += 1
                    exits += 1

                    bar_trades.append({
                        "time": str(bar_time)[11:19], "sym": sym, "action": "EXIT",
                        "side": pos["side"], "entry": entry, "exit": price,
                        "pnl": net, "pnl_pct": pnl_pct * 100, "reason": reason,
                        "held": bar_idx - pos.get("bar_idx", 0),
                    })
                    del positions[sym]

            # Check entries
            elif len(positions) < MAX_POSITIONS:
                direction = None
                conf = 0.0
                if up_prob > CONFIDENCE:
                    direction = "long"
                    conf = up_prob
                elif down_prob > CONFIDENCE:
                    direction = "short"
                    conf = down_prob

                # Volatility filter: skip low-ATR (choppy) conditions
                if direction:
                    atr_val = row.get("atr_pct", 0.0)
                    if atr_val < MIN_ATR_PCT:
                        direction = None

                # Regime gate (with BULL-LONG override)
                if direction and (regime_filter or bull_long):
                    regime = compute_regime(regime_hourly, bar_time)
                    if bull_long and regime == "BULL":
                        if direction == "short":
                            if up_prob > BULL_LONG_CONFIDENCE:
                                direction = "long"
                                conf = up_prob
                            else:
                                regime_blocks += 1
                                direction = None
                        # longs pass through in BULL
                    elif bull_long and regime == "BULL" and direction is None:
                        pass
                    else:
                        if direction == "long" and regime == "BEAR":
                            regime_blocks += 1
                            direction = None
                        elif direction == "short" and regime == "BULL":
                            regime_blocks += 1
                            direction = None
                        elif regime == "NEUTRAL":
                            regime_blocks += 1
                            direction = None

                # BULL-LONG: also try generating longs when no signal
                if bull_long and direction is None and sym not in positions:
                    regime = compute_regime(regime_hourly, bar_time)
                    if regime == "BULL" and up_prob > BULL_LONG_CONFIDENCE:
                        atr_val = row.get("atr_pct", 0.0)
                        if atr_val >= MIN_ATR_PCT:
                            direction = "long"
                            conf = up_prob

                if direction:
                    max_val = cash * MAX_POS_PCT
                    shares = max_val / price
                    if shares * price < 0.50:
                        continue
                    cost = price * shares * COST_PCT
                    debit = price * shares + cost
                    if debit > cash:
                        continue
                    cash -= debit
                    total_costs += cost
                    levels = compute_exit_levels(
                        float(row.get("atr_pct", 0.0)), atr_exits)
                    positions[sym] = {
                        "side": direction, "shares": shares,
                        "entry_price": price, "cost": cost,
                        "entry_time": str(bar_time)[11:19],
                        "bar_idx": bar_idx,
                        "sl": levels["sl"], "tp": levels["tp"],
                        "trail_act": levels["trail_act"],
                        "trail_off": levels["trail_off"],
                    }
                    entries += 1
                    bar_trades.append({
                        "time": str(bar_time)[11:19], "sym": sym, "action": "ENTER",
                        "side": direction, "price": price, "conf": conf,
                        "size": shares * price,
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
    global SYMBOLS, CONFIDENCE, MIN_ATR_PCT
    parser = argparse.ArgumentParser(description="1-min Time-Machine Replay")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD")
    parser.add_argument("--all-days", action="store_true", help="Run all available days")
    parser.add_argument("--backtest-30d", action="store_true", help="Backtest on 5m bars, excluding last 7 days")
    parser.add_argument("--no-regime-filter", action="store_true", help="Disable regime gate")
    parser.add_argument("--bull-long", action="store_true", help="BULL-LONG mode: aggressive longs in bull market")
    parser.add_argument("--atr-exits", action="store_true", help="Scale SL/TP/trail to each symbol's ATR%% at entry")
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers to run (overrides default list)")
    parser.add_argument("--confidence", type=float, default=None, help="Override entry confidence threshold (default %.2f)" % CONFIDENCE)
    parser.add_argument("--min-atr", type=float, default=None, help="Override min ATR%% entry gate as a fraction (default %.4f)" % MIN_ATR_PCT)
    parser.add_argument("--folds", type=int, default=0, help="Partition test dates into N sequential folds and report per-fold totals (robustness check)")
    args = parser.parse_args()

    if args.symbols:
        SYMBOLS = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.confidence is not None:
        CONFIDENCE = args.confidence
    if args.min_atr is not None:
        MIN_ATR_PCT = args.min_atr

    use_regime = not args.no_regime_filter
    use_bull_long = args.bull_long
    use_atr_exits = args.atr_exits

    print("=" * 90)
    print("  VOLATILE STOCKS 1-MIN REPLAY")
    print("=" * 90)

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

    results = []
    for d in dates_to_run:
        if use_bull_long:
            r = run_day(d, data_1m, data_5m, data_1h,
                        regime_filter=True, bull_long=True, verbose=args.verbose,
                        atr_exits=use_atr_exits)
            if r and "error" not in r:
                results.append(r)

        r = run_day(d, data_1m, data_5m, data_1h,
                    regime_filter=use_regime, bull_long=False, verbose=(args.verbose and not use_bull_long),
                    atr_exits=use_atr_exits)
        if r and "error" not in r:
            results.append(r)

        if use_regime:
            r2 = run_day(d, data_1m, data_5m, data_1h,
                         regime_filter=False, bull_long=False, verbose=False,
                         atr_exits=use_atr_exits)
            if r2 and "error" not in r2:
                results.append(r2)

    # ── Multi-day comparison table ──────────────────────────────────────────
    if len(results) > 1:
        print(f"\n\n{'='*90}")
        print(f"  MULTI-DAY COMPARISON")
        print(f"{'='*90}")
        print(f"\n  {'Date':<12} {'Mode':<12} {'Return':>8} {'W/L':>8} {'WR':>6} "
              f"{'MaxDD':>7} {'Entries':>8} {'Blocked':>8}")
        print(f"  {'-'*76}")

        totals = {}
        for r in sorted(results, key=lambda x: (x["date"], x["mode"])):
            wl = f"{r['wins']}W/{r['losses']}L"
            wr = f"{r['win_rate']:.0f}%" if r['exits'] > 0 else "N/A"
            print(f"  {r['date']:<12} {r['mode']:<12} {r['return_pct']:>+7.3f}% "
                  f"{wl:>8} {wr:>6} {r['max_dd']:>6.3f}% "
                  f"{r['entries']:>8} {r['regime_blocks']:>8}")

            m = r["mode"]
            if m not in totals:
                totals[m] = {"ret": 0.0, "wins": 0, "losses": 0}
            totals[m]["ret"] += r["return_pct"]
            totals[m]["wins"] += r["wins"]
            totals[m]["losses"] += r["losses"]

        n_days = len(dates_to_run)
        print()
        for m in sorted(totals.keys()):
            t = totals[m]
            print(f"  {'TOTALS':<12} {m:<12} {t['ret']:>+7.3f}% "
                  f"{t['wins']}W/{t['losses']}L")

        modes = list(totals.keys())
        if len(modes) >= 2:
            best = max(modes, key=lambda m: totals[m]["ret"])
            worst = min(modes, key=lambda m: totals[m]["ret"])
            delta = totals[best]["ret"] - totals[worst]["ret"]
            print(f"\n  Best mode: {best} ({totals[best]['ret']:+.3f}%), "
                  f"beat {worst} by {delta:+.3f}% over {n_days} day(s)")

    # ── Walk-forward fold consistency ────────────────────────────────────────
    if args.folds and args.folds > 1 and results:
        sorted_dates = sorted(set(r["date"] for r in results))
        n = len(sorted_dates)
        k = args.folds
        # contiguous, near-equal folds
        bounds = [sorted_dates[i * n // k] for i in range(k)]
        def fold_of(d: str) -> int:
            f = 0
            for i, b in enumerate(bounds):
                if d >= b:
                    f = i
            return f

        fold_totals = {}  # (fold, mode) -> ret
        for r in results:
            key = (fold_of(r["date"]), r["mode"])
            fold_totals[key] = fold_totals.get(key, 0.0) + r["return_pct"]

        modes = sorted(set(m for _, m in fold_totals.keys()))
        print(f"\n\n{'='*90}")
        print(f"  WALK-FORWARD FOLD CONSISTENCY ({k} folds, {n} test days)")
        print(f"{'='*90}\n")
        header = f"  {'Fold':<6} {'Dates':<26}" + "".join(f"{m:>14}" for m in modes)
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for i in range(k):
            lo = bounds[i]
            hi = bounds[i + 1] if i + 1 < k else sorted_dates[-1]
            cells = "".join(f"{fold_totals.get((i, m), 0.0):>+13.3f}%" for m in modes)
            print(f"  {i+1:<6} {lo}..{hi:<13}{cells}")
        # positive-fold count per mode
        print()
        for m in modes:
            pos = sum(1 for i in range(k) if fold_totals.get((i, m), 0.0) > 0)
            print(f"  {m:<12} positive in {pos}/{k} folds")

    print(f"\n{'='*90}")
    print("  DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
