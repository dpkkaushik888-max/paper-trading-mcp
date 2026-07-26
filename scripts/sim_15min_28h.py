#!/usr/bin/env python3
"""Live Simulation: 15-min bars over the last 28 hours.

Trains on 7 days of 15-min data, then replays the last 28 hours bar-by-bar.
Compares: BASELINE (15-min features only) vs MTF (15-min + 1h features).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/sim_15min_28h.py
    PYTHONPATH=. .venv/bin/python scripts/sim_15min_28h.py --verbose
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

from trading_engine.config import CRYPTO_WATCHLIST
from trading_engine.models.classifiers import SmartLogistic
from trading_engine.price_engine import build_mtf_features

warnings.filterwarnings("ignore")

CAPITAL = 1000.0
CONFIDENCE = 0.65
SL_PCT = 0.02
TP_PCT = 0.03
MAX_POS_PCT = 0.20
COST_PCT = 0.001
MAX_POSITIONS = 6
TRAIN_BARS = 400
LOGISTIC_C = 0.15
SIM_HOURS = 28

TOP_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
]


def fetch_15min_data(symbols: list[str], days: int = 10) -> dict[str, pd.DataFrame]:
    """Fetch 15-min bars (Yahoo max ~60 days for 15m)."""
    data = {}
    period = f"{min(days, 59)}d"
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period=period, interval="15m")
            if df.empty or len(df) < 50:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    return data


def fetch_1h_data(symbols: list[str], days: int = 30) -> dict[str, pd.DataFrame]:
    """Fetch 1h bars for MTF features."""
    data = {}
    period = f"{min(days, 710)}d"
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period=period, interval="1h")
            if df.empty or len(df) < 50:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    return data


def build_15min_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from 15-min OHLCV bars."""
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
    feat["return_4b"] = close.pct_change(4)
    feat["return_8b"] = close.pct_change(8)
    feat["return_16b"] = close.pct_change(16)

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
    feat["minute_bucket"] = df.index.minute // 15

    feat["target"] = close.shift(-1).pct_change(-1) * -1
    feat["target_dir"] = (feat["target"] > 0).astype(int)

    return feat


def add_mtf_to_features(
    feat_15m: pd.DataFrame,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame | None,
    bar_time: pd.Timestamp,
) -> pd.DataFrame:
    """Add MTF hourly features to the current 15-min feature row."""
    if hourly_df is None or len(hourly_df) < 50:
        return feat_15m

    h_before = hourly_df[hourly_df.index < bar_time]
    if len(h_before) < 50:
        return feat_15m

    d_df = daily_df if daily_df is not None else pd.DataFrame()
    if d_df.empty:
        d_df = pd.DataFrame({"Close": [h_before["Close"].iloc[-1]]})

    mtf = build_mtf_features(d_df, h_before)
    for k, v in mtf.items():
        feat_15m.loc[bar_time, k] = v

    return feat_15m


def run_sim(
    data_15m: dict[str, pd.DataFrame],
    data_1h: dict[str, pd.DataFrame] | None,
    add_mtf: bool = False,
    label: str = "BASELINE",
    verbose: bool = False,
) -> dict:
    """Walk-forward simulation on 15-min bars, last 28 hours."""

    cutoff = datetime.now() - timedelta(hours=SIM_HOURS)
    cash = CAPITAL
    positions = {}
    trades = []
    bar_results = []
    total_costs = 0.0

    all_features = {}
    for sym, df in data_15m.items():
        feat = build_15min_features(df)
        feat = feat.dropna(subset=["target_dir"])
        if len(feat) > TRAIN_BARS:
            all_features[sym] = feat

    if not all_features:
        return {"error": "No symbols with enough data"}

    all_bars = sorted(set().union(*(f.index for f in all_features.values())))
    sim_bars = [b for b in all_bars if b >= cutoff]
    train_bars = [b for b in all_bars if b < cutoff]

    if len(train_bars) < TRAIN_BARS or len(sim_bars) < 10:
        return {"error": f"Insufficient bars: {len(train_bars)} train, {len(sim_bars)} sim"}

    feature_cols = None
    train_X_list, train_y_list = [], []
    for sym, feat in all_features.items():
        train_slice = feat[feat.index < cutoff].tail(TRAIN_BARS)
        if len(train_slice) < 50:
            continue

        if add_mtf and data_1h and sym in data_1h:
            h_df = data_1h[sym]
            for t in train_slice.index[-20:]:
                h_before = h_df[h_df.index < t]
                if len(h_before) >= 50:
                    dummy_daily = pd.DataFrame({"Close": [float(train_slice.loc[t, "return_1b"])]}) if "return_1b" in train_slice.columns else pd.DataFrame()
                    mtf = build_mtf_features(dummy_daily, h_before)
                    for k, v in mtf.items():
                        train_slice.loc[t, k] = v

        if feature_cols is None:
            exclude = {"target", "target_dir"}
            feature_cols = [c for c in train_slice.columns if c not in exclude]

        valid = train_slice.dropna(subset=["target_dir"])
        if len(valid) < 30:
            continue
        train_X_list.append(valid.reindex(columns=feature_cols, fill_value=0).fillna(0))
        train_y_list.append(valid["target_dir"])

    if not train_X_list:
        return {"error": "No training data"}

    X_train = pd.concat(train_X_list)
    y_train = pd.concat(train_y_list)
    model = SmartLogistic(params={"C": LOGISTIC_C})
    model.fit(X_train, y_train)

    entry_count = 0
    exit_count = 0
    wins = 0
    losses = 0

    for bar in sim_bars:
        bar_str = str(bar)[:19]
        bar_trades = []
        bar_pnl = 0.0

        for sym, feat in all_features.items():
            if bar not in feat.index:
                continue

            price = float(data_15m[sym].loc[bar, "Close"]) if bar in data_15m[sym].index else 0
            if price <= 0:
                continue

            row_feat = feat.copy()
            if add_mtf and data_1h and sym in data_1h:
                row_feat = add_mtf_to_features(row_feat, data_1h[sym], None, bar)

            if bar not in row_feat.index:
                continue

            row = row_feat.loc[bar]
            row_feats = row.reindex(feature_cols, fill_value=0).fillna(0)
            X_pred = row_feats.values.reshape(1, -1)
            proba = model.predict_proba(X_pred)[0]
            up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            down_prob = 1.0 - up_prob

            if sym in positions:
                pos = positions[sym]
                entry = pos["entry_price"]
                if pos["side"] == "long":
                    pnl_pct = (price - entry) / entry
                    ml_exit = down_prob > CONFIDENCE
                else:
                    pnl_pct = (entry - price) / entry
                    ml_exit = up_prob > CONFIDENCE

                if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT or ml_exit:
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

                    reason = "TP" if pnl_pct >= TP_PCT else ("SL" if pnl_pct <= -SL_PCT else "ML")
                    if net > 0:
                        wins += 1
                    else:
                        losses += 1
                    exit_count += 1

                    bar_trades.append({
                        "time": bar_str, "sym": sym, "action": "EXIT",
                        "side": pos["side"], "entry": round(entry, 2),
                        "exit": round(price, 2), "pnl": round(net, 4),
                        "pnl_pct": round(pnl_pct * 100, 2),
                        "reason": reason, "held_bars": pos.get("bars", 0),
                    })
                    del positions[sym]
                else:
                    if "bars" in positions[sym]:
                        positions[sym]["bars"] += 1

            elif len(positions) < MAX_POSITIONS:
                direction = None
                conf = 0.0
                if up_prob > CONFIDENCE:
                    direction = "long"
                    conf = up_prob
                elif down_prob > CONFIDENCE:
                    direction = "short"
                    conf = down_prob

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
                    positions[sym] = {
                        "side": direction, "shares": shares,
                        "entry_price": price, "cost": cost,
                        "entry_time": bar_str, "bars": 0,
                    }
                    entry_count += 1
                    bar_trades.append({
                        "time": bar_str, "sym": sym, "action": "ENTER",
                        "side": direction, "price": round(price, 4),
                        "conf": round(conf, 4), "size": round(shares * price, 2),
                    })

        port_val = cash + sum(
            p["shares"] * (
                float(data_15m[s].loc[bar, "Close"])
                if s in data_15m and bar in data_15m[s].index
                else p["entry_price"]
            )
            for s, p in positions.items()
        )
        bar_results.append({
            "time": bar_str, "value": round(port_val, 4),
            "pnl": round(bar_pnl, 4), "trades": len(bar_trades),
        })
        trades.extend(bar_trades)

        if verbose and bar_trades:
            for t in bar_trades:
                if t["action"] == "ENTER":
                    print(f"  [{label}] {t['time']} ENTER {t['side']:>5} {t['sym']:<10} "
                          f"@ {t['price']:>10.4f}  conf={t['conf']:.3f}  size=${t['size']:.2f}")
                else:
                    marker = "✅" if t["pnl"] > 0 else "❌"
                    print(f"  [{label}] {t['time']} EXIT  {t['side']:>5} {t['sym']:<10} "
                          f"entry={t['entry']:>10.2f} → {t['exit']:>10.2f}  "
                          f"pnl=${t['pnl']:>+8.4f} ({t['pnl_pct']:>+5.2f}%) "
                          f"[{t['reason']}] {marker}  held={t['held_bars']}bars")

    final_val = cash
    for s, p in positions.items():
        last_bar = sim_bars[-1] if sim_bars else None
        if last_bar and s in data_15m and last_bar in data_15m[s].index:
            cp = float(data_15m[s].loc[last_bar, "Close"])
            if p["side"] == "long":
                final_val += p["shares"] * cp
            else:
                final_val += p["entry_price"] * p["shares"] + (p["entry_price"] - cp) * p["shares"]

    vals = [b["value"] for b in bar_results] if bar_results else [CAPITAL]
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_pnl = final_val - CAPITAL
    closed = exit_count
    return {
        "label": label,
        "final_value": round(final_val, 4),
        "total_pnl": round(total_pnl, 4),
        "return_pct": round(total_pnl / CAPITAL * 100, 4),
        "entries": entry_count,
        "exits": closed,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / closed * 100, 1) if closed else 0,
        "open_positions": len(positions),
        "max_drawdown_pct": round(max_dd, 4),
        "total_costs": round(total_costs, 6),
        "sim_bars": len(sim_bars),
        "train_bars": len(train_bars),
        "bar_results": bar_results,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="15-min Live Simulation (28h)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  LIVE SIMULATION: 15-min bars × 28 hours")
    print(f"  Symbols: {', '.join(TOP_SYMBOLS)}")
    print(f"  Capital: ${CAPITAL:,.0f} | Conf: {CONFIDENCE:.0%} | SL: {SL_PCT:.0%} | TP: {TP_PCT:.0%}")
    print(f"  Comparing: BASELINE (15m only) vs MTF (15m + 1h hourly features)")
    print("=" * 80)

    print("\n  [1/3] Fetching 15-min data (10 days)...")
    t0 = time.time()
    data_15m = fetch_15min_data(TOP_SYMBOLS, days=10)
    print(f"         {len(data_15m)} symbols, "
          f"~{max(len(df) for df in data_15m.values()) if data_15m else 0} bars, "
          f"{time.time() - t0:.0f}s")

    print("\n  [2/3] Fetching 1h data (30 days for MTF)...")
    t0 = time.time()
    data_1h = fetch_1h_data(list(data_15m.keys()), days=30)
    print(f"         {len(data_1h)} symbols, {time.time() - t0:.0f}s")

    cutoff = datetime.now() - timedelta(hours=SIM_HOURS)
    sample_sym = next(iter(data_15m))
    sim_count = len([b for b in data_15m[sample_sym].index if b >= cutoff])
    print(f"\n         Train cutoff: {cutoff.strftime('%Y-%m-%d %H:%M')}")
    print(f"         Sim bars (per symbol): ~{sim_count}")

    print(f"\n  [3/3] Running simulations...")

    print(f"\n  ── A) BASELINE (15-min features only) ──")
    t0 = time.time()
    r_base = run_sim(data_15m, None, add_mtf=False, label="BASELINE", verbose=args.verbose)
    print(f"         {time.time() - t0:.1f}s")

    print(f"\n  ── B) MTF (15-min + 1h hourly features) ──")
    t0 = time.time()
    r_mtf = run_sim(data_15m, data_1h, add_mtf=True, label="MTF", verbose=args.verbose)
    print(f"         {time.time() - t0:.1f}s")

    if "error" in r_base:
        print(f"\n  BASELINE ERROR: {r_base['error']}")
        return
    if "error" in r_mtf:
        print(f"\n  MTF ERROR: {r_mtf['error']}")
        return

    print(f"\n  {'='*80}")
    print(f"  RESULTS — Last {SIM_HOURS}h at 15-min intervals")
    print(f"  {'='*80}")
    print(f"\n  {'METRIC':<24} {'A) BASELINE':>14} {'B) MTF':>14}")
    print(f"  {'-'*54}")

    variants = [r_base, r_mtf]
    rows = [
        ("Sim Bars",        [f"{r['sim_bars']}" for r in variants]),
        ("Train Bars",      [f"{r['train_bars']}" for r in variants]),
        ("Return %",        [f"{r['return_pct']:+.4f}%" for r in variants]),
        ("Net P&L",         [f"${r['total_pnl']:+.4f}" for r in variants]),
        ("Entries",         [f"{r['entries']}" for r in variants]),
        ("Exits (closed)",  [f"{r['exits']}" for r in variants]),
        ("Wins / Losses",   [f"{r['wins']}W/{r['losses']}L" for r in variants]),
        ("Win Rate",        [f"{r['win_rate']:.1f}%" for r in variants]),
        ("Open Positions",  [f"{r['open_positions']}" for r in variants]),
        ("Max Drawdown",    [f"{r['max_drawdown_pct']:.4f}%" for r in variants]),
        ("Total Costs",     [f"${r['total_costs']:.6f}" for r in variants]),
    ]
    for label, vals in rows:
        print(f"  {label:<24} {vals[0]:>14} {vals[1]:>14}")

    print(f"\n  {'='*80}")

    base_trades = [t for t in r_base.get("trades", []) if t["action"] == "EXIT"]
    mtf_trades = [t for t in r_mtf.get("trades", []) if t["action"] == "EXIT"]

    if base_trades:
        print(f"\n  BASELINE Closed Trades:")
        for t in base_trades:
            marker = "✅" if t["pnl"] > 0 else "❌"
            print(f"    {t['time']} {t['side']:>5} {t['sym']:<10} "
                  f"${t['pnl']:>+8.4f} ({t['pnl_pct']:>+5.2f}%) [{t['reason']}] {marker}")

    if mtf_trades:
        print(f"\n  MTF Closed Trades:")
        for t in mtf_trades:
            marker = "✅" if t["pnl"] > 0 else "❌"
            print(f"    {t['time']} {t['side']:>5} {t['sym']:<10} "
                  f"${t['pnl']:>+8.4f} ({t['pnl_pct']:>+5.2f}%) [{t['reason']}] {marker}")

    open_base = [t for t in r_base.get("trades", []) if t["action"] == "ENTER" and t["sym"] in
                 {s for s, p in [] }]  # just show entries if no exits

    print(f"\n  OPEN POSITIONS (BASELINE): {r_base['open_positions']}")
    print(f"  OPEN POSITIONS (MTF):      {r_mtf['open_positions']}")

    base_ret = r_base["return_pct"]
    mtf_ret = r_mtf["return_pct"]
    base_wr = r_base["win_rate"]
    mtf_wr = r_mtf["win_rate"]

    print(f"\n  {'='*80}")
    print(f"  VERDICT")
    print(f"  {'='*80}")

    if mtf_ret > base_ret and r_mtf["exits"] > 0:
        delta = mtf_ret - base_ret
        print(f"\n  ✅ MTF outperformed baseline by {delta:+.4f}% in {SIM_HOURS}h")
    elif r_mtf["exits"] == 0 and r_base["exits"] == 0:
        print(f"\n  📊 No closed trades in {SIM_HOURS}h — market too quiet or confidence too high")
        print(f"     Try lowering --confidence or extending --hours")
    elif mtf_ret == base_ret:
        print(f"\n  📊 Both variants performed identically — MTF features had no impact")
    else:
        delta = mtf_ret - base_ret
        print(f"\n  ❌ Baseline outperformed MTF by {-delta:+.4f}% in this {SIM_HOURS}h window")

    print(f"\n  NOTE: {SIM_HOURS}h is a tiny sample. This shows real-time behavior,")
    print(f"        not statistical significance. Use A/B test for that.")
    print(f"  {'='*80}")


if __name__ == "__main__":
    main()
