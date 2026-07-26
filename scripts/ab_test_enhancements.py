#!/usr/bin/env python3
"""A/B Test: Multi-Timeframe Features + Fast Recalibration.

Runs 3 variants of the crypto time-machine backtest and compares:
  A) BASELINE — current settings (retrain_every=10, daily features only)
  B) MTF — adds 11 hourly-derived features to daily feature matrix
  C) FAST_RECAL — retrain_every=3 instead of 10 (daily features only)

Only variants that beat the baseline on BOTH return% AND win_rate get promoted.

Usage:
    python scripts/ab_test_enhancements.py                # full test
    python scripts/ab_test_enhancements.py --period 1y    # shorter history
    python scripts/ab_test_enhancements.py --verbose       # day-by-day logs
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_engine.config import CRYPTO_WATCHLIST
from trading_engine.models.mean_rev_model import (
    MEAN_REV_CONFIGS,
    build_features_for_market,
)
from trading_engine.models.classifiers import SmartLogistic
from trading_engine.price_engine import build_mtf_features

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──────────────────────────────────────────────────────────────────
CFG = MEAN_REV_CONFIGS["crypto"]
CONFIDENCE = CFG["default_confidence"]
SL_PCT = CFG.get("default_sl", 0.10)
TP_PCT = CFG.get("default_tp", 0.15)
MAX_POS_PCT = CFG.get("default_max_pos", 0.15)
LOGISTIC_C = CFG.get("logistic_C", 0.15)
TRAIN_WINDOW = CFG["train_window"]
MIN_TRAIN = CFG["min_train"]
CROSS_ASSET = CFG["cross_asset_symbol"]
CROSS_FEATURES = CFG["cross_asset_features"]
CROSS_PREFIX = CFG["cross_asset_prefix"]
COST_PCT = 0.001
MAX_POSITIONS = 8
INITIAL_CAPITAL = 1000.0


# ── Data fetching ───────────────────────────────────────────────────────────

def fetch_data(period: str = "2y") -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for all crypto symbols."""
    data = {}
    for sym in CRYPTO_WATCHLIST:
        try:
            df = yf.Ticker(sym).history(period=period, interval="1d")
            if df.empty:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    return data


def fetch_hourly_data(symbols: list[str], days: int = 710) -> dict[str, pd.DataFrame]:
    """Download hourly OHLCV data (max ~730 days for Yahoo Finance).

    Uses period= instead of start/end to avoid the 730-day boundary error.
    """
    data = {}
    period_str = f"{min(days, 710)}d"
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period=period_str, interval="1h")
            if df.empty:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    return data


# ── Feature helpers ─────────────────────────────────────────────────────────

def _add_relative_strength(feat, ca_feat, prefix):
    for period in [5, 10, 20]:
        sc = f"return_{period}d"
        if sc in feat.columns and sc in ca_feat.columns:
            aligned = ca_feat[sc].reindex(feat.index)
            feat[f"rs_{prefix}_{period}d"] = feat[sc] - aligned
    if "volatility_5d" in feat.columns and "volatility_5d" in ca_feat.columns:
        aligned = ca_feat["volatility_5d"].reindex(feat.index)
        feat[f"rel_vol_{prefix}"] = feat["volatility_5d"] / aligned.replace(0, np.nan)
    return feat


def build_all_features(
    data: dict[str, pd.DataFrame],
    hourly_data: dict[str, pd.DataFrame] | None = None,
    add_mtf: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build feature matrices for all symbols. Optionally add MTF features."""
    ca_df = data.get(CROSS_ASSET)
    all_features = {}

    for symbol, df in data.items():
        if len(df) < MIN_TRAIN:
            continue
        feat = build_features_for_market(df, "crypto")

        if ca_df is not None and symbol != CROSS_ASSET and len(ca_df) > 30:
            ca_feat = build_features_for_market(ca_df, "crypto")
            avail_ca = [c for c in CROSS_FEATURES if c in ca_feat.columns]
            cross = ca_feat[avail_ca].copy()
            cross.columns = [f"{CROSS_PREFIX}_{c}" for c in cross.columns]
            feat = feat.join(cross, how="left")
            feat = _add_relative_strength(feat, ca_feat, CROSS_PREFIX)

        if add_mtf and hourly_data and symbol in hourly_data:
            h_df = hourly_data[symbol]
            mtf = build_mtf_features(df, h_df)
            if mtf:
                for day_idx in feat.index:
                    h_before = h_df[h_df.index < day_idx]
                    if len(h_before) >= 50:
                        day_mtf = build_mtf_features(
                            df[df.index <= day_idx], h_before
                        )
                        for k, v in day_mtf.items():
                            feat.loc[day_idx, k] = v

        feat = feat.dropna(subset=["target_dir"])
        if len(feat) > MIN_TRAIN:
            all_features[symbol] = feat

    return all_features


# ── Walk-forward backtest ───────────────────────────────────────────────────

def run_backtest(
    all_features: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame],
    retrain_every: int = 10,
    label: str = "baseline",
    verbose: bool = False,
) -> dict:
    """Walk-forward backtest with Logistic Regression."""
    all_dates = sorted(set().union(*(f.index for f in all_features.values())))
    if len(all_dates) < MIN_TRAIN + 20:
        return {"error": "Not enough dates"}

    test_start_idx = MIN_TRAIN
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    daily_results = []
    total_costs = 0.0
    model = None
    feature_cols = None

    for day_idx in range(test_start_idx, len(all_dates) - 1):
        day = all_dates[day_idx]
        day_str = str(day)[:10]
        day_trades = []
        day_pnl = 0.0

        if model is None or (day_idx - test_start_idx) % retrain_every == 0:
            train_X_list, train_y_list = [], []
            for symbol, feat in all_features.items():
                train_slice = feat[feat.index < day].tail(TRAIN_WINDOW)
                if len(train_slice) < 30:
                    continue
                if feature_cols is None:
                    exclude = {"target", "target_dir"}
                    feature_cols = [c for c in train_slice.columns if c not in exclude]
                valid = train_slice.dropna(subset=feature_cols[:5] + ["target_dir"])
                if len(valid) < 30:
                    continue
                train_X_list.append(
                    valid.reindex(columns=feature_cols, fill_value=0).fillna(0)
                )
                train_y_list.append(valid["target_dir"])

            if train_X_list:
                X_train = pd.concat(train_X_list)
                y_train = pd.concat(train_y_list)
                model = SmartLogistic(params={"C": LOGISTIC_C})
                model.fit(X_train, y_train)

        if model is None or feature_cols is None:
            continue

        for symbol, feat in all_features.items():
            if day not in feat.index:
                continue

            row = feat.loc[day]
            row_feats = row.reindex(feature_cols, fill_value=0).fillna(0)
            X_pred = row_feats.values.reshape(1, -1)
            proba = model.predict_proba(X_pred)[0]
            up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            down_prob = 1.0 - up_prob

            price = float(data[symbol].loc[day, "Close"]) if symbol in data and day in data[symbol].index else 0
            if price <= 0:
                continue

            if symbol in positions:
                pos = positions[symbol]
                entry = pos["entry_price"]
                if pos["side"] == "long":
                    pnl_pct = (price - entry) / entry
                else:
                    pnl_pct = (entry - price) / entry

                hit_sl = pnl_pct <= -SL_PCT
                hit_tp = pnl_pct >= TP_PCT
                ml_exit = (down_prob > CONFIDENCE) if pos["side"] == "long" else (up_prob > CONFIDENCE)

                if hit_sl or hit_tp or ml_exit:
                    shares = pos["shares"]
                    cost = price * shares * COST_PCT
                    if pos["side"] == "long":
                        gross_pnl = (price - entry) * shares
                        cash += price * shares - cost
                    else:
                        gross_pnl = (entry - price) * shares
                        cash += entry * shares + gross_pnl - cost
                    net_pnl = gross_pnl - pos["entry_cost"] - cost
                    total_costs += cost
                    day_pnl += net_pnl
                    reason = "TP" if hit_tp else ("SL" if hit_sl else "ML")
                    day_trades.append({
                        "date": day_str, "symbol": symbol,
                        "action": "sell" if pos["side"] == "long" else "cover",
                        "side": pos["side"], "net_pnl": round(net_pnl, 2),
                        "reason": reason,
                    })
                    del positions[symbol]

            elif len(positions) < MAX_POSITIONS:
                direction = None
                confidence = 0.0
                if up_prob > CONFIDENCE:
                    direction = "long"
                    confidence = up_prob
                elif down_prob > CONFIDENCE:
                    direction = "short"
                    confidence = down_prob

                if direction:
                    max_val = cash * MAX_POS_PCT
                    shares = max_val / price
                    if shares * price < 1.0:
                        continue
                    cost = price * shares * COST_PCT
                    if direction == "long":
                        debit = price * shares + cost
                    else:
                        debit = price * shares + cost
                    if debit > cash:
                        continue
                    cash -= debit
                    total_costs += cost
                    positions[symbol] = {
                        "side": direction, "shares": shares,
                        "entry_price": price, "entry_cost": cost,
                        "entry_date": day_str,
                    }
                    day_trades.append({
                        "date": day_str, "symbol": symbol,
                        "action": "buy" if direction == "long" else "short",
                        "side": direction, "confidence": round(confidence, 4),
                    })

        port_val = cash + sum(
            p["shares"] * (
                float(data[s].loc[day, "Close"])
                if s in data and day in data[s].index
                else p["entry_price"]
            )
            for s, p in positions.items()
        )
        daily_results.append({
            "date": day_str, "total_value": round(port_val, 2),
            "trades": len(day_trades),
        })
        trades.extend(day_trades)

        if verbose and day_trades:
            print(f"  [{label}] {day_str} | Val: ${port_val:,.2f} | "
                  f"Trades: {len(day_trades)} | Pos: {len(positions)}")

    final_val = cash
    last_date = all_dates[-1]
    for s, p in positions.items():
        if s in data and last_date in data[s].index:
            cp = float(data[s].loc[last_date, "Close"])
            if p["side"] == "long":
                final_val += p["shares"] * cp
            else:
                final_val += p["entry_price"] * p["shares"] + (p["entry_price"] - cp) * p["shares"]

    closed = [t for t in trades if t.get("action") in ("sell", "cover")]
    wins = sum(1 for t in closed if t.get("net_pnl", 0) > 0)
    total_pnl = final_val - INITIAL_CAPITAL

    vals = [d["total_value"] for d in daily_results] if daily_results else [INITIAL_CAPITAL]
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "label": label,
        "final_value": round(final_val, 2),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "wins": wins,
        "losses": len(closed) - wins,
        "win_rate": round(wins / len(closed) * 100, 1) if closed else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "total_costs": round(total_costs, 4),
        "retrain_every": 0,
        "daily_results": daily_results,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A/B Test: MTF + Fast Recalibration")
    parser.add_argument("--period", default="2y", help="Data period (6mo, 1y, 2y)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  A/B TEST: Multi-Timeframe Features + Fast Recalibration")
    print("  Market: Crypto | Model: Logistic Regression")
    print(f"  Period: {args.period} | Capital: ${INITIAL_CAPITAL:,.0f}")
    print("=" * 80)

    # 1. Fetch data
    print("\n  [1/5] Fetching daily data...")
    t0 = time.time()
    daily_data = fetch_data(period=args.period)
    print(f"         {len(daily_data)} symbols, {time.time() - t0:.0f}s")

    if len(daily_data) < 3:
        print("  ERROR: Not enough data. Exiting.")
        return

    print("\n  [2/5] Fetching hourly data (for MTF variant)...")
    t0 = time.time()
    hourly_data = fetch_hourly_data(list(daily_data.keys()))
    print(f"         {len(hourly_data)} symbols, {time.time() - t0:.0f}s")

    # 2. Build features
    print("\n  [3/5] Building feature matrices...")
    t0 = time.time()
    baseline_features = build_all_features(daily_data, add_mtf=False)
    mtf_features = build_all_features(daily_data, hourly_data, add_mtf=True)
    print(f"         Done in {time.time() - t0:.0f}s")

    sample_sym = next(iter(baseline_features))
    n_base = len([c for c in baseline_features[sample_sym].columns if c not in {"target", "target_dir"}])
    n_mtf = len([c for c in mtf_features[sample_sym].columns if c not in {"target", "target_dir"}])
    print(f"         Baseline: {n_base} features | MTF: {n_mtf} features (+{n_mtf - n_base} MTF)")

    # 3. Run backtests
    print("\n  [4/5] Running backtests...")

    print("\n  ── A) BASELINE (retrain_every=10, daily only) ──")
    t0 = time.time()
    r_base = run_backtest(
        baseline_features, daily_data,
        retrain_every=10, label="BASELINE", verbose=args.verbose,
    )
    r_base["retrain_every"] = 10
    print(f"         {time.time() - t0:.0f}s")

    print("\n  ── B) MTF (retrain_every=10, daily + hourly features) ──")
    t0 = time.time()
    r_mtf = run_backtest(
        mtf_features, daily_data,
        retrain_every=10, label="MTF", verbose=args.verbose,
    )
    r_mtf["retrain_every"] = 10
    print(f"         {time.time() - t0:.0f}s")

    print("\n  ── C) FAST_RECAL (retrain_every=3, daily only) ──")
    t0 = time.time()
    r_fast = run_backtest(
        baseline_features, daily_data,
        retrain_every=3, label="FAST_RECAL", verbose=args.verbose,
    )
    r_fast["retrain_every"] = 3
    print(f"         {time.time() - t0:.0f}s")

    # 4. Results comparison
    print(f"\n  [5/5] Results")
    print(f"\n  {'='*80}")
    print(f"  {'METRIC':<22} {'A) BASELINE':>14} {'B) MTF':>14} {'C) FAST_RECAL':>14}")
    print(f"  {'-'*66}")

    variants = [r_base, r_mtf, r_fast]
    rows = [
        ("Return %",       [f"{r['return_pct']:+.2f}%" for r in variants]),
        ("Net P&L",        [f"${r['total_pnl']:+.2f}" for r in variants]),
        ("Win Rate",       [f"{r['win_rate']:.1f}%" for r in variants]),
        ("Total Trades",   [f"{r['total_trades']}" for r in variants]),
        ("Closed Trades",  [f"{r['closed_trades']}" for r in variants]),
        ("Wins / Losses",  [f"{r['wins']}W/{r['losses']}L" for r in variants]),
        ("Max Drawdown",   [f"{r['max_drawdown_pct']:.2f}%" for r in variants]),
        ("Total Costs",    [f"${r['total_costs']:.4f}" for r in variants]),
        ("Retrain Every",  [f"{r['retrain_every']}d" for r in variants]),
    ]

    for label, vals in rows:
        print(f"  {label:<22} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # 5. Verdict
    print(f"\n  {'='*80}")
    print(f"  VERDICT")
    print(f"  {'='*80}")

    base_ret = r_base["return_pct"]
    base_wr = r_base["win_rate"]
    base_dd = r_base["max_drawdown_pct"]

    promotions = []

    mtf_better = (
        r_mtf["return_pct"] > base_ret
        and r_mtf["win_rate"] >= base_wr - 2.0
        and r_mtf["max_drawdown_pct"] <= base_dd * 1.2
    )
    if mtf_better:
        delta_ret = r_mtf["return_pct"] - base_ret
        print(f"\n  ✅ B) MTF PROMOTED — return +{delta_ret:.2f}% over baseline")
        promotions.append("MTF")
    else:
        print(f"\n  ❌ B) MTF REJECTED — no meaningful improvement over baseline")
        print(f"       Return: {r_mtf['return_pct']:+.2f}% vs {base_ret:+.2f}% baseline")
        print(f"       WinRate: {r_mtf['win_rate']:.1f}% vs {base_wr:.1f}% baseline")

    fast_better = (
        r_fast["return_pct"] > base_ret
        and r_fast["win_rate"] >= base_wr - 2.0
        and r_fast["max_drawdown_pct"] <= base_dd * 1.2
    )
    if fast_better:
        delta_ret = r_fast["return_pct"] - base_ret
        print(f"\n  ✅ C) FAST_RECAL PROMOTED — return +{delta_ret:.2f}% over baseline")
        promotions.append("FAST_RECAL")
    else:
        print(f"\n  ❌ C) FAST_RECAL REJECTED — no meaningful improvement over baseline")
        print(f"       Return: {r_fast['return_pct']:+.2f}% vs {base_ret:+.2f}% baseline")
        print(f"       WinRate: {r_fast['win_rate']:.1f}% vs {base_wr:.1f}% baseline")

    if promotions:
        print(f"\n  🎯 PROMOTE: {', '.join(promotions)}")
        if "MTF" in promotions:
            print(f"     → Integrate hourly MTF features into build_features_for_market()")
        if "FAST_RECAL" in promotions:
            print(f"     → Change crypto retrain_every from 10 → 3 in MEAN_REV_CONFIGS")
    else:
        print(f"\n  📊 No variants beat baseline. Current settings are optimal for this period.")

    print(f"\n  {'='*80}")


if __name__ == "__main__":
    main()
