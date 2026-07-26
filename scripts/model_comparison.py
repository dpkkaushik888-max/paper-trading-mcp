"""ML Algorithm Comparison -- A/B test multiple classifiers on crypto data.

Usage:
    PYTHONPATH=. python scripts/model_comparison.py
"""
from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from trading_engine.autoresearch import CACHE_DIR
from trading_engine.models.base_model import _SmartLGBM
from trading_engine.models.classifiers import (
    SmartRandomForest,
    SmartLogistic,
    SmartMLP,
)
from trading_engine.models.mean_rev_model import (
    build_features_for_market,
    MEAN_REV_CONFIGS,
)

CAPITAL = 10_000.0
THRESHOLD = 0.60
MAX_POS = 0.10
SL_PCT = 0.10
TP_PCT = 0.15


def load_data():
    f = CACHE_DIR / "crypto_2y.pkl"
    with open(f, "rb") as fh:
        return pickle.load(fh)


def build_features(hist, market="crypto"):
    cfg = MEAN_REV_CONFIGS[market]
    ca_sym = cfg["cross_asset_symbol"]
    ca_fl = cfg["cross_asset_features"]
    ca_pfx = cfg["cross_asset_prefix"]

    cross = None
    if ca_sym in hist:
        ca_f = build_features_for_market(hist[ca_sym], market)
        av = [c for c in ca_fl if c in ca_f.columns]
        cross = ca_f[av].copy()
        cross.columns = [f"{ca_pfx}_{c}" for c in cross.columns]

    feats = {}
    for sym, df in hist.items():
        ft = build_features_for_market(df, market)
        if cross is not None and sym != ca_sym:
            ft = ft.join(cross, how="left")
        ft = ft.dropna()
        if len(ft) > cfg["min_train"]:
            feats[sym] = ft
    return feats, cfg


def run_backtest(factory, name, feats, hist, cfg):
    """Walk-forward backtest with given model factory."""
    tw = cfg["train_window"]
    mt = cfg["min_train"]
    re = cfg["retrain_every"]

    dates = sorted(set().union(*(f.index for f in feats.values())))
    if len(dates) < mt + 20:
        return None

    cash = CAPITAL
    longs = {}
    shorts = {}
    trades = []
    costs = 0.0
    model = None
    fcols = None

    prob_samples = []
    for di in range(mt, len(dates) - 1):
        day = dates[di]
        ds = str(day)[:10]

        # --- retrain ---
        if model is None or (di - mt) % re == 0:
            tX, ty = [], []
            for sym, ft in feats.items():
                ts = ft[ft.index < day].tail(tw)
                if len(ts) < mt:
                    continue
                if fcols is None:
                    fcols = [c for c in ts.columns if c not in {"target", "target_dir"}]
                valid = ts.dropna(subset=["target_dir"])
                if len(valid) < 30:
                    continue
                tX.append(valid.reindex(columns=fcols, fill_value=0).fillna(0))
                ty.append(valid["target_dir"])
            if tX:
                X = pd.concat(tX)
                y = pd.concat(ty)
                try:
                    model = factory()
                    model.fit(X, y)
                except Exception as e:
                    print(f"  [{name}] Train error: {e}")
                    model = None

        if model is None or fcols is None:
            continue

        # --- predict and trade ---
        for sym, ft in feats.items():
            if day not in ft.index:
                continue
            if sym not in hist or day not in hist[sym].index:
                continue

            price = float(hist[sym].loc[day, "Close"])
            if price <= 0:
                continue

            row = ft.loc[day]

            row_feats = row.reindex(fcols, fill_value=0).fillna(0)
            X_pred = row_feats.values.reshape(1, -1)
            try:
                proba = model.predict_proba(X_pred)[0]
                up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                continue
            down_prob = 1.0 - up_prob
            if len(prob_samples) < 500:
                prob_samples.append(up_prob)

            # check exits
            if sym in longs:
                pos = longs[sym]
                pnl_pct = (price - pos["entry"]) / pos["entry"]
                if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT or down_prob > THRESHOLD:
                    net = (price - pos["entry"]) * pos["shares"]
                    cost = price * pos["shares"] * 0.001
                    cash += price * pos["shares"] - cost
                    costs += cost
                    trades.append({"sym": sym, "side": "long", "pnl": net - cost,
                                   "entry": pos["entry"], "exit": price, "date": ds})
                    del longs[sym]

            elif sym in shorts:
                pos = shorts[sym]
                pnl_pct = (pos["entry"] - price) / pos["entry"]
                if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT or up_prob > THRESHOLD:
                    net = (pos["entry"] - price) * pos["shares"]
                    cost = price * pos["shares"] * 0.001
                    cash += pos["margin"] + net - cost
                    costs += cost
                    trades.append({"sym": sym, "side": "short", "pnl": net - cost,
                                   "entry": pos["entry"], "exit": price, "date": ds})
                    del shorts[sym]

            # check entries
            elif len(longs) + len(shorts) < 8:
                if up_prob > THRESHOLD:
                    max_val = cash * MAX_POS
                    shares = int(max_val / price)
                    if shares > 0:
                        cost = price * shares * 0.001
                        total = price * shares + cost
                        if total <= cash:
                            cash -= total
                            costs += cost
                            longs[sym] = {"entry": price, "shares": shares, "date": ds}

                elif down_prob > THRESHOLD:
                    max_val = cash * MAX_POS
                    shares = int(max_val / price)
                    if shares > 0:
                        cost = price * shares * 0.001
                        margin = price * shares
                        total = margin + cost
                        if total <= cash:
                            cash -= total
                            costs += cost
                            shorts[sym] = {"entry": price, "shares": shares,
                                           "margin": margin, "date": ds}

    # mark-to-market open positions
    last_day = dates[-1]
    open_pnl = 0.0
    for sym, pos in longs.items():
        if sym in feats and last_day in feats[sym].index:
            lp = float(feats[sym].loc[last_day].get("close_raw",
                        feats[sym].loc[last_day].get("Close", pos["entry"])))
            open_pnl += (lp - pos["entry"]) * pos["shares"]
    for sym, pos in shorts.items():
        if sym in feats and last_day in feats[sym].index:
            lp = float(feats[sym].loc[last_day].get("close_raw",
                        feats[sym].loc[last_day].get("Close", pos["entry"])))
            open_pnl += (pos["entry"] - lp) * pos["shares"]

    closed_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 1
    pf = (sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses)
          if losses and sum(abs(t["pnl"]) for t in losses) > 0 else 0)

    total_val = cash + open_pnl
    for pos in longs.values():
        total_val += pos["entry"] * pos["shares"]
    for pos in shorts.values():
        total_val += pos["margin"]

    prob_arr = np.array(prob_samples) if prob_samples else np.array([0.5])
    return {
        "name": name,
        "prob_mean": round(float(prob_arr.mean()), 3),
        "prob_std": round(float(prob_arr.std()), 3),
        "prob_max": round(float(prob_arr.max()), 3),
        "prob_min": round(float(prob_arr.min()), 3),
        "pct_above_thr": round(float((prob_arr > THRESHOLD).mean() * 100), 1),
        "pct_below_thr": round(float((prob_arr < (1 - THRESHOLD)).mean() * 100), 1),
        "final_value": round(total_val, 2),
        "return_pct": round((total_val - CAPITAL) / CAPITAL * 100, 2),
        "closed_pnl": round(closed_pnl, 2),
        "open_pnl": round(open_pnl, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "costs": round(costs, 2),
        "open_long": len(longs),
        "open_short": len(shorts),
        "long_trades": len([t for t in trades if t["side"] == "long"]),
        "short_trades": len([t for t in trades if t["side"] == "short"]),
    }


def main():
    print("=" * 70)
    print("ML ALGORITHM COMPARISON — Crypto Walk-Forward Backtest")
    print("=" * 70)
    print(f"Capital: ${CAPITAL:,.0f} | Threshold: {THRESHOLD:.0%} | "
          f"SL: {SL_PCT:.0%} | TP: {TP_PCT:.0%}")
    print()

    print("Loading cached crypto data...")
    hist = load_data()
    print(f"  {len(hist)} assets loaded")

    print("Building features...")
    feats, cfg = build_features(hist)
    print(f"  {len(feats)} assets with sufficient data")
    print()

    # --- model factories ---
    models = [
        ("LightGBM", lambda: _SmartLGBM(params=cfg["lgbm_params"])),
        ("RandomForest", lambda: SmartRandomForest()),
        ("Logistic", lambda: SmartLogistic()),
        ("MLP", lambda: SmartMLP()),
    ]

    # try XGBoost
    try:
        from trading_engine.models.classifiers import SmartXGBoost
        models.insert(1, ("XGBoost", lambda: SmartXGBoost()))
    except ImportError:
        print("  (XGBoost not installed — skipping)")

    results = []
    for name, factory in models:
        print(f"Running {name}...", end=" ", flush=True)
        t0 = time.time()
        r = run_backtest(factory, name, feats, hist, cfg)
        elapsed = time.time() - t0
        if r:
            r["time_s"] = round(elapsed, 1)
            results.append(r)
            print(f"done ({elapsed:.1f}s) — {r['return_pct']:+.2f}% | "
                  f"WR: {r['win_rate']:.1f}% | PF: {r['profit_factor']:.2f}x")
        else:
            print("FAILED")

    # --- comparison table ---
    print()
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Algorithm':<15} {'Return%':>8} {'Trades':>7} {'WR%':>6} " \
             f"{'PF':>6} {'AvgWin':>8} {'AvgLoss':>8} {'Long':>5} {'Short':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["return_pct"], reverse=True):
        print(f"{r['name']:<15} {r['return_pct']:>+7.2f}% {r['total_trades']:>7} "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f}x "
              f"${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} "
              f"{r['long_trades']:>5} {r['short_trades']:>6} {r['time_s']:>5.1f}s")

    print()
    print("PROBABILITY DIAGNOSTICS")
    print("-" * 70)
    pd_header = f"{'Algorithm':<15} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'%>Thr':>6} {'%<1-T':>6}"
    print(pd_header)
    print("-" * len(pd_header))
    for r in results:
        print(f"{r['name']:<15} {r['prob_mean']:>6.3f} {r['prob_std']:>6.3f} "
              f"{r['prob_min']:>6.3f} {r['prob_max']:>6.3f} "
              f"{r['pct_above_thr']:>5.1f}% {r['pct_below_thr']:>5.1f}%")

    print()
    print("DETAILED RESULTS")
    print("-" * 70)
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    Final value: ${r['final_value']:,.2f} ({r['return_pct']:+.2f}%)")
        print(f"    Closed P&L: ${r['closed_pnl']:,.2f} | Open P&L: ${r['open_pnl']:,.2f}")
        print(f"    Trades: {r['total_trades']} (W:{r['wins']} L:{r['losses']}) | "
              f"Open: {r['open_long']}L {r['open_short']}S")
        print(f"    Win Rate: {r['win_rate']:.1f}% | Profit Factor: {r['profit_factor']:.2f}x")
        print(f"    Costs: ${r['costs']:,.2f}")

    # --- winner ---
    if results:
        best = max(results, key=lambda x: x["return_pct"])
        print()
        print(f"WINNER: {best['name']} with {best['return_pct']:+.2f}% return")


if __name__ == "__main__":
    main()
