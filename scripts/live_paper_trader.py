#!/usr/bin/env python3
"""Live Paper Trader — daily crypto trading simulation with real prices.

Runs once per day (via GitHub Actions cron or manually):
1. Loads portfolio state from JSON
2. Fetches latest crypto prices from Yahoo Finance
3. Retrains Logistic Regression model on trailing window (every N days)
4. Checks SL/TP exits for open positions
5. Generates new BUY/SELL signals
6. Persists updated state + appends to trade log CSV

Usage:
    python scripts/live_paper_trader.py              # real run
    python scripts/live_paper_trader.py --dry-run    # no state changes
    python scripts/live_paper_trader.py --verbose    # detailed output
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_engine.config import CRYPTO_WATCHLIST
from trading_engine.models.mean_rev_model import (
    MARKET_CONFIGS,
    build_features_for_market,
)
from trading_engine.models.classifiers import SmartLogistic
from trading_engine.price_engine import build_mtf_features

warnings.filterwarnings("ignore", category=UserWarning)

PAPER_DIR = PROJECT_ROOT / "paper_trading"
STATE_FILE = PAPER_DIR / "portfolio_state.json"
LOG_FILE = PAPER_DIR / "trade_log.csv"

CFG = MARKET_CONFIGS["crypto"]
CONFIDENCE_THRESHOLD = CFG["default_confidence"]  # 0.70
STOP_LOSS_PCT = CFG.get("default_sl", 0.10)
TAKE_PROFIT_PCT = CFG.get("default_tp", 0.15)
MAX_POSITION_PCT = CFG.get("default_max_pos", 0.15)
LOGISTIC_C = CFG.get("logistic_C", 0.15)
TRAIN_WINDOW = CFG["train_window"]  # 150
MIN_TRAIN = CFG["min_train"]  # 60
RETRAIN_EVERY = CFG["retrain_every"]  # 10
CROSS_ASSET = CFG["cross_asset_symbol"]  # BTC-USD
CROSS_FEATURES = CFG["cross_asset_features"]
CROSS_PREFIX = CFG["cross_asset_prefix"]  # btc
COST_PCT = 0.001  # 0.1% taker fee
MAX_POSITIONS = 8


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    return {
        "initial_capital": 1000.0,
        "cash": 1000.0,
        "positions": {},
        "total_costs": 0.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "days_since_retrain": RETRAIN_EVERY,  # force retrain on first run
        "feature_cols": None,
        "start_date": date.today().isoformat(),
        "last_run_date": None,
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return _default_state()


def save_state(state: dict):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_log(rows: list[dict]):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0
    fieldnames = [
        "date", "time", "action", "symbol", "price", "shares",
        "confidence", "pnl", "cost", "cash", "portfolio_value",
        "positions_count", "reason",
    ]
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_crypto_data(period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all crypto symbols."""
    data = {}
    for sym in CRYPTO_WATCHLIST:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    return data


def fetch_hourly_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download hourly OHLCV data for MTF features (max 710 days)."""
    data = {}
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period="710d", interval="1h")
            if df.empty:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    return data


def get_latest_prices(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Get the most recent close price for each symbol."""
    prices = {}
    for sym, df in data.items():
        if not df.empty:
            prices[sym] = float(df["Close"].iloc[-1])
    return prices


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _add_relative_strength(feat: pd.DataFrame, ca_feat: pd.DataFrame,
                           prefix: str) -> pd.DataFrame:
    """Add relative strength features (same logic as time_machine.py)."""
    for period in [5, 10, 20]:
        stock_col = f"return_{period}d"
        idx_col = f"return_{period}d"
        if stock_col in feat.columns and idx_col in ca_feat.columns:
            aligned = ca_feat[idx_col].reindex(feat.index)
            feat[f"rs_{prefix}_{period}d"] = feat[stock_col] - aligned
    if "volatility_5d" in feat.columns and "volatility_5d" in ca_feat.columns:
        aligned = ca_feat["volatility_5d"].reindex(feat.index)
        feat[f"rel_vol_{prefix}"] = feat["volatility_5d"] / aligned.replace(0, np.nan)
    return feat


def train_model(
    data: dict[str, pd.DataFrame],
    feature_cols: list[str] | None = None,
    hourly_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[SmartLogistic | None, list[str] | None]:
    """Train a fresh SmartLogistic on the trailing window of all crypto data."""
    train_X_list = []
    train_y_list = []
    discovered_cols = feature_cols

    ca_df = data.get(CROSS_ASSET)

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

        if hourly_data and symbol in hourly_data:
            h_df = hourly_data[symbol]
            mtf = build_mtf_features(df, h_df)
            for k, v in mtf.items():
                feat.loc[feat.index[-1], k] = v

        if discovered_cols is None and symbol != CROSS_ASSET:
            exclude = {"target", "target_dir"}
            discovered_cols = [c for c in feat.columns if c not in exclude]

        train_slice = feat.tail(TRAIN_WINDOW)
        valid = train_slice.dropna(subset=["target_dir"])
        if len(valid) < 30:
            continue

        if discovered_cols is None:
            exclude = {"target", "target_dir"}
            discovered_cols = [c for c in feat.columns if c not in exclude]

        train_X_list.append(
            valid.reindex(columns=discovered_cols, fill_value=0).fillna(0)
        )
        train_y_list.append(valid["target_dir"])

    if not train_X_list or discovered_cols is None:
        return None, None

    X_train = pd.concat(train_X_list)
    y_train = pd.concat(train_y_list)

    model = SmartLogistic(params={"C": LOGISTIC_C})
    model.fit(X_train, y_train)

    return model, discovered_cols


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_signals(
    model: SmartLogistic,
    feature_cols: list[str],
    data: dict[str, pd.DataFrame],
    hourly_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, dict]:
    """Generate up/down probabilities for each crypto symbol."""
    signals = {}
    ca_df = data.get(CROSS_ASSET)

    for symbol, df in data.items():
        if len(df) < 30:
            continue

        feat = build_features_for_market(df, "crypto")

        if ca_df is not None and symbol != CROSS_ASSET and len(ca_df) > 30:
            ca_feat = build_features_for_market(ca_df, "crypto")
            avail_ca = [c for c in CROSS_FEATURES if c in ca_feat.columns]
            cross = ca_feat[avail_ca].copy()
            cross.columns = [f"{CROSS_PREFIX}_{c}" for c in cross.columns]
            feat = feat.join(cross, how="left")
            feat = _add_relative_strength(feat, ca_feat, CROSS_PREFIX)

        if hourly_data and symbol in hourly_data:
            mtf = build_mtf_features(df, hourly_data[symbol])
            for k, v in mtf.items():
                feat.loc[feat.index[-1], k] = v

        if feat.empty:
            continue

        last_row = feat.iloc[-1]
        row_feats = last_row.reindex(feature_cols, fill_value=0).fillna(0)
        X_pred = row_feats.values.reshape(1, -1)

        proba = model.predict_proba(X_pred)[0]
        up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        down_prob = 1.0 - up_prob
        price = float(df["Close"].iloc[-1])

        signals[symbol] = {
            "up_prob": up_prob,
            "down_prob": down_prob,
            "price": price,
        }

    return signals


# ---------------------------------------------------------------------------
# Portfolio value
# ---------------------------------------------------------------------------

def compute_portfolio_value(state: dict, prices: dict[str, float]) -> float:
    """Compute total portfolio value: cash + position values."""
    value = state["cash"]
    for sym, pos in state["positions"].items():
        current_price = prices.get(sym, pos["entry_price"])
        if pos["side"] == "long":
            value += pos["shares"] * current_price
        else:  # short
            entry_val = pos["shares"] * pos["entry_price"]
            current_val = pos["shares"] * current_price
            value += entry_val + (entry_val - current_val)
    return value


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------

def check_exits(state: dict, prices: dict[str, float], today: str) -> list[dict]:
    """Check SL/TP for all open positions. Return log rows for exits."""
    log_rows = []
    to_close = []

    for sym, pos in state["positions"].items():
        current_price = prices.get(sym)
        if current_price is None:
            continue

        entry_price = pos["entry_price"]

        if pos["side"] == "long":
            pct_change = (current_price - entry_price) / entry_price
            hit_sl = pct_change <= -STOP_LOSS_PCT
            hit_tp = pct_change >= TAKE_PROFIT_PCT
        else:  # short
            pct_change = (entry_price - current_price) / entry_price
            hit_sl = pct_change <= -STOP_LOSS_PCT
            hit_tp = pct_change >= TAKE_PROFIT_PCT

        if hit_sl or hit_tp:
            shares = pos["shares"]
            cost = current_price * shares * COST_PCT

            if pos["side"] == "long":
                gross_pnl = (current_price - entry_price) * shares
            else:
                gross_pnl = (entry_price - current_price) * shares

            net_pnl = gross_pnl - cost - pos["entry_cost"]

            state["cash"] += current_price * shares - cost if pos["side"] == "long" else (
                pos["shares"] * pos["entry_price"] + gross_pnl - cost
            )
            state["total_costs"] += cost
            state["total_trades"] += 1

            if net_pnl > 0:
                state["wins"] += 1
            else:
                state["losses"] += 1

            reason = f"{'TP' if hit_tp else 'SL'} hit ({pct_change:+.1%})"
            to_close.append(sym)

            portfolio_value = compute_portfolio_value(state, prices)
            log_rows.append({
                "date": today, "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "action": "SELL" if pos["side"] == "long" else "COVER",
                "symbol": sym, "price": f"{current_price:.2f}",
                "shares": shares, "confidence": "",
                "pnl": f"{net_pnl:+.2f}", "cost": f"{cost:.4f}",
                "cash": f"{state['cash']:.2f}",
                "portfolio_value": f"{portfolio_value:.2f}",
                "positions_count": len(state["positions"]) - 1,
                "reason": reason,
            })

    for sym in to_close:
        del state["positions"][sym]

    return log_rows


def check_entries(
    state: dict,
    signals: dict[str, dict],
    prices: dict[str, float],
    today: str,
) -> list[dict]:
    """Open new positions based on model signals."""
    log_rows = []

    for sym, sig in sorted(signals.items(), key=lambda x: -x[1]["up_prob"]):
        if len(state["positions"]) >= MAX_POSITIONS:
            break
        if sym in state["positions"]:
            continue

        up_prob = sig["up_prob"]
        down_prob = sig["down_prob"]
        price = sig["price"]

        direction = None
        confidence = 0.0
        if up_prob > CONFIDENCE_THRESHOLD:
            direction = "long"
            confidence = up_prob
        elif down_prob > CONFIDENCE_THRESHOLD:
            direction = "short"
            confidence = down_prob

        if direction is None:
            continue

        max_value = state["cash"] * MAX_POSITION_PCT
        shares = max_value / price
        if shares * price < 1.0:  # minimum $1 position
            continue

        cost = price * shares * COST_PCT

        if direction == "long":
            total_debit = price * shares + cost
            if total_debit > state["cash"]:
                continue
            state["cash"] -= total_debit
        else:  # short
            margin = price * shares
            total_debit = margin + cost
            if total_debit > state["cash"]:
                continue
            state["cash"] -= total_debit

        state["total_costs"] += cost

        state["positions"][sym] = {
            "symbol": sym,
            "side": direction,
            "shares": round(shares, 6),
            "entry_price": price,
            "entry_cost": cost,
            "entry_date": today,
            "confidence": round(confidence, 4),
        }

        portfolio_value = compute_portfolio_value(state, prices)
        log_rows.append({
            "date": today, "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "action": "BUY" if direction == "long" else "SHORT",
            "symbol": sym, "price": f"{price:.2f}",
            "shares": f"{shares:.6f}", "confidence": f"{confidence:.4f}",
            "pnl": "", "cost": f"{cost:.4f}",
            "cash": f"{state['cash']:.2f}",
            "portfolio_value": f"{portfolio_value:.2f}",
            "positions_count": len(state["positions"]),
            "reason": f"ML {direction} ({confidence:.0%} conf)",
        })

    return log_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False, verbose: bool = False):
    today = date.today().isoformat()
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")

    print(f"{'='*60}")
    print(f"  LIVE PAPER TRADER — {today} {now} UTC")
    print(f"  Model: Logistic Regression (C={LOGISTIC_C})")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD:.0%} | SL: {STOP_LOSS_PCT:.0%} | TP: {TAKE_PROFIT_PCT:.0%}")
    if dry_run:
        print(f"  *** DRY RUN — no state changes ***")
    print(f"{'='*60}")

    # 1. Load state
    state = load_state()
    print(f"\n  Portfolio: ${state['cash']:.2f} cash + {len(state['positions'])} positions")
    print(f"  Started: {state['start_date']} | Last run: {state.get('last_run_date', 'never')}")

    # 2. Fetch data
    print(f"\n  Fetching crypto data...")
    data = fetch_crypto_data(period="1y")
    print(f"  Got {len(data)} symbols, ~{max(len(df) for df in data.values()) if data else 0} days")

    print(f"  Fetching hourly data (MTF features)...")
    hourly_data = fetch_hourly_data(list(data.keys()))
    print(f"  Got {len(hourly_data)} symbols hourly")

    if not data:
        print("  ERROR: No data fetched. Exiting.")
        return

    prices = get_latest_prices(data)

    # 3. Show current portfolio value
    portfolio_value = compute_portfolio_value(state, prices)
    pnl = portfolio_value - state["initial_capital"]
    print(f"\n  Portfolio value: ${portfolio_value:.2f} ({pnl:+.2f}, {pnl/state['initial_capital']*100:+.1f}%)")

    if state["positions"]:
        print(f"\n  Open Positions:")
        for sym, pos in state["positions"].items():
            current = prices.get(sym, pos["entry_price"])
            if pos["side"] == "long":
                pos_pnl = (current - pos["entry_price"]) * pos["shares"]
                pct = (current - pos["entry_price"]) / pos["entry_price"] * 100
            else:
                pos_pnl = (pos["entry_price"] - current) * pos["shares"]
                pct = (pos["entry_price"] - current) / pos["entry_price"] * 100
            print(f"    {pos['side'].upper():>5} {sym:<12} entry=${pos['entry_price']:.2f} "
                  f"now=${current:.2f} pnl=${pos_pnl:+.2f} ({pct:+.1f}%)")

    # 4. Retrain if needed
    days_since = state.get("days_since_retrain", RETRAIN_EVERY)
    feature_cols = state.get("feature_cols")

    if days_since >= RETRAIN_EVERY or feature_cols is None:
        print(f"\n  Retraining model (days since last: {days_since})...")
        model, feature_cols = train_model(data, feature_cols, hourly_data=hourly_data)
        if model is None:
            print("  ERROR: Model training failed. Exiting.")
            return
        state["days_since_retrain"] = 0
        state["feature_cols"] = feature_cols
        print(f"  Trained on {len(feature_cols)} features")
    else:
        print(f"\n  Reusing model (retrain in {RETRAIN_EVERY - days_since} days)...")
        model, feature_cols = train_model(data, feature_cols, hourly_data=hourly_data)
        if model is None:
            print("  ERROR: Model training failed. Exiting.")
            return
        state["days_since_retrain"] = days_since + 1

    # 5. Check exits
    log_rows = []
    exit_rows = check_exits(state, prices, today)
    log_rows.extend(exit_rows)
    if exit_rows:
        print(f"\n  EXITS ({len(exit_rows)}):")
        for r in exit_rows:
            print(f"    {r['action']} {r['symbol']} @ ${r['price']} | PnL: {r['pnl']} | {r['reason']}")

    # 6. Check entries
    print(f"\n  Scanning signals...")
    signals = predict_signals(model, feature_cols, data, hourly_data=hourly_data)

    if verbose:
        print(f"\n  Signal Probabilities:")
        for sym, sig in sorted(signals.items(), key=lambda x: -x[1]["up_prob"]):
            marker = ""
            if sig["up_prob"] > CONFIDENCE_THRESHOLD:
                marker = " *** BUY SIGNAL"
            elif sig["down_prob"] > CONFIDENCE_THRESHOLD:
                marker = " *** SHORT SIGNAL"
            print(f"    {sym:<12} up={sig['up_prob']:.3f} down={sig['down_prob']:.3f} "
                  f"${sig['price']:.2f}{marker}")

    entry_rows = check_entries(state, signals, prices, today)
    log_rows.extend(entry_rows)
    if entry_rows:
        print(f"\n  ENTRIES ({len(entry_rows)}):")
        for r in entry_rows:
            print(f"    {r['action']} {r['symbol']} @ ${r['price']} | conf: {r['confidence']} | {r['reason']}")

    # 7. Log HOLD if no actions
    if not log_rows:
        portfolio_value = compute_portfolio_value(state, prices)
        log_rows.append({
            "date": today, "time": now,
            "action": "HOLD", "symbol": "", "price": "",
            "shares": "", "confidence": "",
            "pnl": "", "cost": "",
            "cash": f"{state['cash']:.2f}",
            "portfolio_value": f"{portfolio_value:.2f}",
            "positions_count": len(state["positions"]),
            "reason": "No signals above threshold",
        })

    # 8. Summary
    portfolio_value = compute_portfolio_value(state, prices)
    total_pnl = portfolio_value - state["initial_capital"]
    closed = state["wins"] + state["losses"]
    wr = state["wins"] / closed * 100 if closed else 0

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Portfolio:    ${portfolio_value:.2f} ({total_pnl:+.2f}, {total_pnl/state['initial_capital']*100:+.1f}%)")
    print(f"  Cash:         ${state['cash']:.2f}")
    print(f"  Positions:    {len(state['positions'])} open")
    print(f"  Trades:       {state['total_trades']} closed ({state['wins']}W/{state['losses']}L, {wr:.0f}% WR)")
    print(f"  Total costs:  ${state['total_costs']:.4f}")
    print(f"  Today:        {len(exit_rows)} exits, {len(entry_rows)} entries")
    print(f"  {'='*60}")

    # 9. Persist
    if not dry_run:
        state["last_run_date"] = today
        save_state(state)
        append_log(log_rows)
        print(f"\n  State saved to {STATE_FILE}")
        print(f"  Log appended to {LOG_FILE}")
    else:
        print(f"\n  DRY RUN — no files written")


def main():
    parser = argparse.ArgumentParser(description="Live Paper Trader — Crypto")
    parser.add_argument("--dry-run", action="store_true", help="No state changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show signal details")
    args = parser.parse_args()
    run(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
