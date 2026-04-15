#!/usr/bin/env python3
"""Time-Machine Backtest Runner — day-by-day replay with learning loop.

Compares time-machine (strict temporal isolation + continuous learning)
vs batch walk-forward backtest for both US and India markets.

Usage:
    python time_machine_run.py                    # both markets
    python time_machine_run.py --market india      # India only
    python time_machine_run.py --market us          # US only
    python time_machine_run.py --no-learning        # disable learning loop
    python time_machine_run.py --verbose            # day-by-day output
"""

from __future__ import annotations

import argparse
import time

import yfinance as yf
import pandas as pd

from trading_engine.config import WATCHLIST, CRYPTO_WATCHLIST
from trading_engine.ml_model import MLSignalGenerator
from trading_engine.price_engine import get_history
from trading_engine.time_machine import TimeMachineBacktest

INDIA_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "HCLTECH.NS", "WIPRO.NS", "NIFTYBEES.NS",
]


def fetch_data(symbols: list[str], period: str = "5y", label: str = "") -> dict[str, pd.DataFrame]:
    """Download historical data."""
    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            data[sym] = df
        except Exception:
            pass
    print(f"  {label}: {len(data)} stocks, ~{max(len(df) for df in data.values()) if data else 0} days")
    return data


def run_comparison(
    market: str,
    data: dict[str, pd.DataFrame],
    capital: float,
    confidence: float,
    stop_loss: float,
    take_profit: float,
    enable_learning: bool = True,
    verbose: bool = False,
    dynamic_sl: bool = False,
):
    """Run batch vs time-machine comparison for a single config."""
    print(f"\n  {'='*80}")
    print(f"  {market.upper()} | Capital: {capital:,.0f} | Confidence: {confidence:.0%} | "
          f"SL: {stop_loss:.0%} | TP: {take_profit:.0%}")
    print(f"  {'='*80}")

    print(f"\n  [1/2] Batch walk-forward (original)...")
    t0 = time.time()
    ml = MLSignalGenerator(market=market, confidence_threshold=confidence)
    batch_r = ml.train_and_backtest(
        history_data=data, initial_capital=capital,
        max_position_pct=0.15, stop_loss_pct=stop_loss, take_profit_pct=take_profit,
    )
    batch_time = time.time() - t0

    sl_label = 'ATR-dynamic' if dynamic_sl else 'fixed'
    print(f"\n  [2/2] Time-machine replay (learning={'ON' if enable_learning else 'OFF'}, SL/TP={sl_label})...")
    t0 = time.time()
    tm = TimeMachineBacktest(
        market=market, initial_capital=capital,
        confidence_threshold=confidence,
        max_position_pct=0.15,
        stop_loss_pct=stop_loss, take_profit_pct=take_profit,
        session_id=f"tm_{market}_{confidence:.0%}",
        enable_learning=enable_learning,
        dynamic_sl=dynamic_sl,
    )
    tm_r = tm.run(history_data=data, verbose=verbose)
    tm_time = time.time() - t0

    if "error" in batch_r:
        print(f"  Batch ERROR: {batch_r['error']}")
        return
    if "error" in tm_r:
        print(f"  Time-machine ERROR: {tm_r['error']}")
        return

    print(f"\n  {'METRIC':<30} {'BATCH':>12} {'TIME-MACHINE':>14} {'DELTA':>10}")
    print(f"  {'-'*70}")

    metrics = [
        ("Return %", f"{batch_r['return_pct']:+.2f}%", f"{tm_r['return_pct']:+.2f}%",
         f"{tm_r['return_pct'] - batch_r['return_pct']:+.2f}%"),
        ("Net P&L", f"{batch_r['total_pnl_net']:+,.0f}", f"{tm_r['total_pnl_net']:+,.0f}",
         f"{tm_r['total_pnl_net'] - batch_r['total_pnl_net']:+,.0f}"),
        ("Total trades", f"{batch_r['total_trades']}", f"{tm_r['total_trades']}",
         f"{tm_r['total_trades'] - batch_r['total_trades']:+d}"),
        ("Long trades", f"{batch_r.get('long_trades', 0)}", f"{tm_r.get('long_trades', 0)}",
         f"{tm_r.get('long_trades', 0) - batch_r.get('long_trades', 0):+d}"),
        ("Short trades", f"{batch_r.get('short_trades', 0)}", f"{tm_r.get('short_trades', 0)}",
         f"{tm_r.get('short_trades', 0) - batch_r.get('short_trades', 0):+d}"),
        ("Closed trades", f"{batch_r['closed_trades']}", f"{tm_r['closed_trades']}",
         f"{tm_r['closed_trades'] - batch_r['closed_trades']:+d}"),
        ("Win rate", f"{batch_r['win_rate']:.1f}%", f"{tm_r['win_rate']:.1f}%",
         f"{tm_r['win_rate'] - batch_r['win_rate']:+.1f}%"),
        ("Total costs", f"{batch_r['total_costs']:,.0f}", f"{tm_r['total_costs']:,.0f}", ""),
        ("Runtime (sec)", f"{batch_time:.0f}s", f"{tm_time:.0f}s", ""),
    ]

    for label, batch_val, tm_val, delta in metrics:
        print(f"  {label:<30} {batch_val:>12} {tm_val:>14} {delta:>10}")

    batch_daily = batch_r.get("daily_results", [])
    tm_daily = tm_r.get("daily_results", [])
    batch_vals = [d["total_value"] for d in batch_daily] if batch_daily else [capital]
    tm_vals = [d["total_value"] for d in tm_daily] if tm_daily else [capital]

    def max_dd(vals):
        peak = vals[0]
        dd = 0
        for v in vals:
            if v > peak:
                peak = v
            d = (peak - v) / peak * 100
            if d > dd:
                dd = d
        return dd

    b_dd = max_dd(batch_vals)
    t_dd = max_dd(tm_vals)
    print(f"  {'Max drawdown':<30} {b_dd:>11.2f}% {t_dd:>13.2f}% {t_dd - b_dd:>+9.2f}%")

    cal = tm_r.get("calibration", {})
    if cal.get("total_trades", 0) > 0:
        print(f"\n  CALIBRATION (time-machine):")
        print(f"    Total trades analyzed: {cal.get('total_trades', 0)}")
        print(f"    Win rate:             {cal.get('win_rate', 'N/A')}")
        print(f"    Avg confidence:       {cal.get('avg_confidence', 'N/A')}")
        print(f"    Calibration error:    {cal.get('calibration_error', 'N/A')}")
        print(f"    Overconfident:        {cal.get('overconfident', 'N/A')}")
        print(f"    Recommended threshold: {cal.get('recommended_threshold', 'N/A')}")

        buckets = cal.get("buckets", {})
        if buckets:
            print(f"\n    {'Bucket':<12} {'Trades':>7} {'Win%':>7} {'Conf':>7} {'Gap':>7}")
            print(f"    {'-'*44}")
            for bucket, info in sorted(buckets.items()):
                print(f"    {bucket:<12} {info['trades']:>7} "
                      f"{info['win_rate']:>6.1%} {info['avg_confidence']:>6.1%} "
                      f"{info['calibration_gap']:>+6.1%}")

    cb_log = tm_r.get("circuit_breaker_log", [])
    if cb_log:
        print(f"\n  CIRCUIT BREAKERS ({len(cb_log)} activations):")
        for cb in cb_log:
            print(f"    {cb['date']} | {cb['tier'].upper():>8} | DD: {cb['dd']:.2f}% | Value: {cb['value']:,.0f}")
    else:
        print(f"\n  CIRCUIT BREAKERS: None triggered")



def main():
    parser = argparse.ArgumentParser(description="Time-Machine Backtest Runner")
    parser.add_argument("--market", choices=["us", "india", "crypto", "both"], default="both")
    parser.add_argument("--no-learning", action="store_true", help="Disable learning loop")
    parser.add_argument("--dynamic-sl", action="store_true", help="ATR-based dynamic SL/TP (1.5x/2.5x ATR)")
    parser.add_argument("--verbose", action="store_true", help="Day-by-day output")
    parser.add_argument("--period", default="5y", help="Data period: 6mo, 1y, 2y, 5y")
    args = parser.parse_args()

    print("=" * 90)
    print("  TIME-MACHINE BACKTEST — Batch vs Day-by-Day Replay")
    print("  Strict temporal isolation | Continuous learning | Confidence calibration")
    print("=" * 90)

    period = args.period
    markets = []
    if args.market in ("us", "both"):
        markets.append(("us", WATCHLIST, 10_000, 0.80, 0.03, 0.05, period))
    if args.market in ("india", "both"):
        markets.append(("india", INDIA_WATCHLIST, 100_000, 0.80, 0.05, 0.08, period))
    if args.market == "crypto":
        markets.append(("crypto", CRYPTO_WATCHLIST, 10_000, 0.85, 0.10, 0.15, period))

    for market, watchlist, capital, conf, sl, tp, period in markets:
        print(f"\n  Downloading {market.upper()} data ({period})...")
        t0 = time.time()
        if market == "us":
            data = {}
            for sym in watchlist:
                days_map = {"6mo": 200, "1y": 400, "2y": 600, "5y": 1500}
                df = get_history(sym, days=days_map.get(period, 1500))
                if not df.empty:
                    data[sym] = df
            try:
                vix = yf.Ticker("^VIX").history(period=period, interval="1d")
                if not vix.empty:
                    vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index
                    data["^VIX"] = vix
            except Exception:
                pass
            print(f"  US: {len(data)} stocks, ~{max(len(df) for df in data.values()) if data else 0} days")
        elif market == "crypto":
            data = fetch_data(watchlist, period=period, label="Crypto")
        else:
            data = fetch_data(watchlist, period=period, label="India")
        print(f"  Download time: {time.time() - t0:.0f}s")

        run_comparison(
            market=market, data=data, capital=capital,
            confidence=conf, stop_loss=sl, take_profit=tp,
            enable_learning=not args.no_learning,
            verbose=args.verbose,
            dynamic_sl=args.dynamic_sl,
        )

    print(f"\n{'='*90}")
    print("  DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
