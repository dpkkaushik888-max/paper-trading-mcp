#!/usr/bin/env python3
"""Indian market ML backtest using Zerodha cost structure.

Uses NSE stocks via Yahoo Finance (.NS suffix).
Applies Zerodha fees: zero brokerage (delivery), STT, stamp duty, GST, SEBI charges.
Capital in INR.
"""
import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from trading_engine.ml_model import MLSignalGenerator
from trading_engine.cost_engine import CostEngine

INDIA_WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TATAMOTORS.NS",
    "NIFTYBEES.NS",
]


def fetch_indian_data(symbols: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    """Download Indian stock data from Yahoo Finance."""
    history = {}

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                print(f"    {sym}: no data")
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            history[sym] = df
            print(f"    {sym}: {len(df)} days")
        except Exception as e:
            print(f"    {sym}: ERROR {e}")

    return history


def main():
    print("=" * 110)
    print("  INDIAN MARKET ML BACKTEST — Zerodha (NSE) + LightGBM")
    print("  Zero brokerage delivery | STT + Stamp Duty + GST + SEBI charges")
    print("=" * 110)

    print("\n  Downloading 5 years of NSE data...")
    t0 = time.time()
    cached = fetch_indian_data(INDIA_WATCHLIST, period="5y")
    print(f"  Downloaded {len(cached)} stocks in {time.time() - t0:.0f}s\n")

    if len(cached) < 5:
        print("  ERROR: Not enough stocks with data. Check internet connection.")
        return

    cost_engine = CostEngine("zerodha")

    print("  Cost structure (per ₹1,00,000 trade):")
    sample = cost_engine.calculate_trade_cost("RELIANCE.NS", 2500, 40, "buy", "INR")
    print(f"    Spread:      ₹{sample.spread_cost:.2f}")
    print(f"    Slippage:    ₹{sample.slippage_cost:.2f}")
    stat = cost_engine._calculate_indian_statutory(100000, "buy")
    print(f"    Statutory:   ₹{stat:.2f} (STT + stamp + GST + SEBI)")
    print(f"    Total:       ₹{sample.total_cost:.2f} ({sample.total_cost / sample.gross_value * 100:.3f}%)")

    for capital_inr in [100_000, 500_000, 10_00_000]:
        cap_label = f"₹{capital_inr:,.0f}" if capital_inr < 10_00_000 else f"₹{capital_inr / 10_00_00:.1f}L"
        print(f"\n  Capital: {cap_label}")
        print(f"  {'Conf':>5} {'Total':>6} {'Long':>5} {'Short':>6} {'Closed':>7} {'Win%':>6} {'Net P&L':>12} {'Return':>8} {'MaxDD':>7}")
        print(f"  {'-' * 75}")

        for conf in [0.55, 0.60, 0.65, 0.70]:
            ml = MLSignalGenerator(market="india", confidence_threshold=conf)
            r = ml.train_and_backtest(
                history_data=cached,
                initial_capital=capital_inr,
                max_position_pct=0.15,
                stop_loss_pct=0.05,
                take_profit_pct=0.08,
            )
            if "error" in r:
                print(f"  {conf:>4.0%}  ERROR: {r['error']}")
                continue

            daily = r.get("daily_results", [])
            vals = [d["total_value"] for d in daily] if daily else [capital_inr]
            peak = vals[0]
            max_dd = 0
            for v in vals:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak * 100
                if dd > max_dd:
                    max_dd = dd

            icon = "\U0001f7e2" if r["total_pnl_net"] > 0 else "\U0001f534"
            star = " <<<" if r["return_pct"] > 3 else ""
            print(
                f"  {conf:>4.0%}  {r['total_trades']:>6} {r.get('long_trades',0):>5} "
                f"{r.get('short_trades',0):>6} {r['closed_trades']:>7} "
                f"{r['win_rate']:>5.1f}% {icon} ₹{r['total_pnl_net']:>+10,.0f} "
                f"{r['return_pct']:>+7.2f}% {max_dd:>6.2f}%{star}"
            )

    print(f"\n  Completed in {time.time() - t0:.0f}s")

    print("\n  ZERODHA COST ADVANTAGES:")
    print("  • Zero brokerage on equity delivery trades")
    print("  • No FX conversion costs (trading in INR)")
    print("  • Lower total cost vs eToro: ~0.13% vs ~1.6% per trade")
    print("  • STT is the main cost (~0.1% buy + 0.1% sell)")


if __name__ == "__main__":
    main()
