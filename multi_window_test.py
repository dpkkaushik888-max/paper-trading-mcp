"""Multi-window backtest: test optimized rules across different time windows."""

import json
import random
from datetime import datetime, timedelta

from trading_engine.backtest import run_backtest
from trading_engine.config import PROJECT_ROOT, WATCHLIST
from trading_engine.price_engine import get_history


def main():
    print("=" * 110)
    print("  MULTI-WINDOW BACKTEST — Optimized Long/Short v3.0 across time windows")
    print("  Capital: $1,000 | Broker: eToro | Currency: USD")
    print("=" * 110)

    # Download 1 year of data
    print("\n  Downloading 1 year of historical data...")
    cached = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=400)
        if not df.empty:
            cached[sym] = df
            print(f"    {sym}: {len(df)} days")

    rules_path = str(PROJECT_ROOT / "rules.json")

    # Fixed windows
    windows = [
        ("Last 1 day", 1),
        ("Last 15 days", 15),
        ("Last 30 days", 30),
        ("Last 60 days", 60),
        ("Last 90 days", 90),
    ]

    # 3 random 30-day windows from last year
    # Pick random start offsets between 90 and 330 days ago
    random.seed(42)
    random_offsets = sorted(random.sample(range(90, 330), 3), reverse=True)
    for i, offset in enumerate(random_offsets, 1):
        end_date = datetime.now() - timedelta(days=offset)
        start_date = end_date - timedelta(days=30)
        label = f"Random 30d #{i} ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')})"
        windows.append((label, 30, offset))

    results = []

    for window in windows:
        label = window[0]
        days = window[1]
        offset = window[2] if len(window) > 2 else 0

        if offset > 0:
            # For random windows, slice the cached data to end at the offset
            sliced = {}
            for sym, df in cached.items():
                cutoff_date = datetime.now() - timedelta(days=offset)
                sliced_df = df[df.index <= cutoff_date.strftime("%Y-%m-%d")]
                if not sliced_df.empty:
                    sliced[sym] = sliced_df
            hist = sliced
        else:
            hist = cached

        r = run_backtest(
            days=days,
            rules_path=rules_path,
            initial_capital=1000,
            broker="etoro",
            account_currency="USD",
            cached_history=hist,
        )

        if "error" in r:
            results.append({
                "window": label, "days": days,
                "trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "net_pnl": 0, "return_pct": 0,
                "error": r["error"],
            })
        else:
            results.append({
                "window": label, "days": days,
                "trades": r.get("total_trades", 0),
                "wins": r.get("wins", 0),
                "losses": r.get("losses", 0),
                "win_rate": r.get("win_rate", 0),
                "net_pnl": r.get("total_pnl_net", 0),
                "return_pct": r.get("return_pct", 0),
                "cost_drag": r.get("cost_drag_pct", 0),
                "open_pos": r.get("open_positions", 0),
            })

    # Print results table
    print(f"\n{'=' * 110}")
    print(f"  {'Window':<45} {'Days':>5} {'Trades':>7} {'W/L':>8} {'Win%':>6} {'Net P&L':>10} {'Return':>8} {'Cost%':>8}")
    print(f"  {'-' * 105}")

    for r in results:
        if "error" in r:
            print(f"  {r['window']:<45} {r['days']:>5}   {r['error']}")
            continue

        icon = "\U0001f7e2" if r["net_pnl"] > 0 else "\U0001f534" if r["net_pnl"] < 0 else "\u26aa"
        print(
            f"  {r['window']:<45} {r['days']:>5} {r['trades']:>7} "
            f"{r['wins']}W/{r['losses']}L "
            f"{r['win_rate']:>5.0f}% "
            f"{icon} ${r['net_pnl']:>+8.2f} "
            f"{r['return_pct']:>+7.2f}% "
            f"{r.get('cost_drag', 0):>7.1f}%"
        )

    print(f"  {'=' * 105}")

    # Summary stats
    valid = [r for r in results if "error" not in r and r["trades"] > 0]
    if valid:
        profitable = sum(1 for r in valid if r["net_pnl"] > 0)
        total_pnl = sum(r["net_pnl"] for r in valid)
        avg_return = sum(r["return_pct"] for r in valid) / len(valid)
        avg_winrate = sum(r["win_rate"] for r in valid) / len(valid)
        print(f"\n  SUMMARY:")
        print(f"    Windows tested: {len(valid)}")
        print(f"    Profitable: {profitable}/{len(valid)} ({profitable/len(valid)*100:.0f}%)")
        print(f"    Total P&L across all windows: ${total_pnl:+.2f}")
        print(f"    Avg return per window: {avg_return:+.2f}%")
        print(f"    Avg win rate: {avg_winrate:.0f}%")

    # Save
    out = PROJECT_ROOT / "optimization_results" / "multi_window_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
