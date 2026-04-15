"""Strategy Shootout — compare all strategies across multiple time windows."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from trading_engine.backtest import run_backtest
from trading_engine.config import PROJECT_ROOT, WATCHLIST
from trading_engine.price_engine import get_history


STRATEGIES = {
    "Connors RSI(2)": "strategies/connors_rsi2.json",
    "3-Day High/Low": "strategies/connors_3day_highlow.json",
    "Turnaround Tue": "strategies/turnaround_tuesday.json",
    "IBS Reversion": "strategies/ibs_mean_reversion.json",
    "RSI2+IBS Combo": "strategies/rsi2_ibs_combo.json",
    "Custom L/S v3": "rules.json",
}


def _slice_history(cached, end_days_ago):
    cutoff = datetime.now() - timedelta(days=end_days_ago)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    return {s: df[df.index <= cutoff_str] for s, df in cached.items()
            if not df[df.index <= cutoff_str].empty}


def main():
    print("=" * 130)
    print("  STRATEGY SHOOTOUT — 6 Strategies × 8 Time Windows")
    print("  Capital: $1,000 | Broker: eToro | Currency: USD")
    print("=" * 130)

    print("\n  Downloading 1 year of data...")
    cached = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=400)
        if not df.empty:
            cached[sym] = df
            print(f"    {sym}: {len(df)} days")

    windows = [
        ("Last 15d", 15, 0),
        ("Last 30d", 30, 0),
        ("Last 60d", 60, 0),
        ("Last 90d", 90, 0),
    ]

    random.seed(42)
    offsets = sorted(random.sample(range(90, 330), 4), reverse=True)
    for i, off in enumerate(offsets, 1):
        end = datetime.now() - timedelta(days=off)
        start = end - timedelta(days=30)
        label = f"Rnd #{i} {start.strftime('%b%d')}-{end.strftime('%b%d')}"
        windows.append((label, 30, off))

    all_results = {}

    for strat_name, rules_file in STRATEGIES.items():
        rules_path = str(PROJECT_ROOT / rules_file)
        strat_results = []

        for label, days, offset in windows:
            hist = _slice_history(cached, offset) if offset > 0 else cached

            r = run_backtest(
                days=days, rules_path=rules_path,
                initial_capital=1000, broker="etoro",
                account_currency="USD", cached_history=hist,
            )

            strat_results.append({
                "window": label, "days": days,
                "trades": r.get("total_trades", 0),
                "win_rate": r.get("win_rate", 0),
                "net_pnl": r.get("total_pnl_net", 0),
                "return_pct": r.get("return_pct", 0),
                "error": r.get("error"),
            })

        all_results[strat_name] = strat_results

    # Print comparison table per window
    for wi, (label, _, _) in enumerate(windows):
        print(f"\n  {'─' * 120}")
        print(f"  Window: {label}")
        print(f"  {'Strategy':<20} {'Trades':>7} {'Win%':>6} {'Net P&L':>10} {'Return':>8}")
        print(f"  {'─' * 55}")

        for strat_name in STRATEGIES:
            r = all_results[strat_name][wi]
            if r.get("error"):
                print(f"  {strat_name:<20} {'ERROR':>7}")
                continue
            icon = "\U0001f7e2" if r["net_pnl"] > 0 else "\U0001f534" if r["net_pnl"] < 0 else "\u26aa"
            print(f"  {strat_name:<20} {r['trades']:>7} {r['win_rate']:>5.0f}% "
                  f"{icon} ${r['net_pnl']:>+8.2f} {r['return_pct']:>+7.2f}%")

    # Summary: rank strategies by total P&L and profitability rate
    print(f"\n{'=' * 130}")
    print(f"  OVERALL RANKING")
    print(f"  {'Strategy':<20} {'Total P&L':>10} {'Avg Ret':>8} {'Avg Win%':>9} {'Profitable':>11} {'Avg Trades':>11}")
    print(f"  {'─' * 75}")

    rankings = []
    for strat_name in STRATEGIES:
        results = all_results[strat_name]
        valid = [r for r in results if not r.get("error") and r["trades"] > 0]
        if not valid:
            rankings.append((strat_name, 0, 0, 0, 0, 0))
            continue
        total_pnl = sum(r["net_pnl"] for r in valid)
        avg_ret = sum(r["return_pct"] for r in valid) / len(valid)
        avg_wr = sum(r["win_rate"] for r in valid) / len(valid)
        profitable = sum(1 for r in valid if r["net_pnl"] > 0)
        avg_trades = sum(r["trades"] for r in valid) / len(valid)
        rankings.append((strat_name, total_pnl, avg_ret, avg_wr, profitable, avg_trades))

    rankings.sort(key=lambda x: x[1], reverse=True)
    for name, pnl, ret, wr, prof, trades in rankings:
        valid_count = len([r for r in all_results[name] if not r.get("error") and r["trades"] > 0])
        icon = "\U0001f947" if pnl == rankings[0][1] else "\U0001f948" if pnl == rankings[1][1] else "\U0001f949" if pnl == rankings[2][1] else "  "
        print(f"  {icon} {name:<18} ${pnl:>+9.2f} {ret:>+7.2f}% {wr:>8.0f}% "
              f"{prof}/{valid_count} ({prof/max(valid_count,1)*100:.0f}%) {trades:>10.0f}")

    print(f"{'=' * 130}")

    out = PROJECT_ROOT / "optimization_results" / "strategy_shootout.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
