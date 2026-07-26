"""ML Strategy Backtest — compare AI model vs best classical strategy."""

from trading_engine.config import WATCHLIST
from trading_engine.price_engine import get_history
from trading_engine.ml_model import MLSignalGenerator


def main():
    print("=" * 100)
    print("  ML SIGNAL GENERATOR — Walk-Forward Backtest")
    print("  LightGBM classifier | 20+ features | Retrain every 20 days")
    print("=" * 100)

    print("\n  Downloading 1+ year of data...")
    cached = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=400)
        if not df.empty:
            cached[sym] = df
            print(f"    {sym}: {len(df)} days")

    configs = [
        {"capital": 1000, "threshold": 0.60, "label": "$1K / 60% conf"},
        {"capital": 1000, "threshold": 0.65, "label": "$1K / 65% conf"},
        {"capital": 10000, "threshold": 0.60, "label": "$10K / 60% conf"},
        {"capital": 10000, "threshold": 0.65, "label": "$10K / 65% conf"},
    ]

    print(f"\n  {'Config':<20} {'Trades':>7} {'Closed':>7} {'Win%':>6} "
          f"{'Net P&L':>10} {'Return':>8} {'Costs':>8} {'Open':>5}")
    print(f"  {'-' * 80}")

    for cfg in configs:
        ml = MLSignalGenerator(
            train_window=120,
            min_train=60,
            confidence_threshold=cfg["threshold"],
        )

        r = ml.train_and_backtest(
            history_data=cached,
            initial_capital=cfg["capital"],
            max_position_pct=0.15,
            stop_loss_pct=0.03,
            take_profit_pct=0.05,
        )

        if "error" in r:
            print(f"  {cfg['label']:<20} ERROR: {r['error']}")
            continue

        icon = "\U0001f7e2" if r["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {cfg['label']:<20} {r['total_trades']:>7} {r['closed_trades']:>7} "
              f"{r['win_rate']:>5.1f}% {icon} ${r['total_pnl_net']:>+8.2f} "
              f"{r['return_pct']:>+7.2f}% ${r['total_costs']:>7.2f} {r['open_positions']:>5}")

    # Detailed run for the best config
    print(f"\n{'=' * 100}")
    print("  DETAILED: $10K / 60% confidence")
    print(f"{'=' * 100}")

    ml = MLSignalGenerator(train_window=120, min_train=60, confidence_threshold=0.60)
    r = ml.train_and_backtest(
        history_data=cached, initial_capital=10000,
        max_position_pct=0.15, stop_loss_pct=0.03, take_profit_pct=0.05,
    )

    daily = r.get("daily_results", [])
    if daily:
        max_val = max(d["total_value"] for d in daily)
        min_val = min(d["total_value"] for d in daily)
        dd = (max_val - min_val) / max_val * 100 if max_val > 0 else 0
        print(f"\n  Start:     ${daily[0]['total_value']:,.2f}")
        print(f"  End:       ${daily[-1]['total_value']:,.2f}")
        print(f"  Peak:      ${max_val:,.2f}")
        print(f"  Trough:    ${min_val:,.2f}")
        print(f"  Max DD:    {dd:.2f}%")
        print(f"  P&L:       ${r['total_pnl_net']:+,.2f} ({r['return_pct']:+.2f}%)")
        print(f"  Costs:     ${r['total_costs']:,.2f}")

    # Show last 20 closed trades
    closed = [t for t in r.get("trades", []) if t["action"] == "sell"]
    if closed:
        print(f"\n  Last 20 closed trades:")
        print(f"  {'Date':<12} {'Sym':<5} {'Shares':>6} {'Price':>8} {'Net P&L':>9} {'Conf':>6} {'Reason'}")
        print(f"  {'-' * 75}")
        for t in closed[-20:]:
            icon = "\U0001f7e2" if t["net_pnl"] > 0 else "\U0001f534"
            print(f"  {t['date']:<12} {t['symbol']:<5} {t['shares']:>6} "
                  f"${t['price']:>7.2f} {icon} ${t['net_pnl']:>+7.2f} "
                  f"{t['confidence']:>5.1%}  {t.get('reason', '')}")

    # Monthly P&L
    if daily:
        print(f"\n  Monthly P&L:")
        monthly = {}
        for d in daily:
            month = d["date"][:7]
            if month not in monthly:
                monthly[month] = {"start": d["total_value"], "end": d["total_value"], "trades": 0}
            monthly[month]["end"] = d["total_value"]
            monthly[month]["trades"] += d["trades"]

        for month, m in monthly.items():
            pnl = m["end"] - m["start"]
            icon = "\U0001f7e2" if pnl > 0 else "\U0001f534" if pnl < 0 else "\u26aa"
            print(f"    {month}: {icon} ${pnl:>+8.2f}  ({m['trades']} trades)")


if __name__ == "__main__":
    main()
