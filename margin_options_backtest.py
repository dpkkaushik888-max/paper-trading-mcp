"""Margin & Options Backtest — compare all trading modes."""

from trading_engine.config import WATCHLIST
from trading_engine.price_engine import get_history
from trading_engine.ml_model import MLSignalGenerator
from trading_engine.margin_options_sim import MarginSimulator, OptionsSimulator


def main():
    print("=" * 120)
    print("  MARGIN & OPTIONS vs CASH — ML-Powered Backtest")
    print("  All use same LightGBM model, different trade execution")
    print("=" * 120)

    print("\n  Downloading data (need 500+ days for SMA warmup + training)...")
    cached = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=600)
        if not df.empty:
            cached[sym] = df
            print(f"    {sym}: {len(df)} days")

    for capital in [1000, 10000]:
        print(f"\n  {'=' * 110}")
        print(f"  Capital: ${capital:,}")
        print(f"  {'=' * 110}")
        print(f"  {'Strategy':<25} {'Trades':>7} {'Win%':>6} {'Net P&L':>10} {'Return':>8} "
              f"{'MaxDD':>7} {'Interest':>9} {'Premiums':>9} {'Margin Calls':>12}")
        print(f"  {'-' * 100}")

        # 1. Cash (ML)
        ml = MLSignalGenerator(train_window=200, min_train=80, confidence_threshold=0.65)
        r = ml.train_and_backtest(
            history_data=cached, initial_capital=capital,
            max_position_pct=0.15, stop_loss_pct=0.03, take_profit_pct=0.05,
        )
        daily = r.get("daily_results", [])
        max_v = max(d["total_value"] for d in daily) if daily else capital
        min_v = min(d["total_value"] for d in daily) if daily else capital
        dd = (max_v - min_v) / max_v * 100 if max_v > 0 else 0
        icon = "\U0001f7e2" if r["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {'ML Cash (no leverage)':<25} {r['total_trades']:>7} {r['win_rate']:>5.1f}% "
              f"{icon} ${r['total_pnl_net']:>+8.2f} {r['return_pct']:>+7.2f}% {dd:>6.2f}% "
              f"{'—':>9} {'—':>9} {'—':>12}")

        # 2. Margin 2x
        margin2 = MarginSimulator(leverage=2.0, margin_interest_annual=0.08)
        r2 = margin2.backtest(
            history_data=cached, initial_capital=capital,
            confidence_threshold=0.65, max_position_pct=0.15,
            stop_loss_pct=0.03, take_profit_pct=0.05,
        )
        daily = r2.get("daily_results", [])
        max_v = max(d["total_value"] for d in daily) if daily else capital
        min_v = min(d["total_value"] for d in daily) if daily else capital
        dd = (max_v - min_v) / max_v * 100 if max_v > 0 else 0
        icon = "\U0001f7e2" if r2["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {'ML Margin 2x':<25} {r2['total_trades']:>7} {r2['win_rate']:>5.1f}% "
              f"{icon} ${r2['total_pnl_net']:>+8.2f} {r2['return_pct']:>+7.2f}% {dd:>6.2f}% "
              f"${r2['total_interest']:>7.2f} {'—':>9} {r2['margin_calls']:>12}")

        # 3. Margin 5x
        margin5 = MarginSimulator(leverage=5.0, margin_interest_annual=0.12)
        r5 = margin5.backtest(
            history_data=cached, initial_capital=capital,
            confidence_threshold=0.65, max_position_pct=0.10,
            stop_loss_pct=0.02, take_profit_pct=0.04,
        )
        daily = r5.get("daily_results", [])
        max_v = max(d["total_value"] for d in daily) if daily else capital
        min_v = min(d["total_value"] for d in daily) if daily else capital
        dd = (max_v - min_v) / max_v * 100 if max_v > 0 else 0
        icon = "\U0001f7e2" if r5["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {'ML Margin 5x':<25} {r5['total_trades']:>7} {r5['win_rate']:>5.1f}% "
              f"{icon} ${r5['total_pnl_net']:>+8.2f} {r5['return_pct']:>+7.2f}% {dd:>6.2f}% "
              f"${r5['total_interest']:>7.2f} {'—':>9} {r5['margin_calls']:>12}")

        # 4. Options (5-day expiry)
        opts5 = OptionsSimulator(days_to_expiry=5, option_cost_pct=0.015)
        ro5 = opts5.backtest(
            history_data=cached, initial_capital=capital,
            confidence_threshold=0.65, max_risk_pct=0.05,
        )
        daily = ro5.get("daily_results", [])
        max_v = max(d["total_value"] for d in daily) if daily else capital
        min_v = min(d["total_value"] for d in daily) if daily else capital
        dd = (max_v - min_v) / max_v * 100 if max_v > 0 else 0
        icon = "\U0001f7e2" if ro5["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {'ML Options (5d expiry)':<25} {ro5['total_trades']:>7} {ro5['win_rate']:>5.1f}% "
              f"{icon} ${ro5['total_pnl_net']:>+8.2f} {ro5['return_pct']:>+7.2f}% {dd:>6.2f}% "
              f"{'—':>9} ${ro5.get('total_premiums_paid',0):>7.2f} {'—':>12}")

        # 5. Options (10-day expiry)
        opts10 = OptionsSimulator(days_to_expiry=10, option_cost_pct=0.02)
        ro10 = opts10.backtest(
            history_data=cached, initial_capital=capital,
            confidence_threshold=0.65, max_risk_pct=0.05,
        )
        daily = ro10.get("daily_results", [])
        max_v = max(d["total_value"] for d in daily) if daily else capital
        min_v = min(d["total_value"] for d in daily) if daily else capital
        dd = (max_v - min_v) / max_v * 100 if max_v > 0 else 0
        icon = "\U0001f7e2" if ro10["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {'ML Options (10d expiry)':<25} {ro10['total_trades']:>7} {ro10['win_rate']:>5.1f}% "
              f"{icon} ${ro10['total_pnl_net']:>+8.2f} {ro10['return_pct']:>+7.2f}% {dd:>6.2f}% "
              f"{'—':>9} ${ro10.get('total_premiums_paid',0):>7.2f} {'—':>12}")

        # Show sample trades for best performer
        best = max(
            [("Cash", r), ("Margin 2x", r2), ("Margin 5x", r5),
             ("Options 5d", ro5), ("Options 10d", ro10)],
            key=lambda x: x[1].get("total_pnl_net", -9999)
        )
        print(f"\n  Best: {best[0]} — last 10 closed trades:")
        closed = [t for t in best[1].get("trades", []) if t["action"] in ("sell", "close")]
        for t in closed[-10:]:
            icon = "\U0001f7e2" if t.get("net_pnl", 0) > 0 else "\U0001f534"
            pnl_str = f"${t.get('net_pnl', 0):>+8.2f}" if "net_pnl" in t else "open"
            print(f"    {t['date']:<11} {t['symbol']:<5} {t.get('type', 'stock'):<5} "
                  f"{icon} {pnl_str}  {t.get('reason', '')}")


if __name__ == "__main__":
    main()
