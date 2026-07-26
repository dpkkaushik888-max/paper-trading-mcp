"""Benchmark: Run all strategy variants × capital levels × currencies sequentially.

Outputs a comparison table showing how each lever affects profitability.
"""

import json
from trading_engine.backtest import run_backtest
from trading_engine.config import PROJECT_ROOT, WATCHLIST
from trading_engine.price_engine import get_history

RULES_CONSERVATIVE = {
    "strategy": {"name": "Conservative v1.0", "version": "1.0", "timeframe": "1d"},
    "watchlist": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "GLD", "TLT"],
    "entry_rules": {
        "long": [
            {"condition": "rsi_14 < 40", "weight": 0.30, "description": "RSI(14) oversold"},
            {"condition": "price > ema_20", "weight": 0.15, "description": "EMA(20) uptrend"},
            {"condition": "macd_histogram > 0", "weight": 0.20, "description": "MACD positive"},
            {"condition": "price > bb_lower", "weight": 0.20, "description": "Above lower BB"},
        ],
        "min_score": 0.6,
    },
    "exit_rules": [
        {"condition": "rsi_14 > 70", "description": "RSI overbought"},
        {"condition": "price < ema_200", "description": "Below EMA(200)"},
        {"condition": "macd_histogram < 0", "description": "MACD negative"},
    ],
    "risk_rules": {
        "max_position_pct": 0.10, "max_daily_loss_pct": 0.02,
        "stop_loss_pct": 0.02, "take_profit_pct": 0.05,
    },
}

RULES_SCALPING_OPTIMIZED = {
    "strategy": {"name": "Scalping v2.0 (optimized)", "version": "2.0", "timeframe": "1d"},
    "watchlist": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "GLD", "TLT"],
    "entry_rules": {
        "long": [
            {"condition": "rsi_3 < 15", "weight": 0.35, "description": "RSI(3) deep oversold"},
            {"condition": "price > ema_20", "weight": 0.20, "description": "EMA(20) uptrend"},
            {"condition": "macd_histogram > 0", "weight": 0.20, "description": "MACD positive"},
            {"condition": "price > bb_lower", "weight": 0.25, "description": "Above lower BB"},
        ],
        "min_score": 0.45,
    },
    "exit_rules": [
        {"condition": "rsi_3 > 80", "description": "RSI(3) overbought"},
        {"condition": "price < ema_200", "description": "Below EMA(200)"},
        {"condition": "macd_histogram < 0", "description": "MACD negative"},
    ],
    "risk_rules": {
        "max_position_pct": 0.10, "max_daily_loss_pct": 0.02,
        "stop_loss_pct": 0.03, "take_profit_pct": 0.02,
    },
}

RULES_SCALPING_LOOSE = {
    "strategy": {"name": "Scalping v2.1 (loose)", "version": "2.1", "timeframe": "1d"},
    "watchlist": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "GLD", "TLT"],
    "entry_rules": {
        "long": [
            {"condition": "rsi_3 < 30", "weight": 0.30, "description": "RSI(3) oversold"},
            {"condition": "price > ema_8", "weight": 0.20, "description": "EMA(8) micro-trend"},
            {"condition": "macd_histogram > 0", "weight": 0.15, "description": "MACD positive"},
            {"condition": "price > bb_lower", "weight": 0.25, "description": "Above lower BB"},
        ],
        "min_score": 0.35,
    },
    "exit_rules": [
        {"condition": "rsi_3 > 70", "description": "RSI(3) overbought"},
        {"condition": "price < ema_20", "description": "Below EMA(20)"},
        {"condition": "macd_histogram < 0", "description": "MACD negative"},
    ],
    "risk_rules": {
        "max_position_pct": 0.10, "max_daily_loss_pct": 0.02,
        "stop_loss_pct": 0.02, "take_profit_pct": 0.015,
    },
}

RULES_HYBRID = {
    "strategy": {"name": "Hybrid v3.0 (balanced)", "version": "3.0", "timeframe": "1d"},
    "watchlist": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "GLD", "TLT"],
    "entry_rules": {
        "long": [
            {"condition": "rsi_3 < 20", "weight": 0.30, "description": "RSI(3) oversold"},
            {"condition": "price > ema_50", "weight": 0.25, "description": "EMA(50) trend filter"},
            {"condition": "macd_histogram > 0", "weight": 0.20, "description": "MACD positive"},
            {"condition": "price > bb_lower", "weight": 0.25, "description": "Above lower BB"},
        ],
        "min_score": 0.50,
    },
    "exit_rules": [
        {"condition": "rsi_3 > 80", "description": "RSI(3) overbought"},
        {"condition": "price < ema_200", "description": "Below EMA(200)"},
        {"condition": "macd_histogram < 0", "description": "MACD negative"},
    ],
    "risk_rules": {
        "max_position_pct": 0.10, "max_daily_loss_pct": 0.02,
        "stop_loss_pct": 0.025, "take_profit_pct": 0.03,
    },
}

RULES_LONG_SHORT = {
    "strategy": {"name": "Long/Short v3.0", "version": "3.0", "timeframe": "1d"},
    "watchlist": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "GLD", "TLT"],
    "entry_rules": {
        "long": [
            {"condition": "rsi_3 < 15", "weight": 0.35, "description": "RSI(3) oversold"},
            {"condition": "price > ema_20", "weight": 0.20, "description": "EMA(20) uptrend"},
            {"condition": "macd_histogram > 0", "weight": 0.20, "description": "MACD positive"},
            {"condition": "price > bb_lower", "weight": 0.25, "description": "Above lower BB"},
        ],
        "short": [
            {"condition": "rsi_3 > 85", "weight": 0.35, "description": "RSI(3) overbought"},
            {"condition": "price < ema_20", "weight": 0.20, "description": "Below EMA(20)"},
            {"condition": "macd_histogram < 0", "weight": 0.20, "description": "MACD negative"},
            {"condition": "price < bb_upper", "weight": 0.25, "description": "Below upper BB"},
        ],
        "min_score": 0.45, "short_min_score": 0.45,
    },
    "exit_rules": [
        {"condition": "rsi_3 > 80", "description": "RSI(3) overbought"},
        {"condition": "price < ema_200", "description": "Below EMA(200)"},
        {"condition": "macd_histogram < 0", "description": "MACD negative"},
    ],
    "short_exit_rules": [
        {"condition": "rsi_3 < 20", "description": "RSI(3) oversold — cover"},
        {"condition": "price > ema_200", "description": "Above EMA(200) — cover"},
        {"condition": "macd_histogram > 0", "description": "MACD positive — cover"},
    ],
    "risk_rules": {
        "max_position_pct": 0.10, "max_daily_loss_pct": 0.02,
        "stop_loss_pct": 0.03, "take_profit_pct": 0.02,
        "short_stop_loss_pct": 0.03, "short_take_profit_pct": 0.02,
    },
}

STRATEGIES = [
    ("Conservative v1.0", RULES_CONSERVATIVE),
    ("Scalping v2.0 (opt)", RULES_SCALPING_OPTIMIZED),
    ("Scalping v2.1 (loose)", RULES_SCALPING_LOOSE),
    ("Hybrid v3.0", RULES_HYBRID),
    ("Long/Short v3.0", RULES_LONG_SHORT),
]

CAPITAL_LEVELS = [1000, 2500, 5000, 10000]
CURRENCIES = ["EUR", "USD"]
BACKTEST_DAYS = 90


def main():
    print("=" * 100)
    print("  PAPER TRADING BENCHMARK — Strategy × Capital × Currency")
    print(f"  Backtest period: {BACKTEST_DAYS} days | Broker: eToro")
    print("=" * 100)

    # Download data once
    print("\n📊 Downloading historical data (one-time)...")
    cached = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=BACKTEST_DAYS + 60)
        if not df.empty:
            cached[sym] = df
            print(f"  {sym}: {len(df)} days")

    results = []

    for strat_name, rules in STRATEGIES:
        for capital in CAPITAL_LEVELS:
            for currency in CURRENCIES:
                # Write temp rules
                tmp = PROJECT_ROOT / "optimization_results" / "bench_rules.json"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w") as f:
                    json.dump(rules, f)

                r = run_backtest(
                    days=BACKTEST_DAYS,
                    rules_path=str(tmp),
                    initial_capital=capital,
                    broker="etoro",
                    account_currency=currency,
                    cached_history=cached,
                )

                row = {
                    "strategy": strat_name,
                    "capital": capital,
                    "currency": currency,
                    "trades": r.get("total_trades", 0),
                    "wins": r.get("wins", 0),
                    "losses": r.get("losses", 0),
                    "win_rate": r.get("win_rate", 0),
                    "net_pnl": r.get("total_pnl_net", 0),
                    "return_pct": r.get("return_pct", 0),
                    "cost_drag": r.get("cost_drag_pct", 0),
                    "final_value": r.get("final_value", 0),
                }
                results.append(row)

    # Print table
    print(f"\n{'='*100}")
    print(f"  {'Strategy':<22} {'Cap':>7} {'Cur':>4} {'Trades':>7} {'W/L':>7} {'Win%':>6} {'Net P&L':>10} {'Return':>8} {'Cost%':>8}")
    print(f"  {'-'*95}")

    prev_strat = None
    for r in results:
        if r["strategy"] != prev_strat:
            if prev_strat is not None:
                print(f"  {'-'*95}")
            prev_strat = r["strategy"]

        pnl_icon = "🟢" if r["net_pnl"] > 0 else "🔴" if r["net_pnl"] < 0 else "⚪"
        print(
            f"  {r['strategy']:<22} ${r['capital']:>5,} {r['currency']:>4} "
            f"{r['trades']:>7} {r['wins']}W/{r['losses']}L "
            f"{r['win_rate']:>5.0f}% "
            f"{pnl_icon} ${r['net_pnl']:>+8.2f} "
            f"{r['return_pct']:>+7.2f}% "
            f"{r['cost_drag']:>7.1f}%"
        )

    print(f"  {'='*95}")

    # Summary: best combos
    profitable = [r for r in results if r["net_pnl"] > 0]
    profitable.sort(key=lambda x: x["net_pnl"], reverse=True)

    print(f"\n  TOP 5 PROFITABLE COMBINATIONS:")
    print(f"  {'Strategy':<22} {'Cap':>7} {'Cur':>4} {'Trades':>7} {'Win%':>6} {'Net P&L':>10} {'Return':>8}")
    print(f"  {'-'*70}")
    for r in profitable[:5]:
        print(
            f"  {r['strategy']:<22} ${r['capital']:>5,} {r['currency']:>4} "
            f"{r['trades']:>7} {r['win_rate']:>5.0f}% "
            f"${r['net_pnl']:>+8.2f} {r['return_pct']:>+7.2f}%"
        )

    if not profitable:
        print("  ❌ No profitable combination found in this 90-day window.")

    # Save full results
    out_path = PROJECT_ROOT / "optimization_results" / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
