"""Quick test: verify long+short backtest works."""
from trading_engine.backtest import run_backtest

r = run_backtest(days=90, account_currency="USD")
print(f"Trades: {r['total_trades']}")
print(f"Wins: {r['wins']}, Losses: {r['losses']}")
print(f"Win rate: {r['win_rate']}%")
print(f"Net PnL: ${r['total_pnl_net']:+.2f}")
print(f"Return: {r['return_pct']:+.2f}%")
print(f"Open positions: {r['open_positions']}")
for d in r["daily_results"]:
    if d["trades"] > 0:
        print(f"  {d['date']}: {d['trades']} trades, pnl={d['daily_pnl_net']:+.2f}")
