"""Detailed crypto backtest analysis — trade frequency, W/L breakdown, capital comparison."""

import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from trading_engine.autoresearch import CACHE_DIR
from trading_engine.config import CRYPTO_WATCHLIST
from trading_engine.time_machine import TimeMachineBacktest

PROJECT_ROOT = Path(__file__).parent


def load_cached_data(period="2y"):
    cache_file = CACHE_DIR / f"crypto_{period}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No cached data at {cache_file}. Run autoresearch first.")


def run_backtest(capital: float, data: dict) -> dict:
    tm = TimeMachineBacktest(
        market="crypto",
        initial_capital=capital,
        confidence_threshold=0.85,
        max_position_pct=0.10,
        stop_loss_pct=0.10,
        take_profit_pct=0.15,
        session_id=f"analysis_{capital}",
        enable_learning=True,
    )
    return tm.run(history_data=data)


def analyze(result: dict, capital: float):
    trades = result["trades"]
    daily = result["daily_results"]
    closed = [t for t in trades if t["action"] in ("sell", "cover")]
    opens = [t for t in trades if t["action"] in ("buy", "short")]

    wins = [t for t in closed if t.get("net_pnl", 0) > 0]
    losses = [t for t in closed if t.get("net_pnl", 0) <= 0]

    long_opens = [t for t in opens if t.get("side") == "long"]
    short_opens = [t for t in opens if t.get("side") == "short"]
    long_closes = [t for t in closed if t.get("side") == "long"]
    short_closes = [t for t in closed if t.get("side") == "short"]

    print(f"\n{'='*80}")
    print(f"  CRYPTO BACKTEST ANALYSIS — Capital: ${capital:,.0f}")
    print(f"{'='*80}")

    # ── 1. Overall Performance ──
    print(f"\n  1. OVERALL PERFORMANCE")
    print(f"  {'─'*40}")
    print(f"  Initial capital:    ${capital:,.2f}")
    print(f"  Final value:        ${result['final_value']:,.2f}")
    print(f"  Net P&L:            ${result['total_pnl_net']:,.2f}")
    print(f"  Return:             {result['return_pct']:+.2f}%")
    print(f"  Total costs:        ${result['total_costs']:,.2f}")

    # ── 2. Trade Breakdown ──
    print(f"\n  2. TRADE BREAKDOWN")
    print(f"  {'─'*40}")
    print(f"  Total trades:       {len(trades)}")
    print(f"    Entry trades:     {len(opens)}")
    print(f"    Exit trades:      {len(closed)}")
    print(f"    Still open:       {result['open_positions']}")
    print(f"  ")
    print(f"  Long trades:        {len(long_opens)} entries, {len(long_closes)} exits")
    print(f"  Short trades:       {len(short_opens)} entries, {len(short_closes)} exits")
    long_pct = len(long_opens) / len(opens) * 100 if opens else 0
    short_pct = len(short_opens) / len(opens) * 100 if opens else 0
    print(f"  Long/Short split:   {long_pct:.0f}% / {short_pct:.0f}%")

    # ── 3. Win/Loss Analysis ──
    print(f"\n  3. WIN / LOSS ANALYSIS")
    print(f"  {'─'*40}")
    print(f"  Closed trades:      {len(closed)}")
    print(f"  Wins:               {len(wins)} ({len(wins)/len(closed)*100:.1f}%)" if closed else "  Wins: 0")
    print(f"  Losses:             {len(losses)} ({len(losses)/len(closed)*100:.1f}%)" if closed else "  Losses: 0")

    if wins:
        avg_win = sum(t.get("net_pnl", 0) for t in wins) / len(wins)
        max_win = max(t.get("net_pnl", 0) for t in wins)
        print(f"  Avg win P&L:        ${avg_win:,.2f}")
        print(f"  Max single win:     ${max_win:,.2f}")
    if losses:
        avg_loss = sum(t.get("net_pnl", 0) for t in losses) / len(losses)
        max_loss = min(t.get("net_pnl", 0) for t in losses)
        print(f"  Avg loss P&L:       ${avg_loss:,.2f}")
        print(f"  Max single loss:    ${max_loss:,.2f}")

    if wins and losses:
        profit_factor = abs(sum(t.get("net_pnl", 0) for t in wins)) / abs(sum(t.get("net_pnl", 0) for t in losses)) if losses else float('inf')
        print(f"  Profit factor:      {profit_factor:.2f}x")

    # ── 4. Long vs Short W/L ──
    long_wins = [t for t in long_closes if t.get("net_pnl", 0) > 0]
    long_losses = [t for t in long_closes if t.get("net_pnl", 0) <= 0]
    short_wins = [t for t in short_closes if t.get("net_pnl", 0) > 0]
    short_losses = [t for t in short_closes if t.get("net_pnl", 0) <= 0]

    print(f"\n  4. LONG vs SHORT PERFORMANCE")
    print(f"  {'─'*40}")
    if long_closes:
        long_wr = len(long_wins) / len(long_closes) * 100
        long_pnl = sum(t.get("net_pnl", 0) for t in long_closes)
        print(f"  Long:  {len(long_wins)}W / {len(long_losses)}L ({long_wr:.1f}% WR) — P&L: ${long_pnl:,.2f}")
    else:
        print(f"  Long:  no closed trades")
    if short_closes:
        short_wr = len(short_wins) / len(short_closes) * 100
        short_pnl = sum(t.get("net_pnl", 0) for t in short_closes)
        print(f"  Short: {len(short_wins)}W / {len(short_losses)}L ({short_wr:.1f}% WR) — P&L: ${short_pnl:,.2f}")
    else:
        print(f"  Short: no closed trades")

    # ── 5. Trading Frequency ──
    print(f"\n  5. TRADING FREQUENCY")
    print(f"  {'─'*40}")

    trade_dates = [t["date"] for t in opens]
    if trade_dates:
        trades_per_day = Counter(trade_dates)
        total_days = len(daily)
        active_days = len(trades_per_day)
        idle_days = total_days - active_days

        print(f"  Backtest duration:  {total_days} trading days")
        print(f"  Active days:        {active_days} ({active_days/total_days*100:.1f}%)")
        print(f"  Idle days:          {idle_days} ({idle_days/total_days*100:.1f}%)")
        print(f"  Avg trades/day:     {len(opens)/total_days:.2f} (all days)")
        print(f"  Avg trades/active:  {len(opens)/active_days:.2f} (active days only)")
        print(f"  Max trades in 1 day: {max(trades_per_day.values())}")

        freq_dist = Counter(trades_per_day.values())
        print(f"\n  Distribution of trades per active day:")
        for n_trades in sorted(freq_dist.keys()):
            bar = "█" * freq_dist[n_trades]
            print(f"    {n_trades} trade(s): {freq_dist[n_trades]:3d} days  {bar}")

        # Trading gaps
        sorted_dates = sorted(trades_per_day.keys())
        if len(sorted_dates) > 1:
            gaps = []
            for i in range(1, len(sorted_dates)):
                d1 = pd.Timestamp(sorted_dates[i-1])
                d2 = pd.Timestamp(sorted_dates[i])
                gap = (d2 - d1).days
                gaps.append(gap)
            print(f"\n  Gaps between active days:")
            print(f"    Avg gap:          {sum(gaps)/len(gaps):.1f} days")
            print(f"    Max gap:          {max(gaps)} days")
            print(f"    Min gap:          {min(gaps)} days")

    # ── 6. Monthly Breakdown ──
    print(f"\n  6. MONTHLY BREAKDOWN")
    print(f"  {'─'*40}")
    monthly_trades = defaultdict(lambda: {"entries": 0, "wins": 0, "losses": 0, "pnl": 0.0})
    for t in opens:
        month = t["date"][:7]
        monthly_trades[month]["entries"] += 1
    for t in closed:
        month = t["date"][:7]
        pnl = t.get("net_pnl", 0)
        if pnl > 0:
            monthly_trades[month]["wins"] += 1
        else:
            monthly_trades[month]["losses"] += 1
        monthly_trades[month]["pnl"] += pnl

    print(f"  {'Month':<10} {'Entries':>8} {'Wins':>6} {'Losses':>8} {'WR':>8} {'P&L':>12}")
    print(f"  {'─'*54}")
    for month in sorted(monthly_trades.keys()):
        m = monthly_trades[month]
        total_closed = m["wins"] + m["losses"]
        wr = f"{m['wins']/total_closed*100:.0f}%" if total_closed > 0 else "—"
        print(f"  {month:<10} {m['entries']:>8} {m['wins']:>6} {m['losses']:>8} {wr:>8} ${m['pnl']:>10,.2f}")

    # ── 7. Margin Usage ──
    print(f"\n  7. MARGIN USAGE")
    print(f"  {'─'*40}")
    print(f"  Max position size:  10% of capital (${capital*0.10:,.2f} per trade)")
    if short_opens:
        print(f"  Short selling:      YES ({len(short_opens)} short entries)")
        print(f"  Margin for shorts:  Simulated 100% margin (borrow full shares)")
    else:
        print(f"  Short selling:      NO (long only)")
    print(f"  Leverage:           NONE (1x, no margin leverage)")
    print(f"  Max concurrent:     Limited by capital / position size")

    max_concurrent = max(
        (r.get("long_count", 0) + r.get("short_count", 0)) for r in daily
    ) if daily else 0
    print(f"  Max simultaneous:   {max_concurrent} positions")

    # ── 8. Per-Asset Breakdown ──
    print(f"\n  8. PER-ASSET BREAKDOWN")
    print(f"  {'─'*40}")
    asset_stats = defaultdict(lambda: {"entries": 0, "wins": 0, "losses": 0, "pnl": 0.0})
    for t in opens:
        asset_stats[t["symbol"]]["entries"] += 1
    for t in closed:
        sym = t["symbol"]
        pnl = t.get("net_pnl", 0)
        if pnl > 0:
            asset_stats[sym]["wins"] += 1
        else:
            asset_stats[sym]["losses"] += 1
        asset_stats[sym]["pnl"] += pnl

    print(f"  {'Asset':<12} {'Entries':>8} {'Wins':>6} {'Losses':>8} {'WR':>8} {'P&L':>12}")
    print(f"  {'─'*56}")
    for sym in sorted(asset_stats.keys(), key=lambda x: asset_stats[x]["pnl"], reverse=True):
        a = asset_stats[sym]
        total_closed = a["wins"] + a["losses"]
        wr = f"{a['wins']/total_closed*100:.0f}%" if total_closed > 0 else "—"
        print(f"  {sym:<12} {a['entries']:>8} {a['wins']:>6} {a['losses']:>8} {wr:>8} ${a['pnl']:>10,.2f}")

    # ── 9. Calibration ──
    cal = result.get("calibration", {})
    print(f"\n  9. MODEL CALIBRATION")
    print(f"  {'─'*40}")
    print(f"  Calibration error:  {cal.get('calibration_error', 'N/A')}")
    print(f"  Trades analyzed:    {cal.get('total_trades', 'N/A')}")
    print(f"  Win rate (cal):     {cal.get('win_rate', 'N/A')}")

    print(f"\n{'='*80}")


def main():
    print("Loading cached crypto data...")
    data = load_cached_data("2y")
    print(f"Loaded {len(data)} assets")

    for capital in [1_000, 10_000]:
        print(f"\n\nRunning backtest with ${capital:,.0f} capital...")
        result = run_backtest(capital, data)
        analyze(result, capital)


if __name__ == "__main__":
    main()
