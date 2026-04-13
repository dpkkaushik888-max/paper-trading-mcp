"""CLI for the paper trading engine."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .config import DEFAULT_BROKER, DEFAULT_INITIAL_CAPITAL, DEFAULT_SESSION_ID, WATCHLIST
from .cost_engine import CostEngine
from .portfolio import Portfolio
from .price_engine import get_quote, scan_watchlist
from .risk_manager import RiskManager
from .rule_evaluator import load_rules, scan_signals


def cmd_scan(args):
    """Scan watchlist for signals."""
    print(f"Scanning {len(WATCHLIST)} ETFs...")
    rules = load_rules()

    portfolio = Portfolio(
        session_id=args.session,
        broker=args.broker,
        initial_capital=args.capital,
    )
    open_positions = portfolio.get_open_positions()

    watchlist_data = scan_watchlist()
    signals = scan_signals(watchlist_data, rules, open_positions)

    print(f"\n{'Symbol':<8} {'Price':>10} {'Signal':>8} {'Score':>8}  Reasons")
    print("-" * 80)

    for signal in signals:
        icon = {"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "⚪ HOLD"}.get(
            signal.action, "HOLD"
        )
        print(
            f"{signal.symbol:<8} ${signal.price:>9.2f} {icon:>8} "
            f"{signal.strength:>7.0%}"
        )
        for reason in signal.reasons:
            print(f"         {reason}")
        print()

    buy_signals = [s for s in signals if s.action == "buy"]
    sell_signals = [s for s in signals if s.action == "sell"]
    print(f"Summary: {len(buy_signals)} BUY, {len(sell_signals)} SELL, "
          f"{len(signals) - len(buy_signals) - len(sell_signals)} HOLD")


def cmd_trade(args):
    """Execute a paper trade."""
    portfolio = Portfolio(
        session_id=args.session,
        broker=args.broker,
        initial_capital=args.capital,
    )
    cost_engine = CostEngine(args.broker)
    risk_mgr = RiskManager(portfolio)

    symbol = args.symbol.upper()
    action = args.action.lower()

    if action == "buy":
        quote = get_quote(symbol)
        if "error" in quote:
            print(f"Error: {quote['error']}")
            return

        price = quote["price"]

        if args.shares:
            shares = args.shares
        else:
            suggestion = risk_mgr.suggest_position_size(symbol, price)
            shares = suggestion["suggested_shares"]
            print(f"Auto-sizing: {shares} shares (${shares * price:.2f}, "
                  f"{suggestion['position_pct']:.1f}% of portfolio)")

        costs = cost_engine.calculate_trade_cost(
            symbol, price, shares, "buy", "EUR"
        )

        risk_check = risk_mgr.check_buy(symbol, shares, price, costs.net_value)
        if not risk_check["allowed"]:
            print(f"❌ BLOCKED: {risk_check['reason']}")
            return

        result = portfolio.open_position(
            symbol, shares, price, costs, args.reason or "Manual buy"
        )

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"\n✅ BUY {shares} {symbol} @ ${price:.4f}")
        print(f"   Gross value:      ${costs.gross_value:.2f}")
        print(f"   Spread cost:      -${costs.spread_cost:.4f} ({cost_engine.get_spread_pct(symbol):.2%})")
        print(f"   Slippage cost:    -${costs.slippage_cost:.4f} ({cost_engine.get_slippage_pct():.2%})")
        print(f"   FX cost (EUR→USD):-${costs.fx_cost:.4f} ({cost_engine.get_fx_conversion_pct('EUR'):.2%})")
        print(f"   Total cost:       -${costs.total_cost:.4f}")
        print(f"   Net deducted:     ${costs.net_value:.2f}")
        print(f"   Cash remaining:   ${result['cash_remaining']:.2f}")

    elif action == "sell":
        pos = portfolio.get_position_for_symbol(symbol)
        if not pos:
            print(f"❌ No open position in {symbol}")
            return

        quote = get_quote(symbol)
        if "error" in quote:
            print(f"Error: {quote['error']}")
            return

        price = quote["price"]
        costs = cost_engine.calculate_trade_cost(
            symbol, price, pos["shares"], "sell", "EUR"
        )

        result = portfolio.close_position(
            pos["id"], price, costs, args.reason or "Manual sell"
        )

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        tax = cost_engine.calculate_tax_on_gain(
            max(result["net_pnl"], 0)
        )

        print(f"\n✅ SELL {result['shares']} {symbol} @ ${price:.4f}")
        print(f"   Entry price:      ${result['entry_price']:.4f}")
        print(f"   Gross P&L:        ${result['gross_pnl']:+.4f}")
        print(f"   Entry costs:      -${result['entry_costs']:.4f}")
        print(f"   Exit spread:      -${costs.spread_cost:.4f}")
        print(f"   Exit slippage:    -${costs.slippage_cost:.4f}")
        print(f"   Exit FX:          -${costs.fx_cost:.4f}")
        print(f"   Net P&L:          ${result['net_pnl']:+.4f}")
        print(f"   Est. tax (DE):    -${tax.get('estimated_tax_usd', 0):.2f} ({tax.get('tax_rate_pct', 0):.1f}%)")
        print(f"   After-tax P&L:    ${tax.get('after_tax_gain_usd', result['net_pnl']):+.4f}")
        print(f"   Outcome:          {'🟢 WIN' if result['outcome'] == 'win' else '🔴 LOSS'}")
    else:
        print(f"Unknown action: {action}. Use 'buy' or 'sell'.")


def cmd_portfolio(args):
    """Show portfolio summary."""
    portfolio = Portfolio(
        session_id=args.session,
        broker=args.broker,
        initial_capital=args.capital,
    )

    current_prices = {}
    positions = portfolio.get_open_positions()
    for pos in positions:
        quote = get_quote(pos["symbol"])
        if "error" not in quote:
            current_prices[pos["symbol"]] = quote["price"]

    summary = portfolio.get_portfolio_summary(current_prices)
    s = summary

    print(f"\n{'='*60}")
    print(f"  Paper Portfolio — Session: {s.session_id}")
    print(f"  Broker: {s.broker_profile}")
    print(f"{'='*60}")
    print(f"  Initial Capital:   ${s.initial_capital:>12,.2f}")
    print(f"  Cash:              ${s.cash:>12,.2f}")
    print(f"  Positions Value:   ${s.positions_value:>12,.2f}")
    print(f"  Total Value:       ${s.total_value:>12,.2f}")
    print(f"  Unrealized P&L:    ${s.unrealized_pnl:>+12,.2f}")
    print(f"  Realized P&L:      ${s.realized_pnl:>+12,.2f}")
    print(f"  Total Costs:       ${s.total_costs:>12,.4f}")
    print(f"{'='*60}")

    if s.positions:
        print(f"\n  Open Positions:")
        print(f"  {'Symbol':<8} {'Shares':>8} {'Entry':>10} {'Current':>10} {'P&L':>12}")
        print(f"  {'-'*52}")
        for p in s.positions:
            print(
                f"  {p['symbol']:<8} {p['shares']:>8.1f} "
                f"${p['entry_price']:>9.2f} ${p['current_price']:>9.2f} "
                f"${p['unrealized_pnl']:>+11.2f}"
            )
    else:
        print("\n  No open positions.")


def cmd_report(args):
    """Show daily P&L report."""
    portfolio = Portfolio(
        session_id=args.session,
        broker=args.broker,
        initial_capital=args.capital,
    )
    cost_engine = CostEngine(args.broker)

    today = args.date or date.today().isoformat()
    daily_pnl = portfolio.get_daily_pnl(today)
    trades_today = portfolio.get_daily_trades_count(today)
    costs = portfolio.get_cumulative_costs()

    closed = portfolio.get_closed_positions(limit=100)
    today_closed = [p for p in closed if p.get("exit_time", "").startswith(today)]
    wins = sum(1 for p in today_closed if (p.get("pnl") or 0) > 0)
    losses = sum(1 for p in today_closed if (p.get("pnl") or 0) <= 0)

    tax = cost_engine.calculate_tax_on_gain(max(daily_pnl, 0))

    print(f"\n{'='*60}")
    print(f"  Daily Report — {today}")
    print(f"{'='*60}")
    print(f"  Trades today:      {trades_today}")
    print(f"  Closed positions:  {len(today_closed)}")
    print(f"  Wins / Losses:     {wins}W / {losses}L")
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    print(f"  Win rate:          {win_rate:.0f}%")
    print(f"\n  Daily P&L (net):   ${daily_pnl:>+10.2f}")
    print(f"  Est. tax (DE):     -${tax.get('estimated_tax_usd', 0):>9.2f}")
    print(f"  After-tax P&L:     ${tax.get('after_tax_gain_usd', daily_pnl):>+10.2f}")
    print(f"\n  Cumulative Costs:")
    print(f"    Spread:          ${costs['total_spread_cost']:>10.4f}")
    print(f"    Slippage:        ${costs['total_slippage_cost']:>10.4f}")
    print(f"    FX conversion:   ${costs['total_fx_cost']:>10.4f}")
    print(f"    Total costs:     ${costs['total_all_costs']:>10.4f}")
    print(f"    Cost drag:       {costs['cost_drag_pct']:>9.1f}%")
    print(f"{'='*60}")


def cmd_history(args):
    """Show trade history."""
    portfolio = Portfolio(
        session_id=args.session,
        broker=args.broker,
        initial_capital=args.capital,
    )

    trades = portfolio.get_trades(limit=args.limit, symbol=args.symbol)
    if not trades:
        print("No trades yet.")
        return

    print(f"\n{'Time':<20} {'Symbol':<8} {'Action':<6} {'Shares':>8} "
          f"{'Price':>10} {'Cost':>10}")
    print("-" * 70)

    for t in trades:
        ts = t["timestamp"][:19]
        print(
            f"{ts:<20} {t['symbol']:<8} {t['action']:<6} "
            f"{t['shares']:>8.1f} ${t['price']:>9.2f} ${t['total_cost']:>9.4f}"
        )


def cmd_costs(args):
    """Show cumulative cost summary."""
    portfolio = Portfolio(
        session_id=args.session,
        broker=args.broker,
        initial_capital=args.capital,
    )
    cost_engine = CostEngine(args.broker)
    costs = portfolio.get_cumulative_costs()

    realized_pnl = costs["realized_pnl_net"]
    tax = cost_engine.calculate_tax_on_gain(max(realized_pnl, 0))

    print(f"\n{'='*60}")
    print(f"  Cumulative Cost Summary")
    print(f"{'='*60}")
    print(f"  Total trades:      {costs['trade_count']}")
    print(f"\n  Cost Breakdown:")
    print(f"    Spread:          ${costs['total_spread_cost']:>12.4f}")
    print(f"    Slippage:        ${costs['total_slippage_cost']:>12.4f}")
    print(f"    FX conversion:   ${costs['total_fx_cost']:>12.4f}")
    print(f"    ─────────────────────────────────")
    print(f"    Total costs:     ${costs['total_all_costs']:>12.4f}")
    print(f"\n  P&L:")
    print(f"    Gross P&L:       ${costs['realized_pnl_gross']:>+12.4f}")
    print(f"    - Costs:         -${costs['total_all_costs']:>11.4f}")
    print(f"    Net P&L:         ${costs['realized_pnl_net']:>+12.4f}")
    print(f"    - Est. tax (DE): -${tax.get('estimated_tax_usd', 0):>11.2f}")
    print(f"    After-tax P&L:   ${tax.get('after_tax_gain_usd', realized_pnl):>+12.2f}")
    print(f"\n  Cost drag:         {costs['cost_drag_pct']:>11.1f}% of gross profit")
    print(f"{'='*60}")


def cmd_backtest(args):
    """Run backtest."""
    from .backtest import run_backtest

    print(f"Running backtest ({args.days} days)...")
    result = run_backtest(
        days=args.days,
        initial_capital=args.capital,
        broker=args.broker,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  Backtest Results — {result['strategy']}")
    print(f"  Period: {result['period_days']} days")
    print(f"{'='*60}")
    print(f"  Initial capital:   ${result['initial_capital']:>12,.2f}")
    print(f"  Final value:       ${result['final_value']:>12,.2f}")
    print(f"  Gross P&L:         ${result['total_pnl_gross']:>+12,.2f}")
    print(f"  Net P&L:           ${result['total_pnl_net']:>+12,.2f}")
    print(f"  After-tax P&L:     ${result['total_pnl_after_tax']:>+12,.2f}")
    print(f"  Return:            {result['return_pct']:>+11.2f}%")
    print(f"\n  Trades:            {result['total_trades']}")
    print(f"  Wins / Losses:     {result['wins']}W / {result['losses']}L")
    print(f"  Win rate:          {result['win_rate']:>10.1f}%")
    print(f"  Avg daily P&L:     ${result['avg_daily_pnl']:>+12,.2f}")
    print(f"\n  Total costs:       ${result['total_costs']:>12,.4f}")
    print(f"  Cost drag:         {result['cost_drag_pct']:>11.1f}%")
    print(f"  Open positions:    {result['open_positions']}")
    print(f"{'='*60}")

    if args.verbose:
        print(f"\n  Daily breakdown:")
        print(f"  {'Date':<12} {'Value':>10} {'P&L Net':>10} {'Costs':>10} {'Trades':>7}")
        print(f"  {'-'*52}")
        for day in result["daily_results"]:
            print(
                f"  {day['date']:<12} ${day['total_value']:>9,.2f} "
                f"${day['daily_pnl_net']:>+9.2f} ${day['daily_costs']:>9.4f} "
                f"{day['trades']:>7}"
            )


def cmd_optimize(args):
    """Run strategy autoresearch optimization."""
    from .strategy_optimizer import run_optimization

    run_optimization(
        iterations=args.iterations,
        backtest_days=args.days,
        initial_capital=args.capital,
        broker=args.broker,
        account_currency=args.currency,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Engine — AI Coffee Money",
        prog="python -m trading_engine",
    )
    parser.add_argument("--session", default=DEFAULT_SESSION_ID, help="Session ID")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="Broker profile")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="Initial capital")

    sub = parser.add_subparsers(dest="command", help="Command")

    sub.add_parser("scan", help="Scan watchlist for signals")

    trade_p = sub.add_parser("trade", help="Execute a paper trade")
    trade_p.add_argument("action", choices=["buy", "sell"], help="Buy or sell")
    trade_p.add_argument("symbol", help="ETF symbol")
    trade_p.add_argument("shares", type=float, nargs="?", help="Number of shares (auto if omitted)")
    trade_p.add_argument("--reason", default="", help="Trade reason")

    sub.add_parser("portfolio", help="Show portfolio")

    report_p = sub.add_parser("report", help="Daily P&L report")
    report_p.add_argument("--date", default=None, help="Date (YYYY-MM-DD)")

    history_p = sub.add_parser("history", help="Trade history")
    history_p.add_argument("--limit", type=int, default=50, help="Max trades")
    history_p.add_argument("--symbol", default=None, help="Filter by symbol")

    sub.add_parser("costs", help="Cumulative cost summary")

    bt_p = sub.add_parser("backtest", help="Run backtest")
    bt_p.add_argument("--days", type=int, default=30, help="Backtest period in days")
    bt_p.add_argument("--verbose", "-v", action="store_true", help="Show daily breakdown")

    opt_p = sub.add_parser("optimize", help="Autoresearch: optimize rules.json via backtest")
    opt_p.add_argument("--iterations", type=int, default=20, help="Optimization iterations")
    opt_p.add_argument("--days", type=int, default=90, help="Backtest period")
    opt_p.add_argument("--currency", default="EUR", help="Account currency")

    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "trade": cmd_trade,
        "portfolio": cmd_portfolio,
        "report": cmd_report,
        "history": cmd_history,
        "costs": cmd_costs,
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
