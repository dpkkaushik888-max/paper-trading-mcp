"""MCP server exposing paper trading tools to Cascade/Claude."""

from __future__ import annotations

import json
from datetime import date
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import DEFAULT_BROKER, DEFAULT_INITIAL_CAPITAL, DEFAULT_SESSION_ID, WATCHLIST
from .cost_engine import CostEngine
from .portfolio import Portfolio
from .price_engine import get_quote, scan_watchlist
from .risk_manager import RiskManager
from .rule_evaluator import load_rules, scan_signals

mcp = FastMCP(
    "Paper Trading Engine",
    instructions="AI-driven paper trading for ETFs with full eToro-realistic cost simulation. Target: daily coffee money.",
)


def _get_portfolio(session_id: str = DEFAULT_SESSION_ID, broker: str = DEFAULT_BROKER) -> Portfolio:
    return Portfolio(session_id=session_id, broker=broker, initial_capital=DEFAULT_INITIAL_CAPITAL)


def _get_cost_engine(broker: str = DEFAULT_BROKER) -> CostEngine:
    return CostEngine(broker)


@mcp.tool()
def scan_signals_tool(
    symbols: Optional[list[str]] = None,
    session_id: str = DEFAULT_SESSION_ID,
    broker: str = DEFAULT_BROKER,
) -> str:
    """Scan watchlist ETFs for buy/sell signals based on technical rules.

    Evaluates RSI, EMA, MACD, Bollinger Bands against rules.json.
    Returns signals per symbol with indicator values and reasoning.
    """
    portfolio = _get_portfolio(session_id, broker)
    open_positions = portfolio.get_open_positions()
    rules = load_rules()

    watchlist_data = scan_watchlist(symbols or WATCHLIST)
    signals = scan_signals(watchlist_data, rules, open_positions)

    result = []
    for signal in signals:
        result.append(signal.to_dict())

    return json.dumps({"signals": result, "count": len(result)}, indent=2)


@mcp.tool()
def place_trade(
    symbol: str,
    action: str,
    shares: Optional[float] = None,
    reason: str = "",
    session_id: str = DEFAULT_SESSION_ID,
    broker: str = DEFAULT_BROKER,
) -> str:
    """Execute a paper trade (buy or sell).

    For buys: auto-sizes position if shares not specified (10% of portfolio max).
    For sells: closes the open position in the given symbol.
    Shows full cost breakdown: spread, slippage, FX, tax estimate, net P&L.
    """
    portfolio = _get_portfolio(session_id, broker)
    cost_engine = _get_cost_engine(broker)
    risk_mgr = RiskManager(portfolio)

    symbol = symbol.upper()
    action = action.lower()

    if action == "buy":
        quote = get_quote(symbol)
        if "error" in quote:
            return json.dumps({"error": quote["error"]})

        price = quote["price"]

        if not shares:
            suggestion = risk_mgr.suggest_position_size(symbol, price)
            shares = suggestion["suggested_shares"]

        costs = cost_engine.calculate_trade_cost(symbol, price, shares, "buy", "EUR")

        risk_check = risk_mgr.check_buy(symbol, shares, price, costs.net_value)
        if not risk_check["allowed"]:
            return json.dumps({"error": risk_check["reason"]})

        result = portfolio.open_position(symbol, shares, price, costs, reason or "AI signal")
        return json.dumps(result, indent=2)

    elif action == "sell":
        pos = portfolio.get_position_for_symbol(symbol)
        if not pos:
            return json.dumps({"error": f"No open position in {symbol}"})

        quote = get_quote(symbol)
        if "error" in quote:
            return json.dumps({"error": quote["error"]})

        price = quote["price"]
        costs = cost_engine.calculate_trade_cost(symbol, price, pos["shares"], "sell", "EUR")
        result = portfolio.close_position(pos["id"], price, costs, reason or "AI signal")

        if "error" not in result:
            tax = cost_engine.calculate_tax_on_gain(max(result.get("net_pnl", 0), 0))
            result["tax_estimate"] = tax

        return json.dumps(result, indent=2)

    return json.dumps({"error": f"Unknown action: {action}. Use 'buy' or 'sell'."})


@mcp.tool()
def get_portfolio(
    session_id: str = DEFAULT_SESSION_ID,
    broker: str = DEFAULT_BROKER,
) -> str:
    """Get current portfolio: cash, positions, unrealized P&L, total value.

    Fetches live prices for open positions to calculate unrealized P&L.
    """
    portfolio = _get_portfolio(session_id, broker)

    current_prices = {}
    positions = portfolio.get_open_positions()
    for pos in positions:
        quote = get_quote(pos["symbol"])
        if "error" not in quote:
            current_prices[pos["symbol"]] = quote["price"]

    summary = portfolio.get_portfolio_summary(current_prices)
    return json.dumps(summary.to_dict(), indent=2)


@mcp.tool()
def daily_report(
    report_date: Optional[str] = None,
    session_id: str = DEFAULT_SESSION_ID,
    broker: str = DEFAULT_BROKER,
) -> str:
    """Daily P&L report: today's profit, win rate, cumulative costs, after-tax P&L.

    Shows gross P&L vs net P&L (after all costs) — the difference is "hidden cost drag".
    """
    portfolio = _get_portfolio(session_id, broker)
    cost_engine = _get_cost_engine(broker)

    today = report_date or date.today().isoformat()
    daily_pnl = portfolio.get_daily_pnl(today)
    trades_today = portfolio.get_daily_trades_count(today)
    costs = portfolio.get_cumulative_costs()

    closed = portfolio.get_closed_positions(limit=100)
    today_closed = [p for p in closed if (p.get("exit_time") or "").startswith(today)]
    wins = sum(1 for p in today_closed if (p.get("pnl") or 0) > 0)
    losses = sum(1 for p in today_closed if (p.get("pnl") or 0) <= 0)

    tax = cost_engine.calculate_tax_on_gain(max(daily_pnl, 0))

    return json.dumps({
        "date": today,
        "trades_today": trades_today,
        "closed_positions": len(today_closed),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
        "daily_pnl_net": round(daily_pnl, 4),
        "tax_estimate": tax,
        "cumulative_costs": costs,
    }, indent=2)


@mcp.tool()
def trade_history(
    limit: int = 50,
    symbol: Optional[str] = None,
    session_id: str = DEFAULT_SESSION_ID,
    broker: str = DEFAULT_BROKER,
) -> str:
    """List all executed trades with entry/exit prices, costs, and P&L."""
    portfolio = _get_portfolio(session_id, broker)
    trades = portfolio.get_trades(limit=limit, symbol=symbol)
    return json.dumps({"trades": trades, "count": len(trades)}, indent=2)


@mcp.tool()
def get_quote_tool(symbol: str) -> str:
    """Get real-time price + all calculated technical indicators for a symbol.

    Returns: price, RSI(14), RSI(3), EMA(8/20/50/200), MACD, Bollinger Bands, ATR.
    """
    from .price_engine import calculate_indicators, get_history

    quote = get_quote(symbol.upper())
    if "error" in quote:
        return json.dumps(quote)

    df = get_history(symbol.upper())
    indicators = calculate_indicators(df)
    quote["indicators"] = indicators
    return json.dumps(quote, indent=2)


@mcp.tool()
def cost_summary(
    session_id: str = DEFAULT_SESSION_ID,
    broker: str = DEFAULT_BROKER,
) -> str:
    """Cumulative cost summary: total spread, slippage, FX, tax, cost drag %.

    Shows how much of your gross profit is eaten by trading costs.
    """
    portfolio = _get_portfolio(session_id, broker)
    cost_engine = _get_cost_engine(broker)
    costs = portfolio.get_cumulative_costs()

    realized_pnl = costs["realized_pnl_net"]
    tax = cost_engine.calculate_tax_on_gain(max(realized_pnl, 0))

    return json.dumps({
        "costs": costs,
        "tax_estimate": tax,
        "broker_profile": cost_engine.get_cost_summary(),
    }, indent=2)


@mcp.tool()
def backtest_tool(
    days: int = 30,
    broker: str = DEFAULT_BROKER,
) -> str:
    """Run strategy against historical data and report simulated P&L.

    Includes full cost simulation and tax estimate. Uses daily OHLC data.
    """
    from .backtest import run_backtest

    result = run_backtest(days=days, broker=broker)
    daily = result.pop("daily_results", [])
    result["daily_results_summary"] = {
        "total_days": len(daily),
        "profitable_days": sum(1 for d in daily if d["daily_pnl_net"] > 0),
        "losing_days": sum(1 for d in daily if d["daily_pnl_net"] < 0),
        "flat_days": sum(1 for d in daily if d["daily_pnl_net"] == 0),
        "best_day": max((d["daily_pnl_net"] for d in daily), default=0),
        "worst_day": min((d["daily_pnl_net"] for d in daily), default=0),
    }
    return json.dumps(result, indent=2)


def main():
    """Start the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
