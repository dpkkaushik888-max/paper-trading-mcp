"""Backtest engine — run strategy against historical data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .config import WATCHLIST
from .cost_engine import CostEngine
from .price_engine import get_history, calculate_indicators
from .rule_evaluator import evaluate_entry_rules, evaluate_exit_rules, load_rules


def run_backtest(
    days: int = 30,
    rules_path: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    initial_capital: float = 1000.0,
    broker: str = "etoro",
    account_currency: str = "EUR",
) -> dict:
    """Run a backtest of the strategy against historical data.

    Uses daily OHLC data. Simulates one pass per day:
    1. Check exit conditions for open positions
    2. Check entry conditions for new positions
    """
    symbols = symbols or WATCHLIST
    rules = load_rules(rules_path)
    cost_engine = CostEngine(broker)

    history_data = {}
    for symbol in symbols:
        df = get_history(symbol, days=days + 60)
        if not df.empty:
            history_data[symbol] = df

    if not history_data:
        return {"error": "No historical data available"}

    all_dates = set()
    for df in history_data.values():
        all_dates.update(df.index.date)
    sorted_dates = sorted(all_dates)

    if len(sorted_dates) <= 60:
        return {"error": "Insufficient historical data for backtest"}
    sorted_dates = sorted_dates[-days:]

    cash = initial_capital
    positions = {}
    trades = []
    daily_results = []
    total_costs = 0.0

    for day in sorted_dates:
        day_str = str(day)
        day_trades = []
        day_pnl_gross = 0.0
        day_pnl_net = 0.0
        day_costs = 0.0

        for symbol, df in history_data.items():
            day_data = df[df.index.date <= day]
            if len(day_data) < 50:
                continue

            indicators = calculate_indicators(day_data)
            if "error" in indicators:
                continue
            indicators["symbol"] = symbol
            price = indicators.get("price", 0)
            if price <= 0:
                continue

            if symbol in positions:
                pos = positions[symbol]
                signal = evaluate_exit_rules(rules, indicators, pos)
                if signal.action == "sell":
                    costs = cost_engine.calculate_trade_cost(
                        symbol, price, pos["shares"], "sell", account_currency
                    )
                    gross_pnl = (price - pos["entry_price"]) * pos["shares"]
                    net_pnl = gross_pnl - pos["entry_costs"] - costs.total_cost

                    day_pnl_gross += gross_pnl
                    day_pnl_net += net_pnl
                    day_costs += costs.total_cost

                    day_trades.append({
                        "symbol": symbol,
                        "action": "sell",
                        "price": round(price, 4),
                        "shares": pos["shares"],
                        "gross_pnl": round(gross_pnl, 4),
                        "net_pnl": round(net_pnl, 4),
                        "costs": costs.to_dict(),
                        "reason": "; ".join(signal.reasons),
                    })

                    cash += costs.net_value
                    del positions[symbol]

            else:
                signal = evaluate_entry_rules(rules, indicators)
                if signal.action == "buy":
                    max_position = cash * 0.10
                    shares = int(max_position / price) if price > 0 else 0
                    if shares <= 0:
                        continue

                    costs = cost_engine.calculate_trade_cost(
                        symbol, price, shares, "buy", account_currency
                    )

                    if costs.net_value > cash:
                        continue

                    positions[symbol] = {
                        "symbol": symbol,
                        "shares": shares,
                        "entry_price": price,
                        "entry_costs": costs.total_cost,
                        "entry_date": day_str,
                    }

                    day_costs += costs.total_cost
                    cash -= costs.net_value

                    day_trades.append({
                        "symbol": symbol,
                        "action": "buy",
                        "price": round(price, 4),
                        "shares": shares,
                        "costs": costs.to_dict(),
                        "reason": "; ".join(signal.reasons),
                    })

        total_costs += day_costs
        trades.extend(day_trades)

        positions_value = sum(
            _get_price_on_date(history_data.get(s), day) * p["shares"]
            for s, p in positions.items()
        )

        daily_results.append({
            "date": day_str,
            "cash": round(cash, 2),
            "positions_value": round(positions_value, 2),
            "total_value": round(cash + positions_value, 2),
            "daily_pnl_gross": round(day_pnl_gross, 2),
            "daily_pnl_net": round(day_pnl_net, 2),
            "daily_costs": round(day_costs, 4),
            "trades": len(day_trades),
        })

    final_value = cash
    for symbol, pos in positions.items():
        last_price = _get_last_price(history_data.get(symbol))
        if last_price:
            final_value += last_price * pos["shares"]

    total_pnl_gross = final_value - initial_capital + total_costs
    total_pnl_net = final_value - initial_capital

    wins = sum(1 for t in trades if t.get("net_pnl", 0) > 0)
    losses = sum(1 for t in trades if t.get("net_pnl", 0) < 0)
    sell_trades = [t for t in trades if t["action"] == "sell"]

    tax_info = cost_engine.calculate_tax_on_gain(max(total_pnl_net, 0))

    return {
        "strategy": rules.get("strategy", {}).get("name", "Unknown"),
        "period_days": days,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_pnl_gross": round(total_pnl_gross, 2),
        "total_pnl_net": round(total_pnl_net, 2),
        "total_pnl_after_tax": round(tax_info.get("after_tax_gain_usd", total_pnl_net), 2),
        "total_costs": round(total_costs, 4),
        "cost_drag_pct": round(
            (total_costs / total_pnl_gross * 100) if total_pnl_gross > 0 else 0, 2
        ),
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(sell_trades) * 100, 1) if sell_trades else 0,
        "avg_daily_pnl": round(total_pnl_net / days, 2) if days > 0 else 0,
        "return_pct": round(total_pnl_net / initial_capital * 100, 2),
        "tax_estimate": tax_info,
        "open_positions": len(positions),
        "daily_results": daily_results,
    }


def _get_price_on_date(df: Optional[pd.DataFrame], day) -> float:
    if df is None or df.empty:
        return 0.0
    day_data = df[df.index.date <= day]
    if day_data.empty:
        return 0.0
    return float(day_data["Close"].iloc[-1])


def _get_last_price(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])
