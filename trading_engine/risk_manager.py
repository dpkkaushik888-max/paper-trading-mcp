"""Risk management — position sizing, daily loss limits, exposure checks."""

from __future__ import annotations

from .config import MAX_DAILY_LOSS_PCT, MAX_POSITION_PCT
from .portfolio import Portfolio


class RiskManager:
    """Enforces risk rules before trades execute."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.max_position_pct = MAX_POSITION_PCT
        self.max_daily_loss_pct = MAX_DAILY_LOSS_PCT

    def check_buy(
        self,
        symbol: str,
        shares: float,
        price: float,
        total_cost: float,
    ) -> dict:
        """Check if a buy order passes all risk rules.

        Returns {"allowed": True} or {"allowed": False, "reason": "..."}.
        """
        summary = self.portfolio.get_portfolio_summary()
        total_value = summary.total_value

        position_value = price * shares
        position_pct = position_value / total_value if total_value > 0 else 1.0
        if position_pct > self.max_position_pct:
            return {
                "allowed": False,
                "reason": (
                    f"Position too large: ${position_value:.2f} = "
                    f"{position_pct:.1%} of portfolio (max {self.max_position_pct:.0%}). "
                    f"Max allowed: ${total_value * self.max_position_pct:.2f}"
                ),
            }

        cash = self.portfolio.get_cash()
        if total_cost > cash:
            return {
                "allowed": False,
                "reason": (
                    f"Insufficient cash: need ${total_cost:.2f}, have ${cash:.2f}"
                ),
            }

        daily_pnl = self.portfolio.get_daily_pnl()
        daily_loss_pct = abs(daily_pnl) / summary.initial_capital if daily_pnl < 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct:
            return {
                "allowed": False,
                "reason": (
                    f"Daily loss limit reached: ${daily_pnl:.2f} = "
                    f"{daily_loss_pct:.1%} of capital (max {self.max_daily_loss_pct:.0%}). "
                    f"No more trades today."
                ),
            }

        existing = self.portfolio.get_position_for_symbol(symbol)
        if existing:
            return {
                "allowed": False,
                "reason": (
                    f"Already have an open position in {symbol} "
                    f"(id={existing['id']}, {existing['shares']} shares @ ${existing['entry_price']:.2f}). "
                    f"Close it first or sell before buying more."
                ),
            }

        return {"allowed": True}

    def suggest_position_size(
        self,
        symbol: str,
        price: float,
        risk_pct: float = 0.01,
    ) -> dict:
        """Suggest number of shares based on risk rules.

        risk_pct: max % of portfolio to risk on this trade (default 1%).
        """
        summary = self.portfolio.get_portfolio_summary()
        total_value = summary.total_value

        max_by_position = total_value * self.max_position_pct
        max_by_risk = total_value * risk_pct
        max_by_cash = summary.cash * 0.95

        max_value = min(max_by_position, max_by_cash)
        shares = int(max_value / price) if price > 0 else 0

        return {
            "symbol": symbol,
            "price": round(price, 4),
            "suggested_shares": shares,
            "position_value": round(shares * price, 2),
            "position_pct": round(shares * price / total_value * 100, 1) if total_value > 0 else 0,
            "max_by_position_limit": round(max_by_position, 2),
            "max_by_cash": round(max_by_cash, 2),
        }
