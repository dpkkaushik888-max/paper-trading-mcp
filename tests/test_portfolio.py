"""Tests for the SQLite portfolio — positions, trades, P&L."""

import os
import tempfile

import pytest

from trading_engine.models import CostBreakdown
from trading_engine.portfolio import Portfolio


@pytest.fixture
def portfolio(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Portfolio(
        db_path=db_path,
        session_id="test",
        broker="etoro",
        initial_capital=1000.0,
    )


class TestPortfolioInit:
    def test_initial_cash(self, portfolio):
        assert portfolio.get_cash() == 1000.0

    def test_no_open_positions(self, portfolio):
        assert portfolio.get_open_positions() == []

    def test_no_trades(self, portfolio):
        assert portfolio.get_trades() == []


class TestOpenPosition:
    def test_buy_deducts_cash(self, portfolio):
        costs = CostBreakdown(
            spread_cost=0.15, slippage_cost=0.10, fx_cost=0.75,
            total_cost=1.0, gross_value=100.0, net_value=101.0,
        )
        result = portfolio.open_position("XLF", 2, 50.0, costs, "test buy")
        assert result["status"] == "opened"
        assert portfolio.get_cash() == pytest.approx(899.0, abs=0.01)

    def test_position_created(self, portfolio):
        costs = CostBreakdown(
            spread_cost=0.15, slippage_cost=0.10, fx_cost=0.75,
            total_cost=1.0, gross_value=100.0, net_value=101.0,
        )
        portfolio.open_position("XLF", 2, 50.0, costs, "test buy")
        positions = portfolio.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "XLF"
        assert positions[0]["shares"] == 2.0

    def test_trade_recorded(self, portfolio):
        costs = CostBreakdown(
            spread_cost=0.15, slippage_cost=0.10, fx_cost=0.75,
            total_cost=1.0, gross_value=100.0, net_value=101.0,
        )
        portfolio.open_position("XLF", 2, 50.0, costs, "test buy")
        trades = portfolio.get_trades()
        assert len(trades) == 1
        assert trades[0]["action"] == "buy"
        assert trades[0]["spread_cost"] == pytest.approx(0.15)

    def test_reject_zero_shares(self, portfolio):
        costs = CostBreakdown(gross_value=0, net_value=0)
        result = portfolio.open_position("SPY", 0, 500.0, costs)
        assert "error" in result

    def test_reject_insufficient_cash(self, portfolio):
        costs = CostBreakdown(
            total_cost=50.0, gross_value=2000.0, net_value=2050.0,
        )
        result = portfolio.open_position("SPY", 4, 500.0, costs)
        assert "error" in result
        assert "Insufficient cash" in result["error"]


class TestClosePosition:
    def test_sell_returns_cash(self, portfolio):
        buy_costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.75,
            total_cost=0.80, gross_value=50.0, net_value=50.80,
        )
        portfolio.open_position("XLF", 1, 50.0, buy_costs, "buy")

        sell_costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.78,
            total_cost=0.83, gross_value=52.0, net_value=51.17,
        )
        pos = portfolio.get_open_positions()[0]
        result = portfolio.close_position(pos["id"], 52.0, sell_costs, "sell")

        assert result["status"] == "closed"
        assert result["gross_pnl"] == pytest.approx(2.0, abs=0.01)
        assert result["net_pnl"] == pytest.approx(2.0 - 0.80 - 0.83, abs=0.01)

    def test_position_closed(self, portfolio):
        buy_costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.75,
            total_cost=0.80, gross_value=50.0, net_value=50.80,
        )
        portfolio.open_position("XLF", 1, 50.0, buy_costs)

        sell_costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.75,
            total_cost=0.80, gross_value=50.0, net_value=49.20,
        )
        pos = portfolio.get_open_positions()[0]
        portfolio.close_position(pos["id"], 50.0, sell_costs)

        assert portfolio.get_open_positions() == []
        assert len(portfolio.get_closed_positions()) == 1

    def test_reject_invalid_position(self, portfolio):
        sell_costs = CostBreakdown(gross_value=50.0, net_value=49.0)
        result = portfolio.close_position(999, 50.0, sell_costs)
        assert "error" in result


class TestCumulativeCosts:
    def test_costs_accumulate(self, portfolio):
        for i in range(3):
            costs = CostBreakdown(
                spread_cost=0.10, slippage_cost=0.05, fx_cost=0.50,
                total_cost=0.65, gross_value=50.0, net_value=50.65,
            )
            portfolio.open_position(f"ETF{i}", 1, 50.0, costs, f"buy {i}")

        cumulative = portfolio.get_cumulative_costs()
        assert cumulative["total_spread_cost"] == pytest.approx(0.30, abs=0.01)
        assert cumulative["total_slippage_cost"] == pytest.approx(0.15, abs=0.01)
        assert cumulative["total_fx_cost"] == pytest.approx(1.50, abs=0.01)
        assert cumulative["trade_count"] == 3


class TestPortfolioSummary:
    def test_summary_with_positions(self, portfolio):
        costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.75,
            total_cost=0.80, gross_value=50.0, net_value=50.80,
        )
        portfolio.open_position("XLF", 1, 50.0, costs)

        summary = portfolio.get_portfolio_summary({"XLF": 55.0})
        assert summary.cash == pytest.approx(949.20, abs=0.01)
        assert summary.positions_value == pytest.approx(55.0, abs=0.01)
        assert summary.unrealized_pnl == pytest.approx(5.0, abs=0.01)
        assert summary.total_value == pytest.approx(1004.20, abs=0.01)


class TestSessionIsolation:
    def test_different_sessions_independent(self, tmp_path):
        db_path = str(tmp_path / "shared.db")
        p1 = Portfolio(db_path=db_path, session_id="strat_a", initial_capital=1000.0)
        p2 = Portfolio(db_path=db_path, session_id="strat_b", initial_capital=5000.0)

        assert p1.get_cash() == 1000.0
        assert p2.get_cash() == 5000.0

        costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.75,
            total_cost=0.80, gross_value=50.0, net_value=50.80,
        )
        p1.open_position("XLF", 1, 50.0, costs)

        assert len(p1.get_open_positions()) == 1
        assert len(p2.get_open_positions()) == 0
