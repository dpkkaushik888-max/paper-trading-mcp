"""Tests for the risk manager — position limits, daily loss, exposure."""

import pytest

from trading_engine.models import CostBreakdown
from trading_engine.portfolio import Portfolio
from trading_engine.risk_manager import RiskManager


@pytest.fixture
def portfolio(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Portfolio(
        db_path=db_path,
        session_id="test",
        broker="etoro",
        initial_capital=1000.0,
    )


@pytest.fixture
def risk_mgr(portfolio):
    return RiskManager(portfolio)


class TestPositionLimit:
    def test_within_limit(self, risk_mgr):
        result = risk_mgr.check_buy("XLF", 1, 50.0, 50.80)
        assert result["allowed"] is True

    def test_exceeds_10_pct(self, risk_mgr):
        result = risk_mgr.check_buy("SPY", 1, 500.0, 507.5)
        assert result["allowed"] is False
        assert "too large" in result["reason"].lower()

    def test_exceeds_limit_by_small_amount(self, risk_mgr):
        result = risk_mgr.check_buy("XLF", 2, 51.0, 103.60)
        assert result["allowed"] is False


class TestInsufficientCash:
    def test_not_enough_cash(self, risk_mgr):
        result = risk_mgr.check_buy("ETF", 1, 50.0, 1500.0)
        assert result["allowed"] is False
        assert "Insufficient cash" in result["reason"]


class TestDuplicatePosition:
    def test_reject_duplicate(self, portfolio, risk_mgr):
        costs = CostBreakdown(
            spread_cost=0.04, slippage_cost=0.01, fx_cost=0.75,
            total_cost=0.80, gross_value=50.0, net_value=50.80,
        )
        portfolio.open_position("XLF", 1, 50.0, costs)

        result = risk_mgr.check_buy("XLF", 1, 51.0, 51.80)
        assert result["allowed"] is False
        assert "Already have" in result["reason"]


class TestSuggestPositionSize:
    def test_suggestion_within_limits(self, risk_mgr):
        suggestion = risk_mgr.suggest_position_size("XLF", 50.0)
        assert suggestion["suggested_shares"] >= 1
        assert suggestion["position_pct"] <= 10.0

    def test_expensive_etf_zero_shares(self, risk_mgr):
        suggestion = risk_mgr.suggest_position_size("SPY", 600.0)
        assert suggestion["suggested_shares"] == 0
