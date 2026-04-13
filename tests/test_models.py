"""Tests for data models — serialization, defaults."""

from trading_engine.models import CostBreakdown, Signal, Trade, Position, PortfolioSummary


class TestCostBreakdown:
    def test_to_dict_rounds(self):
        cb = CostBreakdown(
            spread_cost=0.123456, slippage_cost=0.054321,
            fx_cost=0.987654, total_cost=1.165431,
            gross_value=100.0, net_value=101.165431,
        )
        d = cb.to_dict()
        assert d["spread_cost"] == 0.1235
        assert d["slippage_cost"] == 0.0543
        assert d["total_cost"] == 1.1654

    def test_defaults(self):
        cb = CostBreakdown()
        assert cb.total_cost == 0.0
        assert cb.gross_value == 0.0


class TestSignal:
    def test_to_dict(self):
        s = Signal(symbol="SPY", action="buy", strength=0.75, price=500.0)
        d = s.to_dict()
        assert d["symbol"] == "SPY"
        assert d["action"] == "buy"
        assert d["strength"] == 0.75

    def test_default_hold(self):
        s = Signal()
        assert s.action == "hold"
        assert s.strength == 0.0


class TestPosition:
    def test_open_position_dict(self):
        p = Position(symbol="XLF", shares=2, entry_price=50.0, status="open")
        d = p.to_dict()
        assert d["symbol"] == "XLF"
        assert d["exit_price"] is None
        assert d["pnl"] is None


class TestPortfolioSummary:
    def test_to_dict(self):
        ps = PortfolioSummary(
            initial_capital=1000.0, cash=950.0,
            positions_value=55.0, total_value=1005.0,
            unrealized_pnl=5.0,
        )
        d = ps.to_dict()
        assert d["total_value"] == 1005.0
        assert d["unrealized_pnl"] == 5.0
