"""Tests for the cost engine — eToro fee calculations."""

import pytest

from trading_engine.cost_engine import CostEngine


@pytest.fixture
def engine():
    return CostEngine("etoro")


class TestSpread:
    def test_liquid_etf_spread(self, engine):
        pct = engine.get_spread_pct("SPY")
        assert pct == 0.0003

    def test_liquid_etf_qqq(self, engine):
        pct = engine.get_spread_pct("QQQ")
        assert pct == 0.0003

    def test_standard_etf_spread(self, engine):
        pct = engine.get_spread_pct("XLF")
        assert pct == 0.0008

    def test_standard_etf_gld(self, engine):
        pct = engine.get_spread_pct("GLD")
        assert pct == 0.0008

    def test_unknown_symbol_uses_standard(self, engine):
        pct = engine.get_spread_pct("ZZZZ")
        assert pct == 0.0008


class TestSlippage:
    def test_default_slippage(self, engine):
        assert engine.get_slippage_pct() == 0.0002


class TestFXConversion:
    def test_eur_to_usd(self, engine):
        assert engine.get_fx_conversion_pct("EUR") == 0.015

    def test_gbp_to_usd(self, engine):
        assert engine.get_fx_conversion_pct("GBP") == 0.005

    def test_usd_no_conversion(self, engine):
        assert engine.get_fx_conversion_pct("USD") == 0.0


class TestTradeCostCalculation:
    def test_buy_cost_breakdown(self, engine):
        costs = engine.calculate_trade_cost("SPY", 500.0, 10, "buy", "EUR")
        assert costs.gross_value == 5000.0
        assert costs.spread_cost == pytest.approx(1.5, abs=0.01)    # 5000 * 0.0003
        assert costs.slippage_cost == pytest.approx(1.0, abs=0.01)  # 5000 * 0.0002
        assert costs.fx_cost == pytest.approx(75.0, abs=0.01)       # 5000 * 0.015
        assert costs.total_cost == pytest.approx(77.5, abs=0.01)
        assert costs.net_value == pytest.approx(5077.5, abs=0.01)   # buy: gross + costs

    def test_sell_cost_breakdown(self, engine):
        costs = engine.calculate_trade_cost("SPY", 500.0, 10, "sell", "EUR")
        assert costs.gross_value == 5000.0
        assert costs.net_value == pytest.approx(4922.5, abs=0.01)   # sell: gross - costs

    def test_usd_account_no_fx(self, engine):
        costs = engine.calculate_trade_cost("SPY", 500.0, 10, "buy", "USD")
        assert costs.fx_cost == 0.0
        assert costs.total_cost == pytest.approx(2.5, abs=0.01)     # spread + slippage only

    def test_standard_etf_higher_spread(self, engine):
        costs = engine.calculate_trade_cost("XLF", 50.0, 1, "buy", "USD")
        assert costs.spread_cost == pytest.approx(0.04, abs=0.01)   # 50 * 0.0008


class TestTaxCalculation:
    def test_no_tax_on_loss(self, engine):
        tax = engine.calculate_tax_on_gain(-100.0)
        assert tax["estimated_tax_usd"] == 0.0
        assert tax["note"] == "No tax on losses"

    def test_tax_within_exemption(self, engine):
        tax = engine.calculate_tax_on_gain(100.0, cumulative_gains_eur=0.0, eur_usd_rate=1.0)
        assert tax["taxable_gain_eur"] == 0.0
        assert tax["estimated_tax_eur"] == 0.0

    def test_tax_above_exemption(self, engine):
        tax = engine.calculate_tax_on_gain(2000.0, cumulative_gains_eur=1000.0, eur_usd_rate=1.0)
        assert tax["taxable_gain_eur"] == 2000.0
        assert tax["tax_rate_pct"] == pytest.approx(26.375, abs=0.01)
        assert tax["estimated_tax_eur"] == pytest.approx(527.5, abs=1.0)

    def test_tax_partial_exemption(self, engine):
        tax = engine.calculate_tax_on_gain(1500.0, cumulative_gains_eur=500.0, eur_usd_rate=1.0)
        assert tax["exemption_used_eur"] == 500.0
        assert tax["taxable_gain_eur"] == 1000.0


class TestWithdrawalFee:
    def test_usd_account(self, engine):
        assert engine.get_withdrawal_fee("USD") == 5.0

    def test_eur_account(self, engine):
        assert engine.get_withdrawal_fee("EUR") == 0.0


class TestCostSummary:
    def test_summary_structure(self, engine):
        summary = engine.get_cost_summary()
        assert summary["broker"] == "eToro"
        assert summary["etf_commission"] == 0
        assert "spread_liquid" in summary
        assert "fx_eur_usd" in summary
