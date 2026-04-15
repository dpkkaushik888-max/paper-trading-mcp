"""Broker-agnostic cost calculator.

Loads fee profile from broker_profiles/*.json. Ships with eToro profile.
Every cost that would eat into real profit is calculated here — zero surprises.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import BROKER_PROFILES_DIR, DEFAULT_BROKER
from .models import CostBreakdown


class CostEngine:
    """Calculates all trading costs based on a broker fee profile."""

    def __init__(self, broker: str = DEFAULT_BROKER):
        self.broker = broker
        self.profile = self._load_profile(broker)

    def _load_profile(self, broker: str) -> dict:
        profile_path = BROKER_PROFILES_DIR / f"{broker}.json"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Broker profile not found: {profile_path}. "
                f"Available: {[p.stem for p in BROKER_PROFILES_DIR.glob('*.json')]}"
            )
        with open(profile_path) as f:
            return json.load(f)

    def is_indian_market(self) -> bool:
        return self.profile.get("market") == "india"

    def get_spread_pct(self, symbol: str) -> float:
        spread_cfg = self.profile.get("spread", {})
        liquid_symbols = spread_cfg.get("liquid_etf_symbols", [])
        if symbol.upper() in liquid_symbols or symbol in liquid_symbols:
            return spread_cfg.get("liquid_etfs", 0.0003)
        large_cap_symbols = spread_cfg.get("large_cap_symbols", [])
        if symbol in large_cap_symbols:
            return spread_cfg.get("large_cap_stocks", 0.0002)
        return spread_cfg.get("standard_etfs", 0.0008)

    def get_slippage_pct(self) -> float:
        return self.profile.get("slippage", {}).get("default_pct", 0.0002)

    def get_fx_conversion_pct(self, currency: str = "EUR") -> float:
        fx_cfg = self.profile.get("fx_conversion", {})
        if currency.upper() == "EUR":
            return fx_cfg.get("eur_to_usd_pct", 0.015)
        elif currency.upper() == "GBP":
            return fx_cfg.get("gbp_to_usd_pct", 0.005)
        return 0.0

    def get_capital_gains_tax_pct(self) -> float:
        if self.is_indian_market():
            return self.profile.get("tax", {}).get("stcg_pct", 0.20)
        return self.profile.get("tax", {}).get(
            "german_capital_gains_pct", 0.26375
        )

    def get_sparerpauschbetrag(self) -> float:
        return self.profile.get("tax", {}).get(
            "german_sparerpauschbetrag_eur", 1000.0
        )

    def get_dividend_withholding_pct(self, has_w8ben: bool = False) -> float:
        tax_cfg = self.profile.get("tax", {})
        if has_w8ben:
            return tax_cfg.get("us_dividend_withholding_w8ben_pct", 0.15)
        return tax_cfg.get("us_dividend_withholding_pct", 0.30)

    def get_withdrawal_fee(self, currency: str = "USD") -> float:
        wf = self.profile.get("withdrawal_fee", {})
        key = f"{currency.lower()}_account"
        return wf.get(key, 5.0)

    def get_overnight_fee_pct(self) -> float:
        """Daily overnight fee for CFD positions (shorts on eToro)."""
        return self.profile.get("cfd", {}).get(
            "overnight_fee_daily_pct", 0.0002
        )

    def calculate_overnight_cost(
        self, price: float, shares: float, holding_days: int
    ) -> float:
        """Calculate cumulative overnight fees for a CFD/short position."""
        gross_value = price * shares
        daily_fee = gross_value * self.get_overnight_fee_pct()
        return daily_fee * holding_days

    def _calculate_indian_statutory(self, gross_value: float, action: str) -> float:
        """Calculate Indian statutory charges: STT, stamp duty, transaction charges, SEBI, GST."""
        sc = self.profile.get("statutory_charges", {})
        cost = 0.0

        if action == "buy":
            cost += gross_value * sc.get("stt_delivery_buy_pct", 0.001)
            cost += gross_value * sc.get("stamp_duty_buy_pct", 0.00015)
        else:
            cost += gross_value * sc.get("stt_delivery_sell_pct", 0.001)

        txn_charge = gross_value * sc.get("transaction_charge_nse_pct", 0.0000297)
        cost += txn_charge

        sebi_charge = gross_value * sc.get("sebi_turnover_pct", 0.000001)
        cost += sebi_charge

        brokerage = min(
            gross_value * self.profile.get("equity_delivery_brokerage", 0.0),
            self.profile.get("equity_intraday_brokerage_max_per_order", 20.0),
        )
        gst = (brokerage + txn_charge) * sc.get("gst_pct", 0.18)
        cost += gst

        return cost

    def calculate_trade_cost(
        self,
        symbol: str,
        price: float,
        shares: float,
        action: str = "buy",
        account_currency: str = "EUR",
    ) -> CostBreakdown:
        """Calculate all costs for a single trade.

        Returns a CostBreakdown with every fee itemized.
        Supports both international (eToro) and Indian (Zerodha) brokers.
        """
        gross_value = price * shares

        spread_pct = self.get_spread_pct(symbol)
        spread_cost = gross_value * spread_pct

        slippage_pct = self.get_slippage_pct()
        slippage_cost = gross_value * slippage_pct

        if self.is_indian_market():
            fx_cost = 0.0
            statutory_cost = self._calculate_indian_statutory(gross_value, action)
            total_cost = spread_cost + slippage_cost + statutory_cost
        else:
            fx_pct = self.get_fx_conversion_pct(account_currency)
            fx_cost = gross_value * fx_pct
            total_cost = spread_cost + slippage_cost + fx_cost

        if action == "buy":
            net_value = gross_value + total_cost
        else:
            net_value = gross_value - total_cost

        return CostBreakdown(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            fx_cost=fx_cost,
            total_cost=total_cost,
            gross_value=gross_value,
            net_value=net_value,
        )

    def calculate_tax_on_gain(
        self,
        realized_gain_usd: float,
        cumulative_gains_eur: float = 0.0,
        eur_usd_rate: float = 1.08,
    ) -> dict:
        """Calculate estimated capital gains tax on a realized gain.

        Supports both German and Indian tax regimes.
        """
        if self.is_indian_market():
            return self._calculate_indian_tax(realized_gain_usd)

        if realized_gain_usd <= 0:
            return {
                "gross_gain_usd": round(realized_gain_usd, 4),
                "taxable_gain_eur": 0.0,
                "estimated_tax_eur": 0.0,
                "estimated_tax_usd": 0.0,
                "after_tax_gain_usd": round(realized_gain_usd, 4),
                "note": "No tax on losses",
            }

        gain_eur = realized_gain_usd / eur_usd_rate
        exemption = self.get_sparerpauschbetrag()
        remaining_exemption = max(0, exemption - cumulative_gains_eur)
        taxable_eur = max(0, gain_eur - remaining_exemption)

        tax_rate = self.get_capital_gains_tax_pct()
        tax_eur = taxable_eur * tax_rate
        tax_usd = tax_eur * eur_usd_rate

        return {
            "gross_gain_usd": round(realized_gain_usd, 4),
            "gain_eur": round(gain_eur, 2),
            "exemption_used_eur": round(min(gain_eur, remaining_exemption), 2),
            "taxable_gain_eur": round(taxable_eur, 2),
            "tax_rate_pct": round(tax_rate * 100, 3),
            "estimated_tax_eur": round(tax_eur, 2),
            "estimated_tax_usd": round(tax_usd, 2),
            "after_tax_gain_usd": round(realized_gain_usd - tax_usd, 4),
        }

    def _calculate_indian_tax(self, realized_gain_inr: float) -> dict:
        """Calculate Indian STCG tax (short-term, held < 1 year)."""
        tax_cfg = self.profile.get("tax", {})
        if realized_gain_inr <= 0:
            return {
                "gross_gain_inr": round(realized_gain_inr, 2),
                "taxable_gain_inr": 0.0,
                "estimated_tax_inr": 0.0,
                "after_tax_gain_inr": round(realized_gain_inr, 2),
                "note": "Losses can be offset against other STCG/LTCG",
            }

        stcg_rate = tax_cfg.get("stcg_pct", 0.20)
        tax_inr = realized_gain_inr * stcg_rate

        return {
            "gross_gain_inr": round(realized_gain_inr, 2),
            "taxable_gain_inr": round(realized_gain_inr, 2),
            "tax_rate_pct": round(stcg_rate * 100, 1),
            "estimated_tax_inr": round(tax_inr, 2),
            "after_tax_gain_inr": round(realized_gain_inr - tax_inr, 2),
            "note": "STCG @ 20% (Budget 2024). LTCG @ 12.5% above ₹1.25L exemption for holdings > 1 year.",
        }

    def get_cost_summary(self) -> dict:
        """Return the broker profile summary for display."""
        return {
            "broker": self.profile.get("broker_name", self.broker),
            "etf_commission": self.profile.get("etf_commission", 0),
            "spread_liquid": self.profile.get("spread", {}).get("liquid_etfs", 0),
            "spread_standard": self.profile.get("spread", {}).get("standard_etfs", 0),
            "slippage": self.get_slippage_pct(),
            "fx_eur_usd": self.get_fx_conversion_pct("EUR"),
            "capital_gains_tax": self.get_capital_gains_tax_pct(),
            "withdrawal_fee_usd": self.get_withdrawal_fee("USD"),
        }
