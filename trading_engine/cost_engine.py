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

    def get_spread_pct(self, symbol: str) -> float:
        spread_cfg = self.profile.get("spread", {})
        liquid_symbols = spread_cfg.get("liquid_etf_symbols", [])
        if symbol.upper() in liquid_symbols:
            return spread_cfg.get("liquid_etfs", 0.0003)
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
        """
        gross_value = price * shares

        spread_pct = self.get_spread_pct(symbol)
        spread_cost = gross_value * spread_pct

        slippage_pct = self.get_slippage_pct()
        slippage_cost = gross_value * slippage_pct

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
        """Calculate estimated German capital gains tax on a realized gain.

        Returns dict with gross gain, tax estimate, and after-tax gain.
        Applies Sparerpauschbetrag (€1,000 annual exemption).
        """
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
