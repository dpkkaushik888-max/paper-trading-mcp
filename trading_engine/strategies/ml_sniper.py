"""Strategy 1: ML Sniper — High-confidence LightGBM predictions.

Trades rarely (1% of days) but with high conviction (>85% confidence).
This is the existing S10 strategy extracted into the strategy framework.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from trading_engine.ml_model import build_features_for_market
from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal


def _add_relative_strength(feat: pd.DataFrame, ca_feat: pd.DataFrame,
                           prefix: str) -> pd.DataFrame:
    """Add relative strength features (stock return minus index return)."""
    for period in [5, 10, 20]:
        stock_col = f"return_{period}d"
        idx_col = f"return_{period}d"
        if stock_col in feat.columns and idx_col in ca_feat.columns:
            aligned = ca_feat[idx_col].reindex(feat.index)
            feat[f"rs_{prefix}_{period}d"] = feat[stock_col] - aligned
    if "volatility_5d" in feat.columns and "volatility_5d" in ca_feat.columns:
        aligned = ca_feat["volatility_5d"].reindex(feat.index)
        feat[f"rel_vol_{prefix}"] = feat["volatility_5d"] / aligned.replace(0, np.nan)
    return feat


class MLSniperStrategy(BaseStrategy):
    """High-confidence ML model predictions."""

    def __init__(
        self,
        config: StrategyConfig,
        market: str = "crypto",
        confidence_threshold: float = 0.85,
        sl_pct: float = 0.10,
        tp_pct: float = 0.15,
    ):
        super().__init__(config)
        self.market = market
        self.confidence_threshold = confidence_threshold
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct

    def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_day: pd.Timestamp,
        model=None,
        feature_cols: list | None = None,
        cross_asset_data: dict | None = None,
    ) -> Optional[StrategySignal]:
        if model is None or feature_cols is None:
            return None

        if not self.can_open():
            return None

        temporal_df = df[df.index <= current_day]
        if len(temporal_df) < 30:
            return None

        feat = build_features_for_market(temporal_df, self.market)

        if cross_asset_data:
            ca_symbol = cross_asset_data.get("symbol")
            ca_features = cross_asset_data.get("features", [])
            ca_prefix = cross_asset_data.get("prefix", "btc")
            ca_df = cross_asset_data.get("df")
            if ca_df is not None and ca_symbol and symbol != ca_symbol:
                ca_temporal = ca_df[ca_df.index <= current_day]
                if len(ca_temporal) > 30:
                    ca_feat = build_features_for_market(ca_temporal, self.market)
                    avail_ca = [c for c in ca_features if c in ca_feat.columns]
                    cross = ca_feat[avail_ca].copy()
                    cross.columns = [f"{ca_prefix}_{c}" for c in cross.columns]
                    feat = feat.join(cross, how="left")
                    feat = _add_relative_strength(feat, ca_feat, ca_prefix)

        if current_day not in feat.index:
            return None

        row = feat.loc[current_day]
        row_feats = row.reindex(feature_cols, fill_value=0).fillna(0)
        if row_feats.isna().any():
            return None

        X_pred = row_feats.values.reshape(1, -1)
        proba = model.predict_proba(X_pred)[0]
        up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        down_prob = 1.0 - up_prob

        if up_prob > self.confidence_threshold:
            return StrategySignal(
                direction="long",
                confidence=up_prob,
                strategy=self.name,
                reason=f"ML sniper long ({up_prob:.0%} up)",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
                size_pct=self.config.max_position_pct,
            )
        elif down_prob > self.confidence_threshold:
            return StrategySignal(
                direction="short",
                confidence=down_prob,
                strategy=self.name,
                reason=f"ML sniper short ({down_prob:.0%} down)",
                sl_pct=self.sl_pct,
                tp_pct=self.tp_pct,
                size_pct=self.config.max_position_pct,
            )

        return None
