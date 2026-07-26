"""ML Signal Generator — backward-compatibility shim.

All model code has been refactored into trading_engine.models/:
  - models.base_model: SmartLGBM, Ensemble, shared utilities
  - models.mean_rev_model: Mean-reversion features + configs (RSI, IBS, BB)
  - models.trend_model: Trend-following features + configs (EMA, ADX, momentum)

This file re-exports everything so existing imports continue to work:
    from trading_engine.ml_model import MLSignalGenerator
    from trading_engine.ml_model import build_feature_matrix, MARKET_CONFIGS
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .models.base_model import (
    _SmartLGBM,
    _SmartLGBMEnsemble,
    _streak,
    add_earnings_features,
    add_sector_relative_features,
    add_vix_features,
    SIMPLIFIED_FEATURES,
)
from .models.mean_rev_model import (
    build_feature_matrix,
    build_feature_matrix_india,
    build_features_for_market,
    MEAN_REV_CONFIGS,
)
from .models.trend_model import (
    build_trend_features,
    TREND_CONFIGS,
    TrendModel,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Backward compatibility — old code uses MARKET_CONFIGS
MARKET_CONFIGS = MEAN_REV_CONFIGS


class MLSignalGenerator:
    """Walk-forward LightGBM signal generator — market-aware.

    Backward-compatible wrapper: delegates to MeanRevModel internally.
    Supports 'us', 'crypto', and 'india' markets with dedicated feature sets,
    hyperparameters, and cross-asset references.
    """

    def __init__(
        self,
        market: str = "us",
        train_window: int | None = None,
        min_train: int | None = None,
        confidence_threshold: float | None = None,
    ):
        self.market = market
        cfg = MARKET_CONFIGS.get(market, MARKET_CONFIGS["us"])
        self.train_window = train_window or cfg["train_window"]
        self.min_train = min_train or cfg["min_train"]
        self.confidence_threshold = confidence_threshold or cfg["default_confidence"]
        self.retrain_every = cfg["retrain_every"]
        self.lgbm_params = cfg["lgbm_params"]
        self.cross_asset_symbol = cfg["cross_asset_symbol"]
        self.cross_asset_features = cfg["cross_asset_features"]
        self.cross_asset_prefix = cfg["cross_asset_prefix"]
        self.model = None
        self.feature_cols = None
        self.last_train_date = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        exclude = {"target", "target_dir"}
        return [c for c in df.columns if c not in exclude]

    def train_and_backtest(
        self,
        history_data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05,
    ) -> dict:
        """Walk-forward backtest: retrain model every 20 days, predict next day."""
        all_features = {}
        cross_features = None
        ca_sym = self.cross_asset_symbol
        if ca_sym in history_data:
            ca_feat = build_features_for_market(history_data[ca_sym], self.market)
            avail = [c for c in self.cross_asset_features if c in ca_feat.columns]
            cross_features = ca_feat[avail].copy()
            cross_features.columns = [f"{self.cross_asset_prefix}_{c}" for c in cross_features.columns]

        for symbol, df in history_data.items():
            feat = build_features_for_market(df, self.market)
            if cross_features is not None and symbol != ca_sym:
                feat = feat.join(cross_features, how="left")
            feat = feat.dropna()
            if len(feat) > self.min_train:
                all_features[symbol] = feat

        if not all_features:
            return {"error": "Insufficient data to build features"}

        all_dates = sorted(set().union(*(f.index for f in all_features.values())))
        if len(all_dates) < self.min_train + 20:
            return {"error": "Not enough dates for walk-forward"}

        test_start_idx = self.min_train
        retrain_every = self.retrain_every

        cash = initial_capital
        long_positions = {}
        short_positions = {}
        trades = []
        daily_results = []
        total_costs = 0.0

        model = None
        feature_cols = None

        for day_idx in range(test_start_idx, len(all_dates) - 1):
            day = all_dates[day_idx]
            day_str = str(day)[:10]
            day_trades = []
            day_pnl = 0.0

            if model is None or (day_idx - test_start_idx) % retrain_every == 0:
                train_X_list = []
                train_y_list = []

                for symbol, feat in all_features.items():
                    train_slice = feat[feat.index < day].tail(self.train_window)
                    if len(train_slice) < self.min_train:
                        continue

                    if feature_cols is None:
                        feature_cols = self._get_feature_cols(train_slice)

                    avail = [c for c in feature_cols if c in train_slice.columns]
                    valid = train_slice.dropna(subset=avail + ["target_dir"])
                    if len(valid) < 30:
                        continue

                    train_X_list.append(valid[avail].reindex(columns=feature_cols, fill_value=0))
                    train_y_list.append(valid["target_dir"])

                if train_X_list:
                    X_train = pd.concat(train_X_list)
                    y_train = pd.concat(train_y_list)

                    model = _SmartLGBM(params=self.lgbm_params)
                    model.fit(X_train, y_train)

            if model is None or feature_cols is None:
                continue

            for symbol, feat in all_features.items():
                if day not in feat.index:
                    continue

                row = feat.loc[day]
                row_feats = row.reindex(feature_cols, fill_value=0)
                if row_feats.isna().any():
                    continue

                X_pred = row_feats.values.reshape(1, -1)
                proba = model.predict_proba(X_pred)[0]
                up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                down_prob = 1.0 - up_prob

                price = float(history_data[symbol].loc[day, "Close"]) if day in history_data[symbol].index else 0
                if price <= 0:
                    continue

                # --- Exit long position ---
                if symbol in long_positions:
                    pos = long_positions[symbol]
                    pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]

                    should_exit = False
                    reason = ""

                    if pnl_pct <= -stop_loss_pct:
                        should_exit = True
                        reason = f"Stop loss {pnl_pct:.2%}"
                    elif pnl_pct >= take_profit_pct:
                        should_exit = True
                        reason = f"Take profit {pnl_pct:.2%}"
                    elif down_prob > self.confidence_threshold:
                        should_exit = True
                        reason = f"ML bearish ({down_prob:.0%} down)"

                    if should_exit:
                        gross_pnl = (price - pos["entry_price"]) * pos["shares"]
                        cost = price * pos["shares"] * 0.001
                        net_pnl = gross_pnl - pos["entry_cost"] - cost
                        cash += price * pos["shares"] - cost
                        day_pnl += net_pnl
                        total_costs += cost

                        day_trades.append({
                            "date": day_str, "symbol": symbol, "action": "sell",
                            "side": "long", "price": round(price, 2),
                            "shares": pos["shares"],
                            "gross_pnl": round(gross_pnl, 2),
                            "net_pnl": round(net_pnl, 2),
                            "confidence": round(up_prob, 3), "reason": reason,
                        })
                        del long_positions[symbol]

                # --- Exit short position (cover) ---
                elif symbol in short_positions:
                    pos = short_positions[symbol]
                    pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]

                    should_exit = False
                    reason = ""

                    if pnl_pct <= -stop_loss_pct:
                        should_exit = True
                        reason = f"Short stop loss {-pnl_pct:.2%}"
                    elif pnl_pct >= take_profit_pct:
                        should_exit = True
                        reason = f"Short take profit {pnl_pct:.2%}"
                    elif up_prob > self.confidence_threshold:
                        should_exit = True
                        reason = f"ML bullish ({up_prob:.0%} up)"

                    if should_exit:
                        gross_pnl = (pos["entry_price"] - price) * pos["shares"]
                        cost = price * pos["shares"] * 0.001
                        net_pnl = gross_pnl - pos["entry_cost"] - cost
                        cash += pos["entry_price"] * pos["shares"] + gross_pnl - cost
                        day_pnl += net_pnl
                        total_costs += cost

                        day_trades.append({
                            "date": day_str, "symbol": symbol, "action": "cover",
                            "side": "short", "price": round(price, 2),
                            "shares": pos["shares"],
                            "gross_pnl": round(gross_pnl, 2),
                            "net_pnl": round(net_pnl, 2),
                            "confidence": round(down_prob, 3), "reason": reason,
                        })
                        del short_positions[symbol]

                # --- Open new position ---
                else:
                    total_open = len(long_positions) + len(short_positions)

                    if up_prob > self.confidence_threshold and total_open < 8:
                        max_pos_value = cash * max_position_pct
                        shares = int(max_pos_value / price)
                        if shares <= 0:
                            continue

                        cost = price * shares * 0.001
                        total_value = price * shares + cost
                        if total_value > cash:
                            continue

                        cash -= total_value
                        total_costs += cost
                        long_positions[symbol] = {
                            "symbol": symbol, "shares": shares,
                            "entry_price": price, "entry_cost": cost,
                            "entry_date": day_str,
                        }
                        day_trades.append({
                            "date": day_str, "symbol": symbol, "action": "buy",
                            "side": "long", "price": round(price, 2),
                            "shares": shares, "confidence": round(up_prob, 3),
                            "reason": f"ML long ({up_prob:.0%} up)",
                        })

                    elif down_prob > self.confidence_threshold and total_open < 8:
                        max_pos_value = cash * max_position_pct
                        shares = int(max_pos_value / price)
                        if shares <= 0:
                            continue

                        cost = price * shares * 0.001
                        margin_required = price * shares + cost
                        if margin_required > cash:
                            continue

                        cash -= margin_required
                        total_costs += cost
                        short_positions[symbol] = {
                            "symbol": symbol, "shares": shares,
                            "entry_price": price, "entry_cost": cost,
                            "entry_date": day_str,
                        }
                        day_trades.append({
                            "date": day_str, "symbol": symbol, "action": "short",
                            "side": "short", "price": round(price, 2),
                            "shares": shares, "confidence": round(down_prob, 3),
                            "reason": f"ML short ({down_prob:.0%} down)",
                        })

            long_value = sum(
                float(history_data[s].loc[day, "Close"]) * p["shares"]
                for s, p in long_positions.items()
                if day in history_data[s].index
            )
            short_value = sum(
                (p["entry_price"] - float(history_data[s].loc[day, "Close"])) * p["shares"]
                + p["entry_price"] * p["shares"]
                for s, p in short_positions.items()
                if day in history_data[s].index
            )

            daily_results.append({
                "date": day_str,
                "cash": round(cash, 2),
                "positions_value": round(long_value + short_value, 2),
                "total_value": round(cash + long_value + short_value, 2),
                "daily_pnl": round(day_pnl, 2),
                "trades": len(day_trades),
                "long_count": len(long_positions),
                "short_count": len(short_positions),
            })
            trades.extend(day_trades)

        final_value = cash
        for s, p in long_positions.items():
            last_date = all_dates[-1]
            if last_date in history_data[s].index:
                final_value += float(history_data[s].loc[last_date, "Close"]) * p["shares"]
        for s, p in short_positions.items():
            last_date = all_dates[-1]
            if last_date in history_data[s].index:
                final_value += p["entry_price"] * p["shares"] + (p["entry_price"] - float(history_data[s].loc[last_date, "Close"])) * p["shares"]

        closed = [t for t in trades if t["action"] in ("sell", "cover")]
        wins = sum(1 for t in closed if t.get("net_pnl", 0) > 0)
        losses = sum(1 for t in closed if t.get("net_pnl", 0) <= 0)
        long_trades = [t for t in trades if t.get("side") == "long"]
        short_trades = [t for t in trades if t.get("side") == "short"]

        total_pnl = final_value - initial_capital
        return_pct = total_pnl / initial_capital * 100

        return {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_pnl_net": round(total_pnl, 2),
            "return_pct": round(return_pct, 2),
            "total_trades": len(trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "closed_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(closed) * 100, 1) if closed else 0,
            "total_costs": round(total_costs, 2),
            "open_positions": len(long_positions) + len(short_positions),
            "trades": trades,
            "daily_results": daily_results,
        }
