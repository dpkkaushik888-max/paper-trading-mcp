"""Margin & Options Trading Simulator.

Simulates:
1. Margin trading — leveraged stock positions (2x, 5x)
2. Options trading — buying calls/puts based on ML signals

Uses the ML model for signals, adds realistic cost modeling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .ml_model import MLSignalGenerator, build_feature_matrix


# --- Margin Trading ---

class MarginSimulator:
    """Simulates leveraged margin trading.

    Leverage amplifies both gains and losses.
    Margin call if equity drops below maintenance margin (25%).
    Daily interest charged on borrowed amount.
    """

    def __init__(
        self,
        leverage: float = 2.0,
        margin_interest_annual: float = 0.08,
        maintenance_margin: float = 0.25,
        spread_pct: float = 0.001,
    ):
        self.leverage = leverage
        self.daily_interest = margin_interest_annual / 252
        self.maintenance_margin = maintenance_margin
        self.spread_pct = spread_pct

    def backtest(
        self,
        history_data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        confidence_threshold: float = 0.60,
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05,
    ) -> dict:
        ml = MLSignalGenerator(
            train_window=200, min_train=80,
            confidence_threshold=confidence_threshold,
        )

        all_features = {}
        spy_features = None
        if "SPY" in history_data:
            spy_feat = build_feature_matrix(history_data["SPY"])
            spy_features = spy_feat[["rsi_2", "rsi_14", "ibs", "return_1d",
                                     "return_5d", "volatility_5d",
                                     "close_vs_sma_200"]].copy()
            spy_features.columns = [f"spy_{c}" for c in spy_features.columns]

        for symbol, df in history_data.items():
            feat = build_feature_matrix(df)
            if spy_features is not None and symbol != "SPY":
                feat = feat.join(spy_features, how="left")
            feat = feat.dropna()
            if len(feat) > 80:
                all_features[symbol] = feat

        if not all_features:
            return {"error": "Insufficient data"}

        import lightgbm as lgb

        all_dates = sorted(set().union(*(f.index for f in all_features.values())))
        test_start = 80
        feature_cols = None
        model = None

        cash = initial_capital
        positions = {}
        trades = []
        daily_results = []
        total_costs = 0.0
        total_interest = 0.0
        margin_calls = 0

        for day_idx in range(test_start, len(all_dates) - 1):
            day = all_dates[day_idx]
            day_str = str(day)[:10]
            day_trades = []
            day_pnl = 0.0

            if model is None or (day_idx - test_start) % 20 == 0:
                X_list, y_list = [], []
                for sym, feat in all_features.items():
                    train = feat[feat.index < day].tail(200)
                    if feature_cols is None:
                        feature_cols = [c for c in train.columns if c not in {"target", "target_dir"}]
                    valid = train.dropna(subset=feature_cols + ["target_dir"])
                    if len(valid) >= 30:
                        X_list.append(valid[feature_cols])
                        y_list.append(valid["target_dir"])
                if X_list:
                    model = lgb.LGBMClassifier(
                        n_estimators=200, max_depth=3, learning_rate=0.03,
                        subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
                        reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42,
                    )
                    model.fit(pd.concat(X_list), pd.concat(y_list))

            if model is None:
                continue

            # Daily interest on borrowed money
            for sym, pos in list(positions.items()):
                borrowed = pos["position_value"] - pos["equity_used"]
                interest = borrowed * self.daily_interest
                total_interest += interest
                pos["total_interest"] += interest

            # Check margin calls & exits
            for sym in list(positions.keys()):
                if day not in history_data[sym].index:
                    continue
                pos = positions[sym]
                price = float(history_data[sym].loc[day, "Close"])
                current_value = price * pos["shares"]
                pnl = current_value - pos["position_value"]
                equity = pos["equity_used"] + pnl - pos["total_interest"]

                margin_ratio = equity / current_value if current_value > 0 else 0

                should_exit = False
                reason = ""

                pnl_pct = pnl / pos["equity_used"] if pos["equity_used"] > 0 else 0

                if margin_ratio < self.maintenance_margin:
                    should_exit = True
                    reason = f"MARGIN CALL (equity {margin_ratio:.0%})"
                    margin_calls += 1
                elif pnl_pct <= -stop_loss_pct * self.leverage:
                    should_exit = True
                    reason = f"Stop loss {pnl_pct:.1%}"
                elif pnl_pct >= take_profit_pct * self.leverage:
                    should_exit = True
                    reason = f"Take profit {pnl_pct:.1%}"
                else:
                    if day in all_features.get(sym, pd.DataFrame()).index:
                        row = all_features[sym].loc[day]
                        if not row[feature_cols].isna().any():
                            proba = model.predict_proba(row[feature_cols].values.reshape(1, -1))[0]
                            if proba[0] > confidence_threshold:
                                should_exit = True
                                reason = f"ML sell ({proba[0]:.0%} down)"

                if should_exit:
                    cost = current_value * self.spread_pct
                    net_pnl = pnl - pos["entry_cost"] - cost - pos["total_interest"]
                    cash += pos["equity_used"] + pnl - cost - pos["total_interest"]
                    total_costs += cost
                    day_pnl += net_pnl

                    day_trades.append({
                        "date": day_str, "symbol": sym, "action": "sell",
                        "price": round(price, 2), "shares": pos["shares"],
                        "net_pnl": round(net_pnl, 2), "interest": round(pos["total_interest"], 2),
                        "reason": reason, "leverage": self.leverage,
                    })
                    del positions[sym]

            # Entry signals
            for sym, feat in all_features.items():
                if day not in feat.index or sym in positions:
                    continue
                row = feat.loc[day]
                if row[feature_cols].isna().any():
                    continue

                proba = model.predict_proba(row[feature_cols].values.reshape(1, -1))[0]
                if proba[1] <= confidence_threshold:
                    continue

                price = float(history_data[sym].loc[day, "Close"])
                if price <= 0:
                    continue

                equity_to_use = cash * max_position_pct
                position_value = equity_to_use * self.leverage
                shares = int(position_value / price)
                if shares <= 0:
                    continue

                actual_pos_value = price * shares
                actual_equity = actual_pos_value / self.leverage
                cost = actual_pos_value * self.spread_pct
                total_out = actual_equity + cost

                if total_out > cash:
                    continue

                cash -= total_out
                total_costs += cost
                positions[sym] = {
                    "shares": shares, "entry_price": price,
                    "position_value": actual_pos_value,
                    "equity_used": actual_equity,
                    "entry_cost": cost, "total_interest": 0.0,
                }

                day_trades.append({
                    "date": day_str, "symbol": sym, "action": "buy",
                    "price": round(price, 2), "shares": shares,
                    "equity": round(actual_equity, 2),
                    "position_value": round(actual_pos_value, 2),
                    "leverage": self.leverage,
                    "reason": f"ML buy ({proba[1]:.0%} up)",
                })

            pos_value = sum(
                float(history_data[s].loc[day, "Close"]) * p["shares"] - (p["position_value"] - p["equity_used"])
                for s, p in positions.items()
                if day in history_data[s].index
            )

            daily_results.append({
                "date": day_str,
                "total_value": round(cash + pos_value, 2),
                "trades": len(day_trades),
            })
            trades.extend(day_trades)

        final_value = cash
        for s, p in positions.items():
            last = all_dates[-1]
            if last in history_data[s].index:
                cv = float(history_data[s].loc[last, "Close"]) * p["shares"]
                borrowed = p["position_value"] - p["equity_used"]
                final_value += cv - borrowed - p["total_interest"]

        closed = [t for t in trades if t["action"] == "sell"]
        wins = sum(1 for t in closed if t.get("net_pnl", 0) > 0)
        pnl = final_value - initial_capital

        return {
            "strategy": f"Margin {self.leverage:.0f}x",
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_pnl_net": round(pnl, 2),
            "return_pct": round(pnl / initial_capital * 100, 2),
            "total_trades": len(trades),
            "closed_trades": len(closed),
            "wins": wins,
            "win_rate": round(wins / len(closed) * 100, 1) if closed else 0,
            "total_costs": round(total_costs, 2),
            "total_interest": round(total_interest, 2),
            "margin_calls": margin_calls,
            "open_positions": len(positions),
            "trades": trades,
            "daily_results": daily_results,
        }


# --- Options Trading ---

class OptionsSimulator:
    """Simulates buying calls and puts.

    Simplified Black-Scholes-like pricing.
    Models time decay (theta) and directional moves.
    """

    def __init__(
        self,
        days_to_expiry: int = 5,
        option_cost_pct: float = 0.015,
        commission_per_contract: float = 0.65,
    ):
        self.days_to_expiry = days_to_expiry
        self.option_cost_pct = option_cost_pct
        self.commission = commission_per_contract

    def _option_price(self, stock_price: float, strike: float,
                      days_left: int, volatility: float, is_call: bool) -> float:
        """Simplified option pricing."""
        if days_left <= 0:
            if is_call:
                return max(0, stock_price - strike)
            else:
                return max(0, strike - stock_price)

        time_value = stock_price * volatility * np.sqrt(days_left / 252)

        if is_call:
            intrinsic = max(0, stock_price - strike)
        else:
            intrinsic = max(0, strike - stock_price)

        moneyness = abs(stock_price - strike) / stock_price
        otm_decay = np.exp(-moneyness * 10)

        return intrinsic + time_value * otm_decay * 0.3

    def backtest(
        self,
        history_data: dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        confidence_threshold: float = 0.65,
        max_risk_pct: float = 0.05,
    ) -> dict:
        ml = MLSignalGenerator(
            train_window=200, min_train=80,
            confidence_threshold=confidence_threshold,
        )

        all_features = {}
        spy_features = None
        if "SPY" in history_data:
            spy_feat = build_feature_matrix(history_data["SPY"])
            spy_features = spy_feat[["rsi_2", "rsi_14", "ibs", "return_1d",
                                     "return_5d", "volatility_5d",
                                     "close_vs_sma_200"]].copy()
            spy_features.columns = [f"spy_{c}" for c in spy_features.columns]

        for symbol, df in history_data.items():
            feat = build_feature_matrix(df)
            if spy_features is not None and symbol != "SPY":
                feat = feat.join(spy_features, how="left")
            feat = feat.dropna()
            if len(feat) > 80:
                all_features[symbol] = feat

        if not all_features:
            return {"error": "Insufficient data"}

        import lightgbm as lgb

        all_dates = sorted(set().union(*(f.index for f in all_features.values())))
        test_start = 80
        feature_cols = None
        model = None

        cash = initial_capital
        option_positions = []
        trades = []
        daily_results = []
        total_premiums_paid = 0.0
        total_commissions = 0.0

        for day_idx in range(test_start, len(all_dates) - 1):
            day = all_dates[day_idx]
            day_str = str(day)[:10]
            day_trades = []
            day_pnl = 0.0

            if model is None or (day_idx - test_start) % 20 == 0:
                X_list, y_list = [], []
                for sym, feat in all_features.items():
                    train = feat[feat.index < day].tail(200)
                    if feature_cols is None:
                        feature_cols = [c for c in train.columns if c not in {"target", "target_dir"}]
                    valid = train.dropna(subset=feature_cols + ["target_dir"])
                    if len(valid) >= 30:
                        X_list.append(valid[feature_cols])
                        y_list.append(valid["target_dir"])
                if X_list:
                    model = lgb.LGBMClassifier(
                        n_estimators=200, max_depth=3, learning_rate=0.03,
                        subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
                        reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42,
                    )
                    model.fit(pd.concat(X_list), pd.concat(y_list))

            if model is None:
                continue

            # Check existing options
            for opt in list(option_positions):
                if day not in history_data[opt["symbol"]].index:
                    continue

                price = float(history_data[opt["symbol"]].loc[day, "Close"])
                opt["days_left"] -= 1

                vol = all_features.get(opt["symbol"], pd.DataFrame())
                current_vol = 0.15
                if not vol.empty and day in vol.index and "volatility_20d" in vol.columns:
                    v = vol.loc[day, "volatility_20d"]
                    if not np.isnan(v):
                        current_vol = v * np.sqrt(252)

                current_value = self._option_price(
                    price, opt["strike"], opt["days_left"], current_vol, opt["is_call"]
                ) * opt["contracts"] * 100

                should_close = False
                reason = ""

                if opt["days_left"] <= 0:
                    should_close = True
                    reason = "Expired"
                elif current_value >= opt["premium_paid"] * 2:
                    should_close = True
                    reason = f"Take profit ({current_value/opt['premium_paid']:.1f}x)"
                elif current_value <= opt["premium_paid"] * 0.2:
                    should_close = True
                    reason = "Cut loss (80% lost)"

                if should_close:
                    comm = self.commission * opt["contracts"]
                    net_pnl = current_value - opt["premium_paid"] - opt["entry_commission"] - comm
                    cash += current_value - comm
                    total_commissions += comm
                    day_pnl += net_pnl

                    day_trades.append({
                        "date": day_str, "symbol": opt["symbol"],
                        "action": "close",
                        "type": "call" if opt["is_call"] else "put",
                        "strike": opt["strike"],
                        "contracts": opt["contracts"],
                        "premium_paid": round(opt["premium_paid"], 2),
                        "exit_value": round(current_value, 2),
                        "net_pnl": round(net_pnl, 2),
                        "reason": reason,
                    })
                    option_positions.remove(opt)

            # New option entries
            for sym, feat in all_features.items():
                if day not in feat.index:
                    continue

                active_for_sym = [o for o in option_positions if o["symbol"] == sym]
                if len(active_for_sym) >= 2:
                    continue

                row = feat.loc[day]
                if row[feature_cols].isna().any():
                    continue

                proba = model.predict_proba(row[feature_cols].values.reshape(1, -1))[0]
                pred_class = int(np.argmax(proba))
                confidence = float(proba[pred_class])

                if confidence < confidence_threshold:
                    continue

                price = float(history_data[sym].loc[day, "Close"])
                if price <= 0:
                    continue

                is_call = pred_class == 1

                vol = 0.15
                if "volatility_20d" in row.index and not np.isnan(row["volatility_20d"]):
                    vol = row["volatility_20d"] * np.sqrt(252)

                if is_call:
                    strike = round(price * 1.01, 2)
                else:
                    strike = round(price * 0.99, 2)

                premium_per_share = self._option_price(
                    price, strike, self.days_to_expiry, vol, is_call
                )

                max_risk = cash * max_risk_pct
                contracts = max(1, int(max_risk / (premium_per_share * 100 + self.commission)))
                total_premium = premium_per_share * contracts * 100
                comm = self.commission * contracts

                if total_premium + comm > cash * 0.1:
                    contracts = max(1, int(cash * 0.1 / (premium_per_share * 100 + self.commission)))
                    total_premium = premium_per_share * contracts * 100
                    comm = self.commission * contracts

                if total_premium + comm > cash:
                    continue

                cash -= total_premium + comm
                total_premiums_paid += total_premium
                total_commissions += comm

                option_positions.append({
                    "symbol": sym, "is_call": is_call,
                    "strike": strike, "contracts": contracts,
                    "premium_paid": total_premium,
                    "entry_commission": comm,
                    "days_left": self.days_to_expiry,
                    "entry_date": day_str,
                })

                day_trades.append({
                    "date": day_str, "symbol": sym,
                    "action": "buy",
                    "type": "call" if is_call else "put",
                    "strike": strike, "contracts": contracts,
                    "premium": round(total_premium, 2),
                    "confidence": round(confidence, 3),
                    "reason": f"ML {'call' if is_call else 'put'} ({confidence:.0%})",
                })

            opt_value = 0
            for opt in option_positions:
                if day in history_data[opt["symbol"]].index:
                    p = float(history_data[opt["symbol"]].loc[day, "Close"])
                    vol = 0.15
                    feat_df = all_features.get(opt["symbol"], pd.DataFrame())
                    if not feat_df.empty and day in feat_df.index:
                        v = feat_df.loc[day].get("volatility_20d", np.nan)
                        if not np.isnan(v):
                            vol = v * np.sqrt(252)
                    opt_value += self._option_price(
                        p, opt["strike"], opt["days_left"], vol, opt["is_call"]
                    ) * opt["contracts"] * 100

            daily_results.append({
                "date": day_str,
                "total_value": round(cash + opt_value, 2),
                "trades": len(day_trades),
            })
            trades.extend(day_trades)

        final_value = cash
        for opt in option_positions:
            last = all_dates[-1]
            if last in history_data[opt["symbol"]].index:
                p = float(history_data[opt["symbol"]].loc[last, "Close"])
                final_value += self._option_price(
                    p, opt["strike"], max(0, opt["days_left"]), 0.15, opt["is_call"]
                ) * opt["contracts"] * 100

        closed = [t for t in trades if t["action"] == "close"]
        wins = sum(1 for t in closed if t.get("net_pnl", 0) > 0)
        pnl = final_value - initial_capital

        return {
            "strategy": f"Options ({self.days_to_expiry}d expiry)",
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_pnl_net": round(pnl, 2),
            "return_pct": round(pnl / initial_capital * 100, 2),
            "total_trades": len(trades),
            "closed_trades": len(closed),
            "wins": wins,
            "win_rate": round(wins / len(closed) * 100, 1) if closed else 0,
            "total_premiums_paid": round(total_premiums_paid, 2),
            "total_commissions": round(total_commissions, 2),
            "open_options": len(option_positions),
            "trades": trades,
            "daily_results": daily_results,
        }
