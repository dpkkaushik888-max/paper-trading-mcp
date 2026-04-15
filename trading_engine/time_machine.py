"""Time-Machine Backtest — day-by-day replay with strict temporal isolation.

Unlike the batch walk-forward backtest, this engine:
1. Processes ONE day at a time
2. Only passes data[:current_date] to the feature builder — impossible to leak future
3. Persists state after each day (model, trades, portfolio)
4. Integrates the learning loop for confidence calibration + adaptive thresholds
5. Can be paused and resumed (simulates daily cron)

Usage:
    from trading_engine.time_machine import TimeMachineBacktest
    tm = TimeMachineBacktest(market="india", initial_capital=100_000)
    results = tm.run(history_data, start_date="2023-01-01")
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CIRCUIT_BREAKER_TIERS, PROJECT_ROOT
from .learning_loop import LearningLoop
from .ml_model import (
    MARKET_CONFIGS,
    _SmartLGBM,
    _SmartLGBMEnsemble,
    add_sector_relative_features,
    add_vix_features,
    build_features_for_market,
)
from .trade_journal import TradeJournal

warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _add_relative_strength(feat: pd.DataFrame, ca_feat: pd.DataFrame,
                           prefix: str) -> pd.DataFrame:
    """Add relative strength features: stock return minus index return.

    These capture alpha — excess return over the market — which is the
    strongest predictor for individual stock trading.
    """
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


class TimeMachineBacktest:
    """Day-by-day replay engine with learning loop integration."""

    def __init__(
        self,
        market: str = "us",
        initial_capital: float = 10000.0,
        confidence_threshold: float | None = None,
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05,
        session_id: str = "tm_default",
        enable_learning: bool = True,
        dynamic_sl: bool = False,
        journal_db_path: str | None = None,
        simplified_features: list[str] | None = None,
    ):
        self.market = market
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.dynamic_sl = dynamic_sl
        self.session_id = session_id
        self.enable_learning = enable_learning
        cfg = MARKET_CONFIGS.get(market, MARKET_CONFIGS["us"])
        self.train_window = cfg["train_window"]
        self.min_train = cfg["min_train"]
        self.retrain_every = cfg["retrain_every"]
        self.lgbm_params = cfg["lgbm_params"]
        self.cross_asset_symbol = cfg["cross_asset_symbol"]
        self.cross_asset_features = cfg["cross_asset_features"]
        self.cross_asset_prefix = cfg["cross_asset_prefix"]

        self.base_confidence = confidence_threshold or cfg["default_confidence"]

        self.journal = TradeJournal(
            db_path=journal_db_path or str(PROJECT_ROOT / "trade_journal.db"),
            session_id=session_id,
        )
        self.learning = LearningLoop(base_threshold=self.base_confidence)

        self._simplified_features = simplified_features
        self.model: Optional[_SmartLGBM] = None
        self.feature_cols: Optional[list[str]] = None
        self.cash = initial_capital
        self.long_positions: dict = {}
        self.short_positions: dict = {}
        self.trade_id_map: dict = {}
        self.total_costs = 0.0
        self.days_since_retrain = 0
        self._max_price_tracker: dict[str, dict] = {}

        self._circuit_breaker_state = "normal"
        self._peak_value = initial_capital
        self._week_start_value = initial_capital
        self._day_start_value = initial_capital
        self._days_in_week = 0
        self._halted = False
        self._circuit_breaker_log: list[dict] = []

    def run(
        self,
        history_data: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
        verbose: bool = False,
    ) -> dict:
        """Run full time-machine backtest.

        Args:
            history_data: {symbol: OHLCV DataFrame}
            start_date: "YYYY-MM-DD" — skip dates before this
            end_date: "YYYY-MM-DD" — stop after this date
            verbose: print day-by-day progress

        Returns:
            Result dict with performance metrics + calibration data.
        """
        self.journal.clear_session(self.market)

        all_dates = sorted(set().union(
            *(df.index for df in history_data.values())
        ))

        if start_date:
            start_dt = pd.Timestamp(start_date)
            all_dates = [d for d in all_dates if d >= start_dt]
        if end_date:
            end_dt = pd.Timestamp(end_date)
            all_dates = [d for d in all_dates if d <= end_dt]

        if len(all_dates) < self.min_train + 20:
            return {"error": "Not enough dates for time-machine backtest"}

        self.cash = self.initial_capital
        self.long_positions = {}
        self.short_positions = {}
        self.trade_id_map = {}
        self.total_costs = 0.0
        self.days_since_retrain = self.retrain_every
        self._max_price_tracker = {}

        self._circuit_breaker_state = "normal"
        self._peak_value = self.initial_capital
        self._week_start_value = self.initial_capital
        self._day_start_value = self.initial_capital
        self._days_in_week = 0
        self._halted = False
        self._circuit_breaker_log = []

        daily_results = []
        all_trades = []

        for day_idx, day in enumerate(all_dates):
            if day_idx < self.min_train:
                continue

            result = self._process_day(
                day=day,
                day_idx=day_idx,
                all_dates=all_dates,
                history_data=history_data,
                verbose=verbose,
            )

            daily_results.append(result["snapshot"])
            all_trades.extend(result["trades"])

        final_value = self._compute_final_value(history_data, all_dates[-1])

        closed = [t for t in all_trades if t["action"] in ("sell", "cover")]
        wins = sum(1 for t in closed if t.get("net_pnl", 0) > 0)
        long_trades = [t for t in all_trades if t.get("side") == "long"]
        short_trades = [t for t in all_trades if t.get("side") == "short"]

        total_pnl = final_value - self.initial_capital
        return_pct = total_pnl / self.initial_capital * 100

        calibration = self.learning.get_calibration_summary()

        self.journal.close()

        return {
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_pnl_net": round(total_pnl, 2),
            "return_pct": round(return_pct, 2),
            "total_trades": len(all_trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "closed_trades": len(closed),
            "wins": wins,
            "losses": len(closed) - wins,
            "win_rate": round(wins / len(closed) * 100, 1) if closed else 0,
            "total_costs": round(self.total_costs, 2),
            "open_positions": len(self.long_positions) + len(self.short_positions),
            "calibration": calibration,
            "trades": all_trades,
            "daily_results": daily_results,
            "circuit_breaker_log": self._circuit_breaker_log,
        }

    def _process_day(
        self,
        day: pd.Timestamp,
        day_idx: int,
        all_dates: list,
        history_data: dict[str, pd.DataFrame],
        verbose: bool = False,
    ) -> dict:
        """Process a single trading day with strict temporal isolation."""
        day_str = str(day)[:10]
        day_trades = []
        day_pnl = 0.0

        current_value = self._estimate_current_value(history_data, day)
        cb_action = self._check_circuit_breakers(day_str, current_value)

        if cb_action == "halt":
            snapshot = self._build_snapshot(day_str, day_pnl, 0,
                                           self.base_confidence)
            return {"snapshot": snapshot, "trades": []}

        if cb_action == "close_all":
            day_trades, day_pnl = self._force_close_all(history_data, day, day_str)
            snapshot = self._build_snapshot(day_str, day_pnl, len(day_trades),
                                           self.base_confidence)
            return {"snapshot": snapshot, "trades": day_trades}

        if self.enable_learning and self.days_since_retrain >= self.retrain_every:
            cal_data = self.journal.get_recent_calibration_data(self.market)
            if cal_data:
                self.learning.calibrate(cal_data)

        effective_threshold = (
            self.learning.get_adaptive_threshold()
            if self.enable_learning
            else self.base_confidence
        )

        if self.days_since_retrain >= self.retrain_every:
            self._retrain_model(day, history_data)
            self.days_since_retrain = 0
        self.days_since_retrain += 1

        if self.model is None or self.feature_cols is None:
            snapshot = self._build_snapshot(day_str, day_pnl, 0, effective_threshold)
            return {"snapshot": snapshot, "trades": []}

        allow_new_positions = cb_action not in ("no_new_positions",)
        position_size_mult = 0.5 if cb_action == "halve_size" else 1.0

        for symbol, df in history_data.items():
            if symbol.startswith("^"):
                continue
            if day not in df.index:
                continue

            price = float(df.loc[day, "Close"])
            if price <= 0:
                continue

            temporal_df = df[df.index <= day]
            if len(temporal_df) < 30:
                continue

            feat = build_features_for_market(temporal_df, self.market)

            if self.cross_asset_symbol in history_data and symbol != self.cross_asset_symbol:
                ca_df = history_data[self.cross_asset_symbol]
                ca_temporal = ca_df[ca_df.index <= day]
                if len(ca_temporal) > 30:
                    ca_feat = build_features_for_market(ca_temporal, self.market)
                    avail_ca = [c for c in self.cross_asset_features if c in ca_feat.columns]
                    cross = ca_feat[avail_ca].copy()
                    cross.columns = [f"{self.cross_asset_prefix}_{c}" for c in cross.columns]
                    feat = feat.join(cross, how="left")
                    feat = _add_relative_strength(feat, ca_feat, self.cross_asset_prefix)

            if day not in feat.index:
                continue

            row = feat.loc[day]
            row_feats = row.reindex(self.feature_cols, fill_value=0).fillna(0)
            if row_feats.isna().any():
                continue

            X_pred = row_feats.values.reshape(1, -1)
            proba = self.model.predict_proba(X_pred)[0]
            up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            down_prob = 1.0 - up_prob

            self._track_price(symbol, price)

            if symbol in self.long_positions:
                trade = self._check_long_exit(
                    symbol, price, day_str, down_prob, effective_threshold
                )
                if trade:
                    day_pnl += trade.get("net_pnl", 0)
                    day_trades.append(trade)

            elif symbol in self.short_positions:
                trade = self._check_short_exit(
                    symbol, price, day_str, up_prob, effective_threshold
                )
                if trade:
                    day_pnl += trade.get("net_pnl", 0)
                    day_trades.append(trade)

            elif allow_new_positions:
                total_open = len(self.long_positions) + len(self.short_positions)
                atr_pct = self._compute_atr_pct(symbol, history_data, day) if self.dynamic_sl else 0.0
                if up_prob > effective_threshold and total_open < 8:
                    trade = self._open_long(
                        symbol, price, day_str, up_prob,
                        size_mult=position_size_mult,
                        atr_pct=atr_pct,
                    )
                    if trade:
                        day_trades.append(trade)

                elif down_prob > effective_threshold and total_open < 8:
                    trade = self._open_short(
                        symbol, price, day_str, down_prob,
                        size_mult=position_size_mult,
                        atr_pct=atr_pct,
                    )
                    if trade:
                        day_trades.append(trade)

        self._update_circuit_breaker_tracking(day, current_value)

        snapshot = self._build_snapshot(
            day_str, day_pnl, len(day_trades), effective_threshold
        )

        if verbose and day_trades:
            print(f"  {day_str} | Cash: {self.cash:>10,.0f} | "
                  f"L:{len(self.long_positions)} S:{len(self.short_positions)} | "
                  f"PnL: {day_pnl:>+8,.0f} | Trades: {len(day_trades)} | "
                  f"Threshold: {effective_threshold:.0%}")

        return {"snapshot": snapshot, "trades": day_trades}

    def _retrain_model(self, current_day: pd.Timestamp, history_data: dict):
        """Retrain the model using ONLY data before current_day."""
        train_X_list = []
        train_y_list = []

        for symbol, df in history_data.items():
            if symbol.startswith("^"):
                continue
            temporal_df = df[df.index < current_day]
            if len(temporal_df) < self.min_train:
                continue

            feat = build_features_for_market(temporal_df, self.market)

            if self.cross_asset_symbol in history_data and symbol != self.cross_asset_symbol:
                ca_df = history_data[self.cross_asset_symbol]
                ca_temporal = ca_df[ca_df.index < current_day]
                if len(ca_temporal) > 30:
                    ca_feat = build_features_for_market(ca_temporal, self.market)
                    avail_ca = [c for c in self.cross_asset_features if c in ca_feat.columns]
                    cross = ca_feat[avail_ca].copy()
                    cross.columns = [f"{self.cross_asset_prefix}_{c}" for c in cross.columns]
                    feat = feat.join(cross, how="left")
                    feat = _add_relative_strength(feat, ca_feat, self.cross_asset_prefix)

            if self.feature_cols is None and symbol != self.cross_asset_symbol:
                exclude = {"target", "target_dir"}
                cols = [c for c in feat.columns if c not in exclude]
                if self._simplified_features:
                    cols = [c for c in self._simplified_features if c in cols]
                self.feature_cols = cols

            train_slice = feat.tail(self.train_window)
            valid = train_slice.dropna(subset=["target_dir"])
            if len(valid) < 30:
                continue

            if self.feature_cols is None:
                exclude = {"target", "target_dir"}
                cols = [c for c in feat.columns if c not in exclude]
                if self._simplified_features:
                    cols = [c for c in self._simplified_features if c in cols]
                self.feature_cols = cols

            train_X_list.append(
                valid.reindex(columns=self.feature_cols, fill_value=0)
                .fillna(0)
            )
            train_y_list.append(valid["target_dir"])

        if not train_X_list:
            return

        X_train = pd.concat(train_X_list)
        y_train = pd.concat(train_y_list)

        if self.enable_learning:
            trade_outcomes = self.journal.get_recent_calibration_data(self.market)
            sample_weights = self.learning.compute_outcome_weights(
                X_train.values, y_train.values, trade_outcomes
            )
        else:
            sample_weights = None

        self.model = _SmartLGBM(params=self.lgbm_params)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        model_path = MODELS_DIR / f"{self.market}_latest.pkl"
        self.model.save(str(model_path))

        cal_data = self.journal.get_recent_calibration_data(self.market)
        cal_result = self.learning.calibrate(cal_data) if cal_data else None

        self.journal.save_model_snapshot(
            market=self.market,
            snapshot_date=str(current_day)[:10],
            train_samples=len(X_train),
            feature_count=len(self.feature_cols),
            model_path=str(model_path),
            calibration=self.learning.get_calibration_summary() if cal_result else None,
            performance=None,
        )

    def _compute_atr_pct(self, symbol: str, history_data: dict, day: pd.Timestamp) -> float:
        """Compute 14-day ATR as percentage of price for a symbol."""
        if symbol not in history_data:
            return 0.0
        df = history_data[symbol]
        temporal = df[df.index <= day]
        if len(temporal) < 15:
            return 0.0
        high = temporal["High"].tail(15)
        low = temporal["Low"].tail(15)
        close = temporal["Close"].tail(15)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.tail(14).mean()
        return float(atr / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

    def _get_dynamic_sl_tp(self, atr_pct: float) -> tuple[float, float]:
        """Compute dynamic SL/TP from ATR. Returns (sl_pct, tp_pct).

        SL = 2.0 * ATR (adapts to volatility, avoids premature stops)
        TP = 3.0 * ATR (maintains 1.5 R:R)
        Floors: SL >= 1%, TP >= 2%. Caps: SL <= 5%, TP <= 8%.
        """
        sl = max(0.01, min(0.05, 2.0 * atr_pct))
        tp = max(0.02, min(0.08, 3.0 * atr_pct))
        return sl, tp

    def _open_long(self, symbol: str, price: float, day_str: str, confidence: float, size_mult: float = 1.0, atr_pct: float = 0.0) -> Optional[dict]:
        """Open a long position."""
        max_value = self.cash * self.max_position_pct * size_mult
        shares = int(max_value / price)
        if shares <= 0:
            return None

        cost = price * shares * 0.001
        total_debit = price * shares + cost
        if total_debit > self.cash:
            shares = int((self.cash - cost) / price)
            if shares <= 0:
                return None
            cost = price * shares * 0.001
            total_debit = price * shares + cost

        self.cash -= total_debit
        self.total_costs += cost

        trade_id = self.journal.open_trade(
            market=self.market, symbol=symbol, side="long",
            entry_date=day_str, entry_price=price,
            entry_confidence=confidence, shares=shares, entry_cost=cost,
        )

        if self.dynamic_sl and atr_pct > 0:
            sl, tp = self._get_dynamic_sl_tp(atr_pct)
        else:
            sl, tp = self.stop_loss_pct, self.take_profit_pct

        self.long_positions[symbol] = {
            "symbol": symbol, "shares": shares, "entry_price": price,
            "entry_cost": cost, "entry_date": day_str, "trade_id": trade_id,
            "sl_pct": sl, "tp_pct": tp,
        }
        self._max_price_tracker[symbol] = {"max": price, "min": price}

        return {
            "date": day_str, "symbol": symbol, "action": "buy",
            "side": "long", "price": round(price, 2),
            "shares": shares, "confidence": round(confidence, 3),
            "reason": f"ML long ({confidence:.0%} up)",
        }

    def _open_short(self, symbol: str, price: float, day_str: str, confidence: float, size_mult: float = 1.0, atr_pct: float = 0.0) -> Optional[dict]:
        """Open a short position (reserve entry value as margin)."""
        max_value = self.cash * self.max_position_pct * size_mult
        shares = int(max_value / price)
        if shares <= 0:
            return None

        cost = price * shares * 0.001
        margin = price * shares
        total_debit = margin + cost
        if total_debit > self.cash:
            shares = int((self.cash - cost) / price)
            if shares <= 0:
                return None
            cost = price * shares * 0.001
            margin = price * shares
            total_debit = margin + cost

        self.cash -= total_debit
        self.total_costs += cost

        trade_id = self.journal.open_trade(
            market=self.market, symbol=symbol, side="short",
            entry_date=day_str, entry_price=price,
            entry_confidence=confidence, shares=shares, entry_cost=cost,
        )

        if self.dynamic_sl and atr_pct > 0:
            sl, tp = self._get_dynamic_sl_tp(atr_pct)
        else:
            sl, tp = self.stop_loss_pct, self.take_profit_pct

        self.short_positions[symbol] = {
            "symbol": symbol, "shares": shares, "entry_price": price,
            "entry_cost": cost, "entry_date": day_str, "trade_id": trade_id,
            "sl_pct": sl, "tp_pct": tp,
        }
        self._max_price_tracker[symbol] = {"max": price, "min": price}

        return {
            "date": day_str, "symbol": symbol, "action": "short",
            "side": "short", "price": round(price, 2),
            "shares": shares, "confidence": round(confidence, 3),
            "reason": f"ML short ({confidence:.0%} down)",
        }

    def _check_long_exit(
        self, symbol: str, price: float, day_str: str,
        down_prob: float, threshold: float,
    ) -> Optional[dict]:
        """Check if a long position should be exited."""
        pos = self.long_positions[symbol]
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
        sl = pos.get("sl_pct", self.stop_loss_pct)
        tp = pos.get("tp_pct", self.take_profit_pct)

        should_exit = False
        reason = ""

        if pnl_pct <= -sl:
            should_exit = True
            reason = f"Stop loss {pnl_pct:.2%}"
        elif pnl_pct >= tp:
            should_exit = True
            reason = f"Take profit {pnl_pct:.2%}"
        elif down_prob > threshold:
            should_exit = True
            reason = f"ML bearish ({down_prob:.0%} down)"

        if not should_exit:
            return None

        gross_pnl = (price - pos["entry_price"]) * pos["shares"]
        cost = price * pos["shares"] * 0.001
        net_pnl = gross_pnl - pos["entry_cost"] - cost
        self.cash += price * pos["shares"] - cost
        self.total_costs += cost

        tracker = self._max_price_tracker.get(symbol, {})
        mae = (pos["entry_price"] - tracker.get("min", price)) / pos["entry_price"]
        mfe = (tracker.get("max", price) - pos["entry_price"]) / pos["entry_price"]

        self.journal.close_trade(
            trade_id=pos["trade_id"], exit_date=day_str, exit_price=price,
            exit_reason=reason, gross_pnl=gross_pnl, net_pnl=net_pnl,
            max_adverse_excursion=mae, max_favorable_excursion=mfe,
        )

        del self.long_positions[symbol]
        self._max_price_tracker.pop(symbol, None)

        return {
            "date": day_str, "symbol": symbol, "action": "sell",
            "side": "long", "price": round(price, 2),
            "shares": pos["shares"], "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "reason": reason,
        }

    def _check_short_exit(
        self, symbol: str, price: float, day_str: str,
        up_prob: float, threshold: float,
    ) -> Optional[dict]:
        """Check if a short position should be covered."""
        pos = self.short_positions[symbol]
        pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]
        sl = pos.get("sl_pct", self.stop_loss_pct)
        tp = pos.get("tp_pct", self.take_profit_pct)

        should_exit = False
        reason = ""

        if pnl_pct <= -sl:
            should_exit = True
            reason = f"Short stop loss {-pnl_pct:.2%}"
        elif pnl_pct >= tp:
            should_exit = True
            reason = f"Short take profit {pnl_pct:.2%}"
        elif up_prob > threshold:
            should_exit = True
            reason = f"ML bullish ({up_prob:.0%} up)"

        if not should_exit:
            return None

        gross_pnl = (pos["entry_price"] - price) * pos["shares"]
        cost = price * pos["shares"] * 0.001
        net_pnl = gross_pnl - pos["entry_cost"] - cost
        self.cash += pos["entry_price"] * pos["shares"] + gross_pnl - cost  # return margin + P&L
        self.total_costs += cost

        tracker = self._max_price_tracker.get(symbol, {})
        mae = (tracker.get("max", price) - pos["entry_price"]) / pos["entry_price"]
        mfe = (pos["entry_price"] - tracker.get("min", price)) / pos["entry_price"]

        self.journal.close_trade(
            trade_id=pos["trade_id"], exit_date=day_str, exit_price=price,
            exit_reason=reason, gross_pnl=gross_pnl, net_pnl=net_pnl,
            max_adverse_excursion=mae, max_favorable_excursion=mfe,
        )

        del self.short_positions[symbol]
        self._max_price_tracker.pop(symbol, None)

        return {
            "date": day_str, "symbol": symbol, "action": "cover",
            "side": "short", "price": round(price, 2),
            "shares": pos["shares"], "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "reason": reason,
        }

    def _track_price(self, symbol: str, price: float):
        """Track max/min price for MAE/MFE calculation."""
        if symbol in self._max_price_tracker:
            t = self._max_price_tracker[symbol]
            t["max"] = max(t["max"], price)
            t["min"] = min(t["min"], price)

    def _build_snapshot(
        self, day_str: str, day_pnl: float, trades_today: int,
        threshold: float,
    ) -> dict:
        """Build and persist a daily snapshot."""
        long_value = sum(
            p["entry_price"] * p["shares"] for p in self.long_positions.values()
        )
        short_margin = sum(
            p["entry_price"] * p["shares"] for p in self.short_positions.values()
        )
        total = self.cash + long_value + short_margin

        cal = self.learning._calibration
        model_cal = (1.0 - cal.calibration_error) if cal and cal.total_trades > 0 else None

        self.journal.save_daily_snapshot(
            market=self.market, date_str=day_str, cash=self.cash,
            long_value=long_value, short_value=short_margin,
            total_value=total, daily_pnl=day_pnl,
            open_longs=len(self.long_positions),
            open_shorts=len(self.short_positions),
            trades_today=trades_today,
            confidence_threshold=threshold,
            model_calibration=model_cal,
        )

        return {
            "date": day_str, "cash": round(self.cash, 2),
            "long_value": round(long_value, 2),
            "short_value": round(short_margin, 2),
            "total_value": round(total, 2),
            "daily_pnl": round(day_pnl, 2),
            "trades": trades_today,
            "long_count": len(self.long_positions),
            "short_count": len(self.short_positions),
            "threshold": round(threshold, 3),
        }

    def _compute_final_value(self, history_data: dict, last_date) -> float:
        """Compute final portfolio value marking open positions to market."""
        value = self.cash
        for s, p in self.long_positions.items():
            if last_date in history_data[s].index:
                value += float(history_data[s].loc[last_date, "Close"]) * p["shares"]
        for s, p in self.short_positions.items():
            if last_date in history_data[s].index:
                close_price = float(history_data[s].loc[last_date, "Close"])
                value += p["entry_price"] * p["shares"] + \
                    (p["entry_price"] - close_price) * p["shares"]
        return value

    def _estimate_current_value(self, history_data: dict, day: pd.Timestamp) -> float:
        """Estimate current portfolio value using today's prices."""
        value = self.cash
        for s, p in self.long_positions.items():
            if s in history_data and day in history_data[s].index:
                value += float(history_data[s].loc[day, "Close"]) * p["shares"]
            else:
                value += p["entry_price"] * p["shares"]
        for s, p in self.short_positions.items():
            if s in history_data and day in history_data[s].index:
                close_price = float(history_data[s].loc[day, "Close"])
                value += p["entry_price"] * p["shares"] + \
                    (p["entry_price"] - close_price) * p["shares"]
            else:
                value += p["entry_price"] * p["shares"]
        return value

    def _check_circuit_breakers(self, day_str: str, current_value: float) -> str:
        """Check circuit breaker tiers and return action.

        Returns: 'normal', 'halve_size', 'no_new_positions', 'close_all', or 'halt'.
        """
        if self._halted:
            return "halt"

        tiers = CIRCUIT_BREAKER_TIERS

        peak_dd = (self._peak_value - current_value) / self._peak_value if self._peak_value > 0 else 0
        if peak_dd >= tiers["halt"]["peak_dd_pct"]:
            self._halted = True
            self._circuit_breaker_log.append({
                "date": day_str, "tier": "halt", "dd": round(peak_dd * 100, 2),
                "value": round(current_value, 2),
            })
            return "halt"

        weekly_dd = (self._week_start_value - current_value) / self._week_start_value if self._week_start_value > 0 else 0
        if weekly_dd >= tiers["critical"]["weekly_dd_pct"]:
            self._circuit_breaker_log.append({
                "date": day_str, "tier": "critical", "dd": round(weekly_dd * 100, 2),
                "value": round(current_value, 2),
            })
            return "close_all"

        daily_dd = (self._day_start_value - current_value) / self._day_start_value if self._day_start_value > 0 else 0
        if daily_dd >= tiers["danger"]["daily_dd_pct"]:
            self._circuit_breaker_log.append({
                "date": day_str, "tier": "danger", "dd": round(daily_dd * 100, 2),
                "value": round(current_value, 2),
            })
            return "no_new_positions"

        if daily_dd >= tiers["caution"]["daily_dd_pct"]:
            return "halve_size"

        return "normal"

    def _update_circuit_breaker_tracking(self, day: pd.Timestamp, current_value: float):
        """Update peak, weekly, and daily tracking values."""
        if current_value > self._peak_value:
            self._peak_value = current_value

        self._day_start_value = current_value

        self._days_in_week += 1
        if self._days_in_week >= 5:
            self._week_start_value = current_value
            self._days_in_week = 0

    def _force_close_all(self, history_data: dict, day: pd.Timestamp, day_str: str) -> tuple[list, float]:
        """Force-close all positions (circuit breaker CRITICAL)."""
        trades = []
        total_pnl = 0.0

        for symbol in list(self.long_positions.keys()):
            if symbol in history_data and day in history_data[symbol].index:
                price = float(history_data[symbol].loc[day, "Close"])
                trade = self._check_long_exit(symbol, price, day_str, 1.0, 0.0)
                if trade:
                    total_pnl += trade.get("net_pnl", 0)
                    trade["reason"] = "Circuit breaker CRITICAL: force close"
                    trades.append(trade)

        for symbol in list(self.short_positions.keys()):
            if symbol in history_data and day in history_data[symbol].index:
                price = float(history_data[symbol].loc[day, "Close"])
                trade = self._check_short_exit(symbol, price, day_str, 1.0, 0.0)
                if trade:
                    total_pnl += trade.get("net_pnl", 0)
                    trade["reason"] = "Circuit breaker CRITICAL: force close"
                    trades.append(trade)

        return trades, total_pnl

