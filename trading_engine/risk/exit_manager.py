"""Unified exit manager — consolidates all exit logic for the trading engine.

Exit types:
  1. Fixed stop-loss / take-profit (mean-reversion default)
  2. Trailing stop (trend-following default)
  3. Time stop (NEW): close after N bars with no SL/TP hit
  4. ML signal exit: close when model flips direction

Strategy-appropriate exit selection:
  - Mean-reversion → fixed TP + time stop (snap back or get out)
  - Trend-following → trailing stop (ride the trend)

Usage:
    from trading_engine.risk import ExitManager
    em = ExitManager()
    decision = em.check_exit(position, current_price, bars_held, ml_signal)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExitReason(str, Enum):
    """Why a position was closed."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    ML_SIGNAL = "ml_signal"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL = "manual"


@dataclass
class ExitDecision:
    """Decision from the exit manager."""
    should_exit: bool
    reason: Optional[ExitReason] = None
    description: str = ""
    pnl_pct: float = 0.0


class ExitManager:
    """Unified exit logic — strategy-aware.

    Each position carries metadata about which strategy opened it,
    so the exit manager applies the correct exit rules.
    """

    def __init__(
        self,
        default_sl_pct: float = 0.03,
        default_tp_pct: float = 0.05,
        trailing_stop_pct: float = 0.05,
        trailing_activation_pct: float = 0.02,
        time_stop_bars_daily: int = 10,
        time_stop_bars_4h: int = 12,
        ml_exit_threshold: float = 0.60,
    ):
        self.default_sl_pct = default_sl_pct
        self.default_tp_pct = default_tp_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.time_stop_bars_daily = time_stop_bars_daily
        self.time_stop_bars_4h = time_stop_bars_4h
        self.ml_exit_threshold = ml_exit_threshold

    def check_exit(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        bars_held: int = 0,
        max_favorable_price: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        use_trailing: bool = False,
        use_time_stop: bool = True,
        ml_counter_prob: Optional[float] = None,
        timeframe: str = "daily",
    ) -> ExitDecision:
        """Check all exit conditions for a position.

        Args:
            side: "long" or "short"
            entry_price: position entry price
            current_price: current market price
            bars_held: number of bars the position has been open
            max_favorable_price: highest price seen (long) or lowest (short)
            sl_pct: custom stop loss percentage (overrides default)
            tp_pct: custom take profit percentage (overrides default)
            use_trailing: use trailing stop instead of fixed TP
            use_time_stop: enable time stop (default True)
            ml_counter_prob: probability of opposite direction from ML model
            timeframe: "daily" or "4h" (affects time stop bars)

        Returns:
            ExitDecision with should_exit, reason, and description.
        """
        sl = sl_pct if sl_pct is not None else self.default_sl_pct
        tp = tp_pct if tp_pct is not None else self.default_tp_pct

        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # 1. Stop loss — always checked first
        if pnl_pct <= -sl:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                description=f"Stop loss hit: {pnl_pct:.2%} (limit: -{sl:.2%})",
                pnl_pct=pnl_pct,
            )

        # 2. Trailing stop (for trend-following)
        if use_trailing and max_favorable_price is not None:
            if side == "long":
                drawdown = (max_favorable_price - current_price) / max_favorable_price
                activation_met = (max_favorable_price - entry_price) / entry_price >= self.trailing_activation_pct
            else:
                drawdown = (current_price - max_favorable_price) / max_favorable_price
                activation_met = (entry_price - max_favorable_price) / entry_price >= self.trailing_activation_pct

            if activation_met and drawdown >= self.trailing_stop_pct:
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    description=f"Trailing stop: {drawdown:.2%} pullback from best (activation: {self.trailing_activation_pct:.1%})",
                    pnl_pct=pnl_pct,
                )

        # 3. Fixed take profit (for mean-reversion)
        if not use_trailing and pnl_pct >= tp:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                description=f"Take profit hit: {pnl_pct:.2%} (target: +{tp:.2%})",
                pnl_pct=pnl_pct,
            )

        # 4. Time stop — close stale positions
        if use_time_stop:
            max_bars = self.time_stop_bars_4h if timeframe == "4h" else self.time_stop_bars_daily
            if bars_held >= max_bars:
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.TIME_STOP,
                    description=f"Time stop: held {bars_held} bars (max: {max_bars})",
                    pnl_pct=pnl_pct,
                )

        # 5. ML signal exit — model flipped direction
        if ml_counter_prob is not None and ml_counter_prob > self.ml_exit_threshold:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.ML_SIGNAL,
                description=f"ML signal exit: counter-direction prob {ml_counter_prob:.0%} > {self.ml_exit_threshold:.0%}",
                pnl_pct=pnl_pct,
            )

        return ExitDecision(should_exit=False, pnl_pct=pnl_pct)

    def get_strategy_exit_config(self, strategy_name: str) -> dict:
        """Get default exit config for a strategy type.

        Returns dict with keys: use_trailing, use_time_stop, sl_pct, tp_pct.
        """
        configs = {
            "ml_sniper": {
                "use_trailing": False,
                "use_time_stop": True,
                "sl_pct": 0.03,
                "tp_pct": 0.05,
            },
            "mean_reversion": {
                "use_trailing": False,
                "use_time_stop": True,
                "sl_pct": 0.03,
                "tp_pct": 0.05,
            },
            "trend_follower": {
                "use_trailing": True,
                "use_time_stop": False,
                "sl_pct": 0.05,
                "tp_pct": 0.15,
            },
            "candlestick_sr": {
                "use_trailing": False,
                "use_time_stop": True,
                "sl_pct": 0.03,
                "tp_pct": 0.05,
            },
        }
        return configs.get(strategy_name, configs["mean_reversion"])
