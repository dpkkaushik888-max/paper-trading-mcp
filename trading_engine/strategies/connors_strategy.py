"""Strategy A — Connors uptrend pullback, wrapped as a BaseStrategy (S23).

Delegates VERBATIM to the locked ``connors_swing`` functions (the live S18
rules). No rule changes — both the live loop and this wrapper import the same
module, so behavior is identical by construction.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal
from trading_engine.strategies import connors_swing as cs
from trading_engine.engine.indicators import build_indicators


class ConnorsSwingStrategy(BaseStrategy):
    """A: buy deep oversold pullbacks within an established uptrend."""

    priority = 0  # highest priority in D6 conflict resolution (A > B > C)
    reason = "connors_uptrend_pullback"

    # ── lightweight hot-loop interface (used by the orchestrator) ─────────
    def entry(self, close: float, ind_row: pd.Series) -> bool:
        return cs.long_entry(close, ind_row, use_adx_filter=True)

    def exit_reason(
        self, close: float, ind_row: pd.Series,
        entry_price: float, entry_date: pd.Timestamp, today: pd.Timestamp,
    ) -> Optional[str]:
        return cs.long_exit(close, ind_row, entry_price, entry_date, today)

    @property
    def sl_pct(self) -> float:
        return cs.SL_PCT

    # ── BaseStrategy compliance ───────────────────────────────────────────
    def evaluate(
        self, symbol, df, current_day, model=None,
        feature_cols=None, cross_asset_data=None,
    ) -> Optional[StrategySignal]:
        ind = build_indicators(df)
        if current_day not in ind.index:
            return None
        close = float(df.loc[current_day, "Close"])
        if not self.entry(close, ind.loc[current_day]):
            return None
        return StrategySignal(
            direction="long", confidence=1.0, strategy=self.name,
            reason=self.reason, sl_pct=self.sl_pct, tp_pct=0.0,
            size_pct=self.config.max_position_pct,
        )


def default_config() -> StrategyConfig:
    return StrategyConfig(name="A_connors", capital_pct=1.0,
                          max_position_pct=0.12, max_concurrent=4)
