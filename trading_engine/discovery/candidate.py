"""Candidate strategy spec + codegen (S25 D2/D8).

A CandidateSpec is the agent's bounded output: entry conditions (AND), an exit
spec (SL / take-profit / rule-exit OR / max-hold), and an optional regime
hypothesis. `to_strategy()` compiles it into a GeneratedStrategy that implements
the existing BaseStrategy contract, so it drops straight into the orchestrator and
backtests like any hand-written strategy. The spec is JSON-serializable so the
agent can emit it and the manifest can record it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from trading_engine.strategies import BaseStrategy, StrategyConfig, StrategySignal
from trading_engine.discovery.primitives import Condition, build_features


@dataclass
class ExitSpec:
    sl_pct: float = 0.07
    max_hold_days: int = 10
    tp_pct: float = 0.0                       # 0 = no take-profit
    exit_conditions: list[Condition] = field(default_factory=list)  # OR

    def to_dict(self) -> dict:
        return {"sl_pct": self.sl_pct, "max_hold_days": self.max_hold_days,
                "tp_pct": self.tp_pct,
                "exit_conditions": [c.to_dict() for c in self.exit_conditions]}

    @classmethod
    def from_dict(cls, d: dict) -> "ExitSpec":
        return cls(sl_pct=d.get("sl_pct", 0.07), max_hold_days=d.get("max_hold_days", 10),
                   tp_pct=d.get("tp_pct", 0.0),
                   exit_conditions=[Condition.from_dict(c) for c in d.get("exit_conditions", [])])


@dataclass
class CandidateSpec:
    id: str
    name: str
    entry: list[Condition]                    # AND
    exit: ExitSpec
    regime_weights: Optional[dict] = None     # agent's "when to use which" hypothesis
    pos_size_pct: float = 0.12

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "entry": [c.to_dict() for c in self.entry], "exit": self.exit.to_dict(),
                "regime_weights": self.regime_weights, "pos_size_pct": self.pos_size_pct}

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateSpec":
        return cls(id=d["id"], name=d["name"],
                   entry=[Condition.from_dict(c) for c in d["entry"]],
                   exit=ExitSpec.from_dict(d["exit"]),
                   regime_weights=d.get("regime_weights"),
                   pos_size_pct=d.get("pos_size_pct", 0.12))

    def to_strategy(self, priority: int = 10) -> "GeneratedStrategy":
        cfg = StrategyConfig(name=self.name, capital_pct=1.0,
                             max_position_pct=self.pos_size_pct, max_concurrent=4)
        return GeneratedStrategy(cfg, self, priority=priority)


class GeneratedStrategy(BaseStrategy):
    """A compiled candidate. Deterministic — no LLM in entry/exit evaluation."""

    def __init__(self, config: StrategyConfig, spec: CandidateSpec, priority: int = 10):
        super().__init__(config)
        self.spec = spec
        self.priority = priority           # discovered strategies rank after fixed A/B/C
        self.reason = spec.name

    def entry(self, close: float, row: pd.Series) -> bool:
        return all(c.evaluate(row, close) for c in self.spec.entry)

    def exit_reason(self, close, row, entry_price, entry_date, today):
        pnl = (close - entry_price) / entry_price
        if pnl <= -self.spec.exit.sl_pct:
            return "SL"
        if self.spec.exit.tp_pct > 0 and pnl >= self.spec.exit.tp_pct:
            return "TP"
        for c in self.spec.exit.exit_conditions:
            if c.evaluate(row, close):
                return "RULE_EXIT"
        if (today - entry_date).days >= self.spec.exit.max_hold_days:
            return "MAX_HOLD"
        return None

    @property
    def sl_pct(self) -> float:
        return self.spec.exit.sl_pct

    def evaluate(self, symbol, df, current_day, model=None,
                 feature_cols=None, cross_asset_data=None) -> Optional[StrategySignal]:
        feats = build_features(df)
        if current_day not in feats.index:
            return None
        close = float(df.loc[current_day, "Close"])
        if not self.entry(close, feats.loc[current_day]):
            return None
        return StrategySignal(direction="long", confidence=1.0, strategy=self.name,
                              reason=self.reason, sl_pct=self.sl_pct, tp_pct=self.spec.exit.tp_pct,
                              size_pct=self.config.max_position_pct)
