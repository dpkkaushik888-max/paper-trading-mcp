"""Continuous learning loop: confidence calibration + outcome feedback.

Tracks how well-calibrated the model's confidence is and adapts
the confidence threshold automatically. Also provides outcome-weighted
training samples for the retrain cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CalibrationResult:
    """How well-calibrated the model is."""
    total_trades: int = 0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    calibration_error: float = 0.0
    confidence_buckets: dict = field(default_factory=dict)
    recommended_threshold: float = 0.65
    overconfident: bool = False


class LearningLoop:
    """Manages confidence calibration and adaptive thresholds.

    Core idea: if the model says "70% confident" but only wins 50% of the
    time, it's overconfident and we should raise the threshold.
    """

    OUTCOME_WEIGHT_CAP = 0.20
    DECAY_FACTOR = 0.95
    MIN_TRADES_FOR_CALIBRATION = 20
    CONFIDENCE_BUCKETS = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7),
                          (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 1.0)]

    def __init__(self, base_threshold: float = 0.65):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self._calibration: Optional[CalibrationResult] = None

    def calibrate(self, trade_history: list[dict]) -> CalibrationResult:
        """Compute confidence calibration from trade history.

        Args:
            trade_history: list of dicts with keys:
                - entry_confidence: float (0-1)
                - net_pnl: float
                - side: "long" or "short"
        """
        result = CalibrationResult()
        if not trade_history:
            result.recommended_threshold = self.base_threshold
            self._calibration = result
            return result

        result.total_trades = len(trade_history)
        wins = sum(1 for t in trade_history if (t.get("net_pnl") or 0) > 0)
        result.win_rate = wins / len(trade_history) if trade_history else 0

        confidences = [t.get("entry_confidence", 0.5) or 0.5 for t in trade_history]
        result.avg_confidence = sum(confidences) / len(confidences)

        buckets = {}
        for lo, hi in self.CONFIDENCE_BUCKETS:
            bucket_trades = [
                t for t in trade_history
                if lo <= (t.get("entry_confidence") or 0.5) < hi
            ]
            if bucket_trades:
                bucket_wins = sum(1 for t in bucket_trades if (t.get("net_pnl") or 0) > 0)
                bucket_wr = bucket_wins / len(bucket_trades)
                bucket_avg_conf = sum(
                    t.get("entry_confidence", 0.5) or 0.5 for t in bucket_trades
                ) / len(bucket_trades)
                buckets[f"{lo:.2f}-{hi:.2f}"] = {
                    "trades": len(bucket_trades),
                    "win_rate": round(bucket_wr, 3),
                    "avg_confidence": round(bucket_avg_conf, 3),
                    "calibration_gap": round(bucket_avg_conf - bucket_wr, 3),
                }

        result.confidence_buckets = buckets

        total_cal_error = 0.0
        cal_count = 0
        for bucket_info in buckets.values():
            total_cal_error += abs(bucket_info["calibration_gap"]) * bucket_info["trades"]
            cal_count += bucket_info["trades"]
        result.calibration_error = round(total_cal_error / cal_count, 4) if cal_count > 0 else 0

        result.overconfident = result.avg_confidence > result.win_rate + 0.05

        if len(trade_history) >= self.MIN_TRADES_FOR_CALIBRATION:
            result.recommended_threshold = self._compute_adaptive_threshold(
                trade_history, buckets
            )
        else:
            result.recommended_threshold = self.base_threshold

        self._calibration = result
        return result

    def _compute_adaptive_threshold(
        self, trade_history: list[dict], buckets: dict
    ) -> float:
        """Find the confidence level where win rate exceeds 50%.

        We want the threshold where the model actually wins more than it loses.
        """
        best_threshold = self.base_threshold

        for bucket_key, info in sorted(buckets.items()):
            lo = float(bucket_key.split("-")[0])
            if info["trades"] >= 5 and info["win_rate"] > 0.50:
                best_threshold = lo
                break

        best_threshold = max(0.55, min(0.85, best_threshold))
        return round(best_threshold, 2)

    def get_adaptive_threshold(self) -> float:
        """Get the current adaptive threshold."""
        if self._calibration and self._calibration.total_trades >= self.MIN_TRADES_FOR_CALIBRATION:
            self.current_threshold = self._calibration.recommended_threshold
        return self.current_threshold

    def compute_outcome_weights(
        self,
        X_train: "np.ndarray",
        y_train: "np.ndarray",
        trade_outcomes: list[dict],
    ) -> "np.ndarray":
        """Compute sample weights that upweight cases where model was wrong.

        Gives extra weight to training samples that match patterns where
        the model was previously confident but incorrect.

        Returns:
            Sample weights array (same length as X_train).
            Default weight = 1.0, upweighted = up to 1 + OUTCOME_WEIGHT_CAP.
        """
        weights = np.ones(len(X_train), dtype=np.float64)

        if not trade_outcomes:
            return weights

        wrong_high_conf = [
            t for t in trade_outcomes
            if (t.get("entry_confidence") or 0) > 0.65
            and (t.get("net_pnl") or 0) <= 0
        ]

        if not wrong_high_conf:
            return weights

        wrong_fraction = len(wrong_high_conf) / len(trade_outcomes)
        boost = min(self.OUTCOME_WEIGHT_CAP, wrong_fraction)

        recent_count = min(len(trade_outcomes), 50)
        recent_wrong = sum(
            1 for t in trade_outcomes[:recent_count]
            if (t.get("entry_confidence") or 0) > 0.65
            and (t.get("net_pnl") or 0) <= 0
        )

        if recent_count > 0 and recent_wrong / recent_count > 0.4:
            tail_pct = int(len(X_train) * 0.3)
            weights[-tail_pct:] *= (1.0 + boost)

        return weights

    def get_outcome_features(self, journal, market: str, symbol: str) -> dict:
        """Generate features from trade history for a specific symbol.

        These features tell the model about its own recent performance
        on this symbol.

        Args:
            journal: TradeJournal instance
            market: "us" or "india"
            symbol: stock symbol

        Returns:
            Dict of features to append to the feature matrix.
        """
        stats = journal.get_symbol_stats(market, symbol)
        calibration = self._calibration

        features = {
            "prev_trade_pnl": stats.get("last_pnl", 0) or 0,
            "symbol_win_rate": stats.get("win_rate", 50) / 100.0,
            "avg_holding_days": stats.get("avg_holding_days", 0) or 0,
            "symbol_trade_count": min(stats.get("trades", 0), 20) / 20.0,
        }

        if calibration and calibration.total_trades >= self.MIN_TRADES_FOR_CALIBRATION:
            features["model_calibration"] = 1.0 - calibration.calibration_error
            features["model_overconfident"] = float(calibration.overconfident)
        else:
            features["model_calibration"] = 0.5
            features["model_overconfident"] = 0.0

        return features

    def get_calibration_summary(self) -> dict:
        """Get a human-readable calibration summary."""
        if self._calibration is None:
            return {"status": "no_data", "message": "No calibration data yet"}

        cal = self._calibration
        return {
            "total_trades": cal.total_trades,
            "win_rate": f"{cal.win_rate:.1%}",
            "avg_confidence": f"{cal.avg_confidence:.1%}",
            "calibration_error": f"{cal.calibration_error:.3f}",
            "overconfident": cal.overconfident,
            "current_threshold": f"{self.current_threshold:.0%}",
            "recommended_threshold": f"{cal.recommended_threshold:.0%}",
            "buckets": cal.confidence_buckets,
        }
