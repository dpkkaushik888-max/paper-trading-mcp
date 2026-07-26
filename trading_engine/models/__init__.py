"""Models package — ML models + data models.

ML models (per trading thesis):
- base_model: SmartLGBM, SmartLGBMEnsemble, shared feature utilities
- mean_rev_model: Mean-reversion features + configs (RSI, IBS, BB, OBV)
- trend_model: Trend-following features + configs (EMA, ADX, momentum, BB squeeze)

Data models (backward compat — previously models.py):
- data_models: CostBreakdown, Trade, Position, Signal, DailySnapshot, PortfolioSummary
"""

from .base_model import (
    _SmartLGBM,
    _SmartLGBMEnsemble,
    _streak,
    add_earnings_features,
    add_sector_relative_features,
    add_vix_features,
    SIMPLIFIED_FEATURES,
)
from .mean_rev_model import (
    build_feature_matrix,
    build_feature_matrix_india,
    build_features_for_market,
    MEAN_REV_CONFIGS,
    MeanRevModel,
)
from .trend_model import (
    build_trend_features,
    TREND_CONFIGS,
    TrendModel,
)
from .classifiers import (
    SmartClassifier,
    SmartXGBoost,
    SmartRandomForest,
    SmartLogistic,
    SmartMLP,
    AVAILABLE_CLASSIFIERS,
    get_classifier,
)
from .data_models import (
    CostBreakdown,
    Trade,
    Position,
    Signal,
    DailySnapshot,
    PortfolioSummary,
)

__all__ = [
    "_SmartLGBM",
    "_SmartLGBMEnsemble",
    "_streak",
    "add_earnings_features",
    "add_sector_relative_features",
    "add_vix_features",
    "SIMPLIFIED_FEATURES",
    "build_feature_matrix",
    "build_feature_matrix_india",
    "build_features_for_market",
    "MEAN_REV_CONFIGS",
    "MeanRevModel",
    "build_trend_features",
    "TREND_CONFIGS",
    "TrendModel",
    "CostBreakdown",
    "Trade",
    "Position",
    "Signal",
    "DailySnapshot",
    "PortfolioSummary",
    "SmartClassifier",
    "SmartXGBoost",
    "SmartRandomForest",
    "SmartLogistic",
    "SmartMLP",
    "AVAILABLE_CLASSIFIERS",
    "get_classifier",
]
