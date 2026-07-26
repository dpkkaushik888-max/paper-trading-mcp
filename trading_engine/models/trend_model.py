"""Trend-following ML model — features optimized for sustained directional moves.

Features predict: price will continue moving in the same direction for 5-20 bars.
Exit style: trailing stop (locks in profit as trend extends).
Best for: EMA crossovers, ADX breakouts, momentum acceleration, BB squeeze breakouts.

This is a NEW model — does NOT exist in the original ml_model.py.
It uses different features than MeanRevModel because the prediction target is different.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base_model import _SmartLGBM, _streak

warnings.filterwarnings("ignore", category=UserWarning)


TREND_CONFIGS = {
    "crypto": {
        "train_window": 200,
        "min_train": 80,
        "retrain_every": 10,
        "default_confidence": 0.65,
        "cross_asset_symbol": "BTC-USD",
        "cross_asset_features": ["momentum_5d", "momentum_10d", "adx",
                                  "ema_cross_8_20", "volatility_5d"],
        "cross_asset_prefix": "btc",
        "lgbm_params": {
            "n_estimators": 150, "max_depth": 3, "learning_rate": 0.02,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20,
            "reg_alpha": 0.2, "reg_lambda": 1.5,
        },
    },
    "us": {
        "train_window": 300,
        "min_train": 120,
        "retrain_every": 20,
        "default_confidence": 0.60,
        "cross_asset_symbol": "SPY",
        "cross_asset_features": ["momentum_5d", "momentum_10d", "adx",
                                  "ema_cross_8_20", "volatility_5d"],
        "cross_asset_prefix": "spy",
        "lgbm_params": {
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
    },
    "india": {
        "train_window": 250,
        "min_train": 100,
        "retrain_every": 20,
        "default_confidence": 0.60,
        "cross_asset_symbol": "NIFTYBEES.NS",
        "cross_asset_features": ["momentum_5d", "momentum_10d", "adx",
                                  "ema_cross_8_20", "volatility_5d"],
        "cross_asset_prefix": "nifty",
        "lgbm_params": {
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.02,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 25,
            "reg_alpha": 0.3, "reg_lambda": 2.0,
        },
    },
}


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build trend-following feature matrix from OHLCV DataFrame.

    These features are designed to capture SUSTAINED directional moves:
    - Momentum indicators (direction + acceleration)
    - Trend strength (ADX, efficiency ratio)
    - Moving average relationships (crossovers, slopes)
    - Volatility expansion (BB squeeze → breakout)
    - Volume confirmation of trend

    Target: 5-bar forward return direction (not 1-bar like mean-rev).
    """
    import pandas_ta as ta

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    open_ = df["Open"]
    ret_1d = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # === Momentum (core trend signal) ===
    features["momentum_3d"] = close.pct_change(3)
    features["momentum_5d"] = close.pct_change(5)
    features["momentum_10d"] = close.pct_change(10)
    features["momentum_20d"] = close.pct_change(20)
    features["momentum_30d"] = close.pct_change(30)
    features["momentum_accel_5d"] = features["momentum_5d"] - features["momentum_5d"].shift(5)
    features["momentum_accel_10d"] = features["momentum_10d"] - features["momentum_10d"].shift(10)

    # === EMA crossovers (trend direction) ===
    ema_8 = ta.ema(close, length=8)
    ema_20 = ta.ema(close, length=20)
    ema_50 = ta.ema(close, length=50)

    if ema_8 is not None and ema_20 is not None:
        features["ema_cross_8_20"] = (ema_8 - ema_20) / ema_20
        features["ema_8_slope"] = (ema_8 - ema_8.shift(3)) / close
        features["ema_20_slope"] = (ema_20 - ema_20.shift(5)) / close
    if ema_50 is not None and ema_20 is not None:
        features["ema_cross_20_50"] = (ema_20 - ema_50) / ema_50
    if ema_50 is not None:
        features["close_vs_ema_50"] = (close - ema_50) / ema_50
        features["ema_50_slope"] = (ema_50 - ema_50.shift(10)) / close

    # === ADX (trend strength — THE key trend indicator) ===
    adx_result = ta.adx(high, low, close, length=14)
    if adx_result is not None and not adx_result.empty:
        adx_cols = adx_result.columns
        features["adx"] = adx_result[adx_cols[0]]
        features["plus_di"] = adx_result[adx_cols[1]]
        features["minus_di"] = adx_result[adx_cols[2]]
        features["di_spread"] = features["plus_di"] - features["minus_di"]
        features["adx_rising"] = (features["adx"] > features["adx"].shift(3)).astype(float)

    # === MACD (trend momentum) ===
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        features["macd_line"] = macd.iloc[:, 0] / close
        features["macd_hist"] = macd.iloc[:, 1] / close
        features["macd_signal"] = macd.iloc[:, 2] / close
        features["macd_hist_accel"] = features["macd_hist"] - features["macd_hist"].shift(3)

    # === Bollinger Band squeeze (volatility contraction → breakout) ===
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        cols = bb.columns
        bb_width = (bb[cols[2]] - bb[cols[0]]) / close
        features["bb_width"] = bb_width
        features["bb_width_pctile"] = bb_width.rolling(50).rank(pct=True)
        features["bb_squeeze"] = (bb_width < bb_width.rolling(50).quantile(0.2)).astype(float)
        features["bb_pct"] = (close - bb[cols[0]]) / (bb[cols[2]] - bb[cols[0]]).replace(0, np.nan)
        features["bb_breakout_up"] = (close > bb[cols[2]]).astype(float)
        features["bb_breakout_down"] = (close < bb[cols[0]]).astype(float)

    # === Donchian channel (Turtle Trading breakout) ===
    features["donchian_20_high"] = close.rolling(20).max()
    features["donchian_20_low"] = close.rolling(20).min()
    features["donchian_breakout_up"] = (close >= features["donchian_20_high"]).astype(float)
    features["donchian_breakout_down"] = (close <= features["donchian_20_low"]).astype(float)
    features["donchian_position"] = (close - features["donchian_20_low"]) / \
        (features["donchian_20_high"] - features["donchian_20_low"]).replace(0, np.nan)

    # === Supertrend proxy (ATR-based trend direction) ===
    atr = ta.atr(high, low, close, length=14)
    if atr is not None:
        features["atr_pct"] = atr / close
        features["atr_expansion"] = atr / atr.rolling(20).mean().replace(0, np.nan)
        upper_band = close.rolling(10).mean() + 2 * atr
        lower_band = close.rolling(10).mean() - 2 * atr
        features["above_supertrend"] = (close > lower_band).astype(float)

    # === Efficiency ratio (how "trendy" is the price action) ===
    for p in [10, 20, 30]:
        direction = abs(close - close.shift(p))
        volatility_sum = ret_1d.abs().rolling(p).sum()
        features[f"efficiency_ratio_{p}"] = direction / volatility_sum.replace(0, np.nan)

    # === Volume trend confirmation ===
    if volume is not None and not volume.empty:
        avg_vol_5 = volume.rolling(5).mean()
        avg_vol_20 = volume.rolling(20).mean()
        features["volume_ratio"] = volume / avg_vol_20.replace(0, np.nan)
        features["volume_trend"] = avg_vol_5 / avg_vol_20.replace(0, np.nan)
        features["volume_price_confirm"] = ret_1d * (volume / avg_vol_20.replace(0, np.nan))

        obv = ta.obv(close, volume)
        if obv is not None:
            obv_ema = obv.ewm(span=20, adjust=False).mean()
            features["obv_trend"] = (obv - obv_ema) / obv_ema.abs().replace(0, np.nan)
            features["obv_momentum"] = obv.pct_change(10)

    # === Volatility regime ===
    features["volatility_5d"] = ret_1d.rolling(5).std()
    features["volatility_20d"] = ret_1d.rolling(20).std()
    features["vol_expansion"] = features["volatility_5d"] / features["volatility_20d"].replace(0, np.nan)

    # === Price structure (higher highs / higher lows) ===
    features["higher_highs_5"] = (
        (close.rolling(5).max() > close.shift(5).rolling(5).max())
    ).astype(float)
    features["higher_lows_5"] = (
        (close.rolling(5).min() > close.shift(5).rolling(5).min())
    ).astype(float)
    features["higher_highs_10"] = (
        (close.rolling(10).max() > close.shift(10).rolling(10).max())
    ).astype(float)
    features["higher_lows_10"] = (
        (close.rolling(10).min() > close.shift(10).rolling(10).min())
    ).astype(float)

    # === Trend streaks ===
    features["up_streak"] = _streak(close > close.shift(1))
    features["down_streak"] = _streak(close < close.shift(1))
    features["ema_above_streak"] = _streak(close > ema_20) if ema_20 is not None else 0

    # === RSI (for trend strength, not mean-rev) ===
    features["rsi_14"] = ta.rsi(close, length=14)
    features["rsi_trend_zone"] = ((features["rsi_14"] > 50) & (features["rsi_14"] < 70)).astype(float)

    # === Returns ===
    features["return_1d"] = ret_1d
    features["return_5d"] = close.pct_change(5)

    # === Day of week ===
    features["day_of_week"] = df.index.dayofweek

    # === Target: 5-bar forward return direction (trend prediction) ===
    features["target"] = close.shift(-5).pct_change(-5) * -1
    features["target_dir"] = (features["target"] > 0).astype(int)

    return features


class TrendModel:
    """Walk-forward trend-following model — wraps feature building + SmartLGBM.

    Uses trailing stop exits (NOT fixed take-profit).
    Predicts 5-bar forward direction instead of 1-bar.
    """

    def __init__(
        self,
        market: str = "crypto",
        train_window: int | None = None,
        min_train: int | None = None,
        confidence_threshold: float | None = None,
    ):
        self.market = market
        cfg = TREND_CONFIGS.get(market, TREND_CONFIGS["us"])
        self.train_window = train_window or cfg["train_window"]
        self.min_train = min_train or cfg["min_train"]
        self.retrain_every = cfg["retrain_every"]
        self.lgbm_params = cfg["lgbm_params"]
        self.cross_asset_symbol = cfg["cross_asset_symbol"]
        self.cross_asset_features = cfg["cross_asset_features"]
        self.cross_asset_prefix = cfg["cross_asset_prefix"]
        self.default_confidence = cfg["default_confidence"]
        self.confidence_threshold = confidence_threshold or cfg["default_confidence"]
        self.model: Optional[_SmartLGBM] = None
        self.feature_cols: Optional[list[str]] = None

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build trend-following features for the given OHLCV data."""
        return build_trend_features(df)

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get feature column names excluding targets."""
        exclude = {"target", "target_dir"}
        return [c for c in df.columns if c not in exclude]

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        """Train the trend-following model."""
        self.model = _SmartLGBM(params=self.lgbm_params)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        """Get calibrated prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)

    def save(self, path: str):
        """Save model to disk."""
        if self.model:
            self.model.save(path)

    @classmethod
    def load(cls, path: str, market: str = "crypto") -> "TrendModel":
        """Load a saved model."""
        instance = cls(market=market)
        instance.model = _SmartLGBM.load(path)
        return instance
