"""Mean-reversion ML model — features and configs optimized for quick reversals.

Features predict: oversold/overbought snaps back within 1-3 bars.
Exit style: fixed take-profit (NOT trailing stop).
Best for: RSI(2) extremes, IBS extremes, BB band touches, volume spikes.

Extracted from ml_model.py — identical feature output for backward compatibility.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base_model import _SmartLGBM, _streak

warnings.filterwarnings("ignore", category=UserWarning)


MEAN_REV_CONFIGS = {
    "us": {
        "train_window": 300,
        "min_train": 100,
        "retrain_every": 20,
        "default_confidence": 0.60,
        "cross_asset_symbol": "SPY",
        "cross_asset_features": ["rsi_2", "rsi_14", "ibs", "return_1d",
                                  "return_5d", "volatility_5d", "close_vs_sma_100"],
        "cross_asset_prefix": "spy",
        "lgbm_params": {
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
    },
    "crypto": {
        "train_window": 150,
        "min_train": 60,
        "retrain_every": 10,
        "default_confidence": 0.70,
        "cross_asset_symbol": "BTC-USD",
        "cross_asset_features": ["rsi_2", "rsi_14", "ibs", "return_1d",
                                  "return_5d", "volatility_5d"],
        "cross_asset_prefix": "btc",
        "model_type": "logistic",
        "logistic_C": 0.15,
        "default_sl": 0.10,
        "default_tp": 0.15,
        "default_max_pos": 0.15,
        "lgbm_params": {
            "n_estimators": 100, "max_depth": 2, "learning_rate": 0.02,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 15,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
    },
    "india": {
        "train_window": 200,
        "min_train": 80,
        "retrain_every": 20,
        "default_confidence": 0.80,
        "cross_asset_symbol": "NIFTYBEES.NS",
        "cross_asset_features": ["rsi_2", "rsi_14", "ibs", "return_1d",
                                  "return_5d", "volatility_5d", "close_vs_sma_100",
                                  "momentum_10d", "gap"],
        "cross_asset_prefix": "nifty",
        "lgbm_params": {
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.02,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 25,
            "reg_alpha": 0.3, "reg_lambda": 2.0,
        },
    },
}

# Backward-compat alias — old code uses MARKET_CONFIGS
MARKET_CONFIGS = MEAN_REV_CONFIGS


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build mean-reversion feature matrix from OHLCV DataFrame.

    Core: ~30 proven v1 features (technical indicators).
    Enhancement: +5 features that ranked in absolute top-5 across all v2 runs.
    Total: ~35 features — optimal signal-to-noise ratio for 200-day training.
    """
    import pandas_ta as ta

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    features = pd.DataFrame(index=df.index)

    features["rsi_2"] = ta.rsi(close, length=2)
    features["rsi_3"] = ta.rsi(close, length=3)
    features["rsi_14"] = ta.rsi(close, length=14)

    features["ibs"] = (close - low) / (high - low).replace(0, np.nan)

    for p in [5, 8, 20, 50]:
        sma = ta.sma(close, length=p)
        if sma is not None:
            features[f"close_vs_sma_{p}"] = (close - sma) / sma
        else:
            features[f"close_vs_sma_{p}"] = np.nan

    sma_100 = ta.sma(close, length=100)
    if sma_100 is not None:
        features["close_vs_sma_100"] = (close - sma_100) / sma_100
        features["above_sma_100"] = (close > sma_100).astype(int)
    else:
        features["close_vs_sma_100"] = np.nan
        features["above_sma_100"] = np.nan

    for p in [8, 20]:
        ema = ta.ema(close, length=p)
        features[f"close_vs_ema_{p}"] = (close - ema) / ema

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        features["macd_hist"] = macd.iloc[:, 1]
        features["macd_hist_norm"] = macd.iloc[:, 1] / close

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        cols = bb.columns
        features["bb_pct"] = (close - bb[cols[0]]) / (bb[cols[2]] - bb[cols[0]]).replace(0, np.nan)

    features["atr_pct"] = ta.atr(high, low, close, length=14) / close

    features["return_1d"] = close.pct_change(1)
    features["return_2d"] = close.pct_change(2)
    features["return_3d"] = close.pct_change(3)
    features["return_5d"] = close.pct_change(5)

    features["volatility_5d"] = close.pct_change().rolling(5).std()
    features["volatility_20d"] = close.pct_change().rolling(20).std()

    if volume is not None and not volume.empty:
        avg_vol = volume.rolling(20).mean()
        features["volume_ratio"] = volume / avg_vol.replace(0, np.nan)

    features["day_of_week"] = df.index.dayofweek

    features["high_low_range"] = (high - low) / close

    for i in range(1, 4):
        features[f"lower_high_{i}"] = (high < high.shift(1)).rolling(i).sum()
        features[f"lower_low_{i}"] = (low < low.shift(1)).rolling(i).sum()

    features["up_streak"] = _streak(close > close.shift(1))
    features["down_streak"] = _streak(close < close.shift(1))

    features["dist_from_20d_high"] = (close - close.rolling(20).max()) / close
    features["dist_from_20d_low"] = (close - close.rolling(20).min()) / close

    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    features["vol_ratio_5_20"] = close.pct_change().rolling(5).std() / close.pct_change().rolling(20).std().replace(0, np.nan)

    features["gap"] = (df["Open"] - close.shift(1)) / close.shift(1)

    # === v2 additions: top-5 new features proven across all backtest runs ===
    avg_vol_5 = volume.rolling(5).mean() if volume is not None else None
    avg_vol_20 = volume.rolling(20).mean() if volume is not None else None
    if avg_vol_5 is not None and avg_vol_20 is not None:
        features["volume_trend"] = avg_vol_5 / avg_vol_20.replace(0, np.nan)

    obv = ta.obv(close, volume)
    if obv is not None:
        obv_sma = obv.rolling(20).mean()
        features["obv_trend"] = (obv - obv_sma) / obv_sma.replace(0, np.nan)

    ret_1d = close.pct_change()
    for p in [10, 20]:
        direction = abs(close - close.shift(p))
        volatility_sum = ret_1d.abs().rolling(p).sum()
        features[f"efficiency_ratio_{p}"] = direction / volatility_sum.replace(0, np.nan)

    features["target"] = close.shift(-1).pct_change(-1) * -1
    features["target_dir"] = (features["target"] > 0).astype(int)

    return features


def build_feature_matrix_india(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix optimized for Indian (NSE) stocks.

    Indian market characteristics vs US:
    - Momentum-driven (trends persist, less mean-reversion)
    - Gap-up/gap-down patterns are strong signals (pre-market news)
    - Volume spikes indicate institutional activity (FII/DII)
    - Sector rotation is pronounced
    - Intraday range relative to gap is informative

    ~40 features tuned for Indian large-cap and ETF dynamics.
    """
    import pandas_ta as ta

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    open_ = df["Open"]
    ret_1d = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # === Momentum indicators (Indian stocks trend more than US ETFs) ===
    features["rsi_2"] = ta.rsi(close, length=2)
    features["rsi_14"] = ta.rsi(close, length=14)
    features["momentum_5d"] = close.pct_change(5)
    features["momentum_10d"] = close.pct_change(10)
    features["momentum_20d"] = close.pct_change(20)
    features["momentum_accel"] = features["momentum_5d"] - features["momentum_5d"].shift(5)

    # === IBS ===
    features["ibs"] = (close - low) / (high - low).replace(0, np.nan)

    # === Moving averages (use SMA100 not SMA200 — saves 100 warmup days) ===
    for p in [5, 20, 50]:
        sma = ta.sma(close, length=p)
        if sma is not None:
            features[f"close_vs_sma_{p}"] = (close - sma) / sma
        else:
            features[f"close_vs_sma_{p}"] = np.nan
    sma_100 = ta.sma(close, length=100)
    if sma_100 is not None:
        features["close_vs_sma_100"] = (close - sma_100) / sma_100
    else:
        features["close_vs_sma_100"] = np.nan
    features["sma_20_slope"] = (ta.sma(close, length=20) - ta.sma(close, length=20).shift(5)) / close

    ema_8 = ta.ema(close, length=8)
    ema_20 = ta.ema(close, length=20)
    features["close_vs_ema_8"] = (close - ema_8) / ema_8
    features["ema_8_20_cross"] = (ema_8 - ema_20) / ema_20

    # === Gap analysis (critical for Indian markets — pre-market news driven) ===
    features["gap"] = (open_ - close.shift(1)) / close.shift(1)
    features["gap_abs"] = abs(features["gap"])
    features["gap_direction"] = (features["gap"] > 0).astype(float)
    features["gap_fill_ratio"] = np.where(
        features["gap"] > 0,
        (high - open_) / (open_ - close.shift(1)).replace(0, np.nan),
        (open_ - low) / (close.shift(1) - open_).replace(0, np.nan),
    )
    features["gap_fill_ratio"] = features["gap_fill_ratio"].clip(-5, 5)
    features["consecutive_gap_up"] = _streak(features["gap"] > 0.002)
    features["consecutive_gap_down"] = _streak(features["gap"] < -0.002)

    # === MACD ===
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        features["macd_hist_norm"] = macd.iloc[:, 1] / close

    # === Bollinger %B ===
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        cols = bb.columns
        features["bb_pct"] = (close - bb[cols[0]]) / (bb[cols[2]] - bb[cols[0]]).replace(0, np.nan)

    # === Volatility (Indian stocks are more volatile) ===
    features["atr_pct"] = ta.atr(high, low, close, length=14) / close
    features["volatility_5d"] = ret_1d.rolling(5).std()
    features["volatility_20d"] = ret_1d.rolling(20).std()
    features["vol_expansion"] = features["volatility_5d"] / features["volatility_20d"].replace(0, np.nan)

    # === Returns ===
    features["return_1d"] = ret_1d
    features["return_2d"] = close.pct_change(2)
    features["return_3d"] = close.pct_change(3)
    features["return_5d"] = close.pct_change(5)

    # === Volume (institutional activity proxy — FII/DII impact) ===
    if volume is not None and not volume.empty:
        avg_vol_5 = volume.rolling(5).mean()
        avg_vol_20 = volume.rolling(20).mean()
        features["volume_ratio"] = volume / avg_vol_20.replace(0, np.nan)
        features["volume_trend"] = avg_vol_5 / avg_vol_20.replace(0, np.nan)
        features["volume_price_confirm"] = ret_1d * features["volume_ratio"]
        features["volume_spike"] = (volume > avg_vol_20 * 2).astype(float)

    # === OBV trend (strong for Indian delivery-based moves) ===
    obv = ta.obv(close, volume)
    if obv is not None:
        obv_sma = obv.rolling(20).mean()
        features["obv_trend"] = (obv - obv_sma) / obv_sma.replace(0, np.nan)

    # === Price action ===
    features["high_low_range"] = (high - low) / close
    features["body_size"] = abs(close - open_) / close
    features["upper_shadow_ratio"] = (high - close.clip(lower=open_)) / (high - low).replace(0, np.nan)
    features["lower_shadow_ratio"] = (close.clip(upper=open_) - low) / (high - low).replace(0, np.nan)

    # === Streaks ===
    features["up_streak"] = _streak(close > close.shift(1))
    features["down_streak"] = _streak(close < close.shift(1))

    # === Distance from highs/lows ===
    features["dist_from_20d_high"] = (close - close.rolling(20).max()) / close
    features["dist_from_20d_low"] = (close - close.rolling(20).min()) / close
    features["dist_from_50d_high"] = (close - close.rolling(50).max()) / close

    # === Efficiency ratio (trend strength) ===
    for p in [10, 20]:
        direction = abs(close - close.shift(p))
        volatility_sum = ret_1d.abs().rolling(p).sum()
        features[f"efficiency_ratio_{p}"] = direction / volatility_sum.replace(0, np.nan)

    # === Regime detection (bull/bear market filter) ===
    sma_50 = ta.sma(close, length=50)
    if sma_50 is not None:
        features["regime_bull"] = (close > sma_50).astype(float)
        features["regime_trend_strength"] = (close - sma_50) / sma_50
        sma_20_r = ta.sma(close, length=20)
        features["regime_sma20_above_50"] = (sma_20_r > sma_50).astype(float) if sma_20_r is not None else np.nan
    else:
        features["regime_bull"] = np.nan
        features["regime_trend_strength"] = np.nan
        features["regime_sma20_above_50"] = np.nan
    features["regime_higher_highs"] = (
        (close.rolling(10).max() > close.shift(10).rolling(10).max())
    ).astype(float)
    features["regime_higher_lows"] = (
        (close.rolling(10).min() > close.shift(10).rolling(10).min())
    ).astype(float)

    # === FII/DII proxy: Accumulation/Distribution & smart money flow ===
    ad_line = ta.ad(high, low, close, volume)
    if ad_line is not None:
        ad_sma = ad_line.rolling(20).mean()
        features["ad_trend"] = (ad_line - ad_sma) / ad_sma.abs().replace(0, np.nan)
        features["ad_divergence"] = (
            (close.pct_change(10) > 0) & (ad_line.pct_change(10) < 0)
        ).astype(float) - (
            (close.pct_change(10) < 0) & (ad_line.pct_change(10) > 0)
        ).astype(float)

    mfi = ta.mfi(high, low, close, volume, length=14)
    if mfi is not None:
        features["mfi"] = mfi
        features["mfi_overbought"] = (mfi > 80).astype(float)
        features["mfi_oversold"] = (mfi < 20).astype(float)

    # === Market breadth proxy (stock's relative strength vs own history) ===
    features["rel_strength_20d"] = close.pct_change(20).rank(pct=True)
    features["rel_strength_60d"] = close.pct_change(60).rank(pct=True)

    # === Institutional volume patterns ===
    if volume is not None and not volume.empty:
        vol_change = volume.pct_change()
        price_up = (ret_1d > 0).astype(float)
        features["smart_money_flow"] = (vol_change * price_up).rolling(10).sum()
        features["vol_on_up_days"] = (volume * price_up).rolling(20).mean() / \
            (volume * (1 - price_up)).rolling(20).mean().replace(0, np.nan)

    # === Day of week (Monday effect, expiry Thursdays) ===
    features["day_of_week"] = df.index.dayofweek
    features["is_thursday"] = (df.index.dayofweek == 3).astype(float)
    features["is_monday"] = (df.index.dayofweek == 0).astype(float)

    # === Month-end/start effect (FII rebalancing) ===
    features["is_month_start"] = (df.index.day <= 5).astype(float)
    features["is_month_end"] = (df.index.day >= 25).astype(float)

    features["target"] = close.shift(-1).pct_change(-1) * -1
    features["target_dir"] = (features["target"] > 0).astype(int)

    return features


def build_features_for_market(df: pd.DataFrame, market: str = "us") -> pd.DataFrame:
    """Dispatch to the right feature builder based on market."""
    if market == "india":
        return build_feature_matrix_india(df)
    return build_feature_matrix(df)  # 'us' and 'crypto' share same features


class MeanRevModel:
    """Walk-forward mean-reversion model — wraps feature building + SmartLGBM.

    Replaces the generic MLSignalGenerator for mean-reversion trading.
    Uses fixed take-profit exits (NOT trailing stops).
    """

    def __init__(
        self,
        market: str = "crypto",
        train_window: int | None = None,
        min_train: int | None = None,
        confidence_threshold: float | None = None,
    ):
        self.market = market
        cfg = MEAN_REV_CONFIGS.get(market, MEAN_REV_CONFIGS["us"])
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
        """Build mean-reversion features for the given OHLCV data."""
        return build_features_for_market(df, self.market)

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get feature column names excluding targets."""
        exclude = {"target", "target_dir"}
        return [c for c in df.columns if c not in exclude]

    def train(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        """Train the mean-reversion model."""
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
    def load(cls, path: str, market: str = "crypto") -> "MeanRevModel":
        """Load a saved model."""
        instance = cls(market=market)
        instance.model = _SmartLGBM.load(path)
        return instance
