"""ML Signal Generator — market-aware LightGBM model predicts next-day returns.

Supports multiple markets with dedicated feature sets:
  - US: Mean-reversion features (RSI oversold bounces, distance from MA)
  - India (NSE): Momentum/trend features (gap patterns, delivery volume, sector momentum)

Target: next-day close-to-close return direction (up/down)
Training: walk-forward — retrain on rolling window, predict next day
Signal: model confidence → only trade when >threshold

Usage:
    from trading_engine.ml_model import MLSignalGenerator
    ml = MLSignalGenerator(market="us")   # or market="india"
    result = ml.train_and_backtest(history_data)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


MARKET_CONFIGS = {
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


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from OHLCV DataFrame.

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
    return build_feature_matrix(df)


def add_vix_features(feat: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Add VIX level and VIX change as features."""
    if vix_df is None or vix_df.empty:
        return feat
    vix_close = vix_df["Close"].reindex(feat.index)
    feat["vix_level"] = vix_close
    feat["vix_5d_change"] = vix_close.pct_change(5)
    feat["vix_above_25"] = (vix_close > 25).astype(float)
    return feat


def add_sector_relative_features(
    feat: pd.DataFrame, symbol: str, sector_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add sector-relative return: stock return minus sector ETF return."""
    from .config import STOCK_TO_SECTOR
    sector_etf = STOCK_TO_SECTOR.get(symbol)
    if sector_etf is None or sector_etf not in sector_data:
        return feat
    sector_close = sector_data[sector_etf]["Close"].reindex(feat.index)
    for p in [1, 5, 10]:
        stock_ret = feat.get(f"return_{p}d")
        if stock_ret is None:
            continue
        sector_ret = sector_close.pct_change(p)
        feat[f"sector_rel_{p}d"] = stock_ret - sector_ret
    return feat


def _streak(cond: pd.Series) -> pd.Series:
    """Count consecutive True values."""
    groups = (~cond).cumsum()
    return cond.groupby(groups).cumsum()


class _SmartLGBM:
    """Calibrated LightGBM with Platt scaling + early stopping.

    Raw LightGBM predict_proba outputs are NOT calibrated probabilities.
    S05 showed 40% calibration error (says 75% but wins 35%).

    Fix: sigmoid calibration (Platt scaling) on a held-out validation set
    maps raw scores → real probabilities. Early stopping prevents overfitting.
    """

    def __init__(self, params: dict | None = None, calibrate: bool = True):
        import lightgbm as lgb

        defaults = {
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
        if params:
            defaults.update(params)

        defaults.setdefault("random_state", 42)
        self._lgbm_params = defaults
        self.model = lgb.LGBMClassifier(
            **defaults, verbose=-1,
        )
        self._calibrator = None
        self._calibrate = calibrate
        self._raw_model = None

    def fit(self, X, y, sample_weight=None):
        """Fit with purged train/val split + Platt scaling calibration.

        Split: 70% train, 30% calibration (last 30% chronologically).
        The split is purged — a 5-row gap between train and cal to avoid leakage.
        Early stopping on the calibration set prevents overfitting.
        Platt scaling: LogisticRegression on raw log-odds → calibrated probs.
        """
        n = len(X)
        purge_gap = min(5, n // 20)
        cal_size = max(int(n * 0.3), 40)

        if n < 80 or cal_size < 30:
            self.model.fit(X, y, sample_weight=sample_weight)
            self._raw_model = self.model
            self._calibrator = None
            return self

        train_end = n - cal_size - purge_gap
        if train_end < 50:
            self.model.fit(X, y, sample_weight=sample_weight)
            self._raw_model = self.model
            self._calibrator = None
            return self

        X_train = X.iloc[:train_end] if hasattr(X, 'iloc') else X[:train_end]
        y_train = y.iloc[:train_end] if hasattr(y, 'iloc') else y[:train_end]
        X_cal = X.iloc[train_end + purge_gap:] if hasattr(X, 'iloc') else X[train_end + purge_gap:]
        y_cal = y.iloc[train_end + purge_gap:] if hasattr(y, 'iloc') else y[train_end + purge_gap:]

        sw_train = None
        if sample_weight is not None:
            sw_train = sample_weight[:train_end]

        import lightgbm as lgb
        callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]
        self.model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_cal, y_cal)],
            callbacks=callbacks,
        )
        self._raw_model = self.model

        if self._calibrate and len(X_cal) >= 20:
            try:
                from sklearn.linear_model import LogisticRegression

                raw_proba = self.model.predict_proba(X_cal)[:, 1]
                raw_logits = np.log(np.clip(raw_proba, 1e-7, 1 - 1e-7) /
                                     (1 - np.clip(raw_proba, 1e-7, 1 - 1e-7)))
                y_cal_arr = np.asarray(y_cal)

                self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
                self._calibrator.fit(raw_logits.reshape(-1, 1), y_cal_arr)
            except Exception:
                self._calibrator = None

        return self

    def predict_proba(self, X):
        """Return calibrated probabilities if available, else raw."""
        raw = self.model.predict_proba(X)
        if self._calibrator is not None:
            raw_up = raw[:, 1]
            logits = np.log(np.clip(raw_up, 1e-7, 1 - 1e-7) /
                             (1 - np.clip(raw_up, 1e-7, 1 - 1e-7)))
            calibrated = self._calibrator.predict_proba(logits.reshape(-1, 1))
            return calibrated
        return raw

    def predict_proba_raw(self, X):
        """Return raw (uncalibrated) probabilities for comparison."""
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator is not None

    def save(self, path: str):
        """Save model + calibrator to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "calibrator": self._calibrator}, f)

    @classmethod
    def load(cls, path: str) -> "_SmartLGBM":
        """Load model + calibrator from disk."""
        import pickle
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            instance.model = data["model"]
            instance._calibrator = data.get("calibrator")
        else:
            instance.model = data
            instance._calibrator = None
        instance._raw_model = instance.model
        return instance


class _SmartLGBMEnsemble:
    """Ensemble of 3 _SmartLGBM models with different random seeds.

    Averages calibrated probabilities across models. Reduces variance
    and overconfident single-model predictions.
    """

    SEEDS = [42, 123, 777]

    def __init__(self, params: dict | None = None, calibrate: bool = True):
        self._base_params = params or {}
        self._calibrate = calibrate
        self._models: list[_SmartLGBM] = []

    def fit(self, X, y, sample_weight=None):
        self._models = []
        for seed in self.SEEDS:
            p = dict(self._base_params)
            p["random_state"] = seed
            m = _SmartLGBM(params=p, calibrate=self._calibrate)
            m.fit(X, y, sample_weight=sample_weight)
            self._models.append(m)
        return self

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self._models]
        return np.mean(probas, axis=0)

    @property
    def feature_importances_(self):
        imps = [m.feature_importances_ for m in self._models]
        return np.mean(imps, axis=0)

    @property
    def is_calibrated(self) -> bool:
        return any(m.is_calibrated for m in self._models)

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"models": self._models}, f)

    @classmethod
    def load(cls, path: str) -> "_SmartLGBMEnsemble":
        import pickle
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance._models = data["models"]
        return instance


class MLSignalGenerator:
    """Walk-forward LightGBM signal generator — market-aware.

    Supports 'us' and 'india' markets with dedicated feature sets,
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
