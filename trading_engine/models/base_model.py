"""Base ML model components — SmartLGBM, Ensemble, and shared feature utilities.

Extracted from ml_model.py to enable separate mean-rev and trend models
while sharing the calibrated LightGBM infrastructure.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


SIMPLIFIED_FEATURES = [
    "volume_trend", "volume_ratio", "ibs", "close_vs_sma_100",
    "atr_pct", "return_1d", "return_2d", "return_3d",
    "volatility_5d", "dist_from_20d_high", "efficiency_ratio_20",
    "rsi_2", "rsi_14", "bb_pct", "day_of_week",
]


def _streak(cond: pd.Series) -> pd.Series:
    """Count consecutive True values."""
    groups = (~cond).cumsum()
    return cond.groupby(groups).cumsum()


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
    from ..config import STOCK_TO_SECTOR
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


def add_earnings_features(
    feat: pd.DataFrame, symbol: str, earnings_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add days-to-next-earnings and post-earnings-drift features."""
    if symbol not in earnings_cache or earnings_cache[symbol] is None:
        return feat
    edates = earnings_cache[symbol]
    if edates.empty:
        return feat

    earnings_dates = sorted(edates.index.tz_localize(None) if edates.index.tz else edates.index)
    days_to_earn = pd.Series(np.nan, index=feat.index)
    days_since_earn = pd.Series(np.nan, index=feat.index)

    for i, day in enumerate(feat.index):
        future = [e for e in earnings_dates if e > day]
        past = [e for e in earnings_dates if e <= day]
        if future:
            days_to_earn.iloc[i] = (future[0] - day).days
        if past:
            days_since_earn.iloc[i] = (day - past[-1]).days

    feat["days_to_earnings"] = days_to_earn.clip(upper=90)
    feat["days_since_earnings"] = days_since_earn.clip(upper=90)
    feat["near_earnings"] = (days_to_earn <= 5).astype(float)
    return feat


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
