"""Drop-in ML classifier wrappers with Platt calibration.

Each wrapper implements the same interface as _SmartLGBM:
  - fit(X, y, sample_weight=None)
  - predict_proba(X) -> ndarray (n_samples, n_classes)
  - save(path) / load(path)
  - feature_importances_ (property)
  - is_calibrated (property)

All classifiers use the same train/calibration split strategy:
  - 70% train, 30% calibration (chronological)
  - 5-row purge gap between train and cal
  - Platt scaling (LogisticRegression on raw scores) for calibration
"""

from __future__ import annotations

import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


class SmartClassifier(ABC):
    """Abstract base for all ML classifiers with calibration support."""

    def __init__(self, calibrate: bool = True):
        self._calibrate = calibrate
        self._calibrator = None
        self._n_features: int = 0

    @abstractmethod
    def _create_model(self):
        """Create the underlying sklearn/xgboost model instance."""
        ...

    @abstractmethod
    def _fit_model(self, X, y, sample_weight=None, eval_set=None):
        """Fit the underlying model (algorithm-specific)."""
        ...

    @abstractmethod
    def _raw_predict_proba(self, X) -> np.ndarray:
        """Raw (uncalibrated) probability predictions."""
        ...

    @property
    @abstractmethod
    def feature_importances_(self) -> np.ndarray:
        """Feature importance scores."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_calibrated(self) -> bool:
        return self._calibrator is not None

    def fit(self, X, y, sample_weight=None):
        """Fit with purged train/val split + Platt scaling calibration."""
        n = len(X)
        purge_gap = min(5, n // 20)
        cal_size = max(int(n * 0.3), 40)

        if n < 80 or cal_size < 30:
            self._fit_model(X, y, sample_weight=sample_weight)
            self._calibrator = None
            return self

        train_end = n - cal_size - purge_gap
        if train_end < 50:
            self._fit_model(X, y, sample_weight=sample_weight)
            self._calibrator = None
            return self

        X_train = X.iloc[:train_end] if hasattr(X, "iloc") else X[:train_end]
        y_train = y.iloc[:train_end] if hasattr(y, "iloc") else y[:train_end]
        X_cal = X.iloc[train_end + purge_gap:] if hasattr(X, "iloc") else X[train_end + purge_gap:]
        y_cal = y.iloc[train_end + purge_gap:] if hasattr(y, "iloc") else y[train_end + purge_gap:]

        sw_train = None
        if sample_weight is not None:
            sw_train = sample_weight[:train_end]

        self._fit_model(X_train, y_train, sample_weight=sw_train,
                        eval_set=[(X_cal, y_cal)])

        if self._calibrate and len(X_cal) >= 20:
            try:
                from sklearn.linear_model import LogisticRegression

                raw_proba = self._raw_predict_proba(X_cal)[:, 1]
                raw_logits = np.log(
                    np.clip(raw_proba, 1e-7, 1 - 1e-7)
                    / (1 - np.clip(raw_proba, 1e-7, 1 - 1e-7))
                )
                y_cal_arr = np.asarray(y_cal)
                self._calibrator = LogisticRegression(
                    C=1e10, solver="lbfgs", max_iter=1000
                )
                self._calibrator.fit(raw_logits.reshape(-1, 1), y_cal_arr)
            except Exception:
                self._calibrator = None

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated probabilities if available, else raw."""
        raw = self._raw_predict_proba(X)
        if self._calibrator is not None:
            raw_up = raw[:, 1]
            logits = np.log(
                np.clip(raw_up, 1e-7, 1 - 1e-7)
                / (1 - np.clip(raw_up, 1e-7, 1 - 1e-7))
            )
            calibrated = self._calibrator.predict_proba(logits.reshape(-1, 1))
            return calibrated
        return raw

    def predict_proba_raw(self, X) -> np.ndarray:
        """Return raw (uncalibrated) probabilities."""
        return self._raw_predict_proba(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"classifier": self}, f)

    @classmethod
    def load(cls, path: str) -> "SmartClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["classifier"]


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class SmartXGBoost(SmartClassifier):
    """XGBoost classifier with Platt calibration."""

    def __init__(self, params: dict | None = None, calibrate: bool = True):
        super().__init__(calibrate=calibrate)
        self._params = params or {}
        self.model = self._create_model()

    def _create_model(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost>=2.0.0")

        defaults = {
            "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 20,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
            "eval_metric": "logloss", "verbosity": 0,
            "use_label_encoder": False,
        }
        defaults.update(self._params)
        from xgboost import XGBClassifier
        return XGBClassifier(**defaults)

    def _fit_model(self, X, y, sample_weight=None, eval_set=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        self.model.fit(X, y, **fit_kwargs)

    def _raw_predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class SmartRandomForest(SmartClassifier):
    """Random Forest classifier with Platt calibration."""

    def __init__(self, params: dict | None = None, calibrate: bool = True):
        super().__init__(calibrate=calibrate)
        self._params = params or {}
        self.model = self._create_model()

    def _create_model(self):
        from sklearn.ensemble import RandomForestClassifier

        defaults = {
            "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20,
            "max_features": "sqrt", "random_state": 42, "n_jobs": -1,
        }
        defaults.update(self._params)
        return RandomForestClassifier(**defaults)

    def _fit_model(self, X, y, sample_weight=None, eval_set=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def _raw_predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class SmartLogistic(SmartClassifier):
    """Logistic Regression — simple baseline, inherently calibrated."""

    def __init__(self, params: dict | None = None, calibrate: bool = False):
        super().__init__(calibrate=calibrate)
        self._params = params or {}
        self.model = self._create_model()
        self._n_features = 0

    def _create_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        defaults = {
            "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42,
        }
        defaults.update(self._params)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**defaults)),
        ])

    def _fit_model(self, X, y, sample_weight=None, eval_set=None):
        fit_params = {}
        if sample_weight is not None:
            fit_params["lr__sample_weight"] = sample_weight
        self.model.fit(X, y, **fit_params)
        self._n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])

    def _raw_predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        try:
            coefs = np.abs(self.model.named_steps["lr"].coef_[0])
            return coefs / coefs.sum() if coefs.sum() > 0 else coefs
        except Exception:
            return np.zeros(self._n_features)


# ---------------------------------------------------------------------------
# MLP (Neural Network)
# ---------------------------------------------------------------------------

class SmartMLP(SmartClassifier):
    """Multi-Layer Perceptron classifier with Platt calibration."""

    def __init__(self, params: dict | None = None, calibrate: bool = True):
        super().__init__(calibrate=calibrate)
        self._params = params or {}
        self.model = self._create_model()
        self._n_features = 0

    def _create_model(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        defaults = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20,
            "random_state": 42,
        }
        defaults.update(self._params)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**defaults)),
        ])

    def _fit_model(self, X, y, sample_weight=None, eval_set=None):
        self.model.fit(X, y)
        self._n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])

    def _raw_predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        try:
            weights = self.model.named_steps["mlp"].coefs_[0]
            importance = np.abs(weights).sum(axis=1)
            return importance / importance.sum() if importance.sum() > 0 else importance
        except Exception:
            return np.zeros(self._n_features)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AVAILABLE_CLASSIFIERS = {
    "xgboost": SmartXGBoost,
    "random_forest": SmartRandomForest,
    "logistic": SmartLogistic,
    "mlp": SmartMLP,
}


def get_classifier(name: str, params: dict | None = None, calibrate: bool = True) -> SmartClassifier:
    """Factory function to create a classifier by name."""
    if name not in AVAILABLE_CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(AVAILABLE_CLASSIFIERS.keys())}")
    return AVAILABLE_CLASSIFIERS[name](params=params, calibrate=calibrate)
