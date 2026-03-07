"""Persistence probability model wrapper."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FeatureVector


class PersistenceModel:
    CLASSIFIER_MODEL_TYPES = {
        "logistic",
        "gradient_boosting",
        "gradient_boosting_classifier",
        "random_forest_classifier",
    }
    REGRESSOR_MODEL_TYPES = {
        "gradient_boosting_regressor",
        "random_forest_regressor",
    }

    def __init__(
        self,
        model_type: str = "logistic",
        feature_scaling: bool = True,
        random_state: int = 42,
        label_mode: str = "persistence",
    ) -> None:
        self.model_type = model_type
        self.feature_scaling = feature_scaling
        self.random_state = random_state
        self.label_mode = label_mode
        self.task_type = self._infer_task_type()
        self._model = self._build_model()

    @property
    def is_classifier(self) -> bool:
        return self.task_type == "classifier"

    def _infer_task_type(self) -> str:
        if self.model_type in self.CLASSIFIER_MODEL_TYPES:
            return "classifier"
        if self.model_type in self.REGRESSOR_MODEL_TYPES:
            return "regressor"
        return "regressor" if self.label_mode in {"drift", "regression", "continuous_drift"} else "classifier"

    def _build_model(self):
        if self.model_type in {"gradient_boosting", "gradient_boosting_classifier"}:
            return GradientBoostingClassifier(random_state=self.random_state)
        if self.model_type == "random_forest_classifier":
            return RandomForestClassifier(random_state=self.random_state)
        if self.model_type == "gradient_boosting_regressor":
            return GradientBoostingRegressor(random_state=self.random_state)
        if self.model_type == "random_forest_regressor":
            return RandomForestRegressor(random_state=self.random_state)

        estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        if self.feature_scaling:
            return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
        return estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def _to_frame(self, features: FeatureVector | dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(features, FeatureVector):
            return pd.DataFrame([features.__dict__])
        elif isinstance(features, dict):
            return pd.DataFrame([features])
        return features

    def predict_signal(self, features: FeatureVector | dict | pd.DataFrame) -> float:
        payload = self._to_frame(features)
        if self.is_classifier:
            return float(self._model.predict_proba(payload)[0, 1])
        return float(self._model.predict(payload)[0])

    def predict_signals(self, X: pd.DataFrame) -> np.ndarray:
        if self.is_classifier:
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def predict_probability(self, features: FeatureVector | dict | pd.DataFrame) -> float:
        warnings.warn("predict_probability() is deprecated; use predict_signal()", DeprecationWarning, stacklevel=2)
        return self.predict_signal(features)

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        warnings.warn("predict_probabilities() is deprecated; use predict_signals()", DeprecationWarning, stacklevel=2)
        return self.predict_signals(X)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PersistenceModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    @property
    def feature_importance_(self):
        model = self._model
        if hasattr(model, "named_steps"):
            model = model.named_steps["estimator"]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "coef_"):
            return model.coef_[0]
        return None
