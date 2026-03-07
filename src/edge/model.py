"""Persistence probability model wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FeatureVector


class PersistenceModel:
    def __init__(self, model_type: str = "logistic", feature_scaling: bool = True, random_state: int = 42) -> None:
        self.model_type = model_type
        self.feature_scaling = feature_scaling
        self.random_state = random_state
        self._model = self._build_model()

    def _build_model(self):
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.random_state)

        estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        if self.feature_scaling:
            return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
        return estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict_probability(self, features: FeatureVector | dict | pd.DataFrame) -> float:
        if isinstance(features, FeatureVector):
            payload = pd.DataFrame([features.__dict__])
        elif isinstance(features, dict):
            payload = pd.DataFrame([features])
        else:
            payload = features
        return float(self._model.predict_proba(payload)[0, 1])

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

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
