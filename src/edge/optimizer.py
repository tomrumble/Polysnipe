"""Automated random-search optimisation for persistence model + policy settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.edge.cross_validation import chronological_split
from src.edge.dataset_builder import dataset_to_matrices
from src.edge.model import PersistenceModel


@dataclass(frozen=True)
class OptimizationResult:
    params: dict
    score: float


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return abs(float(drawdown.min())) if not drawdown.empty else 0.0


def _objective(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> float:
    selected = frame.copy()
    selected["probability"] = probabilities
    selected = selected[selected["probability"] >= threshold]
    if selected.empty:
        return -1e6

    if "return" not in selected.columns:
        raise RuntimeError("Optimizer requires a market-derived 'return' column.")

    returns = selected["return"]
    expected_return = float(returns.mean())
    drawdown_penalty = _max_drawdown(returns)
    return expected_return - drawdown_penalty


def random_search_optimize(
    dataset: pd.DataFrame,
    iterations: int = 16,
    seed: int = 42,
    label_mode: str = "persistence",
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    split = chronological_split(dataset)

    X_train, y_train, _ = dataset_to_matrices(split.train, label_mode=label_mode)
    X_valid, _, _ = dataset_to_matrices(split.validation, label_mode=label_mode)

    best = OptimizationResult(params={}, score=-1e12)

    classifier_models = ["logistic", "gradient_boosting_classifier", "random_forest_classifier"]
    regressor_models = ["gradient_boosting_regressor", "random_forest_regressor"]

    for _ in range(iterations):
        model_candidates = regressor_models if label_mode in {"drift", "regression", "continuous_drift"} else classifier_models
        params = {
            "model_type": str(rng.choice(model_candidates)),
            "probability_threshold": float(rng.choice([0.9, 0.93, 0.95, 0.97, 0.98])),
            "feature_scaling": bool(rng.choice([True, False])),
            "label_mode": label_mode,
            "lookback_window": int(rng.choice([30, 60, 120])),
        }
        model = PersistenceModel(
            model_type=params["model_type"],
            feature_scaling=params["feature_scaling"],
            random_state=seed,
            label_mode=label_mode,
        )
        model.fit(X_train, y_train)
        signals = model.predict_signals(X_valid)
        score = _objective(split.validation, signals, params["probability_threshold"])

        if score > best.score:
            best = OptimizationResult(params=params, score=float(score))

    return best
