"""Deterministic random-search optimizer for edge model settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.edge.cross_validation import time_based_split
from src.edge.dataset_builder import dataset_to_matrices
from src.edge.model import PersistenceModel


@dataclass(frozen=True)
class OptimizationResult:
    params: dict
    score: float


def _score_frame(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> float:
    selected = frame.copy()
    selected["probability"] = probabilities
    selected = selected[selected["probability"] >= threshold]
    if selected.empty:
        return -1.0
    wins = (selected["persistence"] == 1).mean()
    avg_return = selected["return"].mean() if "return" in selected.columns else (wins - (1 - wins)) * 0.01
    drawdown_penalty = 1.0 - wins
    return float(wins * avg_return - drawdown_penalty)


def random_search_optimize(dataset: pd.DataFrame, iterations: int = 12, seed: int = 42) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    train, valid, _ = time_based_split(dataset)
    X_train, y_train, _ = dataset_to_matrices(train)
    X_valid, _, _ = dataset_to_matrices(valid)

    best = OptimizationResult(params={}, score=-1e9)
    for _ in range(iterations):
        params = {
            "model_type": str(rng.choice(["logistic", "gradient_boosting"])),
            "feature_scaling": bool(rng.choice([True, False])),
            "probability_threshold": float(rng.choice([0.9, 0.93, 0.95, 0.97])),
            "lookback_window": int(rng.choice([20, 30, 40])),
            "entropy_window": int(rng.choice([10, 20, 30])),
            "volatility_window": int(rng.choice([10, 20, 30])),
        }
        model = PersistenceModel(model_type=params["model_type"], feature_scaling=params["feature_scaling"])
        model.fit(X_train, y_train)
        probs = model.predict_probabilities(X_valid)
        score = _score_frame(valid, probs, params["probability_threshold"])
        if score > best.score:
            best = OptimizationResult(params=params, score=score)
    return best
