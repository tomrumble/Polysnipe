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


def random_search_optimize(dataset: pd.DataFrame, iterations: int = 16, seed: int = 42) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    split = chronological_split(dataset)

    X_train, y_train, _ = dataset_to_matrices(split.train)
    X_valid, _, _ = dataset_to_matrices(split.validation)

    best = OptimizationResult(params={}, score=-1e12)

    for _ in range(iterations):
        params = {
            "model_type": str(rng.choice(["logistic", "gradient_boosting"])),
            "probability_threshold": float(rng.choice([0.9, 0.93, 0.95, 0.97, 0.98])),
            "feature_scaling": bool(rng.choice([True, False])),
            "lookback_window": int(rng.choice([30, 60, 120])),
        }
        model = PersistenceModel(
            model_type=params["model_type"],
            feature_scaling=params["feature_scaling"],
            random_state=seed,
        )
        model.fit(X_train, y_train)
        probs = model.predict_probabilities(X_valid)
        score = _objective(split.validation, probs, params["probability_threshold"])

        if score > best.score:
            best = OptimizationResult(params=params, score=float(score))

    return best
