"""Continuous learning pipeline for self-optimising edge discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score

from src.edge.cross_validation import chronological_split
from src.edge.dataset_builder import build_edge_dataset, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.model import PersistenceModel
from src.edge.optimizer import random_search_optimize


LABEL_MODE = "persistence"


def run_edge_pipeline(telemetry: pd.DataFrame) -> dict:
    dataset = build_edge_dataset(telemetry, label_mode=LABEL_MODE)
    if "return" not in dataset.columns:
        raise RuntimeError("Edge pipeline requires a market-derived 'return' column.")

    return_std = float(dataset["return"].std(ddof=0)) if not dataset.empty else 0.0
    if np.isnan(return_std) or return_std == 0.0:
        raise RuntimeError("Degenerate return distribution detected")

    split = chronological_split(dataset)

    opt = random_search_optimize(dataset, label_mode=LABEL_MODE)
    model = PersistenceModel(
        model_type=opt.params.get("model_type", "logistic"),
        feature_scaling=opt.params.get("feature_scaling", True),
        label_mode=LABEL_MODE,
    )

    X_train, y_train, _ = dataset_to_matrices(split.train, label_mode=LABEL_MODE)
    X_valid, y_valid, _ = dataset_to_matrices(split.validation, label_mode=LABEL_MODE)
    X_test, y_test, _ = dataset_to_matrices(split.test, label_mode=LABEL_MODE)

    model.fit(X_train, y_train)
    valid_signals = model.predict_signals(X_valid)
    test_signals = model.predict_signals(X_test)

    threshold = opt.params.get("probability_threshold", 0.97)

    signal_summary = {
        "min": float(np.min(test_signals)),
        "mean": float(np.mean(test_signals)),
        "max": float(np.max(test_signals)),
        "p90": float(np.percentile(test_signals, 90)),
        "p99": float(np.percentile(test_signals, 99)),
    }

    metrics = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "best_params": opt.params,
        "optimizer_score": opt.score,
        "validation": {
            **(
                {
                    "roc_auc": float(roc_auc_score(y_valid, valid_signals)) if len(set(y_valid)) > 1 else 0.5,
                    "precision": float(precision_score(y_valid, (valid_signals >= threshold).astype(int), zero_division=0)),
                    "recall": float(recall_score(y_valid, (valid_signals >= threshold).astype(int), zero_division=0)),
                    "win_rate": float(y_valid.mean()),
                }
                if model.is_classifier
                else {
                    "mse": float(mean_squared_error(y_valid, valid_signals)),
                    "mae": float(mean_absolute_error(y_valid, valid_signals)),
                    "target_mean": float(np.mean(y_valid)),
                }
            ),
        },
        "test": {
            **(
                {
                    "roc_auc": float(roc_auc_score(y_test, test_signals)) if len(set(y_test)) > 1 else 0.5,
                    "precision": float(precision_score(y_test, (test_signals >= threshold).astype(int), zero_division=0)),
                    "recall": float(recall_score(y_test, (test_signals >= threshold).astype(int), zero_division=0)),
                    "win_rate": float(y_test.mean()),
                }
                if model.is_classifier
                else {
                    "mse": float(mean_squared_error(y_test, test_signals)),
                    "mae": float(mean_absolute_error(y_test, test_signals)),
                    "target_mean": float(np.mean(y_test)),
                }
            ),
        },
        "signal_summary": signal_summary,
        "training_samples": int(len(X_train)),
        "label_mode": LABEL_MODE,
    }

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    model_path = Path("models") / f"persistence_model_v{ts}.pkl"
    model.save(model_path)
    model.save(Path("models") / "latest_persistence_model.pkl")

    metrics_path = Path("models/model_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    diagnostics = generate_edge_surfaces(dataset)
    return {"metrics": metrics, "model_path": str(model_path), "diagnostics": diagnostics}
