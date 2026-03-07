"""Continuous learning pipeline for self-optimising edge discovery."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.edge.cross_validation import time_based_split
from src.edge.dataset_builder import build_edge_dataset, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.model import PersistenceModel
from src.edge.optimizer import random_search_optimize


def run_edge_pipeline(telemetry: pd.DataFrame) -> dict:
    dataset = build_edge_dataset(telemetry)
    train, valid, test = time_based_split(dataset)

    opt = random_search_optimize(dataset)
    model = PersistenceModel(
        model_type=opt.params.get("model_type", "logistic"),
        feature_scaling=opt.params.get("feature_scaling", True),
    )

    X_train, y_train, _ = dataset_to_matrices(train)
    X_valid, y_valid, _ = dataset_to_matrices(valid)
    X_test, y_test, _ = dataset_to_matrices(test)

    model.fit(X_train, y_train)
    valid_probs = model.predict_probabilities(X_valid)
    test_probs = model.predict_probabilities(X_test)

    threshold = opt.params.get("probability_threshold", 0.97)
    valid_pred = (valid_probs >= threshold).astype(int)
    test_pred = (test_probs >= threshold).astype(int)

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "best_params": opt.params,
        "optimizer_score": opt.score,
        "validation": {
            "roc_auc": float(roc_auc_score(y_valid, valid_probs)) if len(set(y_valid)) > 1 else 0.5,
            "precision": float(precision_score(y_valid, valid_pred, zero_division=0)),
            "recall": float(recall_score(y_valid, valid_pred, zero_division=0)),
            "win_rate": float(y_valid.mean()),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, test_probs)) if len(set(y_test)) > 1 else 0.5,
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "win_rate": float(y_test.mean()),
        },
    }

    model_path = Path("models") / f"persistence_model_v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pkl"
    model.save(model_path)
    PersistenceModel.load(model_path).save(Path("models") / "latest_persistence_model.pkl")

    metrics_path = Path("models/model_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    diagnostics = generate_edge_surfaces(dataset)
    return {"metrics": metrics, "model_path": str(model_path), "diagnostics": diagnostics}
