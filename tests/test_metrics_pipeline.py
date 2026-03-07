from pathlib import Path

import pandas as pd

from src.edge.metrics_pipeline import build_metrics_payload, load_metrics, persist_metrics


def _sample_telemetry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "persistence_probability": [0.52, 0.67, 0.81, 0.93, 0.74, 0.61],
            "trade_outcome": ["WIN", "LOSS", "WIN", "WIN", "LOSS", "WIN"],
            "return": [0.01, -0.02, 0.015, 0.02, -0.01, 0.008],
        }
    )


def test_build_metrics_payload_contains_required_fields() -> None:
    telemetry = _sample_telemetry()
    traded = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()

    payload = build_metrics_payload(
        telemetry=telemetry,
        traded=traded,
        edge_score=0.72,
        edge_status="Promising Edge",
        calibration_error=0.04,
        spearman_rank_correlation=0.35,
        expected_value=0.005,
        model_version="latest_persistence_model.pkl",
        dataset_size=len(telemetry),
    )

    expected_keys = {
        "training_samples",
        "dataset_size",
        "model_version",
        "edge_score",
        "edge_status",
        "expected_value_recent",
        "trade_selectivity",
        "calibration_error",
        "spearman_rank_correlation",
        "drift_distribution",
        "drift_prediction_correlation",
        "drift_prediction_rmse",
        "drift_signal_histogram",
        "signal_mean",
        "signal_std",
        "signal_percentiles",
    }
    assert expected_keys.issubset(payload.keys())
    assert len(payload["drift_signal_histogram"]) == 10


def test_persist_and_load_metrics_history(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    first = {
        "timestamp": "2025-01-01T00:00:00+00:00",
        "training_samples": 10,
    }
    second = {
        "timestamp": "2025-01-01T00:01:00+00:00",
        "training_samples": 11,
    }

    persist_metrics(first, metrics_path=metrics_path, max_history=5)
    artifact = persist_metrics(second, metrics_path=metrics_path, max_history=5)

    assert artifact["latest"]["training_samples"] == 11
    assert len(artifact["history"]) == 2

    loaded = load_metrics(metrics_path)
    assert loaded["latest"]["training_samples"] == 11
    assert loaded["history"][0]["training_samples"] == 10
