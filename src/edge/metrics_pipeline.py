"""Metrics service for edge diagnostics and dashboard refresh artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: float | int | np.floating | None, default: float = 0.0) -> float:
    if value is None:
        return default
    out = float(value)
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _signal_series(telemetry: pd.DataFrame) -> pd.Series:
    if "persistence_probability" not in telemetry.columns or telemetry.empty:
        return pd.Series(dtype=float)
    signal = pd.to_numeric(telemetry["persistence_probability"], errors="coerce").dropna()
    return signal.astype(float)


def _trade_outcome_series(telemetry: pd.DataFrame) -> pd.Series:
    if "trade_outcome" not in telemetry.columns or telemetry.empty:
        return pd.Series(dtype=float)
    outcome = telemetry["trade_outcome"].map({"WIN": 1.0, "LOSS": 0.0})
    return pd.to_numeric(outcome, errors="coerce").dropna().astype(float)


def _drift_distribution(signal: pd.Series, bins: int = 10) -> float:
    if len(signal) < 40:
        return 0.0
    split = len(signal) // 2
    baseline = signal.iloc[:split]
    recent = signal.iloc[split:]
    if baseline.empty or recent.empty:
        return 0.0

    counts_base, edges = np.histogram(baseline, bins=bins, range=(0.0, 1.0), density=False)
    counts_recent, _ = np.histogram(recent, bins=edges, density=False)

    eps = 1e-6
    p = (counts_base + eps) / (np.sum(counts_base) + eps * bins)
    q = (counts_recent + eps) / (np.sum(counts_recent) + eps * bins)
    psi = np.sum((q - p) * np.log(q / p))
    return _safe_float(float(psi))


def _signal_histogram(signal: pd.Series, bins: int = 10) -> list[dict[str, float | int]]:
    counts, edges = np.histogram(signal, bins=bins, range=(0.0, 1.0), density=False)
    histogram: list[dict[str, float | int]] = []
    for idx, count in enumerate(counts):
        histogram.append(
            {
                "bin_start": _safe_float(edges[idx]),
                "bin_end": _safe_float(edges[idx + 1]),
                "count": int(count),
            }
        )
    return histogram


def build_metrics_payload(
    *,
    telemetry: pd.DataFrame,
    traded: pd.DataFrame,
    edge_score: float,
    edge_status: str,
    calibration_error: float,
    spearman_rank_correlation: float,
    expected_value: float,
    model_version: str,
    dataset_size: int,
) -> dict[str, Any]:
    """Build the latest metrics payload for persistence and visualization."""

    signal = _signal_series(telemetry)
    outcomes = _trade_outcome_series(telemetry)

    trade_selectivity = _safe_float(len(traded) / max(len(telemetry), 1))

    signal_mean = _safe_float(signal.mean() if not signal.empty else 0.0)
    signal_std = _safe_float(signal.std(ddof=0) if not signal.empty else 0.0)
    signal_percentiles = {
        "p5": _safe_float(np.percentile(signal, 5)) if not signal.empty else 0.0,
        "p25": _safe_float(np.percentile(signal, 25)) if not signal.empty else 0.0,
        "p50": _safe_float(np.percentile(signal, 50)) if not signal.empty else 0.0,
        "p75": _safe_float(np.percentile(signal, 75)) if not signal.empty else 0.0,
        "p95": _safe_float(np.percentile(signal, 95)) if not signal.empty else 0.0,
    }

    common = telemetry[["persistence_probability", "return"]].dropna() if {"persistence_probability", "return"}.issubset(telemetry.columns) else pd.DataFrame()
    if len(common) > 1 and common["persistence_probability"].nunique() > 1 and common["return"].nunique() > 1:
        drift_prediction_correlation = _safe_float(common["persistence_probability"].corr(common["return"]))
    else:
        drift_prediction_correlation = 0.0

    if not signal.empty and not outcomes.empty:
        joint = telemetry[["persistence_probability", "trade_outcome"]].copy()
        joint["target"] = joint["trade_outcome"].map({"WIN": 1.0, "LOSS": 0.0})
        joint = joint.dropna(subset=["persistence_probability", "target"])
        rmse = _safe_float(np.sqrt(np.mean((joint["persistence_probability"] - joint["target"]) ** 2))) if not joint.empty else 0.0
    else:
        rmse = 0.0

    expected_value_recent = _safe_float(traded["return"].tail(200).mean()) if not traded.empty and "return" in traded.columns else 0.0

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "training_samples": int(len(telemetry)),
        "dataset_size": int(dataset_size),
        "model_version": model_version,
        "edge_score": _safe_float(edge_score),
        "edge_status": edge_status,
        "expected_value_recent": expected_value_recent,
        "trade_selectivity": trade_selectivity,
        "calibration_error": _safe_float(calibration_error),
        "spearman_rank_correlation": _safe_float(spearman_rank_correlation),
        "drift_distribution": _drift_distribution(signal),
        "drift_prediction_correlation": drift_prediction_correlation,
        "drift_prediction_rmse": rmse,
        "drift_signal_histogram": _signal_histogram(signal),
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "signal_percentiles": signal_percentiles,
    }


def persist_metrics(
    payload: dict[str, Any],
    *,
    metrics_path: Path = Path("metrics.json"),
    max_history: int = 300,
) -> dict[str, Any]:
    """Persist latest metrics and compact history for dashboard charting."""

    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    history = existing.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(payload)
    history = history[-max_history:]

    artifact = {
        "latest": payload,
        "history": history,
    }
    metrics_path.write_text(json.dumps(artifact, indent=2))
    return artifact


def load_metrics(metrics_path: Path = Path("metrics.json")) -> dict[str, Any]:
    """Load persisted metrics artifact."""

    if not metrics_path.exists():
        return {"latest": {}, "history": []}
    try:
        content = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return {"latest": {}, "history": []}
    if not isinstance(content, dict):
        return {"latest": {}, "history": []}
    return {
        "latest": content.get("latest", {}),
        "history": content.get("history", []),
    }
