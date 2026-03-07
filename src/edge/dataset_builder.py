"""Incremental edge dataset construction and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import FeatureVector, extract_features
from src.labels import label_future_drift, label_persistence, label_short_horizon_move

FEATURE_COLUMNS = [
    "entropy",
    "entropy_slope",
    "spread",
    "volatility",
    "volatility_slope",
    "stability_ratio",
    "acceleration",
    "seconds_remaining",
    "distance_to_boundary",
    "regime_label",
]

TARGET_COLUMNS = ["persistence_label", "short_move_label", "drift_10s_pct"]


class EdgeDatasetBuilder:
    def __init__(self, dataset_path: str | Path = "datasets/edge_training_data.parquet") -> None:
        self.dataset_path = Path(dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            return pd.DataFrame()
        data = pd.read_parquet(self.dataset_path)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        if "persistence_label" not in data.columns:
            if "persistence_outcome" in data.columns:
                data["persistence_label"] = pd.to_numeric(data["persistence_outcome"], errors="coerce")
            elif "persistence" in data.columns:
                data["persistence_label"] = pd.to_numeric(data["persistence"], errors="coerce")
        return data

    def append(
        self,
        features: FeatureVector,
        persistence_label: int,
        *,
        short_move_label: int,
        drift_10s_pct: float,
        timestamp: Any,
        symbol: str,
        metadata: dict[str, Any] | None = None,
        trade_return: float | None = None,
    ) -> pd.DataFrame:
        payload = {
            **features.__dict__,
            "features": json.dumps(features.__dict__, sort_keys=True),
            "persistence_label": int(persistence_label),
            "short_move_label": int(short_move_label),
            "drift_10s_pct": float(drift_10s_pct),
            # Backward-compat aliases used by older pipeline/tests.
            "persistence_outcome": int(persistence_label),
            "persistence": int(persistence_label),
            "timestamp": pd.to_datetime(timestamp, utc=True),
            "symbol": symbol,
            "metadata": json.dumps(metadata or {}, sort_keys=True),
        }
        if trade_return is not None:
            payload["return"] = float(trade_return)

        fresh = pd.DataFrame([payload])
        existing = self._load()
        merged = pd.concat([existing, fresh], ignore_index=True)

        merged = merged.drop_duplicates(subset=["timestamp", "symbol", "features"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged = merged.replace([np.inf, -np.inf], np.nan)

        required_cols = FEATURE_COLUMNS + TARGET_COLUMNS + ["timestamp"]
        present_required = [col for col in required_cols if col in merged.columns]
        merged = merged.dropna(subset=present_required)

        merged[FEATURE_COLUMNS] = merged[FEATURE_COLUMNS].astype(float)
        merged["persistence_label"] = merged["persistence_label"].astype(int)
        merged["short_move_label"] = merged["short_move_label"].astype(int)
        merged["drift_10s_pct"] = merged["drift_10s_pct"].astype(float)

        merged.to_parquet(self.dataset_path, index=False)
        return merged

    def append_from_observation(
        self,
        observation: dict[str, Any],
        future_path: list[float],
        *,
        label_mode: str = "persistence",
    ) -> pd.DataFrame:
        fv = extract_features(observation)
        persistence_label = int(label_persistence(observation, future_path))
        short_move_label = int(label_short_horizon_move(observation, future_path))
        drift_10s_pct = float(label_future_drift(observation, future_path, horizon=10))

        if label_mode == "persistence":
            target: float | int = persistence_label
        elif label_mode == "short_move":
            target = short_move_label
        elif label_mode == "drift":
            target = drift_10s_pct
        else:
            raise ValueError(f"Unsupported label_mode '{label_mode}'")

        raw_return = observation.get("return")
        trade_return = float(raw_return) if raw_return is not None and pd.notna(raw_return) else None

        return self.append(
            fv,
            persistence_label,
            short_move_label=short_move_label,
            drift_10s_pct=drift_10s_pct,
            timestamp=observation.get("timestamp"),
            symbol=str(observation.get("symbol", "UNKNOWN")),
            metadata={
                "entry_price": observation.get("entry_price"),
                "boundary_price": observation.get("boundary_price"),
                "label_mode": label_mode,
                "target": target,
            },
            trade_return=trade_return,
        )


def build_edge_dataset(
    telemetry: pd.DataFrame,
    dataset_path: str | Path = "datasets/edge_training_data.parquet",
    append: bool = True,
    label_mode: str = "persistence",
) -> pd.DataFrame:
    builder = EdgeDatasetBuilder(dataset_path=dataset_path)
    if not append and Path(dataset_path).exists():
        Path(dataset_path).unlink()

    for row in telemetry.to_dict(orient="records"):
        raw_path = row.get("price_path_until_expiry", "[]")
        future_path = json.loads(raw_path) if isinstance(raw_path, str) else list(raw_path)
        builder.append_from_observation(row, future_path, label_mode=label_mode)

    return builder._load()


def dataset_to_matrices(
    dataset: pd.DataFrame,
    label_mode: str = "persistence",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = dataset[FEATURE_COLUMNS].copy()

    label_map = {
        "persistence": ["persistence_label", "persistence_outcome", "persistence"],
        "short_move": ["short_move_label"],
        "drift": ["drift_10s_pct"],
    }
    if label_mode not in label_map:
        raise ValueError(f"Unsupported label_mode '{label_mode}'")

    y_col = next((candidate for candidate in label_map[label_mode] if candidate in dataset.columns), None)
    if y_col is None:
        raise KeyError(f"No target column available for label_mode '{label_mode}'")

    y = pd.to_numeric(dataset[y_col], errors="coerce")
    y = y.astype(float if label_mode == "drift" else int)

    metadata_cols = [
        c
        for c in [
            "timestamp",
            "symbol",
            "metadata",
            "trade_id",
            "entry_price",
            "boundary_price",
            "return",
            "persistence_label",
            "short_move_label",
            "drift_10s_pct",
        ]
        if c in dataset.columns
    ]
    metadata = dataset[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=dataset.index)
    return X, y, metadata
