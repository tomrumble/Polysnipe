"""Incremental edge dataset construction and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import FeatureVector, extract_features
from src.labels import label_persistence

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
        return data

    def append(
        self,
        features: FeatureVector,
        outcome: int,
        *,
        timestamp: Any,
        symbol: str,
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        payload = {
            **features.__dict__,
            "features": json.dumps(features.__dict__, sort_keys=True),
            "persistence_outcome": int(outcome),
            "persistence": int(outcome),
            "timestamp": pd.to_datetime(timestamp, utc=True),
            "symbol": symbol,
            "metadata": json.dumps(metadata or {}, sort_keys=True),
        }
        fresh = pd.DataFrame([payload])
        existing = self._load()
        merged = pd.concat([existing, fresh], ignore_index=True)

        dedupe_cols = ["timestamp", "symbol", "features"]
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged.to_parquet(self.dataset_path, index=False)
        return merged

    def append_from_observation(self, observation: dict[str, Any], future_path: list[float]) -> pd.DataFrame:
        fv = extract_features(observation)
        outcome = int(label_persistence(observation, future_path))
        return self.append(
            fv,
            outcome,
            timestamp=observation.get("timestamp"),
            symbol=str(observation.get("symbol", "UNKNOWN")),
            metadata={
                "entry_price": observation.get("entry_price"),
                "boundary_price": observation.get("boundary_price"),
            },
        )


def build_edge_dataset(
    telemetry: pd.DataFrame,
    dataset_path: str | Path = "datasets/edge_training_data.parquet",
    append: bool = True,
) -> pd.DataFrame:
    builder = EdgeDatasetBuilder(dataset_path=dataset_path)
    if not append and Path(dataset_path).exists():
        Path(dataset_path).unlink()

    for row in telemetry.to_dict(orient="records"):
        raw_path = row.get("price_path_until_expiry", "[]")
        future_path = json.loads(raw_path) if isinstance(raw_path, str) else list(raw_path)
        builder.append_from_observation(row, future_path)

    return builder._load()


def dataset_to_matrices(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = dataset[FEATURE_COLUMNS].copy()
    y_col = "persistence_outcome" if "persistence_outcome" in dataset.columns else "persistence"
    y = dataset[y_col].astype(int)

    metadata_cols = [c for c in ["timestamp", "symbol", "metadata", "trade_id", "entry_price", "boundary_price", "return"] if c in dataset.columns]
    metadata = dataset[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=dataset.index)
    return X, y, metadata
