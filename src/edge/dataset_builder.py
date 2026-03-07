"""Build edge training datasets from simulator telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features import extract_features
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
    "regime",
]


def build_edge_dataset(
    telemetry: pd.DataFrame,
    dataset_path: str | Path = "datasets/edge_training_data.parquet",
    append: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for row in telemetry.to_dict(orient="records"):
        path_raw = row.get("price_path_until_expiry", "[]")
        price_path = json.loads(path_raw) if isinstance(path_raw, str) else list(path_raw)
        fv = extract_features(row)
        persistence = label_persistence(row, price_path)

        rows.append(
            {
                **fv.__dict__,
                "persistence": int(persistence),
                "trade_id": row.get("trade_id", ""),
                "timestamp": row.get("timestamp"),
                "entry_price": row.get("entry_price", 0.0),
                "boundary_price": row.get("boundary_price", 0.0),
            }
        )

    new_frame = pd.DataFrame(rows)
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new_frame], ignore_index=True)
        if "trade_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["trade_id"], keep="last")
    else:
        merged = new_frame

    merged.to_parquet(path, index=False)
    return merged


def dataset_to_matrices(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["persistence"].astype(int)
    metadata = dataset[["trade_id", "timestamp", "entry_price", "boundary_price"]].copy()
    return X, y, metadata
