"""Feature dataset persistence helpers for research and live execution modes."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.edge.dataset_builder import FEATURE_COLUMNS
from src.features import extract_features
from src.labels import label_future_drift, label_persistence


FEATURE_DATASET_DIR = Path("datasets/features")
METADATA_DIR = Path("datasets/metadata")


def load_parquet_dataset(
    path: str | Path,
    *,
    target_samples: int | None = None,
    randomized_start: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Load and optionally slice a parquet dataset for fast in-memory replay."""

    frame = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if target_samples is None or target_samples <= 0 or target_samples >= len(frame):
        return frame

    if randomized_start:
        rng = random.Random(random_seed)
        max_start = max(len(frame) - target_samples, 0)
        start_index = rng.randint(0, max_start)
        return frame.iloc[start_index : start_index + target_samples].reset_index(drop=True)

    return frame.tail(target_samples).reset_index(drop=True)


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict("records")


def build_feature_dataset(
    raw_frame: pd.DataFrame,
    *,
    horizon_ticks: int = 60,
) -> pd.DataFrame:
    """Generate feature + label rows from raw market candles."""

    if raw_frame.empty:
        return pd.DataFrame(columns=["timestamp", "price", *FEATURE_COLUMNS, "label_persistence", "label_drift"])

    frame = raw_frame.sort_values("timestamp").reset_index(drop=True)
    price_col = "close" if "close" in frame.columns else "price"
    if price_col not in frame.columns:
        raise ValueError("raw_frame must include a 'close' or 'price' column")

    closes = frame[price_col].astype(float).tolist()
    rows: list[dict[str, Any]] = []
    for idx, observation in enumerate(frame.to_dict("records")):
        future_path = closes[idx + 1 : idx + 1 + horizon_ticks]
        fv = extract_features(observation)
        rows.append(
            {
                "timestamp": observation.get("timestamp"),
                "price": float(observation.get(price_col, observation.get("price", 0.0))),
                **fv.__dict__,
                "label_persistence": int(label_persistence(observation, future_path)),
                "label_drift": float(label_future_drift(observation, future_path, horizon=10)),
            }
        )

    dataset = pd.DataFrame(rows)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce")
    return dataset.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def save_feature_dataset(
    frame: pd.DataFrame,
    dataset_name: str,
    *,
    symbol: str,
    interval: str,
    feature_version: str,
    label_mode: str,
    append: bool = True,
) -> Path:
    """Persist feature dataset parquet and sidecar metadata JSON."""

    FEATURE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = FEATURE_DATASET_DIR / f"{dataset_name}.parquet"
    combined = frame.copy()

    if append and dataset_path.exists():
        existing = pd.read_parquet(dataset_path)
        combined = pd.concat([existing, combined], ignore_index=True)

    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)

    combined.to_parquet(dataset_path, index=False)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_source": "features",
        "symbol": symbol,
        "interval": interval,
        "feature_version": feature_version,
        "label_mode": label_mode,
        "samples": int(len(combined)),
        "last_updated": datetime.now(tz=timezone.utc).isoformat(),
    }
    metadata_path = METADATA_DIR / f"{dataset_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return dataset_path


def load_feature_dataset(dataset_name: str) -> pd.DataFrame:
    return pd.read_parquet(FEATURE_DATASET_DIR / f"{dataset_name}.parquet")


def load_dataset_metadata(dataset_name: str) -> dict[str, Any]:
    metadata_path = METADATA_DIR / f"{dataset_name}.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())
