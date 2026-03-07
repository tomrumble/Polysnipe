from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_dataset_store import load_parquet_dataset, save_feature_dataset


def test_load_parquet_dataset_random_slice(tmp_path: Path):
    path = tmp_path / "raw.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="s", tz="UTC"),
            "close": [float(i) for i in range(100)],
        }
    )
    frame.to_parquet(path, index=False)

    sliced = load_parquet_dataset(path, target_samples=20, randomized_start=True, random_seed=7)

    assert len(sliced) == 20
    assert sliced["timestamp"].is_monotonic_increasing


def test_save_feature_dataset_appends_and_dedupes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    df_one = pd.DataFrame(
        {
            "timestamp": [base, base + pd.Timedelta(seconds=1)],
            "price": [100.0, 101.0],
            "entropy": [0.1, 0.2],
            "entropy_slope": [0.0, 0.1],
            "spread": [0.01, 0.01],
            "volatility": [0.1, 0.1],
            "volatility_slope": [0.0, 0.0],
            "stability_ratio": [2.0, 2.1],
            "acceleration": [0.0, 0.0],
            "seconds_remaining": [10, 9],
            "distance_to_boundary": [1.0, 1.0],
            "regime_label": [0, 0],
            "label_persistence": [1, 0],
            "label_drift": [0.01, -0.01],
        }
    )
    df_two = pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=1), base + pd.Timedelta(seconds=2)],
            "price": [101.5, 102.0],
            "entropy": [0.25, 0.3],
            "entropy_slope": [0.1, 0.2],
            "spread": [0.01, 0.01],
            "volatility": [0.1, 0.1],
            "volatility_slope": [0.0, 0.0],
            "stability_ratio": [2.2, 2.3],
            "acceleration": [0.0, 0.0],
            "seconds_remaining": [8, 7],
            "distance_to_boundary": [1.0, 1.0],
            "regime_label": [0, 0],
            "label_persistence": [1, 1],
            "label_drift": [0.02, 0.03],
        }
    )

    path = save_feature_dataset(df_one, "btc_features_v1", symbol="BTCUSDT", interval="1s", feature_version="v1", label_mode="drift", append=True)
    save_feature_dataset(df_two, "btc_features_v1", symbol="BTCUSDT", interval="1s", feature_version="v1", label_mode="drift", append=True)

    saved = pd.read_parquet(path)
    assert len(saved) == 3
    assert saved["timestamp"].is_monotonic_increasing
    metadata = Path("datasets/metadata/btc_features_v1.json")
    assert metadata.exists()
