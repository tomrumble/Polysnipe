from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.edge.offline_optimize import run_offline_optimization


def test_offline_optimizer_writes_result(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "entropy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "entropy_slope": [0.0] * 6,
            "spread": [0.01] * 6,
            "volatility": [0.02] * 6,
            "volatility_slope": [0.0] * 6,
            "stability_ratio": [2.0] * 6,
            "acceleration": [0.0] * 6,
            "seconds_remaining": [10.0] * 6,
            "distance_to_boundary": [5.0] * 6,
            "regime_label": [0.0] * 6,
            "persistence_outcome": [1, 0, 1, 0, 1, 0],
            "return": [0.05, -0.02, 0.04, -0.01, 0.06, -0.03],
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="s", tz="UTC"),
            "symbol": ["BTCUSDT"] * 6,
        }
    )
    dataset_path = tmp_path / "edge.parquet"
    output_path = tmp_path / "result.json"
    dataset.to_parquet(dataset_path, index=False)

    result = run_offline_optimization(dataset_path=dataset_path, iterations=2, seed=7, output_path=output_path)

    assert output_path.exists()
    payload = pd.read_json(output_path, typ="series")
    assert payload["seed"] == 7
    assert payload["iterations"] == 2
    assert isinstance(result.params, dict)
