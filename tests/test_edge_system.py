from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.edge.dataset_builder import build_edge_dataset, dataset_to_matrices
from src.edge.model import PersistenceModel
from src.edge.pipeline import run_edge_pipeline
from src.edge.policy import TradingPolicy
from src.features import extract_features


def sample_telemetry() -> pd.DataFrame:
    rows = []
    for i in range(60):
        rows.append(
            {
                "trade_id": f"t-{i}",
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
                "entry_price": 100.0,
                "boundary_price": 102.5,
                "directional_entropy": 0.2 + (i % 5) * 0.02,
                "entropy_velocity": -0.1 if i % 2 == 0 else -0.02,
                "spread": 0.01 + (i % 3) * 0.001,
                "volatility": 0.03 + (i % 4) * 0.01,
                "volatility_slope": -0.01,
                "stability_ratio": 2.0 + (i % 4) * 0.4,
                "price_acceleration": 0.01,
                "seconds_remaining": 10,
                "distance_to_boundary_at_entry": 2.5,
                "regime_label": "PERSISTENT_COMPRESSION",
                "return": 0.01 if i % 3 else -0.01,
                "trade_outcome": "WIN" if i % 3 else "LOSS",
                "price_path_until_expiry": "[100.1,100.2,100.3]" if i % 3 else "[101.0,102.6,102.7]",
            }
        )
    return pd.DataFrame(rows)


def test_feature_extractor():
    fv = extract_features({"directional_entropy": 0.3, "regime_label": "LATE_MARKET_FREEZE"})
    assert fv.entropy == 0.3
    assert fv.regime == 1


def test_dataset_builder(tmp_path: Path):
    ds_path = tmp_path / "edge.parquet"
    data = build_edge_dataset(sample_telemetry(), dataset_path=ds_path, append=False)
    assert ds_path.exists()
    X, y, meta = dataset_to_matrices(data)
    assert not X.empty
    assert len(y) == len(meta)


def test_model_training_and_probability(tmp_path: Path):
    data = build_edge_dataset(sample_telemetry(), dataset_path=tmp_path / "edge.parquet", append=False)
    X, y, _ = dataset_to_matrices(data)
    model = PersistenceModel(model_type="logistic", feature_scaling=True)
    model.fit(X, y)
    p = model.predict_probability(X.iloc[[0]])
    assert 0.0 <= p <= 1.0


def test_policy_decision():
    policy = TradingPolicy(confidence_threshold=0.97)
    assert policy.evaluate(0.99).enter is True
    assert policy.evaluate(0.5).enter is False


def test_pipeline_stability(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("models").mkdir(parents=True, exist_ok=True)
    result = run_edge_pipeline(sample_telemetry())
    assert "metrics" in result
    assert Path(result["model_path"]).exists()
    assert Path("models/model_metrics.json").exists()
