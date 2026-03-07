from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.training_engine import RuntimeState, TrainingEngine
from src.tape.market_tape import MarketTape


def test_training_engine_step_consumes_one_preloaded_observation(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="s", tz="UTC"),
            "close": [100, 101, 102, 103, 104],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, index=False)

    tape = MarketTape(raw_path)
    engine = TrainingEngine(tape=tape, retrain_interval=1000, metric_interval=1000)

    records = []
    for i, ts in enumerate(pd.date_range("2024-01-01", periods=3, freq="s", tz="UTC")):
        records.append(
            {
                "timestamp": ts,
                "price": 100 + i,
                "entropy": 0.1,
                "entropy_slope": 0.0,
                "spread": 0.01,
                "volatility": 0.1,
                "volatility_slope": 0.0,
                "stability_ratio": 2.0,
                "acceleration": 0.0,
                "seconds_remaining": 10,
                "distance_to_boundary": 1.0,
                "regime_label": 0,
                "label_persistence": 1,
                "label_drift": 0.0,
                "close": 100 + i,
                "open": 100 + i,
                "high": 100 + i,
                "low": 100 + i,
            }
        )

    engine.load_dataset(records, precomputed_features=True)
    engine.start()

    snap_1 = engine.step()
    snap_2 = engine.step()

    assert engine.state.runtime_state == RuntimeState.RUNNING
    assert snap_1 is not None and snap_2 is not None
    assert snap_1.step_index == 1
    assert snap_2.step_index == 2
    assert engine.cursor == 2
