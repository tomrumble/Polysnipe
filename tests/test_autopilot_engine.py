from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.binance_ingestor import BinanceIngestor
from src.engine import ResearchEngine, TrainingController
from src.tape import MarketTape


def test_binance_ingestor_appends_deduplicates_and_orders(tmp_path: Path):
    ingestor = BinanceIngestor(base_dir=tmp_path)

    frame_a = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"]),
            "open": [1.0, 1.1],
            "high": [1.2, 1.2],
            "low": [0.9, 1.0],
            "close": [1.1, 1.15],
            "volume": [10.0, 11.0],
            "close_time_ms": [0, 0],
            "quote_asset_volume": [0.0, 0.0],
            "trade_count": [1, 1],
            "taker_buy_base_volume": [0.0, 0.0],
            "taker_buy_quote_volume": [0.0, 0.0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
        }
    )
    frame_b = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:01Z", "2024-01-01T00:00:02Z"]),
            "open": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [1.0, 1.1],
            "close": [1.15, 1.25],
            "volume": [11.0, 12.0],
            "close_time_ms": [0, 0],
            "quote_asset_volume": [0.0, 0.0],
            "trade_count": [1, 1],
            "taker_buy_base_volume": [0.0, 0.0],
            "taker_buy_quote_volume": [0.0, 0.0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
        }
    )

    calls = {"n": 0}

    def fake_fetch(*args, **kwargs):
        calls["n"] += 1
        return frame_a if calls["n"] == 1 else frame_b

    ingestor._fetch_range = fake_fetch  # type: ignore[method-assign]
    ingestor.ingest("BTCUSDT", pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:03Z"))
    path = ingestor.ingest("BTCUSDT", pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:03Z"))

    stored = pd.read_parquet(path)
    assert len(stored) == 3
    assert stored["timestamp"].is_monotonic_increasing


def test_market_tape_and_research_engine_loop(tmp_path: Path):
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=140, freq="s", tz="UTC"),
            "close": [100 + i * 0.01 for i in range(140)],
            "entry_price": [100.0] * 140,
            "boundary_price": [105.0] * 140,
            "directional_entropy": [0.2] * 140,
            "entropy_velocity": [0.0] * 140,
            "spread": [0.01] * 140,
            "volatility": [0.03] * 140,
            "volatility_slope": [0.0] * 140,
            "stability_ratio": [2.0] * 140,
            "price_acceleration": [0.0] * 140,
            "seconds_remaining": [10] * 140,
            "distance_to_boundary": [5.0] * 140,
            "regime_label": ["PERSISTENT_COMPRESSION"] * 140,
            "symbol": ["BTCUSDT"] * 140,
        }
    )
    tape_path = tmp_path / "tape.parquet"
    market.to_parquet(tape_path, index=False)

    tape = MarketTape(tape_path)
    controller = TrainingController.load(tmp_path / "training_state.json")
    controller.start_training(reset_progress=False)
    engine = ResearchEngine(tape=tape, retrain_interval=1000, training_controller=controller)
    state = engine.run(max_ticks=120)

    assert state.observations_seen == 120
    assert Path("datasets/edge_training_data.parquet").exists()


def test_training_engine_runtime_states_and_intervals(tmp_path: Path):
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="s", tz="UTC"),
            "close": [100 + i * 0.01 for i in range(40)],
            "entry_price": [100.0] * 40,
            "boundary_price": [101.0] * 40,
            "directional_entropy": [0.2] * 40,
            "entropy_velocity": [0.0] * 40,
            "spread": [0.01] * 40,
            "volatility": [0.03] * 40,
            "volatility_slope": [0.0] * 40,
            "stability_ratio": [2.0] * 40,
            "price_acceleration": [0.0] * 40,
            "seconds_remaining": [10] * 40,
            "distance_to_boundary": [1.0] * 40,
            "regime_label": ["PERSISTENT_COMPRESSION"] * 40,
            "symbol": ["BTCUSDT"] * 40,
        }
    )
    tape_path = tmp_path / "runtime_tape.parquet"
    market.to_parquet(tape_path, index=False)

    dataset_path = tmp_path / "edge_training_data.parquet"
    builder = EdgeDatasetBuilder(dataset_path=dataset_path)
    tape = MarketTape(tape_path)
    engine = TrainingEngine(
        tape=tape,
        dataset_builder=builder,
        retrain_interval=10,
        metric_interval=5,
        horizon_ticks=5,
        poll_interval_seconds=0.0,
    )

    retrain_calls = {"n": 0}

    def fake_retrain() -> None:
        retrain_calls["n"] += 1

    engine._maybe_retrain = fake_retrain  # type: ignore[method-assign]

    paused_state = engine.run(max_iterations=2)
    assert paused_state.runtime_state == RuntimeState.PAUSED
    assert paused_state.dataset_size == 0

    engine.start()
    running_state = engine.run(max_iterations=15)
    assert running_state.observations_seen == 15
    assert running_state.dataset_size == 15
    assert retrain_calls["n"] == 1
    assert running_state.latest_metrics.get("dataset_size") == 15

    engine.pause()
    paused_again = engine.run(max_iterations=1)
    assert paused_again.dataset_size == 15

    engine.stop()
    stopped = engine.run(max_iterations=1)
    assert stopped.runtime_state == RuntimeState.STOPPED
