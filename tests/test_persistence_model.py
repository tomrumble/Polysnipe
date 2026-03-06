from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence_model import PersistenceInputs, PersistenceModel


def test_compute_metrics_end_to_end():
    model = PersistenceModel(center=1.0, slope=2.0)

    inputs = PersistenceInputs(
        current_price=108.0,
        boundary_price=100.0,
        expiry_timestamp=1010.0,
        now_timestamp=1000.0,
        recent_prices=[100.0, 101.0, 102.5, 102.0, 103.0],
    )

    out = model.compute(inputs)

    assert out.volatility > 0
    assert out.distance_to_boundary == 8.0
    assert out.time_remaining == 10.0
    assert out.stability_ratio > 0
    assert 0.0 < out.persistence_probability < 1.0


def test_probability_increases_with_stability_ratio():
    model = PersistenceModel(center=1.0, slope=2.0)

    low = model.map_stability_ratio_to_probability(0.5)
    high = model.map_stability_ratio_to_probability(2.0)

    assert high > low


def test_time_remaining_is_clamped_to_zero():
    model = PersistenceModel()

    assert model.compute_time_remaining(now_timestamp=10.0, expiry_timestamp=5.0) == 0.0


def test_volatility_uses_epsilon_for_insufficient_data():
    model = PersistenceModel(epsilon=1e-6)

    assert model.compute_volatility([]) == 1e-6
    assert model.compute_volatility([100.0]) == 1e-6
