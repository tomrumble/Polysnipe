from src.labels import label_short_horizon_move


def test_short_horizon_label_prefers_larger_up_move() -> None:
    observation = {"entry_price": 100.0}
    future_path = [100.2, 100.5, 99.8]
    assert label_short_horizon_move(observation, future_path, horizon=3) == 1


def test_short_horizon_label_handles_empty_path() -> None:
    observation = {"entry_price": 100.0}
    assert label_short_horizon_move(observation, [], horizon=10) == 0
