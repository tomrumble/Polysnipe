"""Short-horizon percentage drift label."""

from __future__ import annotations

from typing import Any, Sequence


def label_future_drift(observation: Any, future_path: Sequence[float], horizon: int = 10) -> float:
    """Return percentage price drift from entry to a bounded future horizon."""

    entry_price = float(observation["entry_price"] if isinstance(observation, dict) else getattr(observation, "entry_price"))
    if not future_path:
        return 0.0

    bounded_horizon = max(0, min(int(horizon), len(future_path) - 1))
    future_price = float(future_path[bounded_horizon])

    if entry_price == 0.0:
        return 0.0
    return float((future_price - entry_price) / entry_price)

