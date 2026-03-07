"""Alternative short-horizon drift label for edge discovery."""

from __future__ import annotations

from typing import Any, Sequence


def label_short_horizon_move(observation: Any, future_path: Sequence[float], horizon: int = 10) -> int:
    entry = float(observation["entry_price"] if isinstance(observation, dict) else getattr(observation, "entry_price"))
    future = [float(price) for price in future_path[:horizon]]

    if not future:
        return 0

    max_move = max(future) - entry
    min_move = entry - min(future)

    if max_move > min_move:
        return 1
    return 0


def label_drift_10s_pct(observation: Any, future_path: Sequence[float], horizon: int = 10) -> float:
    """10-second drift target as percent move from entry to horizon price."""

    entry = float(observation["entry_price"] if isinstance(observation, dict) else getattr(observation, "entry_price"))
    future = [float(price) for price in future_path[:horizon]]

    if not future or entry == 0.0:
        return 0.0

    horizon_price = future[-1]
    return float((horizon_price - entry) / entry)
