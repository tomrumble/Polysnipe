"""Alternative short-horizon drift label for edge discovery."""

from __future__ import annotations

from typing import Any, Sequence


def _entry_price(observation: Any) -> float:
    """Current/entry price from observation; fallback for tape rows that have close/open but not entry_price."""
    if isinstance(observation, dict):
        v = observation.get("entry_price") or observation.get("close") or observation.get("open") or observation.get("price", 0.0)
        return float(v)
    return float(getattr(observation, "entry_price", None) or getattr(observation, "close", 0.0))


def label_short_horizon_move(observation: Any, future_path: Sequence[float], horizon: int = 10) -> int:
    entry = _entry_price(observation)
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

    entry = _entry_price(observation)
    future = [float(price) for price in future_path[:horizon]]

    if not future or entry == 0.0:
        return 0.0

    horizon_price = future[-1]
    return float((horizon_price - entry) / entry)
