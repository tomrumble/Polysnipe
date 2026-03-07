"""Short-horizon percentage drift label."""

from __future__ import annotations

from typing import Any, Sequence


def _entry_price(observation: Any) -> float:
    """Current/entry price from observation; fallback for tape rows that have close/open but not entry_price."""
    if isinstance(observation, dict):
        v = observation.get("entry_price") or observation.get("close") or observation.get("open") or observation.get("price", 0.0)
        return float(v)
    return float(getattr(observation, "entry_price", None) or getattr(observation, "close", 0.0))


def label_future_drift(observation: Any, future_path: Sequence[float], horizon: int = 10) -> float:
    """Return percentage price drift from entry to a bounded future horizon."""

    entry_price = _entry_price(observation)
    if not future_path:
        return 0.0

    bounded_horizon = max(0, min(int(horizon), len(future_path) - 1))
    future_price = float(future_path[bounded_horizon])

    if entry_price == 0.0:
        return 0.0
    return float((future_price - entry_price) / entry_price)

