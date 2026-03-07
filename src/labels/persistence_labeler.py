"""Outcome labeling for persistence training targets."""

from __future__ import annotations

from typing import Any, Sequence


def label_persistence(observation: Any, future_path: Sequence[float]) -> bool:
    """Persistence is true if price never crosses boundary before expiry."""

    boundary_price = float(observation.get("boundary_price", 0.0)) if isinstance(observation, dict) else float(getattr(observation, "boundary_price", 0.0))
    entry_price = float(observation.get("entry_price", 0.0)) if isinstance(observation, dict) else float(getattr(observation, "entry_price", 0.0))
    side_up = boundary_price >= entry_price

    if side_up:
        return all(float(price) < boundary_price for price in future_path)
    return all(float(price) > boundary_price for price in future_path)
