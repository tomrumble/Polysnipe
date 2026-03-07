"""Outcome labeling for persistence training targets."""

from __future__ import annotations

from typing import Sequence


def label_persistence(observation: dict, price_path_to_expiry: Sequence[float]) -> bool:
    """Return True when price never breaches boundary before expiry."""

    boundary_price = float(observation.get("boundary_price", 0.0))
    entry_price = float(observation.get("entry_price", 0.0))
    boundary_side = 1.0 if boundary_price >= entry_price else -1.0

    if boundary_side > 0:
        return all(float(price) < boundary_price for price in price_path_to_expiry)
    return all(float(price) > boundary_price for price in price_path_to_expiry)
