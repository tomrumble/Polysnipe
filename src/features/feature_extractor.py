"""Feature extraction helpers for edge discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGIME_ENCODING = {
    "PERSISTENT_COMPRESSION": 0,
    "LATE_MARKET_FREEZE": 1,
    "VOLATILE_LIQUIDATION": 2,
    "OSCILLATORY_NOISE": 3,
}


@dataclass(frozen=True)
class FeatureVector:
    entropy: float
    entropy_slope: float
    spread: float
    volatility: float
    volatility_slope: float
    stability_ratio: float
    acceleration: float
    seconds_remaining: int
    distance_to_boundary: float
    regime_label: int

    @property
    def regime(self) -> int:
        """Backward-compatible alias."""
        return self.regime_label


def _read(observation: Any, *keys: str, default: float = 0.0) -> Any:
    for key in keys:
        if isinstance(observation, dict) and key in observation:
            return observation[key]
        if hasattr(observation, key):
            return getattr(observation, key)
    return default


def extract_features(observation: Any) -> FeatureVector:
    regime_name = str(_read(observation, "regime_label", "regime", default="PERSISTENT_COMPRESSION"))
    return FeatureVector(
        entropy=float(_read(observation, "directional_entropy", "entropy", "entropy_at_entry", default=0.0)),
        entropy_slope=float(_read(observation, "entropy_velocity", "entropy_slope", "entropy_slope_before_entry", default=0.0)),
        spread=float(_read(observation, "spread", "spread_at_entry", default=0.0)),
        volatility=float(_read(observation, "volatility", "volatility_at_entry", default=0.0)),
        volatility_slope=float(_read(observation, "volatility_slope", default=0.0)),
        stability_ratio=float(_read(observation, "stability_ratio", "stability_ratio_at_entry", default=0.0)),
        acceleration=float(_read(observation, "price_acceleration", "acceleration", default=0.0)),
        seconds_remaining=int(float(_read(observation, "seconds_remaining", "seconds_to_expiry_at_entry", default=0.0))),
        distance_to_boundary=float(_read(observation, "distance_to_boundary_at_entry", "distance_to_boundary", default=0.0)),
        regime_label=REGIME_ENCODING.get(regime_name, 0),
    )
