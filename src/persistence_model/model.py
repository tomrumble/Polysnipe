"""Persistence probability model.

The core intuition is based on first-passage style scaling under short-horizon
price diffusion:

    remaining_move_capacity = volatility * sqrt(time_remaining)
    stability_ratio = distance_to_boundary / remaining_move_capacity

A larger ``stability_ratio`` means price has less room (in normalized terms) to
reach the boundary before expiry, so persistence probability should be higher.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from statistics import fmean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class PersistenceInputs:
    """Inputs used to evaluate persistence.

    Attributes:
        current_price: Latest observed market price.
        boundary_price: Price level that would invalidate persistence.
        expiry_timestamp: UNIX timestamp in seconds at which the market expires.
        now_timestamp: Current UNIX timestamp in seconds.
        recent_prices: Ordered historical prices (oldest -> newest) used to
            estimate short-horizon volatility.
    """

    current_price: float
    boundary_price: float
    expiry_timestamp: float
    now_timestamp: float
    recent_prices: Sequence[float]


@dataclass(frozen=True)
class PersistenceOutput:
    """Result of the persistence calculation."""

    volatility: float
    distance_to_boundary: float
    time_remaining: float
    stability_ratio: float
    persistence_probability: float


class PersistenceModel:
    """Model for probability that a state persists until expiry.

    The model is intentionally transparent and deterministic, prioritizing
    interpretability for strategy debugging and calibration.

    Probability mapping:
        ``p = 1 / (1 + exp(-slope * (stability_ratio - center)))``

    Where:
    - ``center`` is the stability ratio where probability equals 0.5
    - ``slope`` controls how quickly confidence transitions around ``center``
    """

    def __init__(self, center: float = 1.0, slope: float = 1.5, epsilon: float = 1e-9) -> None:
        """Initialize model calibration parameters.

        Args:
            center: Midpoint for logistic mapping from stability ratio to
                persistence probability.
            slope: Steepness of logistic mapping.
            epsilon: Small positive floor used for numerical stability.
        """

        if slope <= 0:
            raise ValueError("slope must be > 0")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self._center = center
        self._slope = slope
        self._epsilon = epsilon

    def compute(self, inputs: PersistenceInputs) -> PersistenceOutput:
        """Compute persistence metrics and probability from raw inputs."""

        volatility = self.compute_volatility(inputs.recent_prices)
        distance = self.compute_distance_to_boundary(inputs.current_price, inputs.boundary_price)
        time_remaining = self.compute_time_remaining(inputs.now_timestamp, inputs.expiry_timestamp)
        stability_ratio = self.compute_stability_ratio(distance, volatility, time_remaining)
        persistence_probability = self.map_stability_ratio_to_probability(stability_ratio)

        return PersistenceOutput(
            volatility=volatility,
            distance_to_boundary=distance,
            time_remaining=time_remaining,
            stability_ratio=stability_ratio,
            persistence_probability=persistence_probability,
        )

    def compute_volatility(self, recent_prices: Sequence[float]) -> float:
        """Estimate volatility from recent price changes.

        Uses population standard deviation of adjacent differences:

            ``changes[i] = recent_prices[i] - recent_prices[i - 1]``
            ``volatility = pstdev(changes)``

        If insufficient data is provided, returns ``epsilon`` to avoid a
        zero-denominator in downstream calculations.
        """

        if len(recent_prices) < 2:
            return self._epsilon

        changes = [recent_prices[i] - recent_prices[i - 1] for i in range(1, len(recent_prices))]
        if len(changes) == 1:
            return max(abs(changes[0]), self._epsilon)

        return max(pstdev(changes), self._epsilon)

    def compute_distance_to_boundary(self, current_price: float, boundary_price: float) -> float:
        """Compute absolute distance from current price to boundary."""

        return abs(current_price - boundary_price)

    def compute_time_remaining(self, now_timestamp: float, expiry_timestamp: float) -> float:
        """Compute non-negative remaining time to expiry (seconds)."""

        return max(expiry_timestamp - now_timestamp, 0.0)

    def compute_stability_ratio(self, distance_to_boundary: float, volatility: float, time_remaining: float) -> float:
        """Compute normalized stability ratio.

        Formula:
            ``stability_ratio = distance_to_boundary / (volatility * sqrt(time_remaining))``

        Edge handling:
            - If ``time_remaining == 0``, returns ``0`` because persistence is no
              longer a forward-looking quantity.
            - Uses ``epsilon`` floor for volatility in denominator.
        """

        if time_remaining <= 0:
            return 0.0

        denom = max(volatility, self._epsilon) * sqrt(time_remaining)
        return distance_to_boundary / max(denom, self._epsilon)

    def map_stability_ratio_to_probability(self, stability_ratio: float) -> float:
        """Map stability ratio to persistence probability via logistic curve.

        Higher stability ratio -> higher persistence probability, bounded in
        ``(0, 1)``.
        """

        x = self._slope * (stability_ratio - self._center)
        return 1.0 / (1.0 + exp(-x))

    def calibrate_center_from_samples(self, stability_ratio_samples: Sequence[float]) -> float:
        """Optional helper: recenter logistic midpoint to sample mean.

        This is useful when bootstrapping a model from historical data.
        Returns the chosen center for convenience.
        """

        if not stability_ratio_samples:
            return self._center

        self._center = fmean(stability_ratio_samples)
        return self._center
