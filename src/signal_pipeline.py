"""Signal pipeline utilities for late-stage state-collapse detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Sequence


class RegimeLabel(str, Enum):
    PERSISTENT_COMPRESSION = "PERSISTENT_COMPRESSION"
    VOLATILE_LIQUIDATION = "VOLATILE_LIQUIDATION"
    OSCILLATORY_NOISE = "OSCILLATORY_NOISE"
    LATE_MARKET_FREEZE = "LATE_MARKET_FREEZE"


@dataclass(frozen=True)
class SignalConfig:
    stability_ratio_threshold: float = 2.0
    entropy_threshold: float = 0.62
    accel_threshold: float = 0.45
    spread_threshold: float = 0.02
    seconds_remaining_threshold: float = 15.0


@dataclass(frozen=True)
class SignalInputs:
    seconds_remaining: float
    spread: float
    directional_entropy: float
    price_acceleration: float
    stability_ratio: float
    volatility_current: float
    volatility_previous: float


@dataclass(frozen=True)
class SignalDecision:
    should_trade: bool
    signal_reason: str
    veto_reason: str


def directional_entropy(prices: Sequence[float], window: int = 20) -> float:
    """Compute directional entropy from up/down signs of recent price changes."""

    if window <= 1:
        raise ValueError("window must be > 1")

    if len(prices) < 3:
        return 1.0

    recent = prices[-window:]
    diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    signs = [1 if d > 0 else -1 for d in diffs if d != 0]

    if not signs:
        return 0.0

    counts = Counter(signs)
    total = sum(counts.values())

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * log(p)

    return entropy


def classify_regime(
    *,
    volatility: float,
    directional_entropy_value: float,
    price_acceleration: float,
    spread: float,
    seconds_remaining: float,
) -> RegimeLabel:
    """Classify microstructure regime using current market diagnostics."""

    if seconds_remaining <= 5 and spread < 0.015 and volatility < 0.08:
        return RegimeLabel.LATE_MARKET_FREEZE

    if volatility > 0.35 or abs(price_acceleration) > 0.8:
        return RegimeLabel.VOLATILE_LIQUIDATION

    if directional_entropy_value > 0.63 or spread > 0.06:
        return RegimeLabel.OSCILLATORY_NOISE

    return RegimeLabel.PERSISTENT_COMPRESSION


def evaluate_signal(inputs: SignalInputs, config: SignalConfig) -> SignalDecision:
    """Apply state-collapse entry rules and strict veto guards."""

    if inputs.stability_ratio < config.stability_ratio_threshold:
        return SignalDecision(False, "none", "stability_guard")

    if inputs.seconds_remaining >= config.seconds_remaining_threshold:
        return SignalDecision(False, "none", "too_early")

    if inputs.spread >= config.spread_threshold:
        return SignalDecision(False, "none", "spread_guard")

    if inputs.directional_entropy >= config.entropy_threshold:
        return SignalDecision(False, "none", "entropy_guard")

    if abs(inputs.price_acceleration) >= config.accel_threshold:
        return SignalDecision(False, "none", "accel_guard")

    if inputs.volatility_current >= inputs.volatility_previous:
        return SignalDecision(False, "none", "vol_not_declining")

    return SignalDecision(True, "late_state_collapse", "")
