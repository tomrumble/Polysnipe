"""Signal pipeline utilities for late-stage state-collapse detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Sequence

from src.signal_pipeline.collapse_detection import CollapseInputs, evaluate_collapse_stage


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
    regime_label: str


@dataclass(frozen=True)
class SignalDecision:
    should_trade: bool
    signal_reason: str
    veto_reason: str
    collapse_reason: str


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
    """Apply state-collapse entry rules and strict veto guards with diagnostics."""

    if inputs.seconds_remaining >= config.seconds_remaining_threshold:
        return SignalDecision(False, "none", "time_guard", "")

    collapse_decision = evaluate_collapse_stage(
        CollapseInputs(
            spread=inputs.spread,
            directional_entropy=inputs.directional_entropy,
            price_acceleration=inputs.price_acceleration,
            stability_ratio=inputs.stability_ratio,
            volatility_current=inputs.volatility_current,
            volatility_previous=inputs.volatility_previous,
            regime_label=inputs.regime_label,
        ),
        entropy_threshold=config.entropy_threshold,
        spread_threshold=config.spread_threshold,
        accel_threshold=config.accel_threshold,
        stability_ratio_threshold=config.stability_ratio_threshold,
    )

    if not collapse_decision.passed:
        veto_reason_map = {
            "entropy_not_collapsing": "entropy_guard",
            "spread_not_tight": "spread_guard",
            "acceleration_too_high": "acceleration_guard",
            "stability_insufficient": "stability_guard",
            "volatility_not_declining": "volatility_guard",
            "regime_not_supported": "regime_guard",
        }
        return SignalDecision(False, "none", veto_reason_map[collapse_decision.reason], collapse_decision.reason)

    return SignalDecision(True, "late_state_collapse", "", "")


__all__ = [
    "RegimeLabel",
    "SignalConfig",
    "SignalInputs",
    "SignalDecision",
    "classify_regime",
    "directional_entropy",
    "evaluate_signal",
]
