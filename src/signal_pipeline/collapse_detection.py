"""Collapse stage diagnostics and decision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CollapseReason(str, Enum):
    ENTROPY_NOT_COLLAPSING = "entropy_not_collapsing"
    SPREAD_NOT_TIGHT = "spread_not_tight"
    ACCELERATION_TOO_HIGH = "acceleration_too_high"
    STABILITY_INSUFFICIENT = "stability_insufficient"
    VOLATILITY_NOT_DECLINING = "volatility_not_declining"
    REGIME_NOT_SUPPORTED = "regime_not_supported"


SUPPORTED_COLLAPSE_REGIMES = {"PERSISTENT_COMPRESSION", "LATE_MARKET_FREEZE"}


@dataclass(frozen=True)
class CollapseInputs:
    spread: float
    directional_entropy: float
    price_acceleration: float
    stability_ratio: float
    volatility_current: float
    volatility_previous: float
    regime_label: str


@dataclass(frozen=True)
class CollapseDecision:
    passed: bool
    reason: str


def evaluate_collapse_stage(
    inputs: CollapseInputs,
    *,
    entropy_threshold: float,
    spread_threshold: float,
    accel_threshold: float,
    stability_ratio_threshold: float,
) -> CollapseDecision:
    """Return collapse-stage decision and precise blocker reason."""

    if inputs.regime_label not in SUPPORTED_COLLAPSE_REGIMES:
        return CollapseDecision(False, CollapseReason.REGIME_NOT_SUPPORTED.value)

    if inputs.directional_entropy >= entropy_threshold:
        return CollapseDecision(False, CollapseReason.ENTROPY_NOT_COLLAPSING.value)

    if inputs.spread >= spread_threshold:
        return CollapseDecision(False, CollapseReason.SPREAD_NOT_TIGHT.value)

    if abs(inputs.price_acceleration) >= accel_threshold:
        return CollapseDecision(False, CollapseReason.ACCELERATION_TOO_HIGH.value)

    if inputs.stability_ratio < stability_ratio_threshold:
        return CollapseDecision(False, CollapseReason.STABILITY_INSUFFICIENT.value)

    if inputs.volatility_current >= inputs.volatility_previous:
        return CollapseDecision(False, CollapseReason.VOLATILITY_NOT_DECLINING.value)

    return CollapseDecision(True, "")
