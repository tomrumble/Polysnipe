"""Policy layer for model-driven trade entries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    enter: bool
    probability: float


class TradingPolicy:
    def __init__(self, confidence_threshold: float = 0.97) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(self, probability: float) -> PolicyDecision:
        return PolicyDecision(enter=probability >= self.confidence_threshold, probability=probability)
