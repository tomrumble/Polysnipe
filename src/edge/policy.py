"""Policy layer for model-driven trade entries."""

from __future__ import annotations

from dataclasses import dataclass
import random


MIN_EXPLORATION_SAMPLES = 10_000
EXPLORATION_THRESHOLD = 0.55
EXPLOITATION_THRESHOLD = 0.90
EPSILON_EXPLORATION = 0.01


@dataclass(frozen=True)
class PolicyDecision:
    enter: bool
    probability: float


class TradingPolicy:
    def __init__(
        self,
        confidence_threshold: float = 0.97,
        dataset_size: int = 0,
        exploration_enabled: bool = True,
    ) -> None:
        self.dataset_size = dataset_size
        self.confidence_threshold = confidence_threshold
        self.exploration_enabled = exploration_enabled

    def evaluate(self, probability: float) -> PolicyDecision:
        if self.exploration_enabled and self.dataset_size < MIN_EXPLORATION_SAMPLES:
            threshold = EXPLORATION_THRESHOLD
        else:
            threshold = EXPLOITATION_THRESHOLD

        if random.random() < EPSILON_EXPLORATION:
            return PolicyDecision(True, probability)

        return PolicyDecision(probability >= threshold, probability)
