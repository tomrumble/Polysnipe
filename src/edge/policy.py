"""Policy layer for model-driven trade entries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


MIN_EXPLORATION_SAMPLES = 10_000
EXPLORATION_THRESHOLD = 0.55
EXPLOITATION_THRESHOLD = 0.90
EPSILON_EXPLORATION = 0.01


class PolicySide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass(frozen=True)
class PolicyDecision:
    enter: bool
    signal_score: float
    side: PolicySide

    @property
    def probability(self) -> float:
        """Backward-compatible alias for persistence-mode score."""
        return self.signal_score


class TradingPolicy:
    def __init__(
        self,
        confidence_threshold: float = 0.97,
        dataset_size: int = 0,
        exploration_enabled: bool = True,
        mode: str = "persistence",
        drift_threshold: float = 0.001,
    ) -> None:
        self.dataset_size = dataset_size
        self.confidence_threshold = confidence_threshold
        self.exploration_enabled = exploration_enabled
        self.mode = mode
        self.drift_threshold = drift_threshold

    def evaluate(self, probability: float, predicted_drift: float = 0.0) -> PolicyDecision:
        if self.mode == "drift":
            if predicted_drift > self.drift_threshold:
                return PolicyDecision(True, predicted_drift, PolicySide.LONG)
            if predicted_drift < -self.drift_threshold:
                return PolicyDecision(True, predicted_drift, PolicySide.SHORT)
            return PolicyDecision(False, predicted_drift, PolicySide.NONE)

        if self.exploration_enabled and self.dataset_size < MIN_EXPLORATION_SAMPLES:
            threshold = EXPLORATION_THRESHOLD
        else:
            threshold = EXPLOITATION_THRESHOLD

        if random.random() < EPSILON_EXPLORATION:
            return PolicyDecision(True, probability, PolicySide.LONG)

        should_enter = probability >= threshold
        side = PolicySide.LONG if should_enter else PolicySide.NONE
        return PolicyDecision(should_enter, probability, side)
