"""Closed-loop continuous research engine over historical market tape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.edge.cross_validation import chronological_split
from src.edge.dataset_builder import EdgeDatasetBuilder, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.model import PersistenceModel
from src.edge.policy import TradingPolicy
from src.features import extract_features
from src.labels import label_persistence
from src.tape import MarketTape


@dataclass
class EngineState:
    observations_seen: int = 0
    retrains: int = 0
    last_retrain_observation: int = 0
    deployed_model_path: str | None = None


@dataclass(frozen=True)
class FixedTrainingConfig:
    model_type: str = "logistic"
    feature_scaling: bool = True
    random_state: int = 42


class ResearchEngine:
    def __init__(
        self,
        tape: MarketTape,
        dataset_builder: EdgeDatasetBuilder | None = None,
        retrain_interval: int = 10_000,
        horizon_ticks: int = 60,
        confidence_threshold: float = 0.97,
        min_training_samples: int = 100,
        retrain_window_size: int = 5_000,
        training_config: FixedTrainingConfig | None = None,
        enable_retrain_diagnostics: bool = False,
    ) -> None:
        self.tape = tape
        self.dataset_builder = dataset_builder or EdgeDatasetBuilder()
        self.retrain_interval = retrain_interval
        self.horizon_ticks = horizon_ticks
        self.min_training_samples = min_training_samples
        self.retrain_window_size = retrain_window_size
        self.training_config = training_config or FixedTrainingConfig()
        self.enable_retrain_diagnostics = enable_retrain_diagnostics
        self.policy = TradingPolicy(confidence_threshold=confidence_threshold)
        self.model: PersistenceModel = PersistenceModel(
            model_type=self.training_config.model_type,
            feature_scaling=self.training_config.feature_scaling,
            random_state=self.training_config.random_state,
        )
        self.state = EngineState()

    def _evaluate(self, model: PersistenceModel, test_frame: pd.DataFrame) -> dict[str, float]:
        X_test, y_test, _ = dataset_to_matrices(test_frame)
        probs = model.predict_probabilities(X_test)
        auc = float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else 0.5
        brier = float(brier_score_loss(y_test, probs))
        return {"roc_auc": auc, "brier": brier, "calibration": 1.0 - brier}

    def _maybe_retrain(self) -> None:
        if self.state.observations_seen - self.state.last_retrain_observation < self.retrain_interval:
            return

        self.state.last_retrain_observation = self.state.observations_seen

        data = self.dataset_builder._load()
        if len(data) < self.min_training_samples:
            return

        if self.retrain_window_size > 0 and len(data) > self.retrain_window_size:
            data = data.tail(self.retrain_window_size).reset_index(drop=True)

        split = chronological_split(data)
        if split.validation.empty or split.test.empty:
            return

        candidate = PersistenceModel(
            model_type=self.training_config.model_type,
            feature_scaling=self.training_config.feature_scaling,
            random_state=self.training_config.random_state,
        )
        X_train, y_train, _ = dataset_to_matrices(split.train)
        candidate.fit(X_train, y_train)

        new_metrics = self._evaluate(candidate, split.test)
        old_metrics = self._evaluate(self.model, split.test) if self.state.deployed_model_path else {"roc_auc": 0.0, "brier": 1.0}

        improved = (new_metrics["roc_auc"] > old_metrics.get("roc_auc", 0.0)) and (new_metrics["brier"] <= old_metrics.get("brier", 1.0))
        if improved:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
            model_path = Path("models") / f"persistence_model_v{ts}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.save(model_path)
            candidate.save(Path("models/latest_persistence_model.pkl"))
            self.model = candidate
            self.state.deployed_model_path = str(model_path)
            self.state.retrains += 1

            metrics = {
                "timestamp": ts,
                "training_config": {
                    "model_type": self.training_config.model_type,
                    "feature_scaling": self.training_config.feature_scaling,
                    "random_state": self.training_config.random_state,
                },
                "test": new_metrics,
                "observations_seen": self.state.observations_seen,
            }
            Path("models/model_metrics.json").write_text(json.dumps(metrics, indent=2))
            if self.enable_retrain_diagnostics:
                generate_edge_surfaces(data)

    def run(self, max_ticks: int | None = None) -> EngineState:
        seen = 0
        while self.tape.has_next() and (max_ticks is None or seen < max_ticks):
            observation = self.tape.next_tick()
            features = extract_features(observation)
            probability = self.model.predict_probability(features.__dict__) if self.state.deployed_model_path else 0.5
            self.policy.dataset_size = self.state.observations_seen
            _ = self.policy.evaluate(probability)

            future_frame = self.tape.peek_future(self.horizon_ticks)
            future_path = future_frame["close"].tolist() if "close" in future_frame.columns else future_frame.get("price", pd.Series(dtype=float)).tolist()
            outcome = int(label_persistence(observation, future_path))

            self.dataset_builder.append(
                features,
                outcome,
                timestamp=observation.get("timestamp"),
                symbol=str(observation.get("symbol", "UNKNOWN")),
                metadata={"probability": probability},
            )

            seen += 1
            self.state.observations_seen += 1
            self._maybe_retrain()

        return self.state
