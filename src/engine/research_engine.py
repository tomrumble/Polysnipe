"""Closed-loop continuous research engine over historical market tape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from src.edge.cross_validation import chronological_split
from src.edge.dataset_builder import EdgeDatasetBuilder, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.model import PersistenceModel
from src.edge.optimizer import random_search_optimize
from src.edge.policy import TradingPolicy
from src.features import extract_features
from src.labels import label_drift_10s_pct, label_persistence, label_short_horizon_move
from src.tape import MarketTape
from src.engine.training_controller import TrainingController, TrainingLifecycleState


@dataclass
class EngineState:
    observations_seen: int = 0
    retrains: int = 0
    deployed_model_path: str | None = None


class ResearchEngine:
    def __init__(
        self,
        tape: MarketTape,
        dataset_builder: EdgeDatasetBuilder | None = None,
        retrain_interval: int = 10_000,
        horizon_ticks: int = 60,
        confidence_threshold: float = 0.97,
        policy_mode: str = "persistence",
        drift_threshold: float = 0.001,
    ) -> None:
        self.tape = tape
        self.dataset_builder = dataset_builder or EdgeDatasetBuilder()
        self.retrain_interval = retrain_interval
        self.horizon_ticks = horizon_ticks
        self.policy = TradingPolicy(
            confidence_threshold=confidence_threshold,
            mode=policy_mode,
            drift_threshold=drift_threshold,
        )
        self.model: PersistenceModel = PersistenceModel()
        latest_model_path = Path("models/latest_persistence_model.pkl")
        if latest_model_path.exists():
            self.model = PersistenceModel.load(latest_model_path)
        self.state = EngineState(observations_seen=self.training_controller.training_step)

    def _evaluate(self, model: PersistenceModel, test_frame: pd.DataFrame) -> dict[str, float]:
        X_test, y_test, _ = dataset_to_matrices(test_frame, label_mode=model.label_mode)
        signals = model.predict_signals(X_test)
        if model.is_classifier:
            auc = float(roc_auc_score(y_test, signals)) if len(set(y_test)) > 1 else 0.5
            brier = float(brier_score_loss(y_test, signals))
            return {"roc_auc": auc, "brier": brier, "calibration": 1.0 - brier}
        mse = float(mean_squared_error(y_test, signals))
        mae = float(mean_absolute_error(y_test, signals))
        return {"mse": mse, "mae": mae}

    def _maybe_retrain(self) -> None:
        data = self.dataset_builder._load()
        if len(data) < 100:
            return

        split = chronological_split(data)
        if split.validation.empty or split.test.empty:
            return

        opt = random_search_optimize(data, label_mode="persistence")
        candidate = PersistenceModel(
            model_type=opt.params.get("model_type", "logistic"),
            feature_scaling=opt.params.get("feature_scaling", True),
            random_state=42,
            label_mode="persistence",
        )
        X_train, y_train, _ = dataset_to_matrices(split.train, label_mode="persistence")
        candidate.fit(X_train, y_train)

        new_metrics = self._evaluate(candidate, split.test)
        old_metrics = self._evaluate(self.model, split.test) if self.state.deployed_model_path else {"roc_auc": 0.0, "brier": 1.0}

        if candidate.is_classifier:
            improved = (new_metrics["roc_auc"] > old_metrics.get("roc_auc", 0.0)) and (new_metrics["brier"] <= old_metrics.get("brier", 1.0))
        else:
            improved = (new_metrics["mse"] < old_metrics.get("mse", float("inf"))) and (new_metrics["mae"] <= old_metrics.get("mae", float("inf")))
        if improved:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
            model_path = Path("models") / f"persistence_model_v{ts}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.save(model_path)
            candidate.save(Path("models/latest_persistence_model.pkl"))
            self.model = candidate
            self.state.deployed_model_path = str(model_path)
            self.state.retrains += 1
            self.training_controller.mark_retrained(model_path.name)

            metrics = {
                "timestamp": ts,
                "optimizer_score": opt.score,
                "best_params": opt.params,
                "test": new_metrics,
            }
            Path("models/model_metrics.json").write_text(json.dumps(metrics, indent=2))
            generate_edge_surfaces(data)

    def run(self, max_ticks: int | None = None) -> EngineState:
        if self.training_controller.training_state != TrainingLifecycleState.RUNNING:
            return self.state

        seen = 0
        while self.tape.has_next() and (max_ticks is None or seen < max_ticks):
            if self.training_controller.training_state != TrainingLifecycleState.RUNNING:
                break
            observation = self.tape.next_tick()
            features = extract_features(observation)
            signal = self.model.predict_signal(features.__dict__) if self.state.deployed_model_path else 0.5
            self.policy.dataset_size = self.state.observations_seen
            policy_decision = self.policy.evaluate(probability)

            future_frame = self.tape.peek_future(self.horizon_ticks)
            future_path = future_frame["close"].tolist() if "close" in future_frame.columns else future_frame.get("price", pd.Series(dtype=float)).tolist()
            persistence_label = int(label_persistence(observation, future_path))
            short_move_label = int(label_short_horizon_move(observation, future_path))
            drift_10s_pct = float(label_drift_10s_pct(observation, future_path))

            self.dataset_builder.append(
                features,
                persistence_label,
                short_move_label=short_move_label,
                drift_10s_pct=drift_10s_pct,
                timestamp=observation.get("timestamp"),
                symbol=str(observation.get("symbol", "UNKNOWN")),
                metadata={
                    "probability": probability,
                    "policy_signal_score": policy_decision.signal_score,
                    "policy_side": policy_decision.side.value,
                    "policy_enter": policy_decision.enter,
                },
            )

            seen += 1
            self.state.observations_seen += 1
            self.training_controller.update_progress(
                training_step=self.state.observations_seen,
                dataset_size=self.state.observations_seen,
            )
            if self.state.observations_seen % self.retrain_interval == 0:
                self._maybe_retrain()

        return self.state
