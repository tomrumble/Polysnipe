"""Closed-loop continuous research engine over historical market tape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from src.edge.cross_validation import chronological_split
from src.edge.dataset_builder import EdgeDatasetBuilder, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.model import PersistenceModel
from src.edge.policy import TradingPolicy
from src.engine.training_controller import TrainingController, TrainingLifecycleState
from src.features import extract_features
from src.labels import label_future_drift, label_persistence, label_short_horizon_move
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
    label_mode: str = "persistence"


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
        training_controller: TrainingController | None = None,
    ) -> None:
        self.tape = tape
        self.dataset_builder = dataset_builder or EdgeDatasetBuilder()
        self.retrain_interval = int(retrain_interval)
        self.horizon_ticks = int(horizon_ticks)
        self.min_training_samples = int(min_training_samples)
        self.retrain_window_size = int(retrain_window_size)
        self.training_config = training_config or FixedTrainingConfig()
        self.enable_retrain_diagnostics = enable_retrain_diagnostics
        if training_controller is None:
            self.training_controller = TrainingController.load()
            if self.training_controller.training_state == TrainingLifecycleState.STOPPED:
                self.training_controller.start_training(reset_progress=False)
        else:
            self.training_controller = training_controller

        self.policy = TradingPolicy(
            confidence_threshold=confidence_threshold,
            mode="drift" if self.training_config.label_mode == "drift" else "persistence",
        )
        self.model = PersistenceModel(
            model_type=self.training_config.model_type,
            feature_scaling=self.training_config.feature_scaling,
            random_state=self.training_config.random_state,
            label_mode=self.training_config.label_mode,
        )
        self.state = EngineState()

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
        if self.state.observations_seen - self.state.last_retrain_observation < self.retrain_interval:
            return

        self.state.last_retrain_observation = self.state.observations_seen

        data = self.dataset_builder._load()
        if len(data) < self.min_training_samples:
            return

        if self.retrain_window_size > 0 and len(data) > self.retrain_window_size:
            data = data.tail(self.retrain_window_size).reset_index(drop=True)

        split = chronological_split(data)
        if split.validation.empty or split.test.empty or split.train.empty:
            return

        candidate = PersistenceModel(
            model_type=self.training_config.model_type,
            feature_scaling=self.training_config.feature_scaling,
            random_state=self.training_config.random_state,
            label_mode=self.training_config.label_mode,
        )
        X_train, y_train, _ = dataset_to_matrices(split.train, label_mode=candidate.label_mode)
        candidate.fit(X_train, y_train)

        new_metrics = self._evaluate(candidate, split.test)
        old_metrics = self._evaluate(self.model, split.test) if self.state.deployed_model_path else None

        improved = True
        if old_metrics is not None:
            if candidate.is_classifier:
                improved = (new_metrics["roc_auc"] >= old_metrics.get("roc_auc", 0.0)) and (
                    new_metrics["brier"] <= old_metrics.get("brier", 1.0)
                )
            else:
                improved = (new_metrics["mse"] <= old_metrics.get("mse", float("inf"))) and (
                    new_metrics["mae"] <= old_metrics.get("mae", float("inf"))
                )

        if not improved:
            return

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
            "training_config": {
                "model_type": self.training_config.model_type,
                "feature_scaling": self.training_config.feature_scaling,
                "random_state": self.training_config.random_state,
                "label_mode": self.training_config.label_mode,
            },
            "test": new_metrics,
            "observations_seen": self.state.observations_seen,
        }
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("models/model_metrics.json").write_text(json.dumps(metrics, indent=2))

        if self.enable_retrain_diagnostics:
            generate_edge_surfaces(data)

    def run(self, max_ticks: int | None = None) -> EngineState:
        if self.training_controller.training_state == TrainingLifecycleState.STOPPED:
            return self.state

        seen = 0
        while self.tape.has_next() and (max_ticks is None or seen < max_ticks):
            if self.training_controller.training_state != TrainingLifecycleState.RUNNING:
                break

            observation = self.tape.next_tick()
            features = extract_features(observation)
            signal = self.model.predict_signal(features.__dict__) if self.state.deployed_model_path else 0.5

            future_frame = self.tape.peek_future(self.horizon_ticks)
            if "close" in future_frame.columns:
                future_path = future_frame["close"].tolist()
            else:
                future_path = future_frame.get("price", pd.Series(dtype=float)).tolist()

            persistence_label = int(label_persistence(observation, future_path))
            short_move_label = int(label_short_horizon_move(observation, future_path))
            drift_10s_pct = float(label_future_drift(observation, future_path, horizon=10))

            policy_decision = (
                self.policy.evaluate(probability=float(signal), predicted_drift=drift_10s_pct)
                if self.policy.mode == "drift"
                else self.policy.evaluate(probability=float(signal))
            )

            self.dataset_builder.append(
                features,
                persistence_label,
                short_move_label=short_move_label,
                drift_10s_pct=drift_10s_pct,
                timestamp=observation.get("timestamp"),
                symbol=str(observation.get("symbol", "UNKNOWN")),
                metadata={
                    "signal": float(signal),
                    "policy_signal_score": policy_decision.signal_score,
                    "policy_side": policy_decision.side.value,
                    "policy_enter": policy_decision.enter,
                    "label_mode": self.training_config.label_mode,
                },
                trade_return=observation.get("return"),
            )

            seen += 1
            self.state.observations_seen += 1
            self.training_controller.update_progress(
                training_step=self.state.observations_seen,
                dataset_size=self.state.observations_seen,
            )
            self._maybe_retrain()

        return self.state
