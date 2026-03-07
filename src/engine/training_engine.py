"""Runtime training engine for continuous market-event processing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.edge.cross_validation import chronological_split
from src.edge.dataset_builder import EdgeDatasetBuilder, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.model import PersistenceModel
from src.edge.optimizer import random_search_optimize
from src.features import extract_features
from src.labels import label_persistence, label_short_horizon_move
from src.tape import MarketTape


class RuntimeState(str, Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class TrainingEngineState:
    runtime_state: RuntimeState = RuntimeState.PAUSED
    observations_seen: int = 0
    retrains: int = 0
    dataset_size: int = 0
    deployed_model_path: str | None = None
    latest_metrics: dict[str, float | int] = field(default_factory=dict)


class TrainingEngine:
    def __init__(
        self,
        tape: MarketTape,
        dataset_builder: EdgeDatasetBuilder | None = None,
        *,
        retrain_interval: int = 1000,
        metric_interval: int = 100,
        horizon_ticks: int = 60,
        label_mode: str = "persistence",
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self.tape = tape
        self.dataset_builder = dataset_builder or EdgeDatasetBuilder()
        self.retrain_interval = retrain_interval
        self.metric_interval = metric_interval
        self.horizon_ticks = horizon_ticks
        self.label_mode = label_mode
        self.poll_interval_seconds = poll_interval_seconds
        self.model = PersistenceModel()
        self.state = TrainingEngineState(dataset_size=len(self.dataset_builder._load()))

    def start(self) -> None:
        self.state.runtime_state = RuntimeState.RUNNING

    def pause(self) -> None:
        if self.state.runtime_state != RuntimeState.STOPPED:
            self.state.runtime_state = RuntimeState.PAUSED

    def stop(self) -> None:
        self.state.runtime_state = RuntimeState.STOPPED

    def _evaluate(self, model: PersistenceModel, test_frame: pd.DataFrame) -> dict[str, float]:
        X_test, y_test, _ = dataset_to_matrices(test_frame)
        probs = model.predict_probabilities(X_test)
        auc = float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else 0.5
        brier = float(brier_score_loss(y_test, probs))
        return {"roc_auc": auc, "brier": brier, "calibration": 1.0 - brier}

    def _recompute_metrics(self) -> None:
        data = self.dataset_builder._load()
        size = int(len(data))
        self.state.dataset_size = size
        if data.empty:
            self.state.latest_metrics = {"dataset_size": 0}
            return

        y_col = "persistence_outcome" if "persistence_outcome" in data.columns else "persistence"
        positive_rate = float(data[y_col].mean())
        self.state.latest_metrics = {
            "dataset_size": size,
            "positive_rate": positive_rate,
            "retrain_count": self.state.retrains,
        }

    def _maybe_retrain(self) -> None:
        data = self.dataset_builder._load()
        if len(data) < 100:
            return

        split = chronological_split(data)
        if split.validation.empty or split.test.empty:
            return

        X_train, y_train, _ = dataset_to_matrices(split.train)
        if y_train.nunique() < 2:
            return

        opt = random_search_optimize(data)
        candidate = PersistenceModel(
            model_type=opt.params.get("model_type", "logistic"),
            feature_scaling=opt.params.get("feature_scaling", True),
            random_state=42,
        )
        candidate.fit(X_train, y_train)

        new_metrics = self._evaluate(candidate, split.test)
        old_metrics = self._evaluate(self.model, split.test) if self.state.deployed_model_path else {"roc_auc": 0.0, "brier": 1.0}

        improved = (new_metrics["roc_auc"] > old_metrics.get("roc_auc", 0.0)) and (new_metrics["brier"] <= old_metrics.get("brier", 1.0))
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

        metrics = {
            "timestamp": ts,
            "optimizer_score": opt.score,
            "best_params": opt.params,
            "test": new_metrics,
        }
        Path("models/model_metrics.json").write_text(json.dumps(metrics, indent=2))
        generate_edge_surfaces(data)

    def _selected_label(self, observation: dict, future_path: list[float]) -> int:
        if self.label_mode == "short_move":
            return int(label_short_horizon_move(observation, future_path))
        return int(label_persistence(observation, future_path))

    def run(self, *, max_iterations: int | None = None) -> TrainingEngineState:
        iterations = 0
        while self.state.runtime_state != RuntimeState.STOPPED:
            if max_iterations is not None and iterations >= max_iterations:
                break

            if self.state.runtime_state == RuntimeState.PAUSED:
                time.sleep(self.poll_interval_seconds)
                iterations += 1
                continue

            if not self.tape.has_next():
                time.sleep(self.poll_interval_seconds)
                iterations += 1
                continue

            observation = self.tape.next_tick()
            future_frame = self.tape.peek_future(self.horizon_ticks)
            future_path = future_frame["close"].tolist() if "close" in future_frame.columns else future_frame.get("price", pd.Series(dtype=float)).tolist()

            self.dataset_builder.append_from_observation(
                observation,
                future_path,
                label_mode=self.label_mode,
            )
            self.state.observations_seen += 1
            self.state.dataset_size += 1

            if self.state.dataset_size % self.retrain_interval == 0:
                self._maybe_retrain()
            if self.state.dataset_size % self.metric_interval == 0:
                self._recompute_metrics()

            iterations += 1

        return self.state
