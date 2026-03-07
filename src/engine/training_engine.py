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
from src.edge.dataset_builder import FEATURE_COLUMNS, EdgeDatasetBuilder, dataset_to_matrices
from src.edge.diagnostics import generate_edge_surfaces
from src.edge.edge_score import compute_edge_score
from src.edge.model import PersistenceModel
from src.edge.optimizer import random_search_optimize
from src.edge.policy import TradingPolicy
from src.features.feature_extractor import FeatureVector
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


@dataclass
class StateSnapshot:
    step_index: int
    timestamp: str
    price: float
    candle: dict[str, float | str]
    feature_vector: dict[str, float | int]
    dataset_size: int
    trade_executed: bool
    trade_result: str
    edge_score: float
    calibration_error: float
    spearman_rank_correlation: float
    expected_value: float
    win_rate: float
    retrain_event: bool
    trade_count: int
    predicted_signal: float
    outcome_label: int
    feature_importance: list[tuple[str, float]] = field(default_factory=list)


class TrainingEngine:
    def __init__(
        self,
        tape: MarketTape,
        dataset_builder: EdgeDatasetBuilder | None = None,
        *,
        retrain_interval: int = 1000,
        metric_interval: int = 50,
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
        self.policy = TradingPolicy()
        self.trade_probabilities: list[float] = []
        self.trade_outcomes: list[int] = []
        self.trade_returns: list[float] = []
        self.last_feature_importance: list[tuple[str, float]] = []
        self.records: list[dict] = []
        self.cursor: int = 0
        self.dataset_preloaded: bool = False


    def load_dataset(self, records: list[dict], *, precomputed_features: bool = False) -> None:
        self.records = list(records)
        self.cursor = 0
        self.dataset_preloaded = bool(precomputed_features)
        if precomputed_features:
            self.state.dataset_size = len(self.records)

    def has_next(self) -> bool:
        if self.records:
            return self.cursor < len(self.records)
        return self.tape.has_next()

    def next_tick(self) -> dict:
        if self.records:
            if self.cursor >= len(self.records):
                raise StopIteration("Preloaded dataset exhausted")
            observation = self.records[self.cursor]
            self.cursor += 1
            return observation
        return self.tape.next_tick()

    def _observation_has_precomputed_features(self, observation: dict) -> bool:
        required = set(FEATURE_COLUMNS + ["label_persistence", "label_drift"])
        return required.issubset(set(observation.keys()))

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

    def _get_training_dataframe(self) -> pd.DataFrame:
        """
        Return the dataset used for training depending on runtime mode.
        """
        if self.dataset_preloaded:
            if not self.records:
                return pd.DataFrame()
            return pd.DataFrame(self.records)
        return self.dataset_builder._load()

    def _recompute_metrics(self) -> None:
        data = self._get_training_dataframe()
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

    def _maybe_retrain(self) -> bool:
        data = self._get_training_dataframe()
        print(
            f"[TRAINING] dataset_size={len(data)} "
            f"features={len(FEATURE_COLUMNS)} "
            f"preloaded={self.dataset_preloaded}"
        )
        if len(data) < 100:
            return False

        split = chronological_split(data)
        if split.validation.empty or split.test.empty:
            return False

        X_train, y_train, _ = dataset_to_matrices(split.train)
        if y_train.nunique() < 2:
            return False

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
            return False

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
        importance = self.model.feature_importance_
        if importance is not None:
            feature_names = list(dataset_to_matrices(split.train)[0].columns)
            paired = sorted(
                [(feature_names[idx], float(value)) for idx, value in enumerate(importance)],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            self.last_feature_importance = paired[:5]
        return True

    def _compute_live_metrics(self, trade_executed: bool, trade_outcome: int, signal: float) -> dict[str, float]:
        if trade_executed:
            self.trade_probabilities.append(float(signal))
            self.trade_outcomes.append(int(trade_outcome))
            self.trade_returns.append(1.0 if trade_outcome == 1 else -1.0)

        trade_count = len(self.trade_outcomes)
        if trade_count == 0:
            return {
                "edge_score": 0.0,
                "calibration_error": 1.0,
                "spearman_rank_correlation": 0.0,
                "expected_value": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
            }

        probs = pd.Series(self.trade_probabilities, dtype=float)
        outcomes = pd.Series(self.trade_outcomes, dtype=float)
        returns = pd.Series(self.trade_returns, dtype=float)
        calibration_error = float((probs - outcomes).abs().mean())
        if probs.nunique() < 2 or outcomes.nunique() < 2:
            spearman = 0.0
        else:
            corr = probs.corr(outcomes, method="spearman")
            spearman = 0.0 if pd.isna(corr) else float(corr)
        win_rate = float(outcomes.mean())
        expected_value = float(returns.mean())
        equity = (1.0 + returns).replace(0, 1e-9).cumprod()
        drawdown = float((equity / equity.cummax() - 1.0).min())
        edge = compute_edge_score(
            expected_value=expected_value,
            calibration_error=calibration_error,
            probability_rank_correlation=spearman,
            max_drawdown=abs(drawdown),
            trade_rate=float(trade_count / max(self.state.observations_seen, 1)),
        )
        return {
            "edge_score": float(edge["edge_score"]),
            "calibration_error": calibration_error,
            "spearman_rank_correlation": spearman,
            "expected_value": expected_value,
            "win_rate": win_rate,
            "trade_count": trade_count,
        }

    def step(self) -> StateSnapshot | None:
        if self.state.runtime_state != RuntimeState.RUNNING or not self.has_next():
            return None

        # Atomic step: exactly one observation is consumed per call.
        observation = self.next_tick()
        precomputed = self._observation_has_precomputed_features(observation)

        future_path: list[float] = []
        if not precomputed and self.tape is not None:
            future_frame = self.tape.peek_future(self.horizon_ticks)
            future_path = future_frame["close"].tolist() if "close" in future_frame.columns else future_frame.get("price", pd.Series(dtype=float)).tolist()

        if precomputed:
            features = FeatureVector(**{col: observation[col] for col in FEATURE_COLUMNS})
            persistence_label = int(observation.get("label_persistence", 0))
        else:
            features = extract_features(observation)
            persistence_label = self._selected_label(observation, future_path)

        signal = float(self.model.predict_signal(features.__dict__))
        trade_decision = self.policy.evaluate(probability=float(signal))

        if not precomputed:
            self.dataset_builder.append_from_observation(
                observation,
                future_path,
                label_mode=self.label_mode,
            )
            self.state.dataset_size += 1

        self.state.observations_seen += 1

        retrain_event = False
        cadence_count = self.state.observations_seen if precomputed else self.state.dataset_size
        if cadence_count % self.retrain_interval == 0:
            retrain_event = self._maybe_retrain()
        if cadence_count % self.metric_interval == 0:
            if precomputed:
                self.state.latest_metrics.update({"dataset_size": self.state.dataset_size, "retrain_count": self.state.retrains})
            else:
                self._recompute_metrics()

        live_metrics = self._compute_live_metrics(trade_decision.enter, persistence_label, signal)
        self.state.latest_metrics.update(live_metrics)
        candle = {
            "timestamp": str(observation.get("timestamp", datetime.now(tz=timezone.utc).isoformat())),
            "open": float(observation.get("open", observation.get("price", observation.get("close", 0.0)))),
            "high": float(observation.get("high", observation.get("price", observation.get("close", 0.0)))),
            "low": float(observation.get("low", observation.get("price", observation.get("close", 0.0)))),
            "close": float(observation.get("close", observation.get("price", 0.0))),
        }
        return StateSnapshot(
            step_index=self.state.observations_seen,
            timestamp=candle["timestamp"],
            price=float(candle["close"]),
            candle=candle,
            feature_vector=features.__dict__,
            dataset_size=self.state.dataset_size,
            trade_executed=bool(trade_decision.enter),
            trade_result="WIN" if trade_decision.enter and persistence_label == 1 else ("LOSS" if trade_decision.enter else "NO_TRADE"),
            edge_score=float(live_metrics["edge_score"]),
            calibration_error=float(live_metrics["calibration_error"]),
            spearman_rank_correlation=float(live_metrics["spearman_rank_correlation"]),
            expected_value=float(live_metrics["expected_value"]),
            win_rate=float(live_metrics["win_rate"]),
            retrain_event=retrain_event,
            trade_count=int(live_metrics["trade_count"]),
            predicted_signal=float(signal),
            outcome_label=int(persistence_label),
            feature_importance=list(self.last_feature_importance),
        )

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

            if not self.has_next():
                time.sleep(self.poll_interval_seconds)
                iterations += 1
                continue

            self.step()

            iterations += 1

        return self.state
