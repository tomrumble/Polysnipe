"""Persistence-backed training lifecycle controller."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd


class TrainingLifecycleState(str, Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class TrainingController:
    training_state: TrainingLifecycleState = TrainingLifecycleState.STOPPED
    dataset_size: int = 0
    model_version: str = "untrained"
    training_step: int = 0
    start_time: datetime | None = None
    last_retrain_time: datetime | None = None
    state_path: Path = field(default_factory=lambda: Path("models/training_state.json"), repr=False)

    @classmethod
    def load(cls, state_path: str | Path = "models/training_state.json") -> "TrainingController":
        path = Path(state_path)
        if not path.exists():
            controller = cls(state_path=path)
            controller.save()
            return controller

        payload = json.loads(path.read_text())
        return cls(
            training_state=TrainingLifecycleState(payload.get("training_state", TrainingLifecycleState.STOPPED.value)),
            dataset_size=int(payload.get("dataset_size", 0)),
            model_version=str(payload.get("model_version", "untrained")),
            training_step=int(payload.get("training_step", 0)),
            start_time=_parse_datetime(payload.get("start_time")),
            last_retrain_time=_parse_datetime(payload.get("last_retrain_time")),
            state_path=path,
        )

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "training_state": self.training_state.value,
            "dataset_size": int(self.dataset_size),
            "model_version": self.model_version,
            "training_step": int(self.training_step),
            "start_time": _format_datetime(self.start_time),
            "last_retrain_time": _format_datetime(self.last_retrain_time),
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def start_training(self, reset_progress: bool = True) -> None:
        self.training_state = TrainingLifecycleState.RUNNING
        self.start_time = datetime.now(tz=timezone.utc)
        if reset_progress:
            self.training_step = 0
            self.dataset_size = 0
            self.model_version = "untrained"
            self.last_retrain_time = None
        self.save()

    def pause_training(self) -> None:
        self.training_state = TrainingLifecycleState.PAUSED
        self.save()

    def resume_training(self, *, dataset_path: str | Path = "datasets/edge_training_data.parquet", latest_model_path: str | Path = "models/latest_persistence_model.pkl") -> None:
        self.sync_from_artifacts(dataset_path=dataset_path, latest_model_path=latest_model_path)
        if self.start_time is None:
            self.start_time = datetime.now(tz=timezone.utc)
        self.training_state = TrainingLifecycleState.RUNNING
        self.save()

    def stop_training(self) -> None:
        self.training_state = TrainingLifecycleState.STOPPED
        self.save()

    def update_progress(self, *, training_step: int | None = None, dataset_size: int | None = None) -> None:
        if training_step is not None:
            self.training_step = int(training_step)
        if dataset_size is not None:
            self.dataset_size = int(dataset_size)
        self.save()

    def mark_retrained(self, model_version: str) -> None:
        self.model_version = model_version
        self.last_retrain_time = datetime.now(tz=timezone.utc)
        self.save()

    def sync_from_artifacts(
        self,
        *,
        dataset_path: str | Path = "datasets/edge_training_data.parquet",
        latest_model_path: str | Path = "models/latest_persistence_model.pkl",
    ) -> None:
        dataset = Path(dataset_path)
        if dataset.exists():
            data = pd.read_parquet(dataset)
            self.dataset_size = int(len(data))
            self.training_step = max(self.training_step, self.dataset_size)

        model_file = Path(latest_model_path)
        if model_file.exists():
            self.model_version = model_file.name


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
