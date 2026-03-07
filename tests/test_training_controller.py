from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.training_controller import TrainingController, TrainingLifecycleState


def test_training_controller_persists_and_restores_state(tmp_path: Path):
    state_path = tmp_path / "training_state.json"

    controller = TrainingController.load(state_path)
    assert controller.training_state == TrainingLifecycleState.STOPPED

    controller.start_training(reset_progress=False)
    controller.update_progress(training_step=15, dataset_size=12)
    controller.mark_retrained("persistence_model_v1.pkl")
    controller.pause_training()

    reloaded = TrainingController.load(state_path)
    assert reloaded.training_state == TrainingLifecycleState.PAUSED
    assert reloaded.training_step == 15
    assert reloaded.dataset_size == 12
    assert reloaded.model_version == "persistence_model_v1.pkl"
    assert reloaded.last_retrain_time is not None


def test_resume_syncs_with_dataset_and_model_artifacts(tmp_path: Path):
    dataset_path = tmp_path / "edge_training_data.parquet"
    model_path = tmp_path / "latest_persistence_model.pkl"
    state_path = tmp_path / "training_state.json"

    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(dataset_path, index=False)
    model_path.write_bytes(b"model")

    controller = TrainingController.load(state_path)
    controller.resume_training(dataset_path=dataset_path, latest_model_path=model_path)

    assert controller.training_state == TrainingLifecycleState.RUNNING
    assert controller.dataset_size == 3
    assert controller.training_step == 3
    assert controller.model_version == "latest_persistence_model.pkl"
