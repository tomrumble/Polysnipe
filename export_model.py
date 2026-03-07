"""Export deployable persistence model + policy artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.edge.model import PersistenceModel


def export_model(threshold: float = 0.97) -> tuple[Path, Path]:
    model_in = Path("models/latest_persistence_model.pkl")
    if not model_in.exists():
        candidates = sorted(Path("models").glob("persistence_model_v*.pkl"))
        if not candidates:
            raise FileNotFoundError("No trained persistence model found in models/")
        model_in = candidates[-1]

    model = PersistenceModel.load(model_in)
    deploy_model = Path("models/deployment_model.pkl")
    model.save(deploy_model)

    policy_path = Path("models/deployment_policy.json")
    policy_path.write_text(json.dumps({"confidence_threshold": threshold}, indent=2))
    return deploy_model, policy_path


if __name__ == "__main__":
    export_model()
