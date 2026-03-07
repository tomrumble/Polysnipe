"""Offline optimization command for persistence model training configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.edge.optimizer import OptimizationResult, random_search_optimize


def run_offline_optimization(
    dataset_path: str | Path = "datasets/edge_training_data.parquet",
    iterations: int = 64,
    seed: int = 42,
    output_path: str | Path = "models/offline_optimization.json",
) -> OptimizationResult:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = pd.read_parquet(path)
    result = random_search_optimize(dataset, iterations=iterations, seed=seed)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(path),
        "iterations": iterations,
        "seed": seed,
        **asdict(result),
    }
    out.write_text(json.dumps(payload, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline optimizer for persistence model configuration.")
    parser.add_argument("--dataset", default="datasets/edge_training_data.parquet", help="Path to training dataset parquet file.")
    parser.add_argument("--iterations", type=int, default=64, help="Number of random-search iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic optimization.")
    parser.add_argument("--output", default="models/offline_optimization.json", help="Path to write optimization results JSON.")
    args = parser.parse_args()

    result = run_offline_optimization(
        dataset_path=args.dataset,
        iterations=args.iterations,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
