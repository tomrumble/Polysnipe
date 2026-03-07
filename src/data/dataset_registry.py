"""Dataset routing for simulator loaders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRoute:
    dataset_name: str
    loader_name: str
    symbol: str
    source: str
    interval: str = "1s"
    path: str | None = None


# Parquet paths under project root (created by Binance ingestor or similar).
PARQUET_BASE = "datasets/raw"

DATASET_LOADERS: dict[str, str] = {
    "btc_binance_api": "binance_api",
    "eth_binance_api": "binance_api",
    "btc_binance_parquet": "parquet",
    "eth_binance_parquet": "parquet",
    "synthetic": "synthetic",
}


def resolve_dataset_route(dataset_name: str) -> DatasetRoute:
    loader_name = DATASET_LOADERS.get(dataset_name)
    if loader_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "btc_binance_api":
        return DatasetRoute(dataset_name, loader_name, symbol="BTCUSDT", source="binance_api")
    if dataset_name == "eth_binance_api":
        return DatasetRoute(dataset_name, loader_name, symbol="ETHUSDT", source="binance_api")
    if dataset_name == "btc_binance_parquet":
        return DatasetRoute(
            dataset_name, loader_name, symbol="BTCUSDT", source="parquet",
            path=f"{PARQUET_BASE}/BTCUSDT_1s.parquet",
        )
    if dataset_name == "eth_binance_parquet":
        return DatasetRoute(
            dataset_name, loader_name, symbol="ETHUSDT", source="parquet",
            path=f"{PARQUET_BASE}/ETHUSDT_1s.parquet",
        )
    return DatasetRoute(dataset_name, loader_name, symbol="synthetic", source="synthetic")
