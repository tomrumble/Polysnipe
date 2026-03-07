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


DATASET_LOADERS: dict[str, str] = {
    "btc_binance_api": "binance_api",
    "eth_binance_api": "binance_api",
}


def resolve_dataset_route(dataset_name: str) -> DatasetRoute:
    loader_name = DATASET_LOADERS.get(dataset_name)
    if loader_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "btc_binance_api":
        return DatasetRoute(dataset_name, loader_name, symbol="BTCUSDT", source="binance_api")
    return DatasetRoute(dataset_name, loader_name, symbol="ETHUSDT", source="binance_api")
