from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import resolve_dataset_route


def test_btc_binance_api_route_uses_btcusdt_and_binance_loader():
    route = resolve_dataset_route("btc_binance_api")
    assert route.loader_name == "binance_api"
    assert route.symbol == "BTCUSDT"
    assert route.source == "binance_api"


def test_eth_binance_api_route_uses_ethusdt_and_binance_loader():
    route = resolve_dataset_route("eth_binance_api")
    assert route.loader_name == "binance_api"
    assert route.symbol == "ETHUSDT"
    assert route.source == "binance_api"


def test_unknown_dataset_route_raises_value_error():
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dataset_route("synthetic")
