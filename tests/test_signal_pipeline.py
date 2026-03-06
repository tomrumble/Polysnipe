from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal_pipeline import RegimeLabel, classify_regime, directional_entropy


def test_directional_entropy_low_for_trend():
    prices = [100, 101, 102, 103, 104, 105]
    assert directional_entropy(prices, window=6) < 0.01


def test_directional_entropy_high_for_alternation():
    prices = [100, 101, 100, 101, 100, 101, 100, 101]
    assert directional_entropy(prices, window=8) > 0.65


def test_regime_classifier_freeze():
    label = classify_regime(
        volatility=0.05,
        directional_entropy_value=0.1,
        price_acceleration=0.02,
        spread=0.01,
        seconds_remaining=4.0,
    )
    assert label == RegimeLabel.LATE_MARKET_FREEZE


def test_regime_classifier_volatile_liquidation():
    label = classify_regime(
        volatility=0.6,
        directional_entropy_value=0.3,
        price_acceleration=1.2,
        spread=0.01,
        seconds_remaining=12.0,
    )
    assert label == RegimeLabel.VOLATILE_LIQUIDATION
