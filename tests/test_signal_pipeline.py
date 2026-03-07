from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal_pipeline import RegimeLabel, SignalConfig, SignalInputs, classify_regime, directional_entropy, evaluate_signal


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


def test_collapse_blocker_logging_entropy_not_collapsing():
    decision = evaluate_signal(
        SignalInputs(
            seconds_remaining=6,
            spread=0.01,
            directional_entropy=0.7,
            entropy_velocity=-0.02,
            price_acceleration=0.01,
            stability_ratio=3.0,
            volatility_current=0.05,
            volatility_previous=0.1,
            regime_label=RegimeLabel.PERSISTENT_COMPRESSION.value,
        ),
        SignalConfig(),
    )
    assert decision.should_trade is False
    assert decision.collapse_reason == "entropy_not_collapsing"


def test_collapse_blocker_logging_regime_not_supported():
    decision = evaluate_signal(
        SignalInputs(
            seconds_remaining=6,
            spread=0.01,
            directional_entropy=0.1,
            entropy_velocity=-0.02,
            price_acceleration=0.01,
            stability_ratio=3.0,
            volatility_current=0.05,
            volatility_previous=0.1,
            regime_label=RegimeLabel.OSCILLATORY_NOISE.value,
        ),
        SignalConfig(),
    )
    assert decision.should_trade is False
    assert decision.collapse_reason == "regime_not_supported"


def test_entropy_velocity_enables_collapse_candidate():
    decision = evaluate_signal(
        SignalInputs(
            seconds_remaining=6,
            spread=0.01,
            directional_entropy=0.68,
            entropy_velocity=-0.15,
            price_acceleration=0.01,
            stability_ratio=3.0,
            volatility_current=0.05,
            volatility_previous=0.1,
            regime_label=RegimeLabel.PERSISTENT_COMPRESSION.value,
        ),
        SignalConfig(),
    )
    assert decision.should_trade is True
