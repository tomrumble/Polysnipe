from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import generate_simulation_report


class DummyOutput:
    def __init__(
        self,
        telemetry: pd.DataFrame,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        dataset_diagnostics: dict | None = None,
    ) -> None:
        self.telemetry = telemetry
        self.trades = trades
        self.equity_curve = equity_curve
        self.dataset_diagnostics = dataset_diagnostics or {}


def test_generate_simulation_report_with_trades_and_new_diagnostics():
    telemetry = pd.DataFrame(
        [
            {
                "trade_outcome": "WIN",
                "return": 0.01,
                "seconds_to_expiry_at_entry": 4,
                "signal_reason": "late_state_collapse",
                "regime_label": "PERSISTENT_COMPRESSION",
                "veto_reason": "",
                "collapse_reason": "",
                "directional_entropy": 0.5,
                "entropy_at_entry": 0.5,
                "spread": 0.02,
                "spread_at_entry": 0.02,
                "volatility": 0.1,
                "volatility_at_entry": 0.1,
                "stability_ratio": 2.2,
                "distance_to_boundary_at_entry": 2.5,
                "entropy_slope_before_entry": -0.21,
            },
            {
                "trade_outcome": "WIN",
                "return": 0.01,
                "seconds_to_expiry_at_entry": 11,
                "signal_reason": "late_state_collapse",
                "regime_label": "PERSISTENT_COMPRESSION",
                "veto_reason": "entropy_guard",
                "collapse_reason": "entropy_not_collapsing",
                "directional_entropy": 0.7,
                "entropy_at_entry": 0.7,
                "spread": 0.03,
                "spread_at_entry": 0.03,
                "volatility": 0.2,
                "volatility_at_entry": 0.2,
                "stability_ratio": 3.0,
                "distance_to_boundary_at_entry": 2.8,
                "entropy_slope_before_entry": -0.32,
            },
            {
                "trade_outcome": "NO_TRADE",
                "return": 0.0,
                "seconds_to_expiry_at_entry": 33,
                "signal_reason": "none",
                "regime_label": "OSCILLATORY_NOISE",
                "veto_reason": "spread_guard",
                "collapse_reason": "spread_not_tight",
                "directional_entropy": 0.64,
                "spread": 0.07,
                "volatility": 0.25,
                "stability_ratio": 1.1,
            },
        ]
    )
    trades = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    equity_curve = pd.DataFrame({"equity": [1000, 1010, 1020]})

    report = generate_simulation_report(
        DummyOutput(
            telemetry=telemetry,
            trades=trades,
            equity_curve=equity_curve,
            dataset_diagnostics={
                "api_source": "binance_api",
                "dataset_loaded": "btc_binance_api",
                "symbol": "BTCUSDT",
                "interval": "1s",
                "api_limit_per_request": 1000,
                "api_requests_used": 2,
                "candles_loaded": 2200,
                "expected_candles_for_range": 2400,
                "data_truncation_detected": False,
            },
        ),
        {
            "dataset": "btc_binance_api",
            "start": datetime(2026, 3, 7, 6, 23),
            "end": datetime(2026, 3, 7, 7, 23),
            "stream_count": 4,
            "total_capital": 1000,
            "stability_ratio_threshold": 0.8,
            "entropy_threshold": 0.62,
            "accel_threshold": 0.45,
            "spread_threshold": 0.02,
            "seconds_remaining_threshold": 30,
            "acceleration_veto": True,
            "oscillation_veto": True,
            "spread_veto": True,
            "volatility_spike_veto": True,
        },
    )

    assert "FEATURE DISTRIBUTION (ALL OBSERVATIONS)" in report
    assert "entropy_95pct:" in report
    assert "COLLAPSE BLOCKERS" in report
    assert "entropy_not_collapsing: 1" in report
    assert "spread_not_tight: 1" in report
    assert "REGIME DISTRIBUTION (ALL OBSERVATIONS)" in report
    assert "EVALUATION TIMING DISTRIBUTION" in report
    assert "0-5: 1" in report
    assert "10-20: 1" in report
    assert "30-60: 1" in report
    assert "DATASET SOURCE" in report
    assert "dataset_selected: btc_binance_api" in report
    assert "dataset_loaded: btc_binance_api" in report
    assert "symbol: BTCUSDT" in report
    assert "interval: 1s" in report
    assert "DATASET DIAGNOSTICS" in report
    assert "api_limit_per_request: 1000" in report


def test_generate_simulation_report_warning_when_no_collapse_signals():
    telemetry = pd.DataFrame(
        [
            {"signal_reason": "none", "trade_outcome": "NO_TRADE", "seconds_to_expiry_at_entry": 12},
            {"signal_reason": "none", "trade_outcome": "NO_TRADE", "seconds_to_expiry_at_entry": 14},
        ]
    )
    report = generate_simulation_report(
        DummyOutput(telemetry=telemetry, trades=pd.DataFrame(), equity_curve=pd.DataFrame()),
        {},
    )
    assert "collapse_detector_never_triggered: true" in report


def test_generate_simulation_report_handles_empty_frames():
    report = generate_simulation_report(
        DummyOutput(telemetry=pd.DataFrame(), trades=pd.DataFrame(), equity_curve=pd.DataFrame()),
        {},
    )

    assert "trade_count: 0" in report
    assert "none: 0" in report
    assert "observations_evaluated: 0" in report
