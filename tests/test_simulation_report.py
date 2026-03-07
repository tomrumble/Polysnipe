from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import generate_simulation_report


class DummyOutput:
    def __init__(self, telemetry: pd.DataFrame, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
        self.telemetry = telemetry
        self.trades = trades
        self.equity_curve = equity_curve


def test_generate_simulation_report_with_trades():
    telemetry = pd.DataFrame(
        [
            {
                "trade_outcome": "WIN",
                "return": 0.01,
                "seconds_to_expiry_at_entry": 30,
                "signal_reason": "late_state_collapse",
                "regime_label": "PERSISTENT_COMPRESSION",
                "veto_reason": "",
                "entropy_at_entry": 0.5,
                "spread_at_entry": 0.02,
                "volatility_at_entry": 0.1,
                "distance_to_boundary_at_entry": 2.5,
                "entropy_slope_before_entry": -0.21,
            },
            {
                "trade_outcome": "WIN",
                "return": 0.01,
                "seconds_to_expiry_at_entry": 30,
                "signal_reason": "late_state_collapse",
                "regime_label": "PERSISTENT_COMPRESSION",
                "veto_reason": "entropy_guard",
                "entropy_at_entry": 0.7,
                "spread_at_entry": 0.03,
                "volatility_at_entry": 0.2,
                "distance_to_boundary_at_entry": 2.8,
                "entropy_slope_before_entry": -0.32,
            },
            {
                "trade_outcome": "NO_TRADE",
                "return": 0.0,
                "signal_reason": "none",
                "veto_reason": "spread_guard",
            },
        ]
    )
    trades = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    equity_curve = pd.DataFrame({"equity": [1000, 1010, 1020]})

    report = generate_simulation_report(
        DummyOutput(telemetry=telemetry, trades=trades, equity_curve=equity_curve),
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

    assert "POLYSNIPE SIMULATION REPORT" in report
    assert "Dataset: btc_binance_api" in report
    assert "trade_count: 2" in report
    assert "win_loss_ratio: 2.00" in report
    assert "average_return: 1.00%" in report
    assert "30s: 2" in report
    assert "late_state_collapse: 2" in report
    assert "PERSISTENT_COMPRESSION: 2" in report
    assert "FILTER BLOCKERS" in report
    assert "entropy_guard: 1" in report
    assert "spread_guard: 1" in report
    assert "entropy_slope_before_entry_mean: -0.2650" in report


def test_generate_simulation_report_handles_empty_frames():
    report = generate_simulation_report(
        DummyOutput(telemetry=pd.DataFrame(), trades=pd.DataFrame(), equity_curve=pd.DataFrame()),
        {},
    )

    assert "trade_count: 0" in report
    assert "none: 0" in report
    assert "observations_evaluated: 0" in report
