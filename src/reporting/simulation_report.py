"""Utilities for rendering simulation output to deterministic text reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd

REGIME_ORDER = [
    "PERSISTENT_COMPRESSION",
    "OSCILLATORY_NOISE",
    "VOLATILE_LIQUIDATION",
    "LATE_MARKET_FREEZE",
]

VETO_ORDER = [
    "entropy_guard",
    "spread_guard",
    "acceleration_guard",
    "stability_guard",
    "time_guard",
]


def _read_frame(simulation_output: Any, field_name: str) -> pd.DataFrame:
    if simulation_output is None:
        return pd.DataFrame()

    if isinstance(simulation_output, dict):
        frame = simulation_output.get(field_name)
    else:
        frame = getattr(simulation_output, field_name, None)

    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _safe_mean(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    return f"{float(frame[column].mean()):.4f}"


def _safe_min(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    return f"{float(frame[column].min()):.4f}"


def _safe_max(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    return f"{float(frame[column].max()):.4f}"


def _format_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def generate_simulation_report(simulation_output: Any, config: Dict[str, Any]) -> str:
    """Build a compact, deterministic simulation report suitable for clipboard usage."""
    telemetry = _read_frame(simulation_output, "telemetry")
    trades = _read_frame(simulation_output, "trades")

    if trades.empty and not telemetry.empty and "trade_outcome" in telemetry.columns:
        trades = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()

    wins = int((trades["trade_outcome"] == "WIN").sum()) if "trade_outcome" in trades.columns else 0
    losses = int((trades["trade_outcome"] == "LOSS").sum()) if "trade_outcome" in trades.columns else 0
    win_loss_ratio = wins / losses if losses else float(wins)
    avg_return = float(trades["return"].mean()) if "return" in trades.columns and not trades.empty else 0.0

    max_drawdown = 0.0
    equity_curve = _read_frame(simulation_output, "equity_curve")
    if not equity_curve.empty and "equity" in equity_curve.columns:
        running_peak = equity_curve["equity"].cummax()
        max_drawdown = float((equity_curve["equity"] / running_peak - 1.0).min())

    timing_distribution = (
        trades["seconds_to_expiry_at_entry"].fillna(0).round().astype(int).value_counts().sort_index()
        if "seconds_to_expiry_at_entry" in trades.columns
        else pd.Series(dtype="int64")
    )

    signal_distribution = (
        telemetry["signal_reason"].fillna("none").value_counts().sort_index()
        if "signal_reason" in telemetry.columns
        else pd.Series(dtype="int64")
    )

    regime_distribution = (
        trades["regime_label"].fillna("").value_counts().to_dict()
        if "regime_label" in trades.columns
        else {}
    )

    veto_distribution = (
        telemetry["veto_reason"].fillna("").value_counts().to_dict()
        if "veto_reason" in telemetry.columns
        else {}
    )

    filter_blockers = {name: int(veto_distribution.get(name, 0)) for name in VETO_ORDER}

    lines = [
        "POLYSNIPE SIMULATION REPORT",
        "===========================",
        "",
        f"Dataset: {config.get('dataset', '')}",
        f"Start: {_format_time(config.get('start'))}",
        f"End: {_format_time(config.get('end'))}",
        f"Streams: {config.get('stream_count', '')}",
        f"Total Capital: {config.get('total_capital', '')}",
        "",
        "SIGNAL CONFIG",
        "-------------",
        f"stability_ratio_threshold: {float(config.get('stability_ratio_threshold', 0.0)):.2f}",
        f"entropy_threshold: {float(config.get('entropy_threshold', 0.0)):.2f}",
        f"accel_threshold: {float(config.get('accel_threshold', 0.0)):.2f}",
        f"spread_threshold: {float(config.get('spread_threshold', 0.0)):.2f}",
        f"seconds_remaining_threshold: {int(config.get('seconds_remaining_threshold', 0))}",
        "",
        "HEURISTIC GUARDS",
        "----------------",
        f"acceleration_veto: {str(bool(config.get('acceleration_veto', False))).lower()}",
        f"oscillation_veto: {str(bool(config.get('oscillation_veto', False))).lower()}",
        f"spread_veto: {str(bool(config.get('spread_veto', False))).lower()}",
        f"volatility_spike_veto: {str(bool(config.get('volatility_spike_veto', False))).lower()}",
        "",
        "STRATEGY RESULTS",
        "----------------",
        f"trade_count: {len(trades)}",
        f"win_loss_ratio: {win_loss_ratio:.2f}",
        f"average_return: {avg_return * 100:.2f}%",
        f"max_drawdown: {max_drawdown * 100:.2f}%",
        "",
        "ENTRY TIMING",
        "------------",
        "seconds_to_expiry_distribution:",
    ]

    if timing_distribution.empty:
        lines.append("")
    else:
        for second, count in timing_distribution.items():
            lines.append(f"{second}s: {int(count)}")

    lines.extend(["", "SIGNAL DISTRIBUTION", "-------------------"])
    if signal_distribution.empty:
        lines.append("none: 0")
    else:
        for reason, count in signal_distribution.items():
            lines.append(f"{reason}: {int(count)}")

    lines.extend(["", "REGIME DISTRIBUTION", "-------------------"])
    for regime in REGIME_ORDER:
        lines.append(f"{regime}: {int(regime_distribution.get(regime, 0))}")

    lines.extend(
        [
            "",
            "FEATURE SUMMARY (TRADES ONLY)",
            "-----------------------------",
            f"entropy_mean: {_safe_mean(trades, 'entropy_at_entry')}",
            f"entropy_min: {_safe_min(trades, 'entropy_at_entry')}",
            f"entropy_max: {_safe_max(trades, 'entropy_at_entry')}",
            "",
            f"entropy_slope_before_entry_mean: {_safe_mean(trades, 'entropy_slope_before_entry')}",
            f"entropy_slope_before_entry_min: {_safe_min(trades, 'entropy_slope_before_entry')}",
            f"entropy_slope_before_entry_max: {_safe_max(trades, 'entropy_slope_before_entry')}",
            "",
            f"spread_mean: {_safe_mean(trades, 'spread_at_entry')}",
            f"spread_min: {_safe_min(trades, 'spread_at_entry')}",
            f"spread_max: {_safe_max(trades, 'spread_at_entry')}",
            "",
            f"volatility_mean: {_safe_mean(trades, 'volatility_at_entry')}",
            f"volatility_min: {_safe_min(trades, 'volatility_at_entry')}",
            f"volatility_max: {_safe_max(trades, 'volatility_at_entry')}",
            "",
            f"distance_to_boundary_mean: {_safe_mean(trades, 'distance_to_boundary_at_entry')}",
            "",
            "FILTER BLOCKERS",
            "---------------",
        ]
    )

    for veto in VETO_ORDER:
        lines.append(f"{veto}: {filter_blockers[veto]}")

    lines.extend(
        [
            "",
            "DATASET SUMMARY",
            "---------------",
            f"observations_evaluated: {len(telemetry)}",
            f"markets_simulated: {config.get('stream_count', '')}",
            f"time_resolution_seconds: 1",
        ]
    )

    return "\n".join(lines)
