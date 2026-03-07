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

COLLAPSE_BLOCKER_ORDER = [
    "entropy_not_collapsing",
    "spread_not_tight",
    "acceleration_too_high",
    "stability_insufficient",
    "volatility_not_declining",
    "regime_not_supported",
]

EVAL_BUCKETS = ["0-5", "5-10", "10-20", "20-30", "30-60", "60+"]


def _read_frame(simulation_output: Any, field_name: str) -> pd.DataFrame:
    if simulation_output is None:
        return pd.DataFrame()

    if isinstance(simulation_output, dict):
        frame = simulation_output.get(field_name)
    else:
        frame = getattr(simulation_output, field_name, None)

    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _read_meta(simulation_output: Any, field_name: str) -> Dict[str, Any]:
    if simulation_output is None:
        return {}
    if isinstance(simulation_output, dict):
        value = simulation_output.get(field_name, {})
    else:
        value = getattr(simulation_output, field_name, {})
    return value if isinstance(value, dict) else {}


def _safe_stat(frame: pd.DataFrame, column: str, agg: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    series = frame[column].dropna()
    if series.empty:
        return ""
    if agg == "mean":
        value = float(series.mean())
    elif agg == "min":
        value = float(series.min())
    elif agg == "max":
        value = float(series.max())
    elif agg == "p95":
        value = float(series.quantile(0.95))
    else:
        raise ValueError(f"Unsupported aggregate: {agg}")
    return f"{value:.4f}"


def _format_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _eval_timing_distribution(telemetry: pd.DataFrame) -> Dict[str, int]:
    counts = {bucket: 0 for bucket in EVAL_BUCKETS}
    if telemetry.empty or "seconds_to_expiry_at_entry" not in telemetry.columns:
        return counts
    for seconds in telemetry["seconds_to_expiry_at_entry"].fillna(0.0).astype(float):
        if seconds <= 5:
            counts["0-5"] += 1
        elif seconds <= 10:
            counts["5-10"] += 1
        elif seconds <= 20:
            counts["10-20"] += 1
        elif seconds <= 30:
            counts["20-30"] += 1
        elif seconds <= 60:
            counts["30-60"] += 1
        else:
            counts["60+"] += 1
    return counts


def generate_simulation_report(simulation_output: Any, config: Dict[str, Any]) -> str:
    """Build a compact, deterministic simulation report suitable for clipboard usage."""
    telemetry = _read_frame(simulation_output, "telemetry")
    trades = _read_frame(simulation_output, "trades")
    dataset_diagnostics = _read_meta(simulation_output, "dataset_diagnostics")

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
    evaluation_timing_distribution = _eval_timing_distribution(telemetry)

    signal_distribution = (
        telemetry["signal_reason"].fillna("none").value_counts().sort_index()
        if "signal_reason" in telemetry.columns
        else pd.Series(dtype="int64")
    )

    regime_distribution = (
        telemetry["regime_label"].fillna("").value_counts().to_dict()
        if "regime_label" in telemetry.columns
        else {}
    )

    veto_distribution = (
        telemetry["veto_reason"].fillna("").value_counts().to_dict()
        if "veto_reason" in telemetry.columns
        else {}
    )

    collapse_distribution = (
        telemetry["collapse_reason"].fillna("").value_counts().to_dict()
        if "collapse_reason" in telemetry.columns
        else {}
    )

    filter_blockers = {name: int(veto_distribution.get(name, 0)) for name in VETO_ORDER}
    collapse_blockers = {name: int(collapse_distribution.get(name, 0)) for name in COLLAPSE_BLOCKER_ORDER}

    observations = len(telemetry)
    none_signals = int(signal_distribution.get("none", 0)) if not signal_distribution.empty else 0
    collapse_detector_never_triggered = observations > 0 and observations == none_signals

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

    lines.extend(["", "EVALUATION TIMING DISTRIBUTION", "------------------------------", "seconds_remaining_bucket_counts"])
    for bucket in EVAL_BUCKETS:
        lines.append(f"{bucket}: {evaluation_timing_distribution[bucket]}")

    lines.extend(["", "SIGNAL DISTRIBUTION", "-------------------"])
    if signal_distribution.empty:
        lines.append("none: 0")
    else:
        for reason, count in signal_distribution.items():
            lines.append(f"{reason}: {int(count)}")

    lines.extend(["", "REGIME DISTRIBUTION (ALL OBSERVATIONS)", "--------------------------------------"])
    for regime in REGIME_ORDER:
        lines.append(f"{regime}: {int(regime_distribution.get(regime, 0))}")

    lines.extend([
        "",
        "FEATURE SUMMARY (TRADES ONLY)",
        "-----------------------------",
        f"entropy_mean: {_safe_stat(trades, 'entropy_at_entry', 'mean')}",
        f"entropy_min: {_safe_stat(trades, 'entropy_at_entry', 'min')}",
        f"entropy_max: {_safe_stat(trades, 'entropy_at_entry', 'max')}",
        "",
        f"entropy_slope_before_entry_mean: {_safe_stat(trades, 'entropy_slope_before_entry', 'mean')}",
        f"entropy_slope_before_entry_min: {_safe_stat(trades, 'entropy_slope_before_entry', 'min')}",
        f"entropy_slope_before_entry_max: {_safe_stat(trades, 'entropy_slope_before_entry', 'max')}",
        "",
        f"spread_mean: {_safe_stat(trades, 'spread_at_entry', 'mean')}",
        f"spread_min: {_safe_stat(trades, 'spread_at_entry', 'min')}",
        f"spread_max: {_safe_stat(trades, 'spread_at_entry', 'max')}",
        "",
        f"volatility_mean: {_safe_stat(trades, 'volatility_at_entry', 'mean')}",
        f"volatility_min: {_safe_stat(trades, 'volatility_at_entry', 'min')}",
        f"volatility_max: {_safe_stat(trades, 'volatility_at_entry', 'max')}",
        "",
        f"distance_to_boundary_mean: {_safe_stat(trades, 'distance_to_boundary_at_entry', 'mean')}",
        "",
        "FEATURE DISTRIBUTION (ALL OBSERVATIONS)",
        "---------------------------------------",
        f"entropy_mean: {_safe_stat(telemetry, 'directional_entropy', 'mean')}",
        f"entropy_min: {_safe_stat(telemetry, 'directional_entropy', 'min')}",
        f"entropy_max: {_safe_stat(telemetry, 'directional_entropy', 'max')}",
        f"entropy_95pct: {_safe_stat(telemetry, 'directional_entropy', 'p95')}",
        "",
        f"stability_ratio_mean: {_safe_stat(telemetry, 'stability_ratio', 'mean')}",
        f"stability_ratio_min: {_safe_stat(telemetry, 'stability_ratio', 'min')}",
        f"stability_ratio_max: {_safe_stat(telemetry, 'stability_ratio', 'max')}",
        f"stability_ratio_95pct: {_safe_stat(telemetry, 'stability_ratio', 'p95')}",
        "",
        f"spread_mean: {_safe_stat(telemetry, 'spread', 'mean')}",
        f"spread_95pct: {_safe_stat(telemetry, 'spread', 'p95')}",
        "",
        f"volatility_mean: {_safe_stat(telemetry, 'volatility', 'mean')}",
        f"volatility_95pct: {_safe_stat(telemetry, 'volatility', 'p95')}",
        "",
        "FILTER BLOCKERS",
        "---------------",
    ])

    for veto in VETO_ORDER:
        lines.append(f"{veto}: {filter_blockers[veto]}")

    lines.extend(["", "COLLAPSE BLOCKERS", "-----------------"])
    for blocker in COLLAPSE_BLOCKER_ORDER:
        lines.append(f"{blocker}: {collapse_blockers[blocker]}")

    lines.extend(
        [
            "",
            "DATASET SUMMARY",
            "---------------",
            f"observations_evaluated: {observations}",
            f"markets_simulated: {config.get('stream_count', '')}",
            "time_resolution_seconds: 1",
            "",
            "DATASET SOURCE",
            "--------------",
            f"dataset_selected: {config.get('dataset', '')}",
            f"dataset_loaded: {dataset_diagnostics.get('dataset_loaded', dataset_diagnostics.get('api_source', ''))}",
            f"symbol: {dataset_diagnostics.get('symbol', '')}",
            f"interval: {dataset_diagnostics.get('interval', '')}",
            "",
            "DATASET DIAGNOSTICS",
            "-------------------",
            f"api_source: {dataset_diagnostics.get('api_source', '')}",
            f"api_limit_per_request: {dataset_diagnostics.get('api_limit_per_request', '')}",
            f"api_requests_used: {dataset_diagnostics.get('api_requests_used', '')}",
            f"candles_loaded: {dataset_diagnostics.get('candles_loaded', '')}",
            f"expected_candles_for_range: {dataset_diagnostics.get('expected_candles_for_range', '')}",
            f"data_truncation_detected: {str(bool(dataset_diagnostics.get('data_truncation_detected', False))).lower()}",
            "",
            "SIMULATION WARNINGS",
            "-------------------",
            f"collapse_detector_never_triggered: {str(collapse_detector_never_triggered).lower()}",
        ]
    )

    return "\n".join(lines)
