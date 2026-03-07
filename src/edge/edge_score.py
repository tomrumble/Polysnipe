"""Composite edge quality score for dashboard validation panel."""

from __future__ import annotations


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def compute_edge_score(
    *,
    expected_value: float,
    calibration_error: float,
    probability_rank_correlation: float,
    max_drawdown: float,
    trade_rate: float,
) -> dict[str, float]:
    """Compute a bounded edge score from core defensibility components.

    Returns component diagnostics with `edge_score` in [0, 1].
    """

    normalized_expected_value = _clamp((expected_value + 0.01) / 0.03)
    calibration_quality = _clamp(1.0 - (calibration_error / 0.15))
    probability_quality = _clamp((probability_rank_correlation + 0.2) / 0.7)

    drawdown_penalty = _clamp(abs(min(max_drawdown, 0.0)) / 0.3)
    overtrading_penalty = _clamp(max(trade_rate - 0.15, 0.0) / 0.35)

    raw = (
        (0.45 * normalized_expected_value)
        + (0.30 * calibration_quality)
        + (0.25 * probability_quality)
        - (0.20 * drawdown_penalty)
        - (0.10 * overtrading_penalty)
    )
    edge_score = _clamp(raw)

    return {
        "edge_score": edge_score,
        "normalized_expected_value": normalized_expected_value,
        "calibration_quality": calibration_quality,
        "probability_rank_correlation": probability_rank_correlation,
        "probability_quality": probability_quality,
        "drawdown_penalty": drawdown_penalty,
        "overtrading_penalty": overtrading_penalty,
    }
