from src.edge.edge_score import compute_edge_score


def test_edge_score_is_bounded() -> None:
    result = compute_edge_score(
        expected_value=0.05,
        calibration_error=0.01,
        probability_rank_correlation=0.7,
        max_drawdown=-0.05,
        trade_rate=0.05,
    )
    assert 0.0 <= result["edge_score"] <= 1.0


def test_edge_score_penalizes_poor_conditions() -> None:
    strong = compute_edge_score(
        expected_value=0.01,
        calibration_error=0.03,
        probability_rank_correlation=0.4,
        max_drawdown=-0.05,
        trade_rate=0.08,
    )
    weak = compute_edge_score(
        expected_value=-0.01,
        calibration_error=0.2,
        probability_rank_correlation=0.0,
        max_drawdown=-0.35,
        trade_rate=0.7,
    )
    assert strong["edge_score"] > weak["edge_score"]


def test_edge_score_handles_no_trade_activity() -> None:
    result = compute_edge_score(
        expected_value=0.01,
        calibration_error=0.03,
        probability_rank_correlation=0.4,
        max_drawdown=-0.05,
        trade_rate=0.0,
    )
    assert result["edge_score"] == 0.0
    assert result["reason"] == "insufficient_trade_activity"
