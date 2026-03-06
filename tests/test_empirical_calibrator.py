import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import build_stability_ratio_calibration


def test_build_stability_ratio_calibration(tmp_path):
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "calibration.json"

    history_path.write_text(
        "stability_ratio,directional_entropy,seconds_remaining,trade_outcome\n"
        "0.2,0.3,4,WIN\n"
        "0.3,0.4,8,LOSS\n"
        "0.7,0.5,12,WIN\n"
        "1.2,0.6,18,LOSS\n"
        "1.3,0.2,25,WIN\n"
        "2.4,0.1,35,WIN\n"
    )

    payload = build_stability_ratio_calibration([history_path], output_path=output_path, bin_width=0.5)

    assert output_path.exists()
    assert payload
    assert "0.0-0.75" in payload

    saved = json.loads(output_path.read_text())
    first_seconds_bucket = next(iter(next(iter(saved.values())).values()))
    nested = next(iter(first_seconds_bucket.values()))
    assert "trade_count" in nested and "failure_rate" in nested and "persistence_probability" in nested
