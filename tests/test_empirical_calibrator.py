import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import build_stability_ratio_calibration


def test_build_stability_ratio_calibration(tmp_path):
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "calibration.json"

    history_path.write_text(
        "stability_ratio,trade_outcome\n"
        "0.2,WIN\n"
        "0.3,LOSS\n"
        "0.7,WIN\n"
        "1.2,LOSS\n"
        "1.3,WIN\n"
        "2.4,WIN\n"
    )

    payload = build_stability_ratio_calibration([history_path], output_path=output_path, bin_width=0.5)

    assert output_path.exists()
    assert payload
    assert "0.5" in payload

    saved = json.loads(output_path.read_text())
    assert "1.5" in saved
    assert all("trade_count" in bucket and "failure_rate" in bucket for bucket in saved.values())
