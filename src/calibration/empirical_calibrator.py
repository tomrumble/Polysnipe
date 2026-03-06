"""Empirical calibration for persistence probabilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

DEFAULT_OUTPUT_PATH = Path("datasets/calibration/stability_ratio_calibration.json")

STABILITY_BUCKETS: list[tuple[float, float | None, str]] = [
    (0.0, 0.5, "0.5"),
    (0.5, 1.0, "1.0"),
    (1.0, 1.5, "1.5"),
    (1.5, 2.0, "2.0"),
    (2.0, None, "2.0+"),
]


def _load_history(paths: Iterable[str | Path]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for path in paths:
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".csv":
            continue

        with p.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                if "stability_ratio" not in raw:
                    continue
                outcome = raw.get("trade_outcome", "")
                if outcome and outcome not in {"WIN", "LOSS"}:
                    continue

                failure = int(raw["failure"]) if raw.get("failure") not in (None, "") else int(outcome == "LOSS")
                rows.append(
                    {
                        "stability_ratio": float(raw["stability_ratio"]),
                        "failure": failure,
                    }
                )

    return rows


def build_stability_ratio_calibration(
    history_paths: Iterable[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    bin_width: float = 0.5,
) -> dict:
    """Build and save stability-ratio empirical calibration."""

    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")

    history = _load_history(history_paths)
    if not history:
        payload = {"bin_width": bin_width, "buckets": []}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2))
        return payload

    bucket_map: dict[str, dict[str, float | int]] = {
        label: {
            "lower": lower,
            "upper": upper,
            "trade_count": 0,
            "failure_count": 0,
        }
        for lower, upper, label in STABILITY_BUCKETS
    }

    for row in history:
        ratio = float(row["stability_ratio"])
        failure = int(row["failure"])

        selected_label = "2.0+"
        for lower, upper, label in STABILITY_BUCKETS:
            if upper is None and ratio >= lower:
                selected_label = label
                break
            if upper is not None and lower <= ratio < upper:
                selected_label = label
                break

        bucket_map[selected_label]["trade_count"] = int(bucket_map[selected_label]["trade_count"]) + 1
        bucket_map[selected_label]["failure_count"] = int(bucket_map[selected_label]["failure_count"]) + failure

    calibration: dict[str, dict[str, float | int | None]] = {}
    for _, _, label in STABILITY_BUCKETS:
        stats = bucket_map[label]
        trade_count = int(stats["trade_count"])
        failure_count = int(stats["failure_count"])
        if trade_count == 0:
            continue

        failure_rate = failure_count / trade_count
        calibration[label] = {
            "lower": float(stats["lower"]),
            "upper": float(stats["upper"]) if stats["upper"] is not None else None,
            "trade_count": trade_count,
            "failure_count": failure_count,
            "failure_rate": failure_rate,
            "empirical_persistence": 1.0 - failure_rate,
        }

    payload = calibration
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload
