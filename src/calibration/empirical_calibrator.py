"""Empirical calibration for persistence probabilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

DEFAULT_OUTPUT_PATH = Path("datasets/calibration/persistence_surface.json")

STABILITY_BUCKETS: list[tuple[float, float | None, str]] = [
    (0.0, 0.75, "0.0-0.75"),
    (0.75, 1.25, "0.75-1.25"),
    (1.25, 2.0, "1.25-2.0"),
    (2.0, None, "2.0+"),
]
ENTROPY_BUCKETS: list[tuple[float, float | None, str]] = [
    (0.0, 0.35, "0.0-0.35"),
    (0.35, 0.55, "0.35-0.55"),
    (0.55, 0.69, "0.55-0.69"),
    (0.69, None, "0.69+"),
]
SECONDS_BUCKETS: list[tuple[float, float | None, str]] = [
    (0.0, 5.0, "0-5s"),
    (5.0, 10.0, "5-10s"),
    (10.0, 20.0, "10-20s"),
    (20.0, 30.0, "20-30s"),
    (30.0, None, "30s+"),
]


def _select_bucket(value: float, buckets: list[tuple[float, float | None, str]]) -> str:
    for lower, upper, label in buckets:
        if upper is None and value >= lower:
            return label
        if upper is not None and lower <= value < upper:
            return label
    return buckets[-1][2]


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
                if outcome not in {"WIN", "LOSS"}:
                    continue

                rows.append(
                    {
                        "stability_ratio": float(raw.get("stability_ratio", 0.0)),
                        "directional_entropy": float(raw.get("directional_entropy", 0.0)),
                        "seconds_remaining": float(raw.get("seconds_remaining", 0.0)),
                        "failure": int(raw.get("failure", int(outcome == "LOSS"))),
                    }
                )

    return rows


def build_stability_ratio_calibration(
    history_paths: Iterable[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    bin_width: float = 0.5,
) -> dict:
    """Build nested calibration by stability, entropy, and seconds remaining."""

    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")

    history = _load_history(history_paths)
    if not history:
        payload: dict[str, dict] = {}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2))
        return payload

    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for row in history:
        s_bucket = _select_bucket(float(row["stability_ratio"]), STABILITY_BUCKETS)
        e_bucket = _select_bucket(float(row["directional_entropy"]), ENTROPY_BUCKETS)
        t_bucket = _select_bucket(float(row["seconds_remaining"]), SECONDS_BUCKETS)

        node = counts.setdefault(s_bucket, {}).setdefault(e_bucket, {}).setdefault(
            t_bucket,
            {"trade_count": 0, "failure_count": 0},
        )
        node["trade_count"] += 1
        node["failure_count"] += int(row["failure"])

    payload: dict[str, dict] = {}
    for s_bucket, e_map in counts.items():
        payload[s_bucket] = {}
        for e_bucket, t_map in e_map.items():
            payload[s_bucket][e_bucket] = {}
            for t_bucket, stats in t_map.items():
                trade_count = stats["trade_count"]
                failure_rate = stats["failure_count"] / trade_count if trade_count else 0.0
                payload[s_bucket][e_bucket][t_bucket] = {
                    "trade_count": trade_count,
                    "failure_rate": failure_rate,
                    "persistence_probability": 1.0 - failure_rate,
                }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload
