"""Replay a simulated trade from persisted telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def replay_trade(trade_id: str, path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"trade telemetry not found: {path}")

    telemetry = pd.read_csv(path)
    trade = telemetry[telemetry["trade_id"] == trade_id]
    if trade.empty:
        raise ValueError(f"trade_id not found: {trade_id}")

    row = trade.iloc[0]
    price_path = json.loads(row["price_path_until_expiry"])
    entry_price = float(row["entry_price"])
    boundary_price = float(row["boundary_price"])

    print("entry conditions")
    print(
        {
            "entry_timestamp": row["entry_timestamp"],
            "expiry_timestamp": row["expiry_timestamp"],
            "seconds_to_expiry_at_entry": row["seconds_to_expiry_at_entry"],
            "stability_ratio": row["stability_ratio_at_entry"],
            "volatility": row["volatility_at_entry"],
            "entropy": row["entropy_at_entry"],
            "spread": row["spread_at_entry"],
            "distance_to_boundary": row["distance_to_boundary_at_entry"],
        }
    )
    print("regime classification")
    print(row["regime_label"])
    print("veto checks")
    print({"signal_reason": row["signal_reason"], "veto_reason": row["veto_reason"]})
    print("price path until expiry")
    print(price_path)

    xs = list(range(len(price_path) + 1))
    ys = [entry_price] + price_path

    plt.figure(figsize=(6, 3))
    plt.plot(xs, ys, label="price")
    plt.axhline(boundary_price, color="red", linestyle="--", label="boundary")
    plt.title(f"Trade replay: {trade_id}")
    plt.xlabel("seconds from entry")
    plt.ylabel("price")
    plt.legend()

    out_path = Path("datasets/telemetry") / f"replay_{trade_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"saved plot: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade_id", required=True)
    parser.add_argument("--path", default="datasets/telemetry/trade_history.csv")
    args = parser.parse_args()
    replay_trade(args.trade_id, Path(args.path))


if __name__ == "__main__":
    main()
