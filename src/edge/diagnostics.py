"""Diagnostics for feature-wise and joint persistence surfaces."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def _line_surface(dataset: pd.DataFrame, feature: str, out_path: Path) -> None:
    frame = dataset[[feature, "persistence"]].dropna().copy()
    frame["bucket"] = pd.cut(frame[feature], bins=12)
    grouped = frame.groupby("bucket", observed=False)["persistence"].mean().reset_index()
    grouped["bucket"] = grouped["bucket"].astype(str)
    fig = px.line(grouped, x="bucket", y="persistence", title=f"P(persistence) vs {feature}")
    fig.write_html(out_path)


def generate_edge_surfaces(dataset: pd.DataFrame, output_dir: str | Path = "datasets/diagnostics") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    for feature in ["entropy", "spread", "stability_ratio", "seconds_remaining"]:
        path = out / f"persistence_vs_{feature}.html"
        _line_surface(dataset, feature, path)
        files[feature] = str(path)

    heat = dataset[["entropy", "volatility", "persistence"]].dropna().copy()
    heat["entropy_bucket"] = pd.cut(heat["entropy"], bins=10)
    heat["volatility_bucket"] = pd.cut(heat["volatility"], bins=10)
    heat_map = heat.groupby(["entropy_bucket", "volatility_bucket"], observed=False)["persistence"].mean().reset_index()
    heat_map["entropy_bucket"] = heat_map["entropy_bucket"].astype(str)
    heat_map["volatility_bucket"] = heat_map["volatility_bucket"].astype(str)
    fig = px.density_heatmap(
        heat_map,
        x="entropy_bucket",
        y="volatility_bucket",
        z="persistence",
        title="Entropy vs Volatility persistence surface",
    )
    heat_path = out / "entropy_vs_volatility_heatmap.html"
    fig.write_html(heat_path)
    files["entropy_volatility"] = str(heat_path)
    return files
