"""Edge probability surface diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def _line_surface(dataset: pd.DataFrame, feature: str, out_path: Path) -> None:
    frame = dataset[[feature, "persistence_outcome"]].dropna().copy()
    frame["bucket"] = pd.cut(frame[feature], bins=12)
    grouped = frame.groupby("bucket", observed=False)["persistence_outcome"].mean().reset_index()
    grouped["bucket"] = grouped["bucket"].astype(str)
    fig = px.line(grouped, x="bucket", y="persistence_outcome", title=f"P(persistence) vs {feature}")
    fig.write_html(out_path)


def generate_edge_surfaces(dataset: pd.DataFrame, output_dir: str | Path = "datasets/diagnostics") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    if "persistence_outcome" not in dataset.columns and "persistence" in dataset.columns:
        dataset = dataset.rename(columns={"persistence": "persistence_outcome"})

    for feature in ["entropy", "spread", "volatility", "seconds_remaining"]:
        path = out / f"persistence_vs_{feature}.html"
        _line_surface(dataset, feature, path)
        files[feature] = str(path)

    heat = dataset[["entropy", "volatility", "persistence_outcome"]].dropna().copy()
    heat["entropy_bucket"] = pd.cut(heat["entropy"], bins=10)
    heat["volatility_bucket"] = pd.cut(heat["volatility"], bins=10)
    hm = heat.groupby(["entropy_bucket", "volatility_bucket"], observed=False)["persistence_outcome"].mean().reset_index()
    hm["entropy_bucket"] = hm["entropy_bucket"].astype(str)
    hm["volatility_bucket"] = hm["volatility_bucket"].astype(str)
    fig = px.density_heatmap(hm, x="entropy_bucket", y="volatility_bucket", z="persistence_outcome", title="Entropy vs Volatility persistence heatmap")
    heat_path = out / "entropy_vs_volatility_heatmap.html"
    fig.write_html(heat_path)
    files["entropy_vs_volatility"] = str(heat_path)
    return files
