"""Time-based cross-validation utilities."""

from __future__ import annotations

import pandas as pd


def time_based_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + validation_ratio))
    return ordered.iloc[:train_end], ordered.iloc[train_end:valid_end], ordered.iloc[valid_end:]
