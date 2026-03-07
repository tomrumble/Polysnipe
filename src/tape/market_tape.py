"""Historical market tape abstraction for deterministic streaming replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MarketTape:
    def __init__(self, dataset_path: str | Path, start_index: int = 0) -> None:
        self.dataset_path = Path(dataset_path)
        self.dataset = pd.read_parquet(self.dataset_path).sort_values("timestamp").reset_index(drop=True)
        self.pointer = max(start_index - 1, -1)

    def has_next(self) -> bool:
        return self.pointer + 1 < len(self.dataset)

    def next_tick(self) -> dict:
        if not self.has_next():
            raise StopIteration("MarketTape exhausted")
        self.pointer += 1
        row = self.dataset.iloc[self.pointer]
        return row.to_dict()

    def peek_future(self, horizon: int) -> pd.DataFrame:
        start = self.pointer + 1
        end = min(start + horizon, len(self.dataset))
        return self.dataset.iloc[start:end].copy()

    def reset(self, start_index: int = 0) -> None:
        self.pointer = max(start_index - 1, -1)
