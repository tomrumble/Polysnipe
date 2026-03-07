"""Historical Binance candle ingestion into local parquet datasets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


class BinanceIngestor:
    """Fetches and incrementally stores 1-second Binance candles."""

    def __init__(self, base_dir: str | Path = "datasets/raw", timeout: int = 15) -> None:
        self.base_dir = Path(base_dir)
        self.timeout = timeout

    def _dataset_path(self, symbol: str) -> Path:
        return self.base_dir / f"{symbol.upper()}_1s.parquet"

    def _fetch_range(self, symbol: str, start_time: datetime, end_time: datetime, limit: int = 1000) -> pd.DataFrame:
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        cursor = start_ms
        rows: list[dict[str, float | int | datetime]] = []

        while cursor < end_ms:
            query = urlencode(
                {
                    "symbol": symbol.upper(),
                    "interval": "1s",
                    "limit": min(limit, 1000),
                    "startTime": cursor,
                    "endTime": end_ms,
                }
            )
            url = f"https://api.binance.com/api/v3/klines?{query}"
            with urlopen(url, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))

            if not payload:
                break

            for candle in payload:
                open_ms = int(candle[0])
                if open_ms >= end_ms:
                    continue
                rows.append(
                    {
                        "timestamp": pd.to_datetime(open_ms, unit="ms", utc=True),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "close_time_ms": int(candle[6]),
                        "quote_asset_volume": float(candle[7]),
                        "trade_count": int(candle[8]),
                        "taker_buy_base_volume": float(candle[9]),
                        "taker_buy_quote_volume": float(candle[10]),
                        "symbol": symbol.upper(),
                    }
                )

            next_cursor = int(payload[-1][6]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(payload) < limit:
                break

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("timestamp").reset_index(drop=True)

    def ingest(self, symbol: str, start_time: datetime, end_time: datetime) -> Path:
        """Ingest a date range and append only unseen candles."""

        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")

        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self._dataset_path(symbol)

        existing = pd.DataFrame()
        if path.exists():
            existing = pd.read_parquet(path)
            if not existing.empty:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

        fetched = self._fetch_range(symbol=symbol, start_time=start_time, end_time=end_time)
        if fetched.empty and path.exists():
            return path

        merged = pd.concat([existing, fetched], ignore_index=True)
        if merged.empty:
            merged = fetched

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        merged.to_parquet(path, index=False)
        return path


def ingest(symbol: str, start_time: datetime, end_time: datetime) -> Path:
    return BinanceIngestor().ingest(symbol, start_time, end_time)
