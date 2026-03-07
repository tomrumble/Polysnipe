"""Binance kline loader with pagination + dataset diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


@dataclass(frozen=True)
class DatasetDiagnostics:
    api_source: str
    api_limit_per_request: int
    api_requests_used: int
    candles_loaded: int
    expected_candles_for_range: int
    data_truncation_detected: bool


def fetch_binance_klines_paginated(
    *,
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 1000,
    timeout: int = 10,
    request_fn: Callable = urlopen,
) -> tuple[pd.DataFrame, DatasetDiagnostics]:
    """Fetch complete kline range using Binance pagination."""

    if end_time <= start_time:
        raise ValueError("end_time must be after start_time")

    if limit <= 0 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    cursor_ms = start_ms
    rows: list[dict[str, float | int | datetime]] = []
    request_count = 0

    while cursor_ms < end_ms:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": cursor_ms,
                "endTime": end_ms,
            }
        )
        url = f"https://api.binance.com/api/v3/klines?{query}"

        with request_fn(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        request_count += 1
        if not payload:
            break

        for candle in payload:
            open_ms = int(candle[0])
            close_ms = int(candle[6])
            if open_ms >= end_ms:
                continue
            rows.append(
                {
                    "open_time_ms": open_ms,
                    "close_time_ms": close_ms,
                    "timestamp": datetime.fromtimestamp(open_ms / 1000),
                    "price": float(candle[4]),
                }
            )

        next_cursor = int(payload[-1][6]) + 1
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor

        if len(payload) < limit:
            break

    frame = pd.DataFrame(rows, columns=["open_time_ms", "close_time_ms", "timestamp", "price"])
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms").reset_index(drop=True)

    expected_seconds_range = max(int((end_time - start_time).total_seconds()), 0)
    expected_candles = max(1, expected_seconds_range)
    candles_loaded = int(len(frame))
    truncation_detected = candles_loaded < int(0.9 * expected_candles)

    diagnostics = DatasetDiagnostics(
        api_source="binance",
        api_limit_per_request=limit,
        api_requests_used=request_count,
        candles_loaded=candles_loaded,
        expected_candles_for_range=expected_candles,
        data_truncation_detected=truncation_detected,
    )

    return frame, diagnostics
