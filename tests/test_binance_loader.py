from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.binance_loader import fetch_binance_klines_paginated


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_klines(total: int, start_ms: int):
    rows = []
    for i in range(total):
        open_ms = start_ms + i * 1000
        close_ms = open_ms + 999
        rows.append([open_ms, "1", "1", "1", "100", "1", close_ms, "0", 0, "0", "0", "0"])
    return rows


def test_pagination_loader_retrieves_more_than_1000_candles():
    start = datetime(2026, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=2500)
    all_klines = _build_klines(2500, int(start.timestamp() * 1000))

    def fake_request(url, timeout=10):
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(url).query)
        start_ms = int(q["startTime"][0])
        limit = int(q["limit"][0])
        idx = max(0, (start_ms - int(start.timestamp() * 1000)) // 1000)
        payload = all_klines[idx : idx + limit]
        return FakeResponse(payload)

    frame, diagnostics = fetch_binance_klines_paginated(
        symbol="BTCUSDT",
        interval="1s",
        start_time=start,
        end_time=end,
        limit=1000,
        request_fn=fake_request,
    )

    assert len(frame) == 2500
    assert diagnostics.api_requests_used == 3
    assert frame["open_time_ms"].is_monotonic_increasing
    assert frame["open_time_ms"].is_unique


def test_dataset_truncation_detection_true_when_under_ninety_percent():
    start = datetime(2026, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=1000)
    partial = _build_klines(500, int(start.timestamp() * 1000))

    def fake_request(url, timeout=10):
        return FakeResponse(partial)

    frame, diagnostics = fetch_binance_klines_paginated(
        symbol="BTCUSDT",
        interval="1s",
        start_time=start,
        end_time=end,
        limit=1000,
        request_fn=fake_request,
    )

    assert len(frame) == 500
    assert diagnostics.expected_candles_for_range == 1000
    assert diagnostics.data_truncation_detected is True
