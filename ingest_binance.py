#!/usr/bin/env python3
"""Run Binance 1s candle ingestion to populate datasets/binance/*.parquet.

Usage:
    python ingest_binance.py [SYMBOL] [HOURS]
    python ingest_binance.py              # BTCUSDT, last 24 hours
    python ingest_binance.py BTCUSDT 48   # BTCUSDT, last 48 hours
    python ingest_binance.py ETHUSDT 24  # ETHUSDT, last 24 hours
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Project root
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data import BinanceIngestor

def main() -> None:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT").upper()
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    base_dir = _root / "datasets" / "binance"
    base_dir.mkdir(parents=True, exist_ok=True)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    print(f"Ingesting {symbol} 1s candles from {start_time} to {end_time} ({hours}h)...")
    ingestor = BinanceIngestor(base_dir=str(base_dir))
    path = ingestor.ingest(symbol, start_time, end_time)
    print(f"Wrote {path}")

if __name__ == "__main__":
    main()
