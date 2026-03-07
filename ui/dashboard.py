"""Edge Validation dashboard for statistically defensible trading edge monitoring.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when run as streamlit run ui/dashboard.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import base64
import time
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.calibration import build_stability_ratio_calibration
from src.data import (
    BinanceIngestor,
    build_feature_dataset,
    dataframe_to_records,
    fetch_binance_klines_paginated,
    load_dataset_metadata,
    load_feature_dataset,
    load_parquet_dataset,
    resolve_dataset_route,
    save_feature_dataset,
)
from src.persistence_model import PersistenceInputs, PersistenceModel as DiffusionPersistenceModel
from src.edge.model import PersistenceModel
from src.edge.policy import (
    EXPLORATION_THRESHOLD,
    EXPLOITATION_THRESHOLD,
    MIN_EXPLORATION_SAMPLES,
    PolicySide,
    TradingPolicy,
)
from src.engine import ResearchEngine, TrainingController, TrainingEngine, TrainingLifecycleState
from src.tape import MarketTape
from src.edge.dataset_builder import EdgeDatasetBuilder
from src.edge.edge_score import compute_edge_score
from src.edge.metrics_pipeline import build_metrics_payload, load_metrics, persist_metrics
from src.features import extract_features
from src.signal_pipeline import (
    SignalConfig,
    SignalInputs,
    classify_regime,
    directional_entropy,
    evaluate_signal,
)


@dataclass
class SimulationOutputs:
    telemetry: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    dataset_diagnostics: Dict[str, float | int | str | bool]


class ReplaySimulator:
    """Lightweight replay simulator for research experimentation."""

    DATASET_SEEDS: Dict[str, int] = {
        "btc_binance_api": 17,
        "eth_binance_api": 43,
        "btc_binance_parquet": 17,
        "eth_binance_parquet": 43,
        "synthetic": 99,
    }
    API_DATASET_SYMBOLS: Dict[str, str] = {
        "btc_binance_api": "BTCUSDT",
        "eth_binance_api": "ETHUSDT",
    }
    CACHE_DIR = Path("datasets/cache")

    def __init__(
        self,
        dataset: str,
        heuristic_guards: Dict[str, bool],
        signal_config: SignalConfig,
        api_limit: int = 600,
        persistence_mode: str = "model",
        entropy_window: int = 20,
    ) -> None:
        self.dataset = dataset
        self.heuristic_guards = heuristic_guards
        self.signal_config = signal_config
        self.api_limit = api_limit
        self.entropy_window = entropy_window
        self.diffusion_model = DiffusionPersistenceModel(center=1.0, slope=1.7, mode=persistence_mode)
        self.edge_model: PersistenceModel | None = None
        self.policy = TradingPolicy()
        latest_model = Path("models/latest_persistence_model.pkl")
        if latest_model.exists():
            self.edge_model = PersistenceModel.load(latest_model)

    @classmethod
    def _cache_path(cls, symbol: str, limit: int, start_time: datetime, end_time: datetime) -> Path:
        start_token = start_time.strftime("%Y%m%dT%H%M%S")
        end_token = end_time.strftime("%Y%m%dT%H%M%S")
        return cls.CACHE_DIR / f"{symbol.lower()}_1s_{start_token}_{end_token}_{limit}.csv"

    @classmethod
    def _load_cached_binance_prices(
        cls,
        *,
        symbol: str,
        limit: int,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        cache_path = cls._cache_path(symbol, limit, start_time, end_time)
        if not cache_path.exists():
            return pd.DataFrame(columns=["timestamp", "price"])

        cached = pd.read_csv(cache_path)
        if cached.empty:
            return pd.DataFrame(columns=["timestamp", "price"])

        cached["timestamp"] = pd.to_datetime(cached["timestamp"])
        cached = cached[(cached["timestamp"] >= start_time) & (cached["timestamp"] < end_time)].copy()
        return cached[["timestamp", "price"]]

    @classmethod
    def _fetch_binance_prices(
        cls,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int,
    ) -> tuple[pd.DataFrame, Dict[str, float | int | str | bool]]:
        cached = cls._load_cached_binance_prices(symbol=symbol, limit=limit, start_time=start_time, end_time=end_time)
        if not cached.empty:
            expected = max(int((end_time - start_time).total_seconds()), 1)
            diagnostics = {
                "api_source": "binance_api",
                "symbol": symbol,
                "interval": "1s",
                "api_limit_per_request": limit,
                "api_requests_used": 0,
                "candles_loaded": int(len(cached)),
                "expected_candles_for_range": expected,
                "data_truncation_detected": len(cached) < int(0.9 * expected),
            }
            return cached, diagnostics

        frame, diagnostics_obj = fetch_binance_klines_paginated(
            symbol=symbol,
            interval="1s",
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        frame = frame[["timestamp", "price"]].copy()

        if frame.empty:
            return frame, diagnostics_obj.__dict__

        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cls._cache_path(symbol=symbol, limit=limit, start_time=start_time, end_time=end_time), index=False)
        return frame, diagnostics_obj.__dict__

    def _load_binance_dataset(
        self,
        *,
        dataset_name: str,
        start_time: datetime,
        end_time: datetime,
        n: int,
    ) -> tuple[pd.DatetimeIndex, np.ndarray, Dict[str, float | int | str | bool]]:
        symbol = self.API_DATASET_SYMBOLS[dataset_name]
        frame, dataset_diagnostics = self._fetch_binance_prices(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=min(self.api_limit, 1000),
        )
        if len(frame) < 24:
            raise RuntimeError(f"Dataset loader failed: {dataset_name}")
        timestamps = pd.to_datetime(frame["timestamp"])
        path = frame["price"].to_numpy(dtype=float)
        return timestamps, path, dataset_diagnostics

    def _load_synthetic_dataset(
        self,
        *,
        rng: np.random.Generator,
        base_vol: float,
        timestamps: pd.DatetimeIndex,
    ) -> tuple[pd.DatetimeIndex, np.ndarray, Dict[str, float | int | str | bool]]:
        n = len(timestamps)
        noise = rng.normal(0, base_vol, n)
        drift = rng.normal(0.01, 0.05, n).cumsum() / 12
        path = 100 + np.cumsum(noise + drift)
        diagnostics = {
            "api_source": "synthetic",
            "dataset_loaded": "synthetic",
            "symbol": "synthetic",
            "interval": "1s",
            "api_limit_per_request": 0,
            "api_requests_used": 0,
            "candles_loaded": n,
            "expected_candles_for_range": n,
            "data_truncation_detected": False,
        }
        return timestamps, path, diagnostics

    def _load_parquet_dataset(
        self,
        *,
        dataset_path: Path,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[pd.DatetimeIndex, np.ndarray, Dict[str, float | int | str | bool]]:
        if not dataset_path.exists():
            raise RuntimeError(f"Parquet dataset not found: {dataset_path}")

        frame = pd.read_parquet(dataset_path)
        if frame.empty:
            raise RuntimeError(f"Parquet dataset is empty: {dataset_path}")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
        bounded = frame[(frame["timestamp"] >= start_time) & (frame["timestamp"] < end_time)].copy()
        if len(bounded) < 24:
            raise RuntimeError(f"Insufficient candles in parquet dataset for selected range: {dataset_path}")

        price_col = "close" if "close" in bounded.columns else "price"
        if price_col not in bounded.columns:
            raise RuntimeError("Parquet dataset must contain a 'close' or 'price' column")

        bounded = bounded.sort_values("timestamp").reset_index(drop=True)
        timestamps = pd.to_datetime(bounded["timestamp"])
        path = bounded[price_col].to_numpy(dtype=float)

        expected = max(int((end_time - start_time).total_seconds()), 1)
        diagnostics: Dict[str, float | int | str | bool] = {
            "api_source": "parquet",
            "dataset_loaded": self.dataset,
            "symbol": resolve_dataset_route(self.dataset).symbol,
            "interval": "1s",
            "api_limit_per_request": 0,
            "api_requests_used": 0,
            "candles_loaded": int(len(path)),
            "expected_candles_for_range": expected,
            "data_truncation_detected": len(path) < int(0.9 * expected),
        }
        return timestamps, path, diagnostics

    def run(
        self,
        start_time: datetime,
        end_time: datetime,
        stream_count: int,
        total_capital: float,
        transaction_cost: float,
        payout: float = 0.95,
    ) -> SimulationOutputs:
        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")

        seconds = int((end_time - start_time).total_seconds())
        n = max(seconds, 120)

        seed = self.DATASET_SEEDS.get(self.dataset, 7) + stream_count
        rng = np.random.default_rng(seed)

        base_vol = {
            "btc_binance_api": 1.2,
            "eth_binance_api": 1.7,
            "synthetic": 2.6,
        }.get(self.dataset, 1.3)

        timestamps = pd.date_range(start_time, periods=n, freq="s")

        dataset_route = resolve_dataset_route(self.dataset)
        dataset_loader = dataset_route.loader_name

        if dataset_loader == "binance_api":
            try:
                timestamps, path, dataset_diagnostics = self._load_binance_dataset(
                    dataset_name=dataset_route.dataset_name,
                    start_time=start_time,
                    end_time=end_time,
                    n=n,
                )
                n = len(path)
                dataset_diagnostics["dataset_loaded"] = self.dataset
            except Exception as exc:
                raise RuntimeError(f"Dataset loader failed: {self.dataset}") from exc
        elif dataset_loader == "parquet":
            if not dataset_route.path:
                raise RuntimeError(f"Missing parquet path for dataset: {dataset_route.dataset_name}")
            timestamps, path, dataset_diagnostics = self._load_parquet_dataset(
                dataset_path=_project_root / dataset_route.path,
                start_time=start_time,
                end_time=end_time,
            )
            n = len(path)
        else:
            timestamps, path, dataset_diagnostics = self._load_synthetic_dataset(
                rng=rng,
                base_vol=base_vol,
                timestamps=timestamps,
            )
            n = len(path)

        dataset_diagnostics.setdefault(
            "dataset_loaded",
            self.dataset if dataset_loader in {"binance_api", "parquet"} else "synthetic",
        )

        telemetry_rows: List[Dict[str, float | int | str]] = []
        trades_rows: List[Dict[str, float | int | str]] = []

        stream_balances = np.full(stream_count, total_capital / stream_count)
        equity_points: List[float] = []
        last_volatility = 0.0
        entropy_history: List[float] = []

        for i in range(max(self.entropy_window, 20), n - 1):
            current_price = float(path[i])
            boundary_delta = 2.5
            boundary_side = 1.0 if path[i] - path[i - 1] >= 0 else -1.0
            boundary_price = current_price + boundary_side * boundary_delta
            recent_prices = path[i - 20 : i + 1].tolist()

            expiry_idx = min(i + 60, n - 1)
            seconds_remaining = float((timestamps[expiry_idx] - timestamps[i]).total_seconds())
            now_ts = timestamps[i].timestamp()
            expiry_ts = timestamps[expiry_idx].timestamp()

            out = self.diffusion_model.compute(
                PersistenceInputs(
                    current_price=current_price,
                    boundary_price=boundary_price,
                    expiry_timestamp=expiry_ts,
                    now_timestamp=now_ts,
                    recent_prices=recent_prices,
                )
            )

            accel = float((path[i] - path[i - 1]) - (path[i - 1] - path[i - 2]))
            spread = float(abs(rng.normal(0.012 * (1 + out.volatility / 3), 0.004)))
            entropy_value = float(directional_entropy(path[: i + 1].tolist(), window=self.entropy_window))
            entropy_history.append(entropy_value)
            entropy_slope_before_entry = 0.0
            entropy_velocity = 0.0
            if len(entropy_history) >= 4:
                entropy_velocity = float(entropy_history[-1] - entropy_history[-4])
                entropy_slope_before_entry = entropy_velocity
            regime = classify_regime(
                volatility=out.volatility,
                directional_entropy_value=entropy_value,
                price_acceleration=accel,
                spread=spread,
                seconds_remaining=seconds_remaining,
            )

            strict_decision = evaluate_signal(
                SignalInputs(
                    seconds_remaining=seconds_remaining,
                    spread=spread,
                    directional_entropy=entropy_value,
                    entropy_velocity=entropy_velocity,
                    price_acceleration=accel,
                    stability_ratio=out.stability_ratio,
                    volatility_current=out.volatility,
                    volatility_previous=last_volatility if last_volatility > 0 else out.volatility + 1,
                    regime_label=regime.value,
                ),
                self.signal_config,
            )

            observation = {
                "directional_entropy": entropy_value,
                "entropy_velocity": entropy_velocity,
                "spread": spread,
                "volatility": out.volatility,
                "volatility_slope": out.volatility - last_volatility,
                "stability_ratio": out.stability_ratio,
                "price_acceleration": accel,
                "seconds_remaining": seconds_remaining,
                "distance_to_boundary_at_entry": out.distance_to_boundary,
                "regime_label": regime.value,
            }
            features = extract_features(observation)
            model_probability = out.persistence_probability
            if self.edge_model is not None:
                model_probability = self.edge_model.predict_signal(features)
            self.policy.dataset_size = i
            predicted_drift = float(path[i + 1] - path[i]) if i + 1 < len(path) else 0.0
            policy_decision = self.policy.evaluate(model_probability, predicted_drift=predicted_drift)

            guard_accel_fail = self.heuristic_guards["acceleration_veto"] and abs(accel) > 3.5
            guard_spread_fail = self.heuristic_guards["spread_veto"] and spread > 0.22
            guard_vol_fail = self.heuristic_guards["volatility_spike_veto"] and out.volatility > base_vol * 1.8
            guard_osc_fail = self.heuristic_guards["oscillation_veto"] and np.sign(path[i] - path[i - 1]) != np.sign(path[i - 1] - path[i - 2])
            heuristic_blocked = guard_accel_fail or guard_spread_fail or guard_vol_fail or guard_osc_fail

            should_trade = policy_decision.enter and policy_decision.side != PolicySide.NONE and not heuristic_blocked
            veto_reason = ""
            if not policy_decision.enter:
                veto_reason = "policy_threshold"
            elif heuristic_blocked:
                veto_reason = "heuristic_guard"

            signal_reason = "model_persistence_policy" if should_trade else "none"

            future_path = path[i + 1 : expiry_idx + 1]
            max_price_until_expiry = float(np.max(future_path)) if len(future_path) else current_price
            min_price_until_expiry = float(np.min(future_path)) if len(future_path) else current_price
            expiry_iso = timestamps[expiry_idx].isoformat()

            if policy_decision.side == PolicySide.SHORT:
                trade_direction = "DOWN"
            elif policy_decision.side == PolicySide.LONG:
                trade_direction = "UP"
            else:
                trade_direction = "UP" if boundary_side > 0 else "DOWN"

            trade_outcome = "NO_TRADE"
            trade_return = 0.0
            failure = 0
            stream_idx = i % stream_count

            if should_trade:
                won_trade = _is_trade_win(
                    trade_direction=trade_direction,
                    boundary_price=boundary_price,
                    max_price_until_expiry=max_price_until_expiry,
                    min_price_until_expiry=min_price_until_expiry,
                )
                trade_outcome = "WIN" if won_trade else "LOSS"
                trade_return = compute_trade_return(
                    trade_direction=trade_direction,
                    boundary_price=boundary_price,
                    max_price_until_expiry=max_price_until_expiry,
                    min_price_until_expiry=min_price_until_expiry,
                    payout=payout,
                    transaction_cost=transaction_cost,
                )
                failure = 0 if won_trade else 1

                stream_balances[stream_idx] *= 1.0 + trade_return

            trade_id = f"trade-{i}"
            row = {
                "trade_id": trade_id,
                "timestamp": timestamps[i],
                "entry_timestamp": timestamps[i].isoformat(),
                "expiry_timestamp": expiry_iso,
                "stream_id": stream_idx,
                "entry_price": current_price,
                "boundary_price": boundary_price,
                "distance_to_boundary_at_entry": out.distance_to_boundary,
                "seconds_remaining": seconds_remaining,
                "seconds_to_expiry_at_entry": seconds_remaining,
                "volatility": out.volatility,
                "volatility_at_entry": out.volatility,
                "stability_ratio": out.stability_ratio,
                "stability_ratio_at_entry": out.stability_ratio,
                "directional_entropy": entropy_value,
                "entropy_at_entry": entropy_value,
                "entropy_slope_before_entry": entropy_slope_before_entry,
                "entropy_velocity": entropy_velocity,
                "spread": spread,
                "spread_at_entry": spread,
                "price_acceleration": accel,
                "persistence_probability": model_probability,
                "policy_signal_score": policy_decision.signal_score,
                "policy_side": policy_decision.side.value,
                "policy_enter": policy_decision.enter,
                "regime_label": regime.value,
                "signal_reason": signal_reason,
                "veto_reason": veto_reason,
                "collapse_reason": strict_decision.collapse_reason,
                "heuristic_blocked": heuristic_blocked,
                "trade_outcome": trade_outcome,
                "failure": failure,
                "return": trade_return,
                "trade_direction": trade_direction,
                "max_price_until_expiry": max_price_until_expiry,
                "min_price_until_expiry": min_price_until_expiry,
                "price_path_until_expiry": json.dumps([float(x) for x in future_path.tolist()]),
            }

            telemetry_rows.append(row)
            if should_trade:
                trades_rows.append(row)

            equity_points.append(float(stream_balances.sum()))
            last_volatility = out.volatility

        telemetry = pd.DataFrame(telemetry_rows)
        trades = pd.DataFrame(trades_rows)
        equity_curve = pd.DataFrame({"timestamp": telemetry["timestamp"], "equity": equity_points})

        telemetry_path = Path("datasets/telemetry/trade_history.csv")
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry.to_csv(telemetry_path, index=False)
        build_stability_ratio_calibration([telemetry_path], output_path="datasets/calibration/persistence_surface.json")

        return SimulationOutputs(
            telemetry=telemetry,
            trades=trades,
            equity_curve=equity_curve,
            dataset_diagnostics=dataset_diagnostics,
        )




def _is_trade_win(
    *,
    trade_direction: str,
    boundary_price: float,
    max_price_until_expiry: float,
    min_price_until_expiry: float,
) -> bool:
    if trade_direction == "UP":
        return max_price_until_expiry >= boundary_price
    if trade_direction == "DOWN":
        return min_price_until_expiry <= boundary_price
    raise ValueError(f"Unsupported trade_direction: {trade_direction}")


def compute_trade_return(
    *,
    trade_direction: str,
    boundary_price: float,
    max_price_until_expiry: float,
    min_price_until_expiry: float,
    payout: float,
    transaction_cost: float,
) -> float:
    """Compute market-path-derived trade return without using model labels."""
    won_trade = _is_trade_win(
        trade_direction=trade_direction,
        boundary_price=boundary_price,
        max_price_until_expiry=max_price_until_expiry,
        min_price_until_expiry=min_price_until_expiry,
    )
    gross_return = payout if won_trade else -1.0
    return gross_return - transaction_cost


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0



CALIBRATION_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.00001]
CALIBRATION_LABELS = ["0.50–0.60", "0.60–0.70", "0.70–0.80", "0.80–0.90", "0.90–0.95", "0.95–1.00"]
RANKING_BINS = [0.6, 0.7, 0.8, 0.9, 0.95, 1.00001]
RANKING_LABELS = ["0.60–0.70", "0.70–0.80", "0.80–0.90", "0.90–0.95", "0.95–1.00"]


def _trade_table(telemetry: pd.DataFrame) -> pd.DataFrame:
    traded = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    traded["outcome"] = (traded["trade_outcome"] == "WIN").astype(int)
    return traded


def _calibration_frame(traded: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if traded.empty:
        return pd.DataFrame(columns=["bucket", "predicted_probability_mean", "actual_persistence_rate", "count"]), 1.0
    frame = traded[["persistence_probability", "outcome"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "predicted_probability_mean", "actual_persistence_rate", "count"]), 1.0
    frame["bucket"] = pd.cut(frame["persistence_probability"], bins=CALIBRATION_BINS, labels=CALIBRATION_LABELS, include_lowest=True, right=False)
    summary = frame.groupby("bucket", observed=False).agg(
        predicted_probability_mean=("persistence_probability", "mean"),
        actual_persistence_rate=("outcome", "mean"),
        count=("outcome", "count"),
    ).reset_index()
    valid = summary["count"] > 0
    calibration_error = float((summary.loc[valid, "predicted_probability_mean"] - summary.loc[valid, "actual_persistence_rate"]).abs().mean()) if valid.any() else 1.0
    return summary, calibration_error


def _ranking_frame(traded: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if traded.empty:
        return pd.DataFrame(columns=["probability_bucket", "trade_count", "win_rate"]), 0.0
    frame = traded[["persistence_probability", "outcome"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["probability_bucket", "trade_count", "win_rate"]), 0.0
    frame["probability_bucket"] = pd.cut(frame["persistence_probability"], bins=RANKING_BINS, labels=RANKING_LABELS, include_lowest=True, right=False)
    summary = frame.groupby("probability_bucket", observed=False).agg(
        trade_count=("outcome", "count"),
        win_rate=("outcome", "mean"),
    ).reset_index()
    # Avoid ConstantInputWarning: correlation undefined when either series is constant
    if len(frame) < 2 or frame["persistence_probability"].nunique() < 2 or frame["outcome"].nunique() < 2:
        spearman = 0.0
    else:
        r = frame["persistence_probability"].corr(frame["outcome"], method="spearman")
        spearman = 0.0 if pd.isna(r) else float(r)
    return summary, spearman


def _profitability_metrics(traded: pd.DataFrame, transaction_cost: float) -> tuple[dict[str, float], pd.DataFrame]:
    if traded.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "expected_value": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }, pd.DataFrame(columns=["timestamp", "equity"])

    frame = traded.sort_values("timestamp").copy()
    frame["net_return"] = frame["return"]
    wins = frame[frame["net_return"] > 0]["net_return"]
    losses = frame[frame["net_return"] <= 0]["net_return"]
    win_rate = float((frame["net_return"] > 0).mean())
    loss_rate = 1.0 - win_rate
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.abs().mean()) if not losses.empty else 0.0
    expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    max_drawdown = float(_drawdown_series(frame["equity"]).min())
    net_std = float(frame["net_return"].std(ddof=0))
    sharpe = float((frame["net_return"].mean() / max(net_std, 1e-9)) * np.sqrt(len(frame)))
    return {
        "total_trades": float(len(frame)),
        "win_rate": win_rate,
        "avg_return": float(frame["net_return"].mean()),
        "expected_value": float(expected_value),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }, frame[["timestamp", "equity", "net_return"]]


def _edge_validation_metrics_text(
    edge_status: str,
    edge_score: float,
    calibration_error: float,
    spearman: float,
    profitability: dict[str, float],
    stability_df: pd.DataFrame,
    config: dict,
) -> str:
    """Build a compact edge validation metrics string suitable for clipboard."""
    lines = [
        "EDGE VALIDATION METRICS",
        "=======================",
        "",
        f"edge_status: {edge_status}",
        f"edge_score: {edge_score:.3f}",
        "",
        "Calibration",
        "-----------",
        f"calibration_error: {calibration_error:.3f}",
        "",
        "Ranking",
        "--------",
        f"spearman_rank_correlation: {spearman:.3f}",
        "",
        "Profitability",
        "-------------",
        f"total_trades: {int(profitability['total_trades'])}",
        f"win_rate: {profitability['win_rate'] * 100:.2f}%",
        f"avg_return_pct: {profitability['avg_return'] * 100:.3f}%",
        f"expected_value_pct: {profitability['expected_value'] * 100:.3f}%",
        f"max_drawdown_pct: {profitability['max_drawdown'] * 100:.2f}%",
        f"sharpe_ratio: {profitability['sharpe']:.3f}",
    ]
    if not stability_df.empty:
        lines.extend(["", "Stability (last N trades)", "----------------------"])
        for _, row in stability_df.iterrows():
            lines.append(
                f"{row['window']}: trades={int(row['trades'])}, win_rate={row['win_rate']:.2f}, "
                f"expected_value={row['expected_value']:.4f}, drawdown={row['drawdown']:.4f}"
            )
    lines.extend(["", "Config (run)", "------------"])
    for key in ("dataset", "start", "end", "stream_count", "total_capital", "transaction_cost"):
        if key in config:
            lines.append(f"{key}: {config[key]}")
    return "\n".join(lines)


def _window_stability(trade_curve: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []
    for window in windows:
        segment = trade_curve.tail(window)
        if segment.empty:
            rows.append({"window": f"last {window}", "trades": 0, "win_rate": 0.0, "expected_value": 0.0, "drawdown": 0.0})
            continue
        w_rate = float((segment["net_return"] > 0).mean())
        wins = segment[segment["net_return"] > 0]["net_return"]
        losses = segment[segment["net_return"] <= 0]["net_return"]
        expected_value = (w_rate * (float(wins.mean()) if not wins.empty else 0.0)) - ((1.0 - w_rate) * (float(losses.abs().mean()) if not losses.empty else 0.0))
        rows.append(
            {
                "window": f"last {window}",
                "trades": int(len(segment)),
                "win_rate": w_rate,
                "expected_value": float(expected_value),
                "drawdown": float(_drawdown_series((1.0 + segment["net_return"]).cumprod()).min()),
            }
        )
    return pd.DataFrame(rows)


def _resolve_speed_delay(simulation_speed: str) -> float:
    delays = {"1x": 0.30, "5x": 0.12, "10x": 0.06, "50x": 0.015, "100x": 0.0}
    return delays.get(simulation_speed, 0.06)


def _line_fig(df: pd.DataFrame, x: str, y: str, title: str, height: int = 180) -> go.Figure:
    """Plotly line chart from dataframe; uses single (0,0) point when empty to avoid Vega-Lite extent errors."""
    if df.empty or y not in df.columns:
        df = pd.DataFrame({x: [0], y: [0.0]})
    elif x != "step" and x not in df.columns:
        df = pd.DataFrame({x: [0], y: [0.0]})
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines"))
    fig.update_layout(title=title, height=height, margin=dict(t=40, b=30, l=40, r=20))
    return fig


def _init_live_state() -> None:
    defaults = {
        "live_engine": None,
        "live_tape": None,
        "running": False,
        "target_samples": 50_000,
        "processed_steps": 0,
        "market_candles": [],
        "dataset_growth": [],
        "edge_history": [],
        "spearman_history": [],
        "calibration_history": [],
        "signal_history": [],
        "signal_outcome_history": [],
        "event_log": [],
        "feature_importance": [],
        "run_error": "",
        "dataset_panel": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _feature_dataset_name(dataset_route) -> str:
    symbol_token = dataset_route.symbol.lower().replace("usdt", "")
    return f"{symbol_token}_features_v1"


def _build_live_engine(dataset: str, *, mode: str, target_samples: int) -> TrainingEngine:
    dataset_route = resolve_dataset_route(dataset)
    if dataset_route.loader_name == "synthetic":
        raise RuntimeError("Simulation requires parquet/api-backed datasets.")

    parquet_path = _project_root / (dataset_route.path or f"datasets/raw/{dataset_route.symbol}_1s.parquet")
    if not parquet_path.exists():
        raise RuntimeError(f"Dataset not found: {parquet_path}. Run ingestion first.")

    tape = MarketTape(str(parquet_path))
    engine = TrainingEngine(
        tape=tape,
        dataset_builder=EdgeDatasetBuilder(),
        retrain_interval=1000,
        metric_interval=50,
        horizon_ticks=60,
    )

    if mode == "Research":
        dataset_name = _feature_dataset_name(dataset_route)
        try:
            _ = load_feature_dataset(dataset_name)
        except Exception:
            raw_frame = load_parquet_dataset(parquet_path)
            feature_frame = build_feature_dataset(raw_frame)
            save_feature_dataset(
                feature_frame,
                dataset_name,
                symbol=dataset_route.symbol,
                interval=dataset_route.interval,
                feature_version="v1",
                label_mode="drift",
                append=True,
            )

        feature_slice = load_parquet_dataset(
            _project_root / f"datasets/features/{dataset_name}.parquet",
            target_samples=target_samples,
            randomized_start=True,
        )
        if feature_slice.empty:
            raise RuntimeError(f"Feature dataset '{dataset_name}' has no usable rows after schema validation")

        preview = feature_slice.iloc[0].to_dict()
        _push_event(
            "Dataset sanity check | "
            f"rows={len(feature_slice):,} "
            f"features={[preview.get('entropy'), preview.get('volatility'), preview.get('stability_ratio')]} "
            f"label={preview.get('persistence_label', preview.get('label_persistence'))}"
        )

        engine.load_dataset(dataframe_to_records(feature_slice), precomputed_features=True)
        st.session_state["dataset_panel"] = load_dataset_metadata(dataset_name)
    else:
        st.session_state["dataset_panel"] = {
            "dataset_name": dataset_route.dataset_name,
            "dataset_source": dataset_route.source,
            "feature_version": "live",
            "samples": 0,
            "last_updated": datetime.now().isoformat(),
        }

    engine.start()
    st.session_state["live_tape"] = tape
    return engine


def _push_event(message: str) -> None:
    st.session_state["event_log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    st.session_state["event_log"] = st.session_state["event_log"][-200:]


def main() -> None:
    st.set_page_config(page_title="Edge Validation Panel", layout="wide")
    st.title("Edge Validation")
    st.caption("Live scientific instrument panel for observing real-time learning")

    _init_live_state()

    st.sidebar.header("SIMULATOR CONTROL")
    dataset = st.sidebar.selectbox("dataset selector", ["btc_binance_parquet", "eth_binance_parquet", "btc_binance_api", "eth_binance_api", "synthetic"])
    target_samples = int(st.sidebar.number_input("target_samples", min_value=100, value=int(st.session_state["target_samples"]), step=1000))
    simulation_speed = st.sidebar.selectbox("simulation_speed", ["1x", "5x", "10x", "50x", "100x"], index=2)
    chart_update_every = int(st.sidebar.number_input("chart_update_every_n_steps", min_value=1, max_value=100, value=25, step=1))
    execution_mode = st.sidebar.selectbox("execution_mode", ["Research", "Live"], index=0)

    b1, b2, b3 = st.sidebar.columns(3)
    start_clicked = b1.button("Start Simulation", type="primary", use_container_width=True)
    pause_clicked = b2.button("Pause Simulation", use_container_width=True)
    reset_clicked = b3.button("Reset", use_container_width=True)

    if start_clicked:
        st.session_state["target_samples"] = target_samples
        st.session_state["run_error"] = ""
        if st.session_state["live_engine"] is None:
            try:
                st.session_state["live_engine"] = _build_live_engine(dataset, mode=execution_mode, target_samples=target_samples)
                _push_event(f"Simulation started for dataset={dataset}, target_samples={target_samples:,}")
            except Exception as exc:
                st.session_state["run_error"] = str(exc)
        st.session_state["running"] = st.session_state["run_error"] == ""

    if pause_clicked:
        st.session_state["running"] = False
        engine = st.session_state.get("live_engine")
        if engine is not None:
            engine.pause()
        _push_event("Simulation paused")

    if reset_clicked:
        st.session_state["running"] = False
        engine = st.session_state.get("live_engine")
        if engine is not None:
            engine.stop()
        st.session_state["live_engine"] = None
        st.session_state["live_tape"] = None
        st.session_state["processed_steps"] = 0
        st.session_state["market_candles"] = []
        st.session_state["dataset_growth"] = []
        st.session_state["edge_history"] = []
        st.session_state["spearman_history"] = []
        st.session_state["calibration_history"] = []
        st.session_state["signal_history"] = []
        st.session_state["signal_outcome_history"] = []
        st.session_state["feature_importance"] = []
        st.session_state["event_log"] = []
        st.session_state["run_error"] = ""

    # Fixed slot for error so main() always has same block count (avoids setIn index mismatch when fragment runs).
    error_ph = st.empty()
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])

    st.session_state["target_samples"] = target_samples
    st.session_state["simulation_speed"] = simulation_speed
    st.session_state["chart_update_every_n_steps"] = chart_update_every

    # Placeholders live in main() so they are not recreated when the fragment reruns.
    # Fragment only updates these in place -> no DOM replace, no flash.
    controls_ph = st.empty()
    market_chart_ph = st.empty()
    metrics_ph = st.empty()
    learning_charts_ph = st.empty()
    feature_panel_ph = st.empty()
    event_log_ph = st.empty()
    st.session_state["_ph"] = {
        "controls": controls_ph,
        "market_chart": market_chart_ph,
        "metrics": metrics_ph,
        "learning_charts": learning_charts_ph,
        "feature_panel": feature_panel_ph,
        "event_log": event_log_ph,
    }

    panel = st.session_state.get("dataset_panel", {})
    with st.container():
        st.subheader("Dataset Panel")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("dataset_name", str(panel.get("dataset_name", dataset)))
        d2.metric("dataset_source", str(panel.get("dataset_source", resolve_dataset_route(dataset).source)))
        d3.metric("feature_version", str(panel.get("feature_version", "v1")))
        d4.metric("samples", f"{int(panel.get('samples', 0)):,}")
        d5.metric("last_updated", str(panel.get("last_updated", "-"))[:19])

    delta_sec = max(0.05, _resolve_speed_delay(st.session_state.get("simulation_speed", "10x")))

    @st.fragment(run_every=delta_sec)
    def live_simulation_panel():
        ph = st.session_state.get("_ph")
        if ph is None:
            return
        target = int(st.session_state["target_samples"])
        speed = st.session_state.get("simulation_speed", "10x")
        delay = _resolve_speed_delay(speed)
        engine = st.session_state.get("live_engine")

        with ph["controls"].container():
            panel = st.session_state.get("dataset_panel", {})
            st.subheader("Dataset Panel")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("dataset_name", str(panel.get("dataset_name", dataset)))
            d2.metric("dataset_source", str(panel.get("dataset_source", resolve_dataset_route(dataset).source)))
            d3.metric("feature_version", str(panel.get("feature_version", "v1")))
            d4.metric("samples", f"{int(panel.get('samples', 0)):,}")
            d5.metric("last_updated", str(panel.get("last_updated", "-"))[:19])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target Samples", f"{target:,}")
            c2.metric("Processed", f"{st.session_state['processed_steps']:,}")
            c3.metric("Simulation Speed", speed)
            c4.metric("Runtime", "RUNNING" if st.session_state["running"] else "PAUSED")

            if st.session_state["running"] and engine is not None and st.session_state["processed_steps"] < target:
                engine.start()
                steps_per_refresh = max(1, int(0.2 / max(0.01, delay))) if delay > 0 else 20
                steps_per_refresh = min(steps_per_refresh, 20)
                for _ in range(steps_per_refresh):
                    if st.session_state["processed_steps"] >= target:
                        st.session_state["running"] = False
                        _push_event("Target sample size reached")
                        break

                    snapshot = engine.step()
                    if snapshot is None:
                        st.session_state["running"] = False
                        _push_event("Market tape exhausted")
                        break

                    st.session_state["processed_steps"] = snapshot.step_index
                    st.session_state["market_candles"].append(snapshot.candle)
                    st.session_state["market_candles"] = st.session_state["market_candles"][-200:]
                    st.session_state["dataset_growth"].append({"step": snapshot.step_index, "dataset_size": snapshot.dataset_size})
                    st.session_state["edge_history"].append({"step": snapshot.step_index, "edge_score": snapshot.edge_score})
                    st.session_state["spearman_history"].append({"step": snapshot.step_index, "spearman": snapshot.spearman_rank_correlation})
                    st.session_state["calibration_history"].append({"step": snapshot.step_index, "calibration": snapshot.calibration_error})
                    st.session_state["signal_history"].append({"step": snapshot.step_index, "predicted_signal": snapshot.predicted_signal})
                    st.session_state["signal_outcome_history"].append(
                        {"step": snapshot.step_index, "predicted_signal": snapshot.predicted_signal, "outcome": snapshot.outcome_label}
                    )
                    st.session_state["feature_importance"] = snapshot.feature_importance

                    _push_event(f"Processing candle {snapshot.step_index} @ {snapshot.price:.4f}")
                    _push_event(f"Feature vector generated: {snapshot.feature_vector}")
                    if snapshot.trade_executed:
                        _push_event(f"Trade executed ({snapshot.trade_result})")
                    if snapshot.retrain_event:
                        _push_event(
                            f"Step {snapshot.step_index}: Retraining model | Edge={snapshot.edge_score:.3f} Spearman={snapshot.spearman_rank_correlation:.3f}"
                        )

                    panel_data = st.session_state.get("dataset_panel", {})
                    panel_data["samples"] = int(max(panel_data.get("samples", 0), snapshot.dataset_size))
                    panel_data["last_updated"] = datetime.now().isoformat()
                    st.session_state["dataset_panel"] = panel_data

        candles_df = pd.DataFrame(st.session_state["market_candles"])
        with ph["market_chart"].container():
            if not candles_df.empty:
                fig = go.Figure(
                    data=[go.Candlestick(
                        x=pd.to_datetime(candles_df["timestamp"]),
                        open=candles_df["open"],
                        high=candles_df["high"],
                        low=candles_df["low"],
                        close=candles_df["close"],
                    )]
                )
                fig.update_layout(title="Market Replay (Rolling 200 Candles)", xaxis_rangeslider_visible=False, height=320)
                st.plotly_chart(fig, use_container_width=True)

        edge_value = st.session_state["edge_history"][-1]["edge_score"] if st.session_state["edge_history"] else 0.0
        spearman_value = st.session_state["spearman_history"][-1]["spearman"] if st.session_state["spearman_history"] else 0.0
        calibration_value = st.session_state["calibration_history"][-1]["calibration"] if st.session_state["calibration_history"] else 1.0
        dataset_size = st.session_state["dataset_growth"][-1]["dataset_size"] if st.session_state["dataset_growth"] else 0
        trade_count = engine.state.latest_metrics.get("trade_count", 0) if engine is not None else 0

        with ph["metrics"].container():
            st.subheader("Live Learning Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("edge_score", f"{edge_value:.3f}")
            m2.metric("spearman_rank_correlation", f"{spearman_value:.3f}")
            m3.metric("calibration_error", f"{calibration_value:.3f}")
            m4.metric("dataset_size", f"{dataset_size:,}")
            m5.metric("trade_count", f"{int(trade_count)}")

        growth = pd.DataFrame(st.session_state["dataset_growth"])
        edge_hist = pd.DataFrame(st.session_state["edge_history"])
        spearman_hist = pd.DataFrame(st.session_state["spearman_history"])
        calibration_hist = pd.DataFrame(st.session_state["calibration_history"])
        signal_hist = pd.DataFrame(st.session_state["signal_history"])
        signal_outcomes = pd.DataFrame(st.session_state["signal_outcome_history"])

        with ph["learning_charts"].container():
            st.subheader("Learning Curves")
            growth_plot = growth.rename(columns={"dataset_size": "y"})[["step", "y"]] if not growth.empty and "dataset_size" in growth.columns else pd.DataFrame({"step": [0], "y": [0.0]})
            st.plotly_chart(_line_fig(growth_plot, "step", "y", "Dataset size", 180), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(_line_fig(edge_hist, "step", "edge_score", "Edge score", 180), use_container_width=True)
            c2.plotly_chart(_line_fig(spearman_hist, "step", "spearman", "Spearman", 180), use_container_width=True)
            c3.plotly_chart(_line_fig(calibration_hist, "step", "calibration", "Calibration error", 180), use_container_width=True)
            s1, s2 = st.columns(2)
            sig_hist = signal_hist if not signal_hist.empty else pd.DataFrame({"predicted_signal": [0.0]})
            s1.plotly_chart(
                px.histogram(sig_hist, x="predicted_signal", nbins=30, title="Predicted Signal Distribution"),
                use_container_width=True,
            )
            if not signal_outcomes.empty:
                binned = signal_outcomes.copy()
                binned["signal_bucket"] = pd.cut(
                    binned["predicted_signal"],
                    bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.00001],
                    labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
                    include_lowest=True,
                )
                relation = (
                    binned.groupby("signal_bucket", observed=False)["outcome"]
                    .mean()
                    .reset_index()
                    .rename(columns={"outcome": "success_rate"})
                )
                s2.plotly_chart(
                    px.bar(relation, x="signal_bucket", y="success_rate", title="Predicted Signal vs Outcome"),
                    use_container_width=True,
                )
            else:
                s2.plotly_chart(
                    px.bar(pd.DataFrame({"signal_bucket": ["0.0-0.2"], "success_rate": [0.0]}), x="signal_bucket", y="success_rate", title="Predicted Signal vs Outcome"),
                    use_container_width=True,
                )

        with ph["feature_panel"].container():
            st.subheader("Top Features (latest retrain)")
            feature_df = (
                pd.DataFrame(st.session_state["feature_importance"], columns=["feature", "importance"])
                if st.session_state["feature_importance"]
                else pd.DataFrame(columns=["feature", "importance"])
            )
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

        with ph["event_log"].container():
            st.subheader("Event Log")
            st.text("\n".join(st.session_state["event_log"][-60:]))

    live_simulation_panel()


if __name__ == "__main__":
    main()
