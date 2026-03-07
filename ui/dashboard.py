"""Edge Validation dashboard for statistically defensible trading edge monitoring.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.calibration import build_stability_ratio_calibration
from src.data import fetch_binance_klines_paginated, resolve_dataset_route
from src.persistence_model import PersistenceInputs, PersistenceModel as DiffusionPersistenceModel
from src.edge.model import PersistenceModel
from src.edge.policy import (
    EXPLORATION_THRESHOLD,
    EXPLOITATION_THRESHOLD,
    MIN_EXPLORATION_SAMPLES,
    TradingPolicy,
)
from src.engine import TrainingEngine
from src.tape import MarketTape
from src.edge.dataset_builder import EdgeDatasetBuilder
from src.edge.edge_score import compute_edge_score
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
        else:
            timestamps, path, dataset_diagnostics = self._load_synthetic_dataset(
                rng=rng,
                base_vol=base_vol,
                timestamps=timestamps,
            )
            n = len(path)

        dataset_diagnostics.setdefault("dataset_loaded", self.dataset if dataset_loader == "binance_api" else "synthetic")

        if dataset_diagnostics.get("api_source") == "synthetic" and self.dataset != "synthetic":
            raise RuntimeError("Dataset mismatch detected")

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
                model_probability = self.edge_model.predict_probability(features)
            self.policy.dataset_size = i
            policy_decision = self.policy.evaluate(model_probability)

            guard_accel_fail = self.heuristic_guards["acceleration_veto"] and abs(accel) > 3.5
            guard_spread_fail = self.heuristic_guards["spread_veto"] and spread > 0.22
            guard_vol_fail = self.heuristic_guards["volatility_spike_veto"] and out.volatility > base_vol * 1.8
            guard_osc_fail = self.heuristic_guards["oscillation_veto"] and np.sign(path[i] - path[i - 1]) != np.sign(path[i - 1] - path[i - 2])
            heuristic_blocked = guard_accel_fail or guard_spread_fail or guard_vol_fail or guard_osc_fail

            should_trade = policy_decision.enter and not heuristic_blocked
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


def main() -> None:
    st.set_page_config(page_title="Edge Validation Panel", layout="wide")
    st.title("Edge Validation")
    st.caption("Scientific instrument panel for statistically defensible edge validation")

    st.sidebar.header("TRAINING ENGINE CONTROL")
    dataset = st.sidebar.selectbox("dataset selector", ["btc_binance_api", "eth_binance_api", "synthetic"])

    now = datetime.now().replace(microsecond=0)
    start_default = now - timedelta(hours=1)
    end_default = now

    if "start_date" not in st.session_state:
        st.session_state["start_date"] = start_default.date()
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = start_default.time()
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = end_default.date()
    if "end_time" not in st.session_state:
        st.session_state["end_time"] = end_default.time()

    start_date = st.sidebar.date_input("start date", value=st.session_state["start_date"], key="start_date")
    start_clock = st.sidebar.time_input("start time", value=st.session_state["start_time"], key="start_time")
    end_date = st.sidebar.date_input("end date", value=st.session_state["end_date"], key="end_date")
    end_clock = st.sidebar.time_input("end time", value=st.session_state["end_time"], key="end_time")

    stream_count = st.sidebar.slider("stream_count", min_value=1, max_value=20, value=4)
    total_capital = st.sidebar.number_input("total_capital", min_value=100.0, value=1000.0, step=100.0)
    transaction_cost = st.sidebar.number_input("transaction_cost_per_trade", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001, format="%.4f")
    persistence_mode = st.sidebar.selectbox("persistence mode", ["model", "empirical"], index=0)
    api_limit = st.sidebar.number_input("binance api candle limit", min_value=24, max_value=1000, value=600, step=1)

    st.sidebar.subheader("state-collapse parameters")
    stability_ratio_threshold = st.sidebar.slider("stability_ratio_threshold", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    entropy_threshold = st.sidebar.slider("entropy_threshold", min_value=0.05, max_value=0.69, value=0.62, step=0.01)
    accel_threshold = st.sidebar.slider("accel_threshold", min_value=0.05, max_value=2.0, value=0.45, step=0.05)
    spread_threshold = st.sidebar.slider("spread_threshold", min_value=0.005, max_value=0.05, value=0.02, step=0.001)

    st.sidebar.subheader("heuristic guard toggles")
    heuristics = {
        "acceleration_veto": st.sidebar.checkbox("acceleration_veto", value=True),
        "oscillation_veto": st.sidebar.checkbox("oscillation_veto", value=True),
        "spread_veto": st.sidebar.checkbox("spread_veto", value=True),
        "volatility_spike_veto": st.sidebar.checkbox("volatility_spike_veto", value=True),
    }

    include_simulation = st.sidebar.checkbox("Enable replay simulation section", value=False)
    run_clicked = st.sidebar.button("Run Training Engine", type="primary")

    if "sim_outputs" not in st.session_state:
        st.session_state["sim_outputs"] = None
    if "last_simulation_output" not in st.session_state:
        st.session_state["last_simulation_output"] = None
    if "last_simulation_config" not in st.session_state:
        st.session_state["last_simulation_config"] = None
    if "run_error" not in st.session_state:
        st.session_state["run_error"] = ""
    if "engine_state" not in st.session_state:
        st.session_state["engine_state"] = None

    if run_clicked:
        start_dt = datetime.combine(start_date, start_clock)
        end_dt = datetime.combine(end_date, end_clock)

        try:
            dataset_route = resolve_dataset_route(dataset)
            if dataset_route.loader_name != "parquet":
                raise RuntimeError("Training engine requires a parquet-backed dataset route")

            tape = MarketTape(dataset_route.path)
            engine = TrainingEngine(
                tape=tape,
                dataset_builder=EdgeDatasetBuilder(),
                retrain_interval=1000,
                metric_interval=100,
                horizon_ticks=60,
            )
            engine.start()
            max_iterations = max(int((end_dt - start_dt).total_seconds()), 1)
            st.session_state["engine_state"] = engine.run(max_iterations=max_iterations)
            engine.stop()
            st.session_state["run_error"] = ""

            if include_simulation:
                simulator = ReplaySimulator(
                    dataset=dataset,
                    heuristic_guards=heuristics,
                    signal_config=SignalConfig(
                        stability_ratio_threshold=stability_ratio_threshold,
                        entropy_threshold=entropy_threshold,
                        accel_threshold=accel_threshold,
                        spread_threshold=spread_threshold,
                        evaluation_window_seconds=60.0,
                    ),
                    api_limit=int(api_limit),
                    persistence_mode=persistence_mode,
                )
                st.session_state["sim_outputs"] = simulator.run(
                    start_time=start_dt,
                    end_time=end_dt,
                    stream_count=stream_count,
                    total_capital=total_capital,
                    transaction_cost=transaction_cost,
                )
                st.session_state["last_simulation_output"] = st.session_state["sim_outputs"]
                st.session_state["last_simulation_config"] = {
                    "dataset": dataset,
                    "start": start_dt,
                    "end": end_dt,
                    "stream_count": stream_count,
                    "total_capital": total_capital,
                    "stability_ratio_threshold": stability_ratio_threshold,
                    "entropy_threshold": entropy_threshold,
                    "accel_threshold": accel_threshold,
                    "spread_threshold": spread_threshold,
                    "evaluation_window_seconds": 60.0,
                    "transaction_cost": transaction_cost,
                    **heuristics,
                }
            else:
                st.session_state["sim_outputs"] = None
                st.session_state["last_simulation_output"] = None
                st.session_state["last_simulation_config"] = None
        except Exception as exc:
            st.session_state["sim_outputs"] = None
            st.session_state["engine_state"] = None
            st.session_state["run_error"] = str(exc)

    engine_state = st.session_state.get("engine_state")
    outputs: SimulationOutputs | None = st.session_state["sim_outputs"]
    if outputs is None:
        if st.session_state.get("run_error"):
            st.error(f"DATASET ERROR: {st.session_state['run_error']}")
        elif engine_state is None:
            st.info("Set parameters and click **Run Training Engine**. Optional: enable replay simulation for full analytics.")
        else:
            st.success("Training engine completed. Enable replay simulation section for analytics panels.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Observations Seen", f"{engine_state.observations_seen:,}")
            c2.metric("Dataset Size", f"{engine_state.dataset_size:,}")
            c3.metric("Retrains", f"{engine_state.retrains:,}")
            c4.metric("Runtime State", str(engine_state.runtime_state))
            if engine_state.latest_metrics:
                st.json(engine_state.latest_metrics)
        return

    loaded_source = outputs.dataset_diagnostics.get("api_source", "")
    selected_dataset = st.session_state.get("last_simulation_config", {}).get("dataset", "")
    if loaded_source == "synthetic" and selected_dataset != "synthetic":
        st.warning(
            f"DATASET WARNING\nDataset selected: {selected_dataset}\nDataset loaded: {loaded_source}"
        )

    telemetry = outputs.telemetry
    trades = outputs.trades
    traded = _trade_table(telemetry)

    calibration_df, calibration_error = _calibration_frame(traded)
    ranking_df, spearman = _ranking_frame(traded)
    profitability, trade_curve = _profitability_metrics(traded, transaction_cost=transaction_cost)
    stability_df = _window_stability(trade_curve, windows=[500, 1000, 5000])
    trade_rate = float(len(trades) / max(len(telemetry), 1))

    edge_components = compute_edge_score(
        expected_value=profitability["expected_value"],
        calibration_error=calibration_error,
        probability_rank_correlation=spearman,
        max_drawdown=profitability["max_drawdown"],
        trade_rate=trade_rate,
    )
    edge_score = edge_components["edge_score"]
    recent_stability = float(stability_df["expected_value"].iloc[0]) if not stability_df.empty else 0.0

    if edge_score >= 0.8 and profitability["expected_value"] > 0 and calibration_error < 0.05 and recent_stability > 0:
        edge_status = "Strong Edge Candidate"
        status_color = "#0f9d58"
    elif edge_score >= 0.6 and profitability["expected_value"] > 0 and calibration_error < 0.1:
        edge_status = "Promising Edge"
        status_color = "#1f77b4"
    elif edge_score >= 0.3 and profitability["expected_value"] >= 0:
        edge_status = "Weak Signal"
        status_color = "#f9a825"
    else:
        edge_status = "No Edge Detected"
        status_color = "#c62828"

    st.markdown(
        f"<div style='padding:0.9rem;border-radius:0.5rem;background:{status_color};color:white;font-weight:700;'>EDGE STATUS: {edge_status}</div>",
        unsafe_allow_html=True,
    )

    gauge = pd.DataFrame({"edge_score": [edge_score]})
    st.plotly_chart(
        px.bar(gauge, x=["Edge Score"], y="edge_score", range_y=[0, 1], title=f"Edge Score Gauge: {edge_score:.2f}"),
        use_container_width=True,
    )

    run_config = st.session_state.get("last_simulation_config", {})
    run_config.setdefault("transaction_cost", transaction_cost)
    metrics_text = _edge_validation_metrics_text(
        edge_status=edge_status,
        edge_score=edge_score,
        calibration_error=calibration_error,
        spearman=spearman,
        profitability=profitability,
        stability_df=stability_df,
        config=run_config,
    )
    metrics_b64 = base64.b64encode(metrics_text.encode()).decode()
    payload_attr = metrics_b64.replace("&", "&amp;").replace('"', "&quot;")
    copy_html = f"""
    <div id="edge-metrics-b64" data-payload="{payload_attr}"></div>
    <button id="copy-edge-metrics" style="
        padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
        cursor: pointer;
        border-radius: 0.35rem;
        border: 1px solid #ccc;
        background: #f0f2f6;
    ">Copy edge validation metrics</button>
    <span id="copy-feedback" style="margin-left: 0.5rem; font-size: 0.9rem; color: #0f9d58;"></span>
    <script>
    (function() {{
        var el = document.getElementById('edge-metrics-b64');
        var b64 = el && el.getAttribute('data-payload') || '';
        var text = b64 ? atob(b64) : '';
        var btn = document.getElementById('copy-edge-metrics');
        var feedback = document.getElementById('copy-feedback');
        if (btn && text) {{
            btn.onclick = function() {{
                navigator.clipboard.writeText(text).then(function() {{
                    feedback.textContent = 'Copied!';
                    setTimeout(function() {{ feedback.textContent = ''; }}, 2000);
                }});
            }};
        }}
    }})();
    </script>
    """
    components.html(copy_html, height=50)

    st.subheader("Model Calibration")
    st.metric("Calibration Error (mean absolute gap)", f"{calibration_error:.3f}")
    if calibration_error < 0.05:
        st.success("Good calibration: error < 0.05")
    elif calibration_error > 0.10:
        st.error("Poor calibration: error > 0.10")
    else:
        st.info("Calibration is marginal; monitor drift closely.")
    if not calibration_df.empty:
        st.plotly_chart(
            px.line(calibration_df, x="predicted_probability_mean", y="actual_persistence_rate", markers=True, title="Calibration Curve: Expected vs Actual"),
            use_container_width=True,
        )

    st.subheader("Probability Ranking Quality")
    st.metric("Spearman Rank Correlation", f"{spearman:.3f}")
    if spearman > 0.3:
        st.success("Strong ranking signal: correlation > 0.3")
    else:
        st.warning("Ranking signal is weak: correlation <= 0.3")
    monotonic = True
    non_empty_ranking = ranking_df[ranking_df["trade_count"] > 0]
    if len(non_empty_ranking) > 1:
        monotonic = bool(non_empty_ranking["win_rate"].fillna(0.0).is_monotonic_increasing)
    st.caption("Monotonic win-rate progression by bucket is expected.")
    if monotonic:
        st.info("Win-rate buckets are monotonic increasing.")
    else:
        st.warning("Win-rate buckets are not monotonic; ranking quality may be unstable.")
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(ranking_df, x="probability_bucket", y="win_rate", title="Win Rate by Probability Bucket"), use_container_width=True)

    st.subheader("Profitability Metrics")
    p1, p2, p3 = st.columns(3)
    p1.metric("Total Trades", f"{int(profitability['total_trades'])}")
    p2.metric("Win Rate", f"{profitability['win_rate'] * 100:.2f}%")
    p3.metric("Average Return / Trade", f"{profitability['avg_return'] * 100:.3f}%")
    p4, p5, p6 = st.columns(3)
    p4.metric("Expected Value / Trade", f"{profitability['expected_value'] * 100:.3f}%")
    p5.metric("Max Drawdown", f"{profitability['max_drawdown'] * 100:.2f}%")
    p6.metric("Sharpe Ratio", f"{profitability['sharpe']:.2f}")
    st.plotly_chart(px.line(trade_curve, x="timestamp", y="equity", title="Simulated Strategy Equity Curve"), use_container_width=True)
    if profitability["expected_value"] > 0:
        st.success("Positive edge candidate: expected value is above zero.")
    else:
        st.error("No positive expectancy detected: expected value is <= 0.")

    st.subheader("Edge Stability")
    st.dataframe(stability_df, use_container_width=True, hide_index=True)
    if not stability_df.empty and float(stability_df.iloc[0]["expected_value"]) < 0:
        st.warning("Recent-window expected value is negative; edge may be decaying.")
    if not trade_curve.empty:
        rolling = trade_curve[["timestamp", "net_return"]].copy()
        rolling["rolling_expected_value_500"] = rolling["net_return"].rolling(window=500, min_periods=50).mean()
        st.plotly_chart(px.line(rolling, x="timestamp", y="rolling_expected_value_500", title="Rolling Expected Value (500 Trades)"), use_container_width=True)

    st.subheader("Trade Selectivity")
    training_samples = len(telemetry)
    exploration_mode = training_samples < MIN_EXPLORATION_SAMPLES
    threshold = EXPLORATION_THRESHOLD if exploration_mode else EXPLOITATION_THRESHOLD
    metrics_path = Path("models/model_metrics.json")
    model_metrics: dict = {}
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text())
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Observations Evaluated", f"{training_samples}")
    t2.metric("Total Trades Executed", f"{len(trades)}")
    t3.metric("Trade Frequency", f"{trade_rate * 100:.2f}%")
    t4.metric("Policy Probability Threshold", f"{threshold:.2f}")

    t5, t6, t7 = st.columns(3)
    t5.metric("training_samples", f"{training_samples:,}")
    t6.metric("exploration_mode", "true" if exploration_mode else "false")
    t7.metric("policy_threshold", f"{threshold:.2f}")

    if engine_state is not None:
        st.subheader("Runtime Training Engine")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations Seen", f"{engine_state.observations_seen:,}")
        c2.metric("Dataset Size", f"{engine_state.dataset_size:,}")
        c3.metric("Retrains", f"{engine_state.retrains:,}")
        c4.metric("Runtime State", str(engine_state.runtime_state))
        if engine_state.latest_metrics:
            st.json(engine_state.latest_metrics)

    with st.expander("Research Diagnostics (secondary)"):
        st.dataframe(telemetry, use_container_width=True, height=280)


if __name__ == "__main__":
    main()
