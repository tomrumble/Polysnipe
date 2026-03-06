"""Research dashboard for simulator + persistence diagnostics.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.calibration import build_stability_ratio_calibration
from src.persistence_model import PersistenceInputs, PersistenceModel


@dataclass
class SimulationOutputs:
    telemetry: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


class ReplaySimulator:
    """Lightweight replay simulator for research experimentation."""

    DATASET_SEEDS: Dict[str, int] = {
        "btc_1s_sample": 17,
        "eth_1s_sample": 43,
        "macro_high_vol": 99,
    }
    API_DATASET_SYMBOLS: Dict[str, str] = {
        "btc_binance_api": "BTCUSDT",
        "eth_binance_api": "ETHUSDT",
    }
    CACHE_DIR = Path("datasets/cache")

    def __init__(
        self,
        dataset: str,
        stability_ratio_threshold: float,
        heuristic_guards: Dict[str, bool],
        api_limit: int = 600,
        persistence_mode: str = "model",
    ) -> None:
        self.dataset = dataset
        self.stability_ratio_threshold = stability_ratio_threshold
        self.heuristic_guards = heuristic_guards
        self.api_limit = api_limit
        self.model = PersistenceModel(center=1.0, slope=1.7, mode=persistence_mode)

    @classmethod
    def _cache_path(cls, symbol: str, limit: int) -> Path:
        return cls.CACHE_DIR / f"{symbol.lower()}_1s_{limit}.csv"

    @classmethod
    def _load_cached_binance_prices(cls, symbol: str, limit: int) -> pd.DataFrame:
        cache_path = cls._cache_path(symbol, limit)
        if not cache_path.exists():
            return pd.DataFrame(columns=["timestamp", "price"])

        cached = pd.read_csv(cache_path)
        if cached.empty:
            return pd.DataFrame(columns=["timestamp", "price"])

        cached["timestamp"] = pd.to_datetime(cached["timestamp"])
        return cached[["timestamp", "price"]]

    @classmethod
    def _fetch_binance_prices(cls, symbol: str, limit: int) -> pd.DataFrame:
        cached = cls._load_cached_binance_prices(symbol=symbol, limit=limit)
        if not cached.empty:
            return cached

        query = urlencode({"symbol": symbol, "interval": "1s", "limit": limit})
        url = f"https://api.binance.com/api/v3/klines?{query}"

        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        rows = [
            {
                "timestamp": datetime.fromtimestamp(float(row[0]) / 1000),
                "price": float(row[4]),
            }
            for row in payload
        ]
        frame = pd.DataFrame(rows, columns=["timestamp", "price"])

        if frame.empty:
            return frame

        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cls._cache_path(symbol=symbol, limit=limit), index=False)
        return frame

    def run(
        self,
        start_time: datetime,
        end_time: datetime,
        stream_count: int,
        total_capital: float,
    ) -> SimulationOutputs:
        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")

        seconds = int((end_time - start_time).total_seconds())
        n = max(seconds, 120)

        seed = self.DATASET_SEEDS.get(self.dataset, 7) + stream_count
        rng = np.random.default_rng(seed)

        base_vol = {
            "btc_1s_sample": 1.2,
            "btc_binance_api": 1.2,
            "eth_1s_sample": 1.7,
            "eth_binance_api": 1.7,
            "macro_high_vol": 2.6,
        }.get(self.dataset, 1.3)

        timestamps = pd.date_range(start_time, periods=n, freq="s")
        path: np.ndarray

        if self.dataset in self.API_DATASET_SYMBOLS:
            symbol = self.API_DATASET_SYMBOLS[self.dataset]
            try:
                frame = self._fetch_binance_prices(symbol=symbol, limit=min(self.api_limit, 1000))
                if len(frame) >= 24:
                    timestamps = pd.to_datetime(frame["timestamp"])
                    path = frame["price"].to_numpy(dtype=float)
                    n = len(path)
                else:
                    raise ValueError("Binance returned insufficient candles for simulation")
            except Exception:
                noise = rng.normal(0, base_vol, n)
                drift = rng.normal(0.01, 0.05, n).cumsum() / 12
                path = 100 + np.cumsum(noise + drift)
        else:
            noise = rng.normal(0, base_vol, n)
            drift = rng.normal(0.01, 0.05, n).cumsum() / 12
            path = 100 + np.cumsum(noise + drift)

        telemetry_rows: List[Dict[str, float | int | str]] = []
        trades_rows: List[Dict[str, float | int | str]] = []

        stream_balances = np.full(stream_count, total_capital / stream_count)
        equity_points: List[float] = []

        for i in range(20, n - 1):
            current_price = float(path[i])
            boundary_delta = 2.5
            boundary_side = 1.0 if path[i] - path[i - 1] >= 0 else -1.0
            boundary_price = current_price + boundary_side * boundary_delta
            recent_prices = path[i - 20 : i + 1].tolist()

            expiry_idx = min(i + 30, n - 1)
            seconds_remaining = float((timestamps[expiry_idx] - timestamps[i]).total_seconds())
            now_ts = timestamps[i].timestamp()
            expiry_ts = timestamps[expiry_idx].timestamp()

            out = self.model.compute(
                PersistenceInputs(
                    current_price=current_price,
                    boundary_price=boundary_price,
                    expiry_timestamp=expiry_ts,
                    now_timestamp=now_ts,
                    recent_prices=recent_prices,
                )
            )

            acceleration = float((path[i] - path[i - 1]) - (path[i - 1] - path[i - 2]))
            spread = float(abs(rng.normal(0.08 * (1 + out.volatility / 3), 0.03)))
            distance = float(out.distance_to_boundary)

            guard_accel_fail = self.heuristic_guards["acceleration_veto"] and abs(acceleration) > 3.5
            guard_spread_fail = self.heuristic_guards["spread_veto"] and spread > 0.22
            guard_vol_fail = self.heuristic_guards["volatility_spike_veto"] and out.volatility > base_vol * 1.8
            guard_osc_fail = self.heuristic_guards["oscillation_veto"] and np.sign(path[i] - path[i - 1]) != np.sign(path[i - 1] - path[i - 2])

            heuristic_blocked = guard_accel_fail or guard_spread_fail or guard_vol_fail or guard_osc_fail
            qualifies = out.stability_ratio >= self.stability_ratio_threshold and not heuristic_blocked

            future_path = path[i + 1 : expiry_idx + 1]
            max_price_until_expiry = float(np.max(future_path)) if len(future_path) else current_price
            min_price_until_expiry = float(np.min(future_path)) if len(future_path) else current_price
            expiry_timestamp = timestamps[expiry_idx].isoformat()

            hit_boundary = (max_price_until_expiry >= boundary_price) if boundary_side > 0 else (min_price_until_expiry <= boundary_price)

            trade_outcome = "NO_TRADE"
            trade_return = 0.0
            failure = 0

            if qualifies:
                stream_idx = i % stream_count
                if hit_boundary:
                    trade_outcome = "LOSS"
                    trade_return = -0.01
                    failure = 1
                else:
                    trade_outcome = "WIN"
                    trade_return = 0.01

                stream_balances[stream_idx] *= 1.0 + trade_return
                trades_rows.append(
                    {
                        "timestamp": timestamps[i],
                        "stream_id": stream_idx,
                        "stability_ratio": out.stability_ratio,
                        "persistence_probability": out.persistence_probability,
                        "seconds_remaining": seconds_remaining,
                        "boundary_price": boundary_price,
                        "expiry_timestamp": expiry_timestamp,
                        "max_price_until_expiry": max_price_until_expiry,
                        "min_price_until_expiry": min_price_until_expiry,
                        "return": trade_return,
                        "trade_outcome": trade_outcome,
                    }
                )

            telemetry_rows.append(
                {
                    "timestamp": timestamps[i],
                    "stability_ratio": out.stability_ratio,
                    "persistence_probability": out.persistence_probability,
                    "volatility": out.volatility,
                    "distance_to_boundary": distance,
                    "seconds_remaining": seconds_remaining,
                    "boundary_price": boundary_price,
                    "expiry_timestamp": expiry_timestamp,
                    "max_price_until_expiry": max_price_until_expiry,
                    "min_price_until_expiry": min_price_until_expiry,
                    "acceleration": acceleration,
                    "spread": spread,
                    "trade_outcome": trade_outcome,
                    "failure": failure,
                    "heuristic_blocked": heuristic_blocked,
                }
            )

            equity_points.append(float(stream_balances.sum()))

        telemetry = pd.DataFrame(telemetry_rows)
        trades = pd.DataFrame(trades_rows)
        equity_curve = pd.DataFrame(
            {
                "timestamp": telemetry["timestamp"],
                "equity": equity_points,
            }
        )

        telemetry_path = Path("datasets/telemetry/trade_history.csv")
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry.to_csv(telemetry_path, index=False)
        build_stability_ratio_calibration([telemetry_path])

        return SimulationOutputs(telemetry=telemetry, trades=trades, equity_curve=equity_curve)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def _strategy_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_loss_ratio": 0.0,
            "avg_return": 0.0,
            "max_drawdown": 0.0,
        }

    wins = (trades["trade_outcome"] == "WIN").sum()
    losses = (trades["trade_outcome"] == "LOSS").sum()
    win_loss_ratio = float(wins / max(losses, 1))
    avg_return = float(trades["return"].mean())
    max_drawdown = float(_drawdown_series(equity_curve["equity"]).min())

    return {
        "trade_count": int(len(trades)),
        "win_loss_ratio": win_loss_ratio,
        "avg_return": avg_return,
        "max_drawdown": max_drawdown,
    }


def _failure_rate_by_stability_bucket(telemetry: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    traded = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    if traded.empty:
        return pd.DataFrame(columns=["bucket_center", "trade_count", "failure_count", "failure_rate"])

    bins = [0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ["0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"]
    traded["stability_bucket"] = pd.cut(
        traded["stability_ratio"], bins=bins, include_lowest=True, right=False, labels=labels
    )

    grouped = traded.groupby("stability_bucket", observed=False).agg(
        trade_count=("trade_outcome", "count"),
        failure_count=("failure", "sum"),
    )
    grouped = grouped[grouped["trade_count"] > 0].reset_index()
    grouped["failure_rate"] = grouped["failure_count"] / grouped["trade_count"]
    grouped["bucket_center"] = grouped["stability_bucket"].astype(str)

    return grouped[["bucket_center", "trade_count", "failure_count", "failure_rate"]]


def _failure_rate_by_seconds_remaining(telemetry: pd.DataFrame, second_buckets: list[int]) -> pd.DataFrame:
    traded = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    if traded.empty:
        return pd.DataFrame(columns=["seconds_remaining_bucket", "trade_count", "failure_rate"])

    traded["seconds_remaining_bucket"] = traded["seconds_remaining"].apply(
        lambda x: min(second_buckets, key=lambda y: abs(y - x))
    )
    grouped = traded.groupby("seconds_remaining_bucket", observed=False).agg(
        trade_count=("failure", "count"),
        failure_count=("failure", "sum"),
    ).reset_index()
    grouped["failure_rate"] = grouped["failure_count"] / grouped["trade_count"]
    return grouped


def _persistence_surface(telemetry: pd.DataFrame) -> pd.DataFrame:
    traded = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    if traded.empty:
        return pd.DataFrame(columns=["stability_ratio_bucket", "seconds_remaining_bucket", "failure_rate", "trade_count"])

    stability_bins = [0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
    stability_labels = ["0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"]
    seconds_buckets = [5, 10, 15, 20, 30]

    traded["stability_ratio_bucket"] = pd.cut(
        traded["stability_ratio"], bins=stability_bins, labels=stability_labels, include_lowest=True, right=False
    )
    traded["seconds_remaining_bucket"] = traded["seconds_remaining"].apply(
        lambda x: f"{min(seconds_buckets, key=lambda y: abs(y - x))}s"
    )

    grouped = traded.groupby(["stability_ratio_bucket", "seconds_remaining_bucket"], observed=False).agg(
        trade_count=("failure", "count"),
        failures=("failure", "sum"),
    ).reset_index()
    grouped["failure_rate"] = np.where(grouped["trade_count"] > 0, grouped["failures"] / grouped["trade_count"], np.nan)
    return grouped


def main() -> None:
    st.set_page_config(page_title="Trading Simulator Research Dashboard", layout="wide")
    st.title("Trading Simulator Research Dashboard")
    st.caption("Interactive experimentation for persistence model + replay simulator")

    st.sidebar.header("SIMULATOR CONTROL")
    dataset = st.sidebar.selectbox(
        "dataset selector",
        ["btc_1s_sample", "eth_1s_sample", "macro_high_vol", "btc_binance_api", "eth_binance_api"],
    )

    now = datetime.now().replace(microsecond=0)
    start_default = now - timedelta(hours=1)
    end_default = now

    start_date = st.sidebar.date_input("start date", value=start_default.date())
    start_clock = st.sidebar.time_input("start time", value=start_default.time())
    end_date = st.sidebar.date_input("end date", value=end_default.date())
    end_clock = st.sidebar.time_input("end time", value=end_default.time())

    stream_count = st.sidebar.slider("stream_count", min_value=1, max_value=20, value=4)
    total_capital = st.sidebar.number_input("total_capital", min_value=100.0, value=1000.0, step=100.0)
    stability_ratio_threshold = st.sidebar.slider("stability_ratio_threshold", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    persistence_mode = st.sidebar.selectbox("persistence mode", ["model", "empirical"], index=0)
    api_limit = st.sidebar.number_input("binance api candle limit", min_value=24, max_value=1000, value=600, step=1)

    st.sidebar.subheader("heuristic guard toggles")
    heuristics = {
        "acceleration_veto": st.sidebar.checkbox("acceleration_veto", value=True),
        "oscillation_veto": st.sidebar.checkbox("oscillation_veto", value=True),
        "spread_veto": st.sidebar.checkbox("spread_veto", value=True),
        "volatility_spike_veto": st.sidebar.checkbox("volatility_spike_veto", value=True),
    }

    run_clicked = st.sidebar.button("Run Simulation", type="primary")

    if "sim_outputs" not in st.session_state:
        st.session_state["sim_outputs"] = None

    if run_clicked:
        start_dt = datetime.combine(start_date, start_clock)
        end_dt = datetime.combine(end_date, end_clock)

        simulator = ReplaySimulator(
            dataset=dataset,
            stability_ratio_threshold=stability_ratio_threshold,
            heuristic_guards=heuristics,
            api_limit=int(api_limit),
            persistence_mode=persistence_mode,
        )
        st.session_state["sim_outputs"] = simulator.run(
            start_time=start_dt,
            end_time=end_dt,
            stream_count=stream_count,
            total_capital=total_capital,
        )

    outputs: SimulationOutputs | None = st.session_state["sim_outputs"]
    if outputs is None:
        st.info("Set parameters and click **Run Simulation**.")
        return

    telemetry = outputs.telemetry
    trades = outputs.trades
    equity_curve = outputs.equity_curve

    st.header("PERSISTENCE DIAGNOSTICS")
    c1, c2 = st.columns(2)
    with c1:
        stability_histogram = px.histogram(telemetry, x="stability_ratio", nbins=50, title="stability_ratio histogram")
        stability_histogram.add_vline(
            x=stability_ratio_threshold,
            line_color="red",
            line_width=2,
            line_dash="dash",
            annotation_text="threshold",
            annotation_position="top right",
        )
        st.plotly_chart(stability_histogram, use_container_width=True)
        st.plotly_chart(px.histogram(telemetry, x="volatility", nbins=50, title="volatility distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(
            px.histogram(telemetry, x="persistence_probability", nbins=50, title="persistence_probability histogram"),
            use_container_width=True,
        )
        st.plotly_chart(px.histogram(telemetry, x="distance_to_boundary", nbins=50, title="distance_to_boundary distribution"), use_container_width=True)

    st.header("STRATEGY RESULTS")
    metrics = _strategy_metrics(trades, equity_curve)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("trade count", f"{metrics['trade_count']}")
    m2.metric("win/loss ratio", f"{metrics['win_loss_ratio']:.2f}")
    m3.metric("average return", f"{metrics['avg_return'] * 100:.2f}%")
    m4.metric("max drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")

    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(px.line(equity_curve, x="timestamp", y="equity", title="equity curve"), use_container_width=True)
    with e2:
        dd = equity_curve.copy()
        dd["drawdown"] = _drawdown_series(dd["equity"])
        st.plotly_chart(px.area(dd, x="timestamp", y="drawdown", title="drawdown"), use_container_width=True)

    st.header("FAILURE ANALYSIS")
    failure_df = telemetry[
        [
            "timestamp",
            "stability_ratio",
            "seconds_remaining",
            "boundary_price",
            "expiry_timestamp",
            "max_price_until_expiry",
            "min_price_until_expiry",
            "volatility",
            "distance_to_boundary",
            "acceleration",
            "spread",
            "trade_outcome",
        ]
    ].copy()
    st.dataframe(failure_df, use_container_width=True, height=320)

    st.header("Empirical failure rate vs stability_ratio")
    bucket_df = _failure_rate_by_stability_bucket(telemetry, bin_width=0.5)

    if bucket_df.empty:
        st.warning("No executed trades for current parameters. Lower threshold or disable guard vetoes.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bucket_df["bucket_center"],
                y=bucket_df["failure_rate"],
                mode="lines+markers",
                name="empirical_failure_rate",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Bar(
                x=bucket_df["bucket_center"],
                y=bucket_df["trade_count"],
                name="trade_count",
                yaxis="y2",
                opacity=0.35,
            )
        )

        fig.update_layout(
            xaxis_title="stability_ratio_bucket",
            yaxis=dict(title="empirical_failure_rate", rangemode="tozero"),
            yaxis2=dict(title="trade_count", overlaying="y", side="right", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bucket_df, use_container_width=True, height=220)

    st.header("Empirical failure rate vs seconds_remaining")
    seconds_df = _failure_rate_by_seconds_remaining(telemetry, second_buckets=[5, 10, 15, 20, 30])
    if seconds_df.empty:
        st.warning("No executed trades for current parameters.")
    else:
        st.plotly_chart(
            px.bar(
                seconds_df,
                x="seconds_remaining_bucket",
                y="failure_rate",
                title="Empirical failure rate by seconds remaining bucket",
                labels={"seconds_remaining_bucket": "seconds_remaining_bucket", "failure_rate": "failure_rate"},
            ),
            use_container_width=True,
        )
        st.dataframe(seconds_df, use_container_width=True, height=220)

    st.header("PERSISTENCE SURFACE")
    surface_df = _persistence_surface(telemetry)
    if surface_df.empty:
        st.warning("No executed trades for persistence surface.")
    else:
        heatmap_df = surface_df.pivot(index="stability_ratio_bucket", columns="seconds_remaining_bucket", values="failure_rate")
        count_df = surface_df.pivot(index="stability_ratio_bucket", columns="seconds_remaining_bucket", values="trade_count")
        heatmap_fig = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            labels={"x": "seconds_remaining_bucket", "y": "stability_ratio_bucket", "color": "failure_rate"},
            title="Empirical persistence surface (failure rate)",
        )
        heatmap_fig.update_traces(
            customdata=np.expand_dims(count_df.to_numpy(), axis=-1),
            hovertemplate="stability=%{y}<br>seconds=%{x}<br>failure_rate=%{z:.3f}<br>trade_count=%{customdata[0]:.0f}<extra></extra>",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.dataframe(surface_df, use_container_width=True, height=260)


if __name__ == "__main__":
    main()
