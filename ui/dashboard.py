"""Research dashboard for simulator + persistence diagnostics.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.persistence_model import PersistenceInputs, PersistenceModel


@dataclass
class SimulationOutputs:
    telemetry: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


class ReplaySimulator:
    """Lightweight replay simulator for research experimentation.

    This simulator generates deterministic pseudo-market trajectories per dataset
    and evaluates the persistence model at each step.
    """

    DATASET_SEEDS: Dict[str, int] = {
        "btc_1s_sample": 17,
        "eth_1s_sample": 43,
        "macro_high_vol": 99,
    }

    def __init__(self, dataset: str, stability_ratio_threshold: float, heuristic_guards: Dict[str, bool]) -> None:
        self.dataset = dataset
        self.stability_ratio_threshold = stability_ratio_threshold
        self.heuristic_guards = heuristic_guards
        self.model = PersistenceModel(center=1.0, slope=1.7)

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

        timestamps = pd.date_range(start_time, periods=n, freq="s")

        base_vol = {
            "btc_1s_sample": 1.2,
            "eth_1s_sample": 1.7,
            "macro_high_vol": 2.6,
        }.get(self.dataset, 1.3)

        noise = rng.normal(0, base_vol, n)
        drift = rng.normal(0.01, 0.05, n).cumsum() / 12
        path = 100 + np.cumsum(noise + drift)

        telemetry_rows: List[Dict[str, float | int | str]] = []
        trades_rows: List[Dict[str, float | int | str]] = []

        stream_balances = np.full(stream_count, total_capital / stream_count)
        equity_points: List[float] = []

        for i in range(20, n - 3):
            current_price = float(path[i])
            boundary_price = current_price + rng.normal(2.5, 1.2)
            recent_prices = path[i - 20 : i + 1].tolist()

            now_ts = timestamps[i].timestamp()
            expiry_ts = (timestamps[i] + timedelta(seconds=30)).timestamp()

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

            trade_outcome = "NO_TRADE"
            trade_return = 0.0
            failure = 0

            if qualifies:
                stream_idx = i % stream_count
                failure_prob = float(1.0 - out.persistence_probability)
                failed = rng.random() < failure_prob
                if failed:
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

    min_val = np.floor(traded["stability_ratio"].min() / bin_width) * bin_width
    max_val = np.ceil(traded["stability_ratio"].max() / bin_width) * bin_width + bin_width
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    traded["stability_bucket"] = pd.cut(traded["stability_ratio"], bins=bins, include_lowest=True)

    grouped = traded.groupby("stability_bucket", observed=False).agg(
        trade_count=("trade_outcome", "count"),
        failure_count=("failure", "sum"),
    )
    grouped = grouped[grouped["trade_count"] > 0].reset_index()
    grouped["failure_rate"] = grouped["failure_count"] / grouped["trade_count"]
    grouped["bucket_center"] = grouped["stability_bucket"].apply(lambda x: x.mid)

    return grouped[["bucket_center", "trade_count", "failure_count", "failure_rate"]]


def main() -> None:
    st.set_page_config(page_title="Trading Simulator Research Dashboard", layout="wide")
    st.title("Trading Simulator Research Dashboard")
    st.caption("Interactive experimentation for persistence model + replay simulator")

    st.sidebar.header("SIMULATOR CONTROL")
    dataset = st.sidebar.selectbox("dataset selector", ["btc_1s_sample", "eth_1s_sample", "macro_high_vol"])

    now = datetime.now().replace(microsecond=0)
    start_default = now - timedelta(hours=1)
    end_default = now

    start_time = st.sidebar.text_input("start time (YYYY-MM-DD HH:MM:SS)", value=start_default.strftime("%Y-%m-%d %H:%M:%S"))
    end_time = st.sidebar.text_input("end time (YYYY-MM-DD HH:MM:SS)", value=end_default.strftime("%Y-%m-%d %H:%M:%S"))
    stream_count = st.sidebar.slider("stream_count", min_value=1, max_value=20, value=4)
    total_capital = st.sidebar.number_input("total_capital", min_value=100.0, value=1000.0, step=100.0)
    stability_ratio_threshold = st.sidebar.slider("stability_ratio_threshold", min_value=0.5, max_value=6.0, value=2.0, step=0.1)

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
        try:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            st.error("Invalid datetime format. Use YYYY-MM-DD HH:MM:SS.")
            return

        simulator = ReplaySimulator(
            dataset=dataset,
            stability_ratio_threshold=stability_ratio_threshold,
            heuristic_guards=heuristics,
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
        st.plotly_chart(px.histogram(telemetry, x="stability_ratio", nbins=50, title="stability_ratio histogram"), use_container_width=True)
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
            "volatility",
            "distance_to_boundary",
            "acceleration",
            "spread",
            "trade_outcome",
        ]
    ].copy()
    st.dataframe(failure_df, use_container_width=True, height=320)

    st.header("Empirical Failure Rate vs Stability Ratio")
    bin_width = st.slider("stability ratio bin width", min_value=0.1, max_value=1.0, step=0.1, value=0.5)
    bucket_df = _failure_rate_by_stability_bucket(telemetry, bin_width=bin_width)

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
            xaxis_title="stability_ratio_bucket_center",
            yaxis=dict(title="empirical_failure_rate", rangemode="tozero"),
            yaxis2=dict(title="trade_count", overlaying="y", side="right", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bucket_df, use_container_width=True, height=220)


if __name__ == "__main__":
    main()
