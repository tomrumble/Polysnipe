"""Research dashboard for simulator + persistence diagnostics.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import json
import html
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.calibration import build_stability_ratio_calibration
from src.persistence_model import PersistenceInputs, PersistenceModel
from src.reporting import generate_simulation_report
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
        last_volatility = 0.0
        entropy_history: List[float] = []

        for i in range(max(self.entropy_window, 20), n - 1):
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

            accel = float((path[i] - path[i - 1]) - (path[i - 1] - path[i - 2]))
            spread = float(abs(rng.normal(0.012 * (1 + out.volatility / 3), 0.004)))
            entropy_value = float(directional_entropy(path[: i + 1].tolist(), window=self.entropy_window))
            entropy_history.append(entropy_value)
            entropy_slope_before_entry = 0.0
            if len(entropy_history) >= 4:
                entropy_slope_before_entry = float(entropy_history[-1] - entropy_history[-4])
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
                    price_acceleration=accel,
                    stability_ratio=out.stability_ratio,
                    volatility_current=out.volatility,
                    volatility_previous=last_volatility if last_volatility > 0 else out.volatility + 1,
                ),
                self.signal_config,
            )

            guard_accel_fail = self.heuristic_guards["acceleration_veto"] and abs(accel) > 3.5
            guard_spread_fail = self.heuristic_guards["spread_veto"] and spread > 0.22
            guard_vol_fail = self.heuristic_guards["volatility_spike_veto"] and out.volatility > base_vol * 1.8
            guard_osc_fail = self.heuristic_guards["oscillation_veto"] and np.sign(path[i] - path[i - 1]) != np.sign(path[i - 1] - path[i - 2])
            heuristic_blocked = guard_accel_fail or guard_spread_fail or guard_vol_fail or guard_osc_fail

            should_trade = strict_decision.should_trade and not heuristic_blocked
            veto_reason = strict_decision.veto_reason
            if not strict_decision.should_trade:
                veto_reason = strict_decision.veto_reason
            elif heuristic_blocked:
                veto_reason = "heuristic_guard"

            signal_reason = strict_decision.signal_reason if should_trade else "none"

            future_path = path[i + 1 : expiry_idx + 1]
            max_price_until_expiry = float(np.max(future_path)) if len(future_path) else current_price
            min_price_until_expiry = float(np.min(future_path)) if len(future_path) else current_price
            expiry_iso = timestamps[expiry_idx].isoformat()

            hit_boundary = (max_price_until_expiry >= boundary_price) if boundary_side > 0 else (min_price_until_expiry <= boundary_price)

            trade_outcome = "NO_TRADE"
            trade_return = 0.0
            failure = 0
            stream_idx = i % stream_count

            if should_trade:
                if hit_boundary:
                    trade_outcome = "LOSS"
                    trade_return = -0.01
                    failure = 1
                else:
                    trade_outcome = "WIN"
                    trade_return = 0.01

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
                "spread": spread,
                "spread_at_entry": spread,
                "price_acceleration": accel,
                "persistence_probability": out.persistence_probability,
                "regime_label": regime.value,
                "signal_reason": signal_reason,
                "veto_reason": veto_reason,
                "heuristic_blocked": heuristic_blocked,
                "trade_outcome": trade_outcome,
                "failure": failure,
                "return": trade_return,
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

        return SimulationOutputs(telemetry=telemetry, trades=trades, equity_curve=equity_curve)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def _strategy_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"trade_count": 0, "win_loss_ratio": 0.0, "avg_return": 0.0, "max_drawdown": 0.0}

    wins = (trades["trade_outcome"] == "WIN").sum()
    losses = (trades["trade_outcome"] == "LOSS").sum()
    return {
        "trade_count": int(len(trades)),
        "win_loss_ratio": float(wins / max(losses, 1)),
        "avg_return": float(trades["return"].mean()),
        "max_drawdown": float(_drawdown_series(equity_curve["equity"]).min()),
    }


def _bucket_seconds(seconds: float) -> str:
    if seconds <= 5:
        return "0-5s"
    if seconds <= 10:
        return "5-10s"
    if seconds <= 20:
        return "10-20s"
    if seconds <= 30:
        return "20-30s"
    return "30s+"


def _success_vs_feature(traded: pd.DataFrame, feature: str, bins: int = 8) -> pd.DataFrame:
    frame = traded[[feature, "failure"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "success_rate"])
    frame["bucket"] = pd.cut(frame[feature], bins=bins, include_lowest=True)
    grouped = frame.groupby("bucket", observed=False).agg(failure_rate=("failure", "mean")).reset_index()
    grouped["success_rate"] = 1.0 - grouped["failure_rate"]
    grouped["bucket"] = grouped["bucket"].astype(str)
    return grouped[["bucket", "success_rate"]]


def _copy_metrics_component(payload: str) -> None:
    """Render a copy-to-clipboard button that runs in the browser with user gesture."""
    # Embed payload in a data attribute (HTML-escaped). Use base64 to avoid quote/newline issues.
    import base64
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <div id="copy-metrics-container">
            <button id="copy-metrics-btn" style="
                padding: 0.4rem 0.8rem;
                font-size: 0.9rem;
                cursor: pointer;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #f0f2f6;
            ">Copy to clipboard</button>
            <span id="copy-metrics-msg" style="margin-left: 8px; font-size: 0.9rem;"></span>
        </div>
        <script>
            (function() {{
                var btn = document.getElementById('copy-metrics-btn');
                var msg = document.getElementById('copy-metrics-msg');
                var payloadB64 = '{payload_b64}';
                btn.addEventListener('click', function() {{
                    try {{
                        var text = atob(payloadB64);
                        navigator.clipboard.writeText(text).then(function() {{
                            msg.textContent = 'Copied!';
                            msg.style.color = 'green';
                        }}, function() {{
                            msg.textContent = 'Clipboard failed (e.g. not HTTPS)';
                            msg.style.color = 'red';
                        }});
                    }} catch (e) {{
                        msg.textContent = 'Error: ' + e.message;
                        msg.style.color = 'red';
                    }}
                }});
            }})();
        </script>
        """,
        height=60,
    )


def main() -> None:
    st.set_page_config(page_title="Trading Simulator Research Dashboard", layout="wide")
    st.title("Trading Simulator Research Dashboard")
    st.caption("Late-stage market collapse detector diagnostics")

    st.sidebar.header("SIMULATOR CONTROL")
    dataset = st.sidebar.selectbox("dataset selector", ["btc_1s_sample", "eth_1s_sample", "macro_high_vol", "btc_binance_api", "eth_binance_api"])

    now = datetime.now().replace(microsecond=0)
    start_default = now - timedelta(hours=1)
    end_default = now

    start_date = st.sidebar.date_input("start date", value=start_default.date())
    start_clock = st.sidebar.time_input("start time", value=start_default.time())
    end_date = st.sidebar.date_input("end date", value=end_default.date())
    end_clock = st.sidebar.time_input("end time", value=end_default.time())

    stream_count = st.sidebar.slider("stream_count", min_value=1, max_value=20, value=4)
    total_capital = st.sidebar.number_input("total_capital", min_value=100.0, value=1000.0, step=100.0)
    persistence_mode = st.sidebar.selectbox("persistence mode", ["model", "empirical"], index=0)
    api_limit = st.sidebar.number_input("binance api candle limit", min_value=24, max_value=1000, value=600, step=1)

    st.sidebar.subheader("state-collapse parameters")
    stability_ratio_threshold = st.sidebar.slider("stability_ratio_threshold", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    entropy_threshold = st.sidebar.slider("entropy_threshold", min_value=0.05, max_value=0.69, value=0.62, step=0.01)
    accel_threshold = st.sidebar.slider("accel_threshold", min_value=0.05, max_value=2.0, value=0.45, step=0.05)
    spread_threshold = st.sidebar.slider("spread_threshold", min_value=0.005, max_value=0.05, value=0.02, step=0.001)
    seconds_threshold = st.sidebar.slider("seconds_remaining_threshold", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

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
    if "last_simulation_output" not in st.session_state:
        st.session_state["last_simulation_output"] = None
    if "last_simulation_config" not in st.session_state:
        st.session_state["last_simulation_config"] = None

    if run_clicked:
        start_dt = datetime.combine(start_date, start_clock)
        end_dt = datetime.combine(end_date, end_clock)

        simulator = ReplaySimulator(
            dataset=dataset,
            heuristic_guards=heuristics,
            signal_config=SignalConfig(
                stability_ratio_threshold=stability_ratio_threshold,
                entropy_threshold=entropy_threshold,
                accel_threshold=accel_threshold,
                spread_threshold=spread_threshold,
                seconds_remaining_threshold=seconds_threshold,
            ),
            api_limit=int(api_limit),
            persistence_mode=persistence_mode,
        )
        st.session_state["sim_outputs"] = simulator.run(
            start_time=start_dt,
            end_time=end_dt,
            stream_count=stream_count,
            total_capital=total_capital,
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
            "seconds_remaining_threshold": seconds_threshold,
            **heuristics,
        }

    outputs: SimulationOutputs | None = st.session_state["sim_outputs"]
    if outputs is None:
        st.info("Set parameters and click **Run Simulation**.")
        return

    if st.session_state.get("last_simulation_output") is not None:
        report = generate_simulation_report(
            st.session_state["last_simulation_output"],
            st.session_state.get("last_simulation_config") or {},
        )
        _copy_metrics_component(report)

    telemetry = outputs.telemetry
    trades = outputs.trades
    traded = telemetry[telemetry["trade_outcome"].isin(["WIN", "LOSS"])].copy()
    metrics = _strategy_metrics(trades, outputs.equity_curve)

    st.header("Strategy Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("trade count", f"{metrics['trade_count']}")
    m2.metric("win/loss ratio", f"{metrics['win_loss_ratio']:.2f}")
    m3.metric("average return", f"{metrics['avg_return'] * 100:.2f}%")
    m4.metric("max drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")

    st.plotly_chart(px.histogram(telemetry, x="seconds_to_expiry_at_entry", nbins=30, title="Trade entry timing histogram"), use_container_width=True)

    st.plotly_chart(px.bar(telemetry["signal_reason"].value_counts().reset_index(), x="signal_reason", y="count", title="Signal reason distribution"), use_container_width=True)

    if not traded.empty:
        regime_matrix = traded.groupby(["regime_label", "trade_outcome"]).size().reset_index(name="count")
        st.plotly_chart(px.density_heatmap(regime_matrix, x="trade_outcome", y="regime_label", z="count", title="Regime success matrix"), use_container_width=True)

        heat = traded.copy()
        heat["vol_bucket"] = pd.cut(heat["volatility"], bins=8)
        heat["ent_bucket"] = pd.cut(heat["directional_entropy"], bins=8)
        failure_heat = heat.groupby(["vol_bucket", "ent_bucket"], observed=False).agg(failure_rate=("failure", "mean")).reset_index()
        failure_heat["vol_bucket"] = failure_heat["vol_bucket"].astype(str)
        failure_heat["ent_bucket"] = failure_heat["ent_bucket"].astype(str)
        st.plotly_chart(px.density_heatmap(failure_heat, x="ent_bucket", y="vol_bucket", z="failure_rate", title="Failure heatmap (volatility vs entropy)"), use_container_width=True)

        sec_diag = traded.groupby(traded["seconds_remaining"].apply(_bucket_seconds), observed=False).agg(failure_rate=("failure", "mean")).reset_index(names="seconds_bucket")
        st.plotly_chart(px.bar(sec_diag, x="seconds_bucket", y="failure_rate", title="Failure rate vs seconds_remaining"), use_container_width=True)

        for feature in ["directional_entropy", "volatility", "spread"]:
            chart_df = _success_vs_feature(traded, feature)
            st.plotly_chart(px.line(chart_df, x="bucket", y="success_rate", title=f"{feature} vs success_rate"), use_container_width=True)

    st.header("Trade Telemetry")
    st.dataframe(telemetry, use_container_width=True, height=340)


if __name__ == "__main__":
    main()
