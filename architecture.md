# Polysnipe Architecture

## 1. System Objective

The system estimates the probability that a market state will not cross a boundary within a short time horizon and deploys capital across segmented bankroll streams to exploit high-persistence opportunities while minimizing catastrophic loss.

Formally the bot estimates:

P(state persists | d, σ, T, μ, microstructure)

Where:

- **d** = distance to boundary
- **σ** = instantaneous volatility
- **T** = time remaining
- **μ** = short-term drift
- **microstructure signals** = orderbook and price path features

The system trades only when:

```text
estimated_failure_probability < risk_threshold
```

---

## 2. High-Level Architecture

The system is composed of seven primary modules.

```text
Market Feed
    ↓
Market State Extractor
    ↓
Persistence Model
    ↓
Heuristic Guard Layer
    ↓
Opportunity Ranking
    ↓
Stream Allocator
    ↓
Execution Engine
```

Parallel to this:

- Telemetry System
- Replay Simulator
- Model Calibration

---

## 3. Core System Principles

### Principle 1 — State Persistence

The system detects when system evolution cannot realistically cross the boundary in remaining time.

### Principle 2 — Asymmetric Risk Control

Because losses are catastrophic, the system optimizes tail risk reduction, not raw return.

### Principle 3 — Capital Segmentation

Capital is divided into independent streams to prevent a single failure from collapsing the entire bankroll.

### Principle 4 — Opportunity Ranking

Capital is deployed only to the highest persistence probability opportunities.

---

## 4. Data Inputs

The system consumes the following real-time data streams.

### Market price updates

- timestamp
- price
- trade_size
- trade_side

### Orderbook data

- best_bid
- best_ask
- bid_depth
- ask_depth
- spread

### Market metadata

- market_id
- expiry_time
- underlying_asset

---

## 5. Market State Extractor

The extractor converts raw data into state variables used by the persistence model.

For each market tick compute:

- price
- time_remaining
- distance_to_boundary
- volatility_estimate
- drift_estimate
- price_path_features
- orderbook_features

### 5.1 Distance to Boundary

For up/down markets:

```text
distance_to_flip = abs(current_price - strike_price)
```

### 5.2 Time Remaining

```text
T = expiry_timestamp - current_timestamp
```

Only evaluate when:

```text
0 < T < entry_window
```

Example: `entry_window = 25` seconds

### 5.3 Volatility Estimate

Volatility is estimated from recent price movement.

Example:

```text
σ = std(price_change over last 10 seconds)
```

Alternative robust estimator:

```text
σ = sqrt(mean(square(price_returns)))
```

### 5.4 Drift Estimate

Short-term drift captures directional movement.

```text
μ = mean(price_change over last 5 seconds)
```

### 5.5 Price Path Features

These capture shape of the last movement.

Examples:

- direction_flips_last_20s
- max_move_last_10s
- acceleration
- momentum_decay

### 5.6 Orderbook Features

Examples:

```text
spread_ratio = (ask - bid) / mid
depth_imbalance = (bid_depth - ask_depth) / total_depth
```

---

## 6. Persistence Model

The model estimates probability the price does not cross the boundary before expiry.

Base formula derived from diffusion first-passage theory:

```text
remaining_move_capacity = σ * sqrt(T)
stability_ratio = distance_to_flip / remaining_move_capacity
```

Higher ratios imply greater persistence.

### 6.1 Drift Adjustment

If drift moves away from boundary:

```text
adjusted_distance = distance_to_flip - μ * T
```

Then:

```text
stability_ratio = adjusted_distance / (σ * sqrt(T))
```

### 6.2 Persistence Probability

Approximate mapping:

```text
persistence_probability = erf(stability_ratio)
```

or logistic approximation.

### 6.3 Decision Threshold

Example threshold:

```text
trade if persistence_probability > 0.995
```

This corresponds roughly to ≤0.5% failure probability. Threshold will later be calibrated from telemetry.

---

## 7. Heuristic Guard Layer

This layer catches conditions where the diffusion model fails.

Rules include:

- **Late acceleration veto** — `if abs(price_change_last_5s) > acceleration_threshold` → reject trade
- **Oscillation veto** — `if direction_flips_last_20s ≥ N` → reject trade
- **Spread veto** — `if spread_ratio > threshold` → reject trade
- **Whale trade veto** — `if trade_size_last_tick > size_threshold` → reject trade
- **Volatility spike veto** — `if volatility_last_3s > volatility_last_15s * factor` → reject trade

---

## 8. Opportunity Ranking

All valid candidates are ranked by persistence_probability (descending order).

Example:

| Market | Persistence |
| ------ | ----------- |
| A      | 0.9991      |
| B      | 0.9987      |
| C      | 0.9981      |

---

## 9. Stream Architecture

Capital is divided into independent trading streams.

Example:

```text
total_capital = $1000
streams = 4
stream_capital = $250 each
```

Each stream behaves as an independent agent. Each stream maintains:

- stream_balance
- active_trade
- trade_history
- loss_count

---

## 10. Stream Allocation Rules

Rules ensure streams avoid correlated risk.

- **Rule 1** — One stream per market instance: no two streams trade the same candle.
- **Rule 2** — Streams only hold one open trade.
- **Rule 3** — Streams receive trades from the opportunity ranking queue.

Example: stream A → highest persistence, stream B → next.

---

## 11. Execution Engine

Each stream executes trades independently.

Execution logic:

```text
place IOC order at target price
if not filled:
    fallback to GTC
```

Track: order_id, fill_price, fill_size, timestamp.

---

## 12. Trade Resolution

At market settlement determine:

- **WIN** → `balance *= 1.01`
- **LOSS** → balance reset to stage start

Loss only affects the stream balance, not other streams.

---

## 13. Capital Ladder

Withdrawals occur at stage milestones.

Example:

- $20 → $500 → withdraw $250
- $250 → $1000 → withdraw $500

Streams inherit the new stage bankroll.

---

## 14. Telemetry System

Every candidate trade is logged.

**Fields:** timestamp, market_id, distance_to_flip, volatility, time_remaining, stability_ratio, persistence_probability, heuristic_flags, decision, trade_result, pnl

This data supports later calibration.

---

## 15. Replay Simulation Engine

Historical tick data can be replayed.

Simulation loop:

```text
for tick in historical_data:
    update market state
    evaluate persistence model
    simulate execution
    record outcome
```

Replay runs faster than real time to generate thousands of trades quickly.

---

## 16. Model Calibration

Telemetry is analyzed to estimate loss_rate vs persistence_probability.

Example output:

| Persistence  | Loss   |
| ------------ | ------ |
| 0.98–0.99    | 3%     |
| 0.99–0.995   | 0.7%   |
| 0.995–0.999  | 0.15%  |

The system threshold is adjusted accordingly.

---

## 17. Optional ML Layer

ML can refine probability estimates.

- **Features:** stability_ratio, volatility, spread, imbalance, price_path_shape, time_remaining
- **Model predicts:** actual_failure_probability
- **Final decision:**

```text
trade if predicted_failure_probability < risk_threshold
```

---

## 18. Global Risk Guard

The system halts trading if extreme conditions appear:

- exchange outage
- latency spike
- market volatility shock

---

## 19. Expected System Behaviour

Over time the system becomes:

- multiple independent compounding streams
- selecting only high persistence opportunities

Loss events degrade individual streams but do not collapse total capital.

---

## 20. Key Success Metric

The system's viability depends almost entirely on tail loss probability.

**Target range:** 0.1% – 0.3% per trade

At that level the ladder + stream structure makes the million-dollar path statistically plausible.
