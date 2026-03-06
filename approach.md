# Polysnipe Approach

## 1. Start with a clean Git repo

Codex works best when it can inspect and modify a repository structure.

Create something like:

```text
state-persistence-bot/
│
├── README.md
├── architecture.md
├── requirements.txt
├── config/
│   └── config.yaml
│
├── src/
│   ├── market_feed/
│   ├── state_extractor/
│   ├── persistence_model/
│   ├── heuristics/
│   ├── streams/
│   ├── execution/
│   └── telemetry/
│
├── simulator/
│   ├── replay_engine.py
│   └── datasets/
│
└── tests/
```

Paste the full architecture spec we wrote into `architecture.md`. That becomes Codex's ground truth design document.

---

## 2. First prompt to Codex

When you open the repo, start with something like:

> Read architecture.md and scaffold the full project structure for a Python state persistence trading bot.
>
> The bot should include:
>
> - market state extractor
> - persistence probability model
> - heuristic guard layer
> - stream-based capital segmentation
> - opportunity ranking
> - execution layer
> - telemetry logger
> - replay simulation engine
>
> Do not implement exchange APIs yet. Focus on architecture, interfaces, and type-safe module boundaries.

This will generate:

- class skeletons
- interfaces
- module wiring

That's the hardest structural step.

---

## 3. Build the system in phases

Do not try to build the entire bot in one shot. Break it into stages.

### Phase 1 — Core model

Implement:

- state_extractor
- persistence_model
- heuristics

You should be able to run:

```text
python test_persistence_model.py
```

on simulated data.

### Phase 2 — Replay simulator

This is critical. You want:

```text
historical_tick_data
→ replay engine
→ run persistence model
→ measure failure probability
```

This will generate your training dataset.

### Phase 3 — Stream engine

Implement:

- StreamManager
- StreamAgent
- OpportunityAllocator

Test with simulated trades.

### Phase 4 — Exchange execution

Finally integrate:

- market API
- order placement
- fills
- trade settlement

---

## 4. Codex workflow trick

The best pattern with Codex is:

```text
Architecture prompt
→ review output
→ narrow prompts for each module
```

Example:

> Implement the persistence_model module using the stability ratio:
>
> ```text
> stability_ratio = distance / (volatility * sqrt(time_remaining))
> ```
>
> Include drift adjustment and probability mapping.

---

## 5. Keep Codex inside the repo context

Codex performs best when it can read the whole repository state.

Typical useful prompts:

- "Summarize how the persistence model interacts with streams."
- "Refactor the stream allocator so no two streams trade the same market instance."

---

## 6. One important suggestion

Have Codex build two modes from day one:

- **LIVE MODE**
- **SIMULATION MODE**

Most systems fail because they only test live. Your replay simulator will be one of the most valuable parts of this project.

---

## 7. Your biggest advantage

You are doing something most trading systems skip: you started from first principles (state persistence) instead of indicators. That gives you a much cleaner system.

Most people build something like:

```text
indicator
+ indicator
+ indicator
```

You're building:

```text
probability model
+ capital reliability system
```

Those architectures tend to scale much better.

---

## 8. One more suggestion before you start

Create this file in the repo: `design_invariants.md`

Put rules like:

1. No two streams may trade the same market instance.
2. All trades must pass persistence threshold.
3. Heuristic veto overrides persistence model.
4. Telemetry must record every candidate trade.

Codex will follow these rules surprisingly well.
