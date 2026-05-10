# Backtester Engine — Architecture

## Design Principles

1. **Event-Driven Lifecycle** — Trade lifecycle modeled as events:
   `on_signal → on_entry → on_bar → on_partial → on_exit`
   Strategies subscribe to events. Same lifecycle works for backtest + live.

2. **Live-Ready** — The `Trade`, `ExitManager`, and `PositionSizer` are designed
   to work identically in backtest (historical replay) and live (real-time bar feed).
   The only difference is the data source: historical candles vs websocket ticks.

3. **Engine-per-Config** — Each parameter combination gets its own stateless engine
   instance. Grid search = loop of engine instantiations. Parallelizable.

4. **Composition over Inheritance** — Strategies are composed from building blocks
   (entry detector + exit checks + position sizer), not subclassed.

5. **Config-as-Code** — Every backtest is fully defined by a `BacktestConfig` object.
   New strategy = new config + new `EntryDetector`. Zero engine code changes.

---

## Module Map

```
trading/backtester/
│
├── types.py            Shared enums, dataclasses, protocols
│   ├── PnLMode         POINTS | RUPEES
│   ├── TradeSide       BUY | SHORT
│   ├── TradeState      PENDING | OPEN | PARTIAL | CLOSED
│   ├── Bar             Normalized OHLCV candle
│   ├── EntrySignal     Strategy-agnostic trade signal
│   └── ExitSignal      (price, reason, partial?)
│
├── trade.py            Trade with event-driven state machine
│   └── Trade           on_entry() → on_bar() → on_exit()
│                       Tracks excursions, P&L, partial exits
│
├── exits.py            Composable exit strategies
│   ├── ExitCheck       ABC: check(trade, bar) → ExitSignal?
│   ├── StoplossExit    Current SL (which breakeven/trail may have moved)
│   ├── TargetExit      Fixed target price
│   ├── BreakevenExit   Move SL to entry ± buffer at trigger_r
│   ├── TrailingSLExit   ATR-based trail with tightening
│   ├── EODExit         Hard close at configurable time
│   ├── MaxHoldExit     Close after N bars
│   └── ExitManager     Composes checks + smart SL/target priority
│
├── sizing.py           Position sizing
│   └── PositionSizer   risk% → qty → cap by notional%
│                       update_capital() for equity tracking
│
├── stats.py            Unified statistics computation
│   └── StatsAggregator BacktestStats from Trade list
│                       Per-phase, weekly, equity curve, drawdown
│
├── report.py           Output formatting
│   └── ReportFormatter CLI summary + Telegram HTML + trade table
│
├── entry.py            Entry detector protocol + adapters
│   ├── EntryDetector   Protocol: detect(symbol, bars, idx, ctx)
│   ├── OKCycleAdapter  Wraps CycleDetector.analyze()
│   ├── IntradayAdapter Wraps IntradayCycleDetector.scan()
│   └── ScreenerAdapter Wraps screener engine signals
│
├── engine.py           Core engine (walk-forward + replay)
│   ├── EngineConfig    All engine parameters as dataclass
│   └── BacktestEngine  run(data, context) → BacktestStats
│                       _walk_forward() for daily
│                       _replay() for intraday
│
├── compat.py           Drop-in replacements for old functions
│   ├── run_ok_backtest()
│   ├── run_intraday_backtest()
│   └── run_screener_backtest()
│
└── __init__.py         Public API exports
```

---

## Event Flow

```
                    ┌──────────────┐
  EntryDetector ──→ │ EntrySignal  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
  PositionSizer ──→ │  Trade.OPEN  │ ←── on_entry(signal, qty, slippage)
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  for each subsequent bar │ ←── on_bar(bar)
              │                          │
              │  ExitManager.process()   │
              │    ├─ BreakevenExit      │ → may move SL
              │    ├─ TrailingSLExit     │ → may move SL
              │    ├─ StoplossExit       │ → may trigger exit
              │    ├─ TargetExit         │ → may trigger exit
              │    ├─ EODExit            │ → may trigger exit
              │    └─ MaxHoldExit        │ → may trigger exit
              │                          │
              │  Smart Priority:         │
              │    if SL+Target both hit │
              │    → check closer to     │
              │      bar.open first      │
              └────────────┬─────────────┘
                           │
                    ┌──────▼───────┐
                    │ Trade.CLOSED │ ←── on_exit(price, reason)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ StatsAggregator │
                    └──────────────┘
```

---

## How Each Strategy Composes

### OK Swing (Daily)
```python
engine = BacktestEngine(
    config=EngineConfig(pnl_mode=RUPEES, mode=DAILY, capital=500_000),
    entry=OKCycleAdapter(CycleDetector()),
    exits=[
        BreakevenExit(trigger_r=0.5, buffer=0.5),
        TrailingSLExit(trigger_r=1.0, atr_factor=0.3, tight_r=1.5, tight_factor=0.2),
        MaxHoldExit(10),
        StoplossExit(),
        TargetExit(),
    ],
)
```

### OK Intraday (15m)
```python
engine = BacktestEngine(
    config=EngineConfig(pnl_mode=RUPEES, mode=INTRADAY, eod_close=(15,20), cooldown=5),
    entry=IntradayAdapter(IntradayCycleDetector(sl_atr_mult=1.5), min_rr=2.5),
    exits=[
        BreakevenExit(trigger_r=1.0, buffer=0.5),
        StoplossExit(),
        TargetExit(),
        EODExit(15, 20),
    ],
)
```

### Screener (Intraday)
```python
engine = BacktestEngine(
    config=EngineConfig(pnl_mode=POINTS, mode=INTRADAY),
    entry=ScreenerAdapter(strategies=STRATEGIES),
    exits=[
        BreakevenExit(trigger_r=1.0, buffer=0.5),
        StoplossExit(),
        TargetExit(),
        EODExit(15, 25),
    ],
)
```

### Future: Mean Reversion
```python
engine = BacktestEngine(
    config=EngineConfig(pnl_mode=RUPEES, mode=INTRADAY, cooldown=10),
    entry=MeanRevAdapter(bb_period=20, bb_std=2.0),
    exits=[
        StoplossExit(),
        TargetExit(),
        MaxHoldExit(30),
        EODExit(15, 20),
    ],
)
```

---

## Live-Ready Design

The same `Trade` + `ExitManager` work in live mode:

```python
# Backtest: engine feeds historical bars
for bar in historical_bars:
    trade.on_bar(bar)
    exit_manager.process(trade, bar)

# Live: websocket feeds real-time bars
def on_new_bar(bar: Bar):
    for trade in open_trades:
        trade.on_bar(bar)
        signal = exit_manager.process(trade, bar)
        if signal:
            broker.place_exit_order(trade, signal)
```

No behavioral difference. The exit logic, breakeven rules, trailing — all identical.

---

## Grid Search Pattern

```python
results = []
for tf in ["5m", "15m"]:
    for sl in [1.0, 1.5, 2.0]:
        for rr in [1.5, 2.0, 2.5]:
            engine = BacktestEngine(
                config=EngineConfig(..., cooldown=5),
                entry=IntradayAdapter(IntradayCycleDetector(sl_atr_mult=sl), min_rr=rr),
                exits=[BreakevenExit(1.0), StoplossExit(), TargetExit(), EODExit(15,20)],
            )
            stats = engine.run(data[tf])
            results.append((tf, sl, rr, stats))

best = max(results, key=lambda r: r[3].profit_factor)
```

Each engine is stateless after `run()`. Data fetching + caching stays outside.
