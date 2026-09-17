# Custom Live Screener — Technical Specification

## Problem Statement

Find intraday trading opportunities in real-time across 50+ stocks with:
- Sub-second price awareness (websocket LTP stream)
- Multi-timeframe indicator computation (1m, 5m, 15m)
- Condition-based signal detection with small, precise stoploss levels
- Decisive alerts the moment conditions align — no manual chart watching

---

## Architecture Overview

```
                        Angel One SmartAPI
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         WebSocket       REST API        REST API
        (live ticks)    (candles)      (batch OHLC)
              │               │               │
              ▼               ▼               ▼
        ┌─────────────────────────────────────────┐
        │           DATA INGESTION LAYER          │
        │  TickAggregator (builds candles from    │
        │  ticks) + CandleFetcher (bootstrap)     │
        └─────────────┬───────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │           CANDLE STORE (in-memory)       │
        │  Per-symbol rolling buffers:             │
        │    1m × 375 bars │ 5m × 78 │ 15m × 26   │
        │  Auto-aggregated from 1m base            │
        └─────────────┬───────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │         INDICATOR ENGINE                 │
        │  Incremental computation on new bar:     │
        │  SMA, EMA, BB, RSI, VWAP, ATR, Pivots   │
        │  Cached per-symbol, per-timeframe        │
        └─────────────┬───────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │        CONDITION ENGINE                  │
        │  Rule sets (strategies) evaluated on     │
        │  every new bar across all symbols        │
        │  Multi-timeframe alignment checks        │
        └─────────────┬───────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │        SIGNAL OUTPUT                     │
        │  Ranked alerts → Dashboard + CLI + Log   │
        │  Entry price, SL, target, R:R, reason    │
        └─────────────────────────────────────────┘
```

---

## Layer 1: Data Ingestion

### 1a. WebSocket Feed (NEW — does not exist today)

Angel One SmartAPI provides `SmartWebSocket` / `SmartWebSocketV2` for live tick data.

```python
class TickStream:
    """Manages websocket connection to Angel One for live LTP/quotes."""

    def __init__(self, symbols: list[str], on_tick: Callable):
        self.ws = SmartWebSocketV2(auth_token, api_key, client_code, feed_token)
        self.symbols = symbols       # up to 50 instruments
        self.on_tick = on_tick        # callback: (symbol, tick_data) -> None

    def start(self):
        """Subscribe and begin streaming."""
        # Mode 1: LTP only (least bandwidth)
        # Mode 2: LTP + Quote (OHLC, volume)
        # Mode 3: Full (+ OI, best 5 bid/ask) — for options
        tokens = [{"exchangeType": 1, "tokens": [...]}]  # NSE=1
        self.ws.subscribe(correlation_id, mode=2, tokens=tokens)

    def on_data(self, ws, message):
        """Parse binary frame → dict, forward to on_tick callback."""
        tick = parse_binary_tick(message)  # {token, ltp, open, high, low, close, volume, oi}
        self.on_tick(tick["token"], tick)
```

**Key design decisions:**
- **Mode 2 (Quote)** for equity screener — gives OHLC + volume per tick, enough to build candles
- **Mode 3 (Full)** reserved for options (OI + depth) — not needed for equity screener
- **50 instrument limit per connection** — matches NIFTY 50 universe perfectly
- **Reconnection:** Auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s)
- **Heartbeat:** Ping every 30s to detect dead connections

### 1b. REST API Bootstrap (existing infra)

On startup, fetch historical candles to seed the rolling buffers:
- `BrokerClient.fetch_candles(symbol, "FIVE_MINUTE", from_date, to_date)` — last 5 days
- Build 1m candles from today's ticks after bootstrap

### 1c. REST API Fallback

If websocket drops, fall back to polling `market_data_batch()` every 5 seconds (existing infra, 50 stocks in 1 call).

---

## Layer 2: Candle Store

### Design: In-Memory Rolling Buffers

```python
@dataclass
class CandleBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float          # running VWAP for the bar
    is_complete: bool     # False = still forming

class CandleStore:
    """Per-symbol, multi-timeframe candle storage with tick aggregation."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bars: dict[str, deque[CandleBar]] = {
            "1m": deque(maxlen=375),    # full trading day
            "5m": deque(maxlen=78),     # full day
            "15m": deque(maxlen=26),    # full day
        }
        self._current_1m: CandleBar | None = None  # forming bar
        self._tick_count = 0
        self._cum_vol_price = 0.0     # for VWAP
        self._cum_vol = 0

    def ingest_tick(self, tick: dict) -> list[str]:
        """
        Process a tick, return list of timeframes that got a new completed bar.
        e.g., ["1m", "5m"] means both 1m and 5m bars just closed.
        """
        completed = []
        minute = tick_minute(tick["timestamp"])

        if self._current_1m is None or minute != self._current_1m.timestamp.minute:
            # Close previous bar, start new one
            if self._current_1m:
                self._current_1m.is_complete = True
                self.bars["1m"].append(self._current_1m)
                completed.append("1m")
                completed += self._try_aggregate(["5m", "15m"])
            self._current_1m = CandleBar(...)

        # Update forming bar
        self._current_1m.high = max(self._current_1m.high, tick["ltp"])
        self._current_1m.low = min(self._current_1m.low, tick["ltp"])
        self._current_1m.close = tick["ltp"]
        self._current_1m.volume += tick.get("volume_delta", 0)

        return completed

    def _try_aggregate(self, timeframes: list[str]) -> list[str]:
        """Aggregate 1m bars into higher timeframes when period completes."""
        completed = []
        for tf in timeframes:
            n = {"5m": 5, "15m": 15}[tf]
            recent_1m = list(self.bars["1m"])[-n:]
            if len(recent_1m) == n and recent_1m[0].timestamp.minute % n == 0:
                agg = aggregate_bars(recent_1m)
                self.bars[tf].append(agg)
                completed.append(tf)
        return completed

    def get_closes(self, tf: str, n: int = 0) -> list[float]:
        """Get last N closing prices for a timeframe. 0 = all."""
        bars = self.bars[tf]
        subset = list(bars)[-n:] if n else list(bars)
        return [b.close for b in subset]

    def prev_day_ohlc(self) -> dict:
        """Previous day OHLC for pivot calculation (from daily candle bootstrap)."""
        ...
```

**Why in-memory, not DB:**
- 50 symbols × 375 bars × 3 timeframes = ~56K bars max — fits in <10MB RAM
- Microsecond access vs millisecond DB queries
- Rebuilt from REST on every startup — no persistence needed
- No concurrent writers — single event loop

---

## Layer 3: Indicator Engine

### Design: Incremental + Cached

Indicators are recomputed only when a new bar completes, not on every tick.

```python
@dataclass(frozen=True)
class IndicatorSnapshot:
    """All indicators for one symbol at one timeframe, at one point in time."""
    sma_9: float
    sma_20: float
    ema_9: float
    ema_21: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_bandwidth: float
    bb_squeeze: bool
    rsi_14: float
    atr_14: float
    vwap: float

    # Pivots (daily — same all day)
    pivot: float
    r1: float; r2: float; r3: float
    s1: float; s2: float; s3: float

    # Derived
    price_vs_sma9: str       # "above" | "below" | "crossing_up" | "crossing_down"
    price_vs_bb: str         # "above_upper" | "below_lower" | "inside"
    rsi_zone: str            # "overbought" | "oversold" | "neutral"
    candle_size_vs_atr: float  # current bar range / ATR — measures "big candle"

class IndicatorEngine:
    """Computes and caches indicators per symbol per timeframe."""

    def __init__(self):
        self._cache: dict[tuple[str, str], IndicatorSnapshot] = {}

    def update(self, symbol: str, tf: str, store: CandleStore) -> IndicatorSnapshot:
        """Recompute indicators for symbol@timeframe. Called on bar close."""
        closes = store.get_closes(tf)
        bars = list(store.bars[tf])

        snap = IndicatorSnapshot(
            sma_9=sma(closes, 9),
            sma_20=sma(closes, 20),
            bb_upper=..., bb_middle=..., bb_lower=...,  # bollinger_bands(closes, 20)
            rsi_14=rsi(closes, 14),
            atr_14=atr(bars, 14),
            vwap=vwap(bars),
            # ... pivots from prev_day_ohlc (computed once at open)
            # ... derived fields
        )
        self._cache[(symbol, tf)] = snap
        return snap

    def get(self, symbol: str, tf: str) -> IndicatorSnapshot | None:
        return self._cache.get((symbol, tf))
```

**Indicators included (v1):**
| Indicator | Period | Purpose |
|-----------|--------|---------|
| SMA | 9, 20 | Trend direction, momentum fade detection |
| EMA | 9, 21 | Faster trend, crossover signals |
| Bollinger Bands | 20, 2σ | Volatility squeeze/expansion, mean reversion |
| RSI | 14 | Overbought/oversold confirmation |
| ATR | 14 | Stoploss sizing, "big candle" detection |
| VWAP | session | Institutional bias, fade setups |
| Classic Pivots | daily | Key support/resistance levels |
| Camarilla Pivots | daily | Intraday range levels (existing) |

**Extensibility:** New indicators added by extending `IndicatorSnapshot` + adding computation in `update()`.

---

## Layer 4: Condition Engine (The Core)

### Philosophy

A **Strategy** is a named set of **Conditions** evaluated across timeframes. Each condition is a simple boolean check on the indicator snapshot. When ALL conditions of a strategy are met, a **Signal** is emitted with a precise entry, stoploss, and target.

### Condition Types

```python
class ConditionType(Enum):
    PRICE_ABOVE = "price_above"           # price > level
    PRICE_BELOW = "price_below"           # price < level
    PRICE_CROSSES_ABOVE = "crosses_above" # was below, now above
    PRICE_CROSSES_BELOW = "crosses_below" # was above, now below
    INDICATOR_COMPARE = "indicator_gt"    # indicator > threshold
    CANDLE_PATTERN = "candle_pattern"     # big candle, doji, engulfing
    TIME_WINDOW = "time_window"           # only between 09:30-10:30
    BARS_SINCE = "bars_since"             # N bars since condition X was true
    SEQUENCE = "sequence"                 # condition A happened before condition B

@dataclass
class Condition:
    type: ConditionType
    timeframe: str          # "1m", "5m", "15m"
    params: dict            # type-specific parameters
    description: str        # human-readable for alerts

    def evaluate(self, snap: IndicatorSnapshot, bars: list[CandleBar],
                 prev_snap: IndicatorSnapshot | None) -> bool:
        ...
```

### Strategy Definition

```python
@dataclass
class Strategy:
    name: str
    description: str
    conditions: list[Condition]       # ALL must be True to fire
    entry_rule: EntryRule             # how to compute entry price
    stoploss_rule: StoplossRule       # how to compute SL
    target_rule: TargetRule           # how to compute target
    min_rr: float = 2.0              # minimum risk:reward to emit signal
    active_window: tuple[time, time] = (time(9, 20), time(15, 0))
    cooldown_bars: int = 5           # min bars between signals for same stock

@dataclass
class EntryRule:
    method: str   # "market" | "limit_at_level" | "limit_at_indicator"
    params: dict  # e.g., {"indicator": "bb_upper", "offset_pct": 0.1}

@dataclass
class StoplossRule:
    method: str   # "atr_multiple" | "swing_low" | "indicator_level" | "fixed_points"
    params: dict  # e.g., {"indicator": "bb_middle", "buffer_pct": 0.2}

@dataclass
class TargetRule:
    method: str   # "rr_multiple" | "indicator_level" | "atr_multiple"
    params: dict  # e.g., {"rr": 2.0} or {"indicator": "r2"}
```

### Example: The Bollinger Band Fade Strategy (from your screenshot)

```python
bb_fade_after_momentum = Strategy(
    name="BB Fade After Momentum",
    description="After big morning move fades below SMA9, buy on upper BB break",
    conditions=[
        # 1. Morning had a big candle (range > 1.5x ATR on 15m)
        Condition(
            type=ConditionType.CANDLE_PATTERN,
            timeframe="15m",
            params={"pattern": "big_candle", "range_vs_atr": 1.5,
                    "lookback_bars": 4, "time_before": "11:00"},
            description="Big morning candle (>1.5x ATR on 15m)"
        ),
        # 2. Price has since fallen below SMA 9 on 5m (momentum faded)
        Condition(
            type=ConditionType.PRICE_BELOW,
            timeframe="5m",
            params={"level": "sma_9"},
            description="Price below SMA9 on 5m (momentum faded)"
        ),
        # 3. BARS_SINCE: at least 6 bars (30min) since the big candle
        Condition(
            type=ConditionType.BARS_SINCE,
            timeframe="5m",
            params={"condition": "big_candle_15m", "min_bars": 6},
            description="At least 30min since big candle"
        ),
        # 4. Price now crosses above upper Bollinger Band on 5m
        Condition(
            type=ConditionType.PRICE_CROSSES_ABOVE,
            timeframe="5m",
            params={"level": "bb_upper"},
            description="Price breaks above upper BB on 5m"
        ),
        # 5. RSI not extreme (avoid chasing)
        Condition(
            type=ConditionType.INDICATOR_COMPARE,
            timeframe="5m",
            params={"indicator": "rsi_14", "op": "<", "value": 75},
            description="RSI < 75 (not overextended)"
        ),
        # 6. Time window: only afternoon (after initial fade)
        Condition(
            type=ConditionType.TIME_WINDOW,
            timeframe="5m",
            params={"after": "12:00", "before": "14:30"},
            description="Afternoon session only"
        ),
    ],
    entry_rule=EntryRule(
        method="limit_at_indicator",
        params={"indicator": "bb_upper", "offset_pct": 0.1}
    ),
    stoploss_rule=StoplossRule(
        method="indicator_level",
        params={"indicator": "sma_9", "buffer_pct": 0.3}
        # SL just below SMA 9 — small, precise
    ),
    target_rule=TargetRule(
        method="rr_multiple",
        params={"rr": 2.0}
    ),
    min_rr=1.5,
    active_window=(time(12, 0), time(14, 30)),
    cooldown_bars=10,
)
```

### More Strategy Templates (v1)

| # | Strategy | Entry Trigger | SL Method | Timeframes |
|---|----------|--------------|-----------|------------|
| 1 | **BB Fade After Momentum** | Upper BB break after SMA9 fade | Below SMA 9 | 5m + 15m |
| 2 | **VWAP Bounce** | Price touches VWAP from above + RSI > 40 | Below VWAP - 0.3% | 5m |
| 3 | **Pivot Rejection Long** | Bullish candle at S1/S2 + RSI oversold | Below pivot level - ATR×0.3 | 5m + 15m |
| 4 | **Squeeze Breakout** | BB squeeze release + volume spike + close > SMA20 | Below BB middle | 15m |
| 5 | **Morning Range Break** | 9:30-10:00 high break + VWAP support | Below ORB low | 5m |
| 6 | **SMA 9/21 Crossover** | 9 crosses above 21 + price > VWAP | Below SMA 21 | 15m |

### Evaluation Loop

```python
class ConditionEngine:
    """Evaluates all strategies against all symbols on every bar close."""

    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies
        self._state: dict[tuple[str, str], StrategyState] = {}
        # tracks per (symbol, strategy): last_signal_bar, conditions_met history

    def on_bar_close(self, symbol: str, tf: str,
                     store: CandleStore, engine: IndicatorEngine) -> list[Signal]:
        """Called when a bar completes. Returns any triggered signals."""
        signals = []
        now = datetime.now(IST)

        for strategy in self.strategies:
            # Only evaluate if this timeframe is relevant to ANY condition
            if not any(c.timeframe == tf for c in strategy.conditions):
                continue

            # Check time window
            if not (strategy.active_window[0] <= now.time() <= strategy.active_window[1]):
                continue

            # Check cooldown
            state = self._state.get((symbol, strategy.name))
            if state and state.bars_since_last_signal < strategy.cooldown_bars:
                continue

            # Evaluate ALL conditions
            all_met = True
            reasons = []
            for cond in strategy.conditions:
                snap = engine.get(symbol, cond.timeframe)
                prev_snap = engine.get_previous(symbol, cond.timeframe)
                if snap is None:
                    all_met = False
                    break
                if not cond.evaluate(snap, list(store.bars[cond.timeframe]), prev_snap):
                    all_met = False
                    break
                reasons.append(cond.description)

            if all_met:
                signal = self._build_signal(symbol, strategy, store, engine, reasons)
                if signal and signal.risk_reward >= strategy.min_rr:
                    signals.append(signal)

        return signals
```

---

## Layer 5: Signal Output

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    strategy: str
    side: str                # "BUY" | "SELL"
    entry: float             # precise entry price
    stoploss: float          # precise SL price
    target: float            # target price
    risk_reward: float       # computed R:R
    risk_points: float       # entry - SL (absolute)
    quantity: int            # from risk engine (% of capital / risk_points)
    reasons: list[str]       # which conditions triggered
    timeframe_alignment: dict  # {"5m": "bullish", "15m": "neutral"}
    confidence: float        # 0-1 based on how many optional boosters match

    # For dashboard display
    indicators_at_signal: dict  # snapshot of all indicators when triggered
```

### Output Channels

| Channel | How | Latency |
|---------|-----|---------|
| **Telegram** | Bot API with rate limiting, dedup, digest mode | < 1 sec |
| **Dashboard** | Streamlit `st.session_state` + auto-refresh | 1-5 sec |
| **CLI stdout** | Formatted log with entry/SL/target/R:R | Instant |
| **Django model** | `AuditLog` for signal history | Async write |

---

## Layer 6: Telegram Alerts

### Architecture

```
Signal → TelegramAlertService
           ├── Controls check:
           │     ├── Kill switch (SystemControl DB)
           │     ├── Quiet hours (outside market)
           │     ├── Per-strategy mute
           │     ├── Rate limit (5/min, 30/hr)
           │     └── Dedup (same symbol+strategy within 5min)
           ├── Format: Markdown with entry/SL/target/R:R
           └── Send: async HTTP POST (non-blocking, never stops screener)
```

### Controls

| Control | Method | Description |
|---------|--------|-------------|
| **Kill switch** | `SystemControl(key="screener_alerts", value="disabled")` | DB-level emergency off |
| **Enable/disable** | `alerts.enable()` / `alerts.disable()` | Runtime toggle |
| **Mute strategy** | `alerts.mute_strategy("VWAP Bounce Long")` | Silence one strategy |
| **Rate limit** | `max_per_minute=5, max_per_hour=30` | Prevent spam |
| **Dedup** | `signal_cooldown=300` | Same symbol+strategy not repeated within 5 min |
| **Quiet hours** | `quiet_start=15:35, quiet_end=9:10` | No alerts outside market |
| **Digest mode** | `--digest --digest-interval 15` | Batch signals into periodic summaries |

### Message Format (Markdown)

```
🟢 *RELIANCE* — BB Fade After Momentum
Side: *BUY*
Entry: `2850.50` | SL: `2840.00` | Target: `2871.00`
R:R: *1.9* | Risk: 10.5 pts (0.37%)
Confidence: 80%

_Reasons:_ Big morning candle, BB upper break
```

### Env vars

```bash
TELEGRAM_BOT_TOKEN=your_bot_token    # from @BotFather
TELEGRAM_CHAT_ID=your_chat_id       # your personal/group chat ID
```

---

## Layer 7: Backtest Engine

Replays historical 1m candles through the same ScreenerEngine, captures signals,
simulates trade outcomes (SL hit / target hit / EOD close), computes stats.

### Usage

```bash
python manage.py run_screener --backtest --from 2026-03-10 --to 2026-03-20
python manage.py run_screener --backtest --from 2026-03-10 --to 2026-03-20 --only "BB Fade After Momentum"
python manage.py run_screener --backtest --from 2026-03-10 --to 2026-03-20 --symbols RELIANCE,TCS
```

### Output

```
═══ Backtest Results ═══
Total signals: 42
Trades taken:  38
Winners:       22 (58%)
Losers:        16
Total P&L:     +234.50 pts
Profit Factor: 1.85
Avg R:R:       1.42
Max Drawdown:  78.30 pts

── Per Strategy ──
  BB Fade After Momentum: 8 trades, 62% win, +89.20 pts
  VWAP Bounce Long: 12 trades, 58% win, +67.40 pts
  ...
```

### Simulation rules
- Entry at signal price + slippage (0.05%)
- Walk forward bar-by-bar: first to hit SL or target wins
- EOD close at 15:25 if neither hit
- Tracks max favorable/adverse excursion per trade

---

## Implementation — Final File Map

```
trading/screener/
├── __init__.py              # Package docstring
├── candle_store.py          # In-memory rolling buffers, tick→1m→5m/15m aggregation
├── indicator_engine.py      # Incremental indicator computation on bar close
├── conditions.py            # 10 condition types, pure evaluator functions
├── strategies.py            # 6 built-in strategies with entry/SL/target rules
├── engine.py                # Orchestrator: ticks → indicators → conditions → signals
├── tick_stream.py           # SmartWebSocket + REST polling fallback
├── backtest.py              # Historical replay, trade simulation, performance stats
├── telegram.py              # Telegram Bot API alerts with full controls
└── signals.py               # Signal dataclass with CLI + Telegram formatters

trading/management/commands/
└── run_screener.py          # Django management command (live + backtest)

trading/config.py
└── ScreenerConfig            # All tunable parameters (poll interval, R:R, periods, Telegram limits)
```

---

## Universe: Multi-Connection Support

NIFTY 50 is the default universe (50 symbols = 1 websocket connection).

For broader screening (NIFTY 200, sectoral indices):
- **Multiple websocket connections** — each handles 50 instruments
- `TickStream` accepts a `connection_id` parameter, engine manages N streams
- Symbols partitioned round-robin across connections
- All feed into the same `ScreenerEngine` — strategies are symbol-agnostic

```bash
# NIFTY 50 (default)
python manage.py run_screener

# Custom universe
python manage.py run_screener --symbols HAPPSTMNDS,ZOMATO,PAYTM,DMART

# Future: NIFTY 200 (4 connections)
python manage.py run_screener --universe nifty200
```

---

## Auto-Execution Pipeline

Signals can optionally be auto-executed in paper mode:

```
Signal → RiskEngine validation → BrokerService.place_order(paper)
           → TradeJournal entry → Active trade monitoring (existing TradeManager)
```

Enabled via `--auto-execute` flag (paper mode only). Live auto-execution requires
explicit `TRADING_MODE=live` + `--auto-execute --confirm-live`.

The existing `RiskEngine` validates position sizing, daily loss limits, and max positions
before any order goes through — same guardrails as the equity workflow.

---

## Options Screening (NFO Extension)

For options screening, extend with:
- **WebSocket Mode 3** (Full) — includes OI + bid/ask depth
- **OI-based conditions**: OI buildup, PCR change, max pain proximity
- **Separate strategy templates**: straddle entry signals, iron condor timing
- **OptionsDataService** integration for Greeks (IV, delta)

New condition types:
```python
ConditionType.OI_BUILDUP       # OI increase > threshold
ConditionType.PCR_EXTREME       # Put-call ratio < 0.7 or > 1.3
ConditionType.IV_RANK           # IV rank > 80 (good for selling)
ConditionType.MAX_PAIN_DISTANCE # Price near max pain (reversion)
```

---

## Integration with Existing Infra

| Existing Component | How Screener Uses It |
|-------------------|---------------------|
| `BrokerClient` singleton | Auth token for websocket + REST fallback + bootstrap candles |
| `TickerService` | Symbol → token mapping for subscriptions |
| `indicators.py` | Reuse existing SMA, EMA, RSI, BB, ATR, VWAP, pivots |
| `RiskEngine` | Position sizing for auto-execute signals |
| `config.py` | `ScreenerConfig` dataclass (poll interval, R:R, periods, Telegram limits) |
| `AuditLog` | Log every signal for backtest validation |
| `TradeJournal` | Record auto-executed trades |
| `SystemControl` | Kill switch for Telegram alerts |
| `time_utils` | Market hours, session phase, candle date ranges |
| `market_scanner` | NIFTY 50 symbol list, sector mapping |

---

## Performance Budget

| Metric | Target |
|--------|--------|
| Tick-to-signal latency | < 100ms |
| Memory (50 symbols, 3 TFs) | < 50MB |
| API calls at startup | 50 (candle bootstrap) + 1 (websocket) |
| API calls during market | 0 (websocket) or 10/min (polling fallback) |
| Signals per day (estimate) | 5-15 across all strategies |
| Telegram alerts per day | 5-15 (rate limited) |
| Backtest speed | ~1 min per 10 trading days × 50 symbols |

---

## Key Design Decisions

1. **WebSocket first, polling fallback** — websocket gives instant ticks; if it drops, batch-poll every 5s
2. **1m as base timeframe** — all higher TFs aggregated from 1m bars, ensuring consistency
3. **Evaluate on bar close, not every tick** — reduces noise, strategies work on completed bars
4. **Conditions are pure functions** — no side effects, easy to backtest
5. **Small stoploss by design** — SL is always tied to a nearby technical level (SMA, BB middle, pivot), not arbitrary %
6. **Multi-timeframe alignment** — signal only fires when conditions across TFs agree
7. **No LLM in the hot path** — screener is pure Python, deterministic, fast
8. **Strategy cooldown** — prevents repeated signals on the same stock within N bars
9. **Telegram alerts are non-blocking** — HTTP POST in daemon thread, failures logged but never stop screener
10. **Same engine for live + backtest** — strategies are validated on historical data before going live
11. **Multi-connection websocket** — supports screening beyond NIFTY 50 via parallel connections
12. **Auto-execute gated by RiskEngine** — same guardrails as equity workflow, paper mode by default
