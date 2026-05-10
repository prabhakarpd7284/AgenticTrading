# The AlphaDesk Mind Palace

_Close your eyes. You're standing outside an 8-story glass tower in Mumbai's BKC district. The sign above the entrance reads **ALPHADESK**. Each floor is a timeframe. Each room is a brain. Each object is a number you must never forget. Walk with me._

> **Last updated:** 2026-05 — added Floor 7 (Pyramid Room), the Library (Feedback Archive on Floor 6.5), v2 backend wing, BFO/SENSEX pipes in the basement.

---

## Ground Floor — THE VAULT (Capital & Risk)

You push through the revolving door into a marble lobby. In the center: a glass vault, illuminated from below, containing exactly **5,00,000 rupees** in neat bundles.

### The Vault Rules (etched into the glass)

| Object | Rule | Number |
|--------|------|--------|
| **Red laser grid** around the vault | Max risk per trade | **1% of capital** (Rs 5,000) |
| **Daily alarm clock** on the wall (set to stop at 3%) | Max daily loss before shutdown | **3%** (Rs 15,000) |
| **10 numbered slots** on the vault door | Max position size | **10% of capital** (Rs 50,000) |
| **3 parking spots** painted on the lobby floor | Max simultaneous open positions | **3** |
| **A balance scale** (both sides must be ≥1.5:1) | Minimum Risk:Reward ratio | **1.5** |
| **A confidence meter** on the wall (needle must pass 0.55) | Minimum confidence threshold | **0.55** |

There is a **single guard** standing in front of the vault. His name tag says **@RiskGuard**. He is not AI. He is a calculator. He checks **10 gates** in exact order, and he **never makes exceptions**. No trade touches the vault without his stamp.

Behind him, a screen shows the **Regime Signal** — a traffic light:
- **GREEN**: VIX calm, market tradeable. Proceed.
- **YELLOW**: Elevated vol. Smaller sizing.
- **RED (EXTREME)**: VIX > 35. **All entries blocked.** Go home.

_Remember: @RiskGuard is the last door before every rupee moves. He cannot be sweet-talked, bribed, or bypassed. He is the reason you sleep at night._

---

## Floor 1 — THE OBSERVATORY (Swing / Weekly View)

Take the elevator up. The doors open to a quiet room with **floor-to-ceiling windows** overlooking the Nifty 50 — each stock is a star in the sky, colored by its cycle phase.

### Oliver Kell's 8-Phase Wheel

Mounted on the north wall is a giant **circular wheel**, divided into 8 phases. Stocks orbit this wheel over weeks/months:

```
         ┌─── RE (Reversal Extension) ───┐
         │   potential bottom, WATCH       │
    BB_BEAR                            WP (Wedge Pop)
  continuation ↓                     BUY ↑ momentum entry
         │                               │
   EC_BEAR                            EC (EMA Crossback)
  failed bounce                     BUY ↑ pullback entry
         │                               │
    WD (Wedge Drop)                   BB (Basin Break)
   breakdown, AVOID                  BUY ↑ continuation
         │                               │
         └─── EX (Exhaustion Extension) ──┘
                   potential top, SELL
```

**The telescope** in the corner: `OKScanner` — fetches 120 days of daily candles, aggregates to weekly, runs `CycleDetector`. Each star (stock) gets tagged with its phase.

**The actionable filter** (a red velvet rope): only stars in WP/EC/BB (bullish) or WD/EC_BEAR/BB_BEAR (bearish) AND where daily + weekly trends **align** pass through to the watchlist.

**The radio** on the desk: sends Telegram alerts when a star transitions between phases. Phase emojis blink on a notification board.

_This room answers one question: "Which stocks are in the right part of their cycle RIGHT NOW?" It runs daily, usually at night. Its output feeds tomorrow's premarket scan._

---

## Floor 2 — THE WAR ROOM (Premarket Scan / Daily Setup)

Walk up one flight. This room is buzzing — screens everywhere, coffee cups, newspapers. It's 8:30 AM. Market opens in 45 minutes.

### The Premarket Scanner Table

A long table with **50 stock dossiers** (NIFTY 50), each scored 0-100:

| Scoring Factor | Max Points | What It Measures |
|----------------|-----------|------------------|
| ATR volatility | 20 | Is this stock moving enough to trade? (>2% sweet spot) |
| Volume vs average | 15 | Is today unusual? (1.5x+ spike) |
| Narrow range (NR4/NR7) | 20 | Has it been coiling? (Compression = explosion) |
| Proximity to PDH/PDL | 20 | Is it near yesterday's high or low? (Level confluence) |
| Trend clarity | 15 | Is the daily trend clean? (EMAs stacked) |
| Gap potential | 10 | Will it gap at open? |

The top-10 dossiers are placed on a **brass tray** labeled `StockSetup[]`. Each dossier includes Camarilla pivots pre-computed (S1-S4, R1-R4) — the day's battleground levels.

**The AI Strategist's notepad** (optional): If LLM premarket analysis is enabled, Claude reads all 10 dossiers and writes a morning brief. Otherwise, this notepad stays blank — the numbers speak for themselves.

_This room answers: "Of everything in the universe, which 10 stocks should I watch TODAY?" It runs once, before 9:15._

---

## Floor 3 — THE TRADING FLOOR (Intraday Execution)

The elevator doors open to controlled chaos. This is where money is made or lost. The room has **5 desks**, each running a different detection algorithm, plus a central **command console**.

### Desk 1: Level Map Builder

A massive **whiteboard** covering one wall, updated every 5 minutes. For each watched stock, it plots every significant price level:

| Level Type | Base Score | Source |
|-----------|-----------|--------|
| Swing highs/lows | 30 | Daily chart extremes |
| Previous day high/low | 25 | Yesterday's range |
| Camarilla pivots | 25 | Math from prev close |
| Round numbers (100s) | 20 | Psychology (23000, 23100...) |
| VWAP | 20 | Today's fair value anchor |
| Weekly high/low | 20 | Multi-day context |
| ORB high/low | 15 | First 15-min range |
| Gap zones | 15 | Unfilled gaps |

Levels within **30 points** of each other **cluster** into a single super-level (scores add). The whiteboard shows levels as horizontal lines — thicker = higher score.

### Desk 2: The Structure Detectors (5 specialists)

Five analysts sit at this desk, each watching for their specific pattern:

**Level Bounce** (the star performer — 70% WR, +Rs 83,850/lot backtested):
- Watches for price hitting a scored level, making a 3-candle local extreme, then reversing 30+ points
- Entry: next candle open after reversal confirmed
- SL: 20 points beyond the extreme
- Target: next significant level
- Max 11 signals per cycle

**Level Retest** (the patient one):
- Watches for a level to break (price stays beyond for 2 candles), then price returns to within 0.15 x ATR of the level and holds for 2 candles
- Entry: on the hold confirmation
- SL: 0.10 x ATR beyond level
- Min R:R: 2.0
- Max 2 signals

**VWAP Fade** (the contrarian):
- Entry when price deviates >0.8% from VWAP AND RSI confirms (>65 for shorts, <35 for longs)
- Target: VWAP (mean reversion)
- Max 1 signal per cycle

**ORB / PDH / PDL detectors** (the classics):
- Opening Range Breakout, Previous Day High/Low breaks
- Standard setups, well-understood edge

### Desk 3: The Sweet Spot Filter

A **sieve** sitting between the detectors and execution. Every signal must pass through:
- Is the entry price within **0.5 x ATR** of a level scoring **>=20**?
- If not: **REJECTED** (trading at random prices = negative edge)
- If yes: confidence boosted up to **+25%** based on level score
- SL/Target improved to actual support/resistance levels
- R:R recalculated with improved levels

_This filter is why AlphaDesk doesn't take dumb trades. No level = no trade._

### Desk 4: The Regime Classifier

A traffic light panel on the wall shows today's market character:

| Regime | Condition | Effect on Signals |
|--------|-----------|-------------------|
| TRENDING | NIFTY range > 1.5% | PDL_BREAK confidence **x1.15** (best edge) |
| RANGE | NIFTY range < 1.3% | PDH_BREAK confidence **x0.5** (kills false breakouts) |
| GAP_UP / GAP_DOWN | Gap > 0.5% | Adjusts setup expectations |
| NORMAL | Default | No adjustment |

_The regime desk prevents the classic mistake: trading breakouts in a range day, or fading trends on a trending day._

### The Command Console (center of the room)

`IntradayMonitor` — runs a scan cycle every **5 minutes** from 9:30 to 15:15:

```
Fetch 5-min candles (cached)
  → Build LevelMap
    → Run all structure detectors
      → Sweet Spot filter
        → Regime adjustment
          → Risk Engine validation (10 gates)
            → BrokerService.place_order()
```

### Desk 5: The Trade Manager (babysitter for open positions)

Once a trade is open, this desk manages it through staged exits:

| Stage | Trigger | Action |
|-------|---------|--------|
| Entry | Signal approved | Position opened |
| Stage 2 | Price moves +0.5R | **Move SL to breakeven** (free trade) |
| Stage 3 | Price moves +1.0R | **Take 50% profit**, trail remainder at 0.3 x ATR |
| Stage 4 | Price moves +1.5R | **Tighten trail** to 0.2 x ATR |
| Time Stop | <0.3R progress in 10 candles | **Exit** (trade is dead) |

_This desk answers: "I'm in a trade. Now what?" The 0.5R breakeven move is the most important — it turns every winner into a risk-free position._

---

## Floor 4 — THE OPTIONS ROOM (Straddle Management)

A quieter floor. Darker. Two massive screens dominate: **NIFTY spot** on the left, **VIX** on the right. A short straddle position is always displayed in the center — two legs, CE and PE, with their P&L ticking in real time.

### The Straddle Anatomy

```
        NIFTY Spot: 24,200
        ┌─────────────────┐
        │  SHORT STRADDLE  │
        │                   │
  CE Sold @ 394.85    PE Sold @ 138.35
  CE Now: ???         PE Now: ???
        │                   │
        │  Combined Sold:   │
        │  533.20 pts       │
        │                   │
        └─────────────────┘
```

### The Lifecycle Engine (5 simple rules on the wall)

A poster with **5 rules**, checked every 15-30 minutes:

| # | Condition | Action |
|---|-----------|--------|
| 1 | Combined current > **1.3x** combined sold | **CLOSE_BOTH** (hard stop — you're bleeding) |
| 2 | Expiry day AND time >= **15:00** | **CLOSE_BOTH** (gamma risk exponential) |
| 3 | NIFTY drifted > **250 pts** from straddle center | **SHIFT_TO_ATM** (re-center, max 2/day) |
| 4 | Max shifts (2) reached | **HOLD** (no more adjustments today) |
| 5 | Everything else | **HOLD** (theta is working for you) |

### The Analyzer's Desk (pure math, no LLM)

`StraddleAnalyzer` computes every cycle:
- **P&L**: combined sold vs current, net pnl in points and INR
- **Delta**: approximate Black-Scholes delta from moneyness x DTE → net portfolio delta
- **VIX Phase**: CALM (<15) / ELEVATED (<22) / SPIKE (>=22)
- **Market Phase**: CRASH / CHOP / RECOVERY / RALLY / CLOSE (from 6-candle structure)
- **Expiry Scenarios**: P&L at 6 NIFTY offsets (+300, +150, 0, -150, -400, -700)

### The LLM Advisor's Chair

`@OptionsStrategist` (Claude) receives all the analyzer's output and recommends an action. But the recommendation passes through **@RiskGuard** (downstairs in the vault) before any execution.

The straddle's **management_log** (a leather-bound ledger on the desk) records every cycle: time, action, NIFTY level, P&L, notes. This is the position's complete history.

_The options room makes money from **theta decay** — time is on your side as long as NIFTY doesn't move too much. The 1.3x hard stop is your fire exit. The 250-point shift is your earthquake protocol._

---

## Floor 5 — THE SCREENER (Live Pattern Detection)

A room full of screens — **98 stocks** (NIFTY 50 + Next 50) monitored simultaneously. No LLM here. Pure Python rule evaluation at machine speed.

### The Engine

`ScreenerEngine` — event-driven loop:
```
Tick arrives (every 5s poll)
  → CandleStore aggregates (1m → 5m → 15m)
    → IndicatorEngine computes snapshot (SMA/EMA/RSI/BB/MACD/ATR/VWAP/Pivots)
      → 8 Strategies evaluate conditions
        → Signal emitted (if ALL conditions true + cooldowns clear)
```

### The 8 Active Strategies (displayed as boxing match posters on the walls)

| Strategy | Setup | Key Conditions | Cooldown |
|----------|-------|---------------|----------|
| BREAKOUT_LONG | 10-bar high break | Above VWAP + RSI 50-80 + 15m uptrend | 30 bars |
| BREAKDOWN_SHORT | 10-bar low break | Below VWAP + RSI 20-50 + 15m downtrend | 30 bars |
| VWAP_BOUNCE_LONG | Price crosses above VWAP | Above SMA9 + MACD > 0 + RSI 45-65 | 60 bars (once/session) |
| VWAP_REJECTION_SHORT | Price crosses below VWAP | Below SMA9 + MACD < 0 + RSI 35-55 | 60 bars |
| PIVOT_REJECTION_LONG | Price reclaims S1 | RSI < 45, window 10:00-14:00 | 30 bars |
| PIVOT_REJECTION_SHORT | Price fails at R1 | RSI > 55, window 10:00-14:00 | 30 bars |
| SQUEEZE_BREAKOUT | Price > BB upper (15m) | Above SMA20 (5m) + RSI > 55 | 30 bars |
| SMA_CROSSOVER_TREND | Price crosses EMA9 (15m) | MACD > 0 + above VWAP (5m) | 30 bars |

Two posters are faded / crossed out: `BB_FADE_AFTER_MOMENTUM` and `MORNING_RANGE_BREAK` — **negative edge in backtest, disabled.**

### The Volatility Bouncer

Every strategy shares a common **volatility filter** (a velvet rope at the entrance): ATR must be between **0.15% and 2.0%** of price. Too quiet = no edge. Too wild = uncontrollable.

### The Telegram Hotline

A red phone on the wall: signals are sent to Telegram with chart images, entry/SL/target, confidence bars, R:R stars. Rate-limited: **5/min, 30/hr, 300s cooldown per symbol+strategy**.

---

## Floor 6 — THE BRIDGE (Dashboard & Control)

The top floor. Panoramic views. Two control stations:

### Station A: Streamlit Dashboard (The Original Bridge)

11 pages, connected **directly** to the Django ORM (no HTTP layer):

```
Command Center ─── Market Pulse (NIFTY/BANKNIFTY/VIX), KPIs, Alerts, Event Log
Intraday Agent ─── Run/monitor the AI (scan → trade → review)
Market Scanner ─── NIFTY 50 batch scan with TradingView charts + "APPROVE & EXECUTE" button
Screener ──────── 98-symbol live breadth chart + buy/sell potential
Swing Scanner ──── Oliver Kell cycle results
Trade Workflow ─── Manual LangGraph pipeline (step-by-step)
Straddle Console ─ Register/analyze/status/execute options positions
Journal ────────── Win rate, streaks, calibration, per-symbol P&L
Risk Control ───── Gauges + PAUSE AI / FORCE CLOSE ALL buttons
Backtest ───────── Historical strategy replay
Settings ───────── Config viewer, strategy library, portfolio init
```

The **big red button** on the Risk Control page: `PAUSE AI TRADING`. Writes to `SystemControl` table. Every trading loop checks this flag before each cycle.

### Station B: React Frontend (AlphaDesk SPA)

The newer, sleeker interface. JWT auth, WebSocket real-time updates. Vite + React 18 + TanStack Query + Tailwind:

```
The Cascade (pre-trade)
  /pulse ────────── Market Pulse — Stages 1+2 (regime, VIX, indices, sector heatmap)
  /rotation ─────── Stage 3 — sector drill-in
  /shortlist ────── Stage 4 — filtered watchlist
  /setup/:symbol ── Stage 5 — single-stock setup preview + @RiskGuard breakdown

Operational
  /dashboard ────── Capital, equity curve (live WebSocket MTM), AI activity
  /positions ────── Open equity + straddles + closed (live tick updates)
  /agents ───────── Agent console (start runs, watch live token stream)
  /strategies ───── Strategy library
  /backtester ───── Historical testing
  /brokers ──────── Angel One API key linking

Strategies
  /pyramid ──────── Pyramid options strategy (Floor 7) — backtest + Go Live
  /swing-scanner ── Oliver Kell cycle scanner UI

Post-trade feedback (the Library on Floor 6.5)
  /monthly ──────── Monthly Feedback Report:
                      • Headline P&L + YTD bar chart
                      • Equity curve + drawdown
                      • Capture Rate Matrix per stock
                      • Signal Audit (outcomes + by-strategy)
                      • Risk Rejections w/ hindsight
                      • Time-of-day + day-of-week + sector analytics
                      • Benchmark vs NIFTY50 (alpha)
                      • AI-generated lessons
```

**Live index ticker** lives in the page header on `/pyramid` (NIFTY · BANKNIFTY · SENSEX with live LTP and % change), driven by `useMarketPulse()` and refreshed every 15s during market hours.

**Single source of truth** for index metadata — `frontend/src/lib/market-config.ts` — holds lot sizes, expiry weekdays, Angel One tokens, exchange (NSE/BSE), and monthly-only flag. Every page imports from here; never hardcode 65 / 30 / 20.

---

## Floor 6.5 — THE LIBRARY (Post-Trade Feedback Archive)

A mezzanine between Floor 6 (Bridge) and Floor 7. Quiet, lined with leather-bound ledgers. This room exists for one reason: **close the learning loop**. The Cascade (Floors 2-3) answers _"what should I trade today?"_. The Library answers _"how well did I trade vs what the market actually offered?"_

### The SignalLog Ledger

A tall shelf of bound volumes — one per trading day. Every signal fired by **any** screener/scanner/premarket detector is recorded here, regardless of whether @RiskGuard approved it or the trader skipped it.

| Field | Why it matters |
|-------|---------------|
| `source` | SCREENER (intraday) · OK_SCANNER (swing cycles) · PREMARKET (basket) |
| `strategy` | Which detector fired (ORB_LONG, VWAP_RECLAIM, WEDGE_POP, etc.) |
| `entry/sl/target/confidence/risk_reward` | The trade plan as it existed at signal time |
| `outcome` | PENDING → TRADED / REJECTED / SKIPPED / EXPIRED |
| `eod_price` + `max_favorable_move` + `max_adverse_move` | **Filled by the EOD enrichment job** — what would have happened |
| `trade_journal` FK | Links to the actual TradeJournal row if we took the trade |

_Without this ledger, every signal that didn't become a trade was lost forever. Now we have a record of every opportunity AlphaDesk noticed but didn't act on._

### The Enrichment Clerk

Every evening at market close, `python manage.py enrich_signals` runs. It walks the day's SignalLog entries, fetches intraday candles from Angel One, and stamps each signal with what the market actually did after the signal fired. Unactioned signals get marked EXPIRED. Risk-rejected signals get linked to their TradeJournal entry.

### The Monthly Feedback Report

The librarian's masterpiece. Open `/monthly` in the React UI to see:

| Section | Question it answers |
|---------|---------------------|
| **Month summary** | What did I earn this month? Win rate, capital deployed, asset-class split |
| **YTD strip** | 12-month bar chart — best/worst month, capital base, total P&L |
| **Equity curve + drawdown** | Day-by-day cumulative P&L with peak-to-trough drawdown |
| **Capture Rate Matrix** | Per stock: market moved X%, signals fired Y, captured Z% — sorted by potential |
| **Signal Audit** | Outcomes (TRADED/REJECTED/SKIPPED/EXPIRED), by source, by strategy with win-rate + avg R:R |
| **Risk Rejections** | What @RiskGuard blocked + would it have been profitable in hindsight? |
| **Analytics** | Time-of-day heatmap (which hour wins), day-of-week, sector attribution |
| **Benchmark** | Portfolio % vs NIFTY50 % → alpha |
| **Lessons** | AI-generated bullets: "Strategy X has 75% WR — lean in", "Risk gate Y blocked 4 winners" |

Backend: `backend/apps/portfolio/services/monthly_report.py` builds the full payload from `Position`, `TradeJournal`, `SignalLog`, `AuditLog`, and the `Candle` table. Cached 120s.

_The Library is what makes this trading desk a learning system instead of a one-way pipe. Every month, it tells you what worked, what didn't, and what your @RiskGuard let through that it shouldn't have (or blocked that it shouldn't have)._

---

## Floor 7 — THE PYRAMID ROOM (Aggressive Options Pyramiding)

A small, focused room on the 7th floor. One desk, three giant screens. The sign on the door reads:

> **PYRAMID STRATEGY**
> _Aggressive momentum pyramiding on options_

This room is where small initial bets become big winning ones — by adding to winners only when structure confirms it.

### The Three Underlyings

A live ticker runs across the top of the room, always visible:

```
NIFTY 50    24,265   +0.42%
BANK NIFTY  55,140   +0.18%
SENSEX      80,210   +0.31%
```

(Sourced from `useMarketPulse()` → `quotes.indices_in`. Index metadata — lot sizes, expiry days, exchange — comes from `frontend/src/lib/market-config.ts`, the single source of truth.)

### The Strategy Engine

`trading/pyramid/strategy.py` — pure deterministic Python, no LLM. Processes 5-minute option candles bar-by-bar.

**Entry (initial)**:
- close > EMA5 (price above fast trend)
- close > BB middle (above mean)
- RSI > RSI_EMA3 (momentum up)
- RSI > 50 (bullish zone)
- RSI > RSI_WMA21 (above slow trend)
- SL distance ≤ `max_risk_pct_of_price` (default 50% — wide enough for SENSEX premiums)

**Sizing**: `init_lots = max_risk / (risk_per_lot × lot_size)` where `max_risk = capital × initial_risk_pct`.

**Pyramid adds (up to 5)**:
- Unrealized profit > `min_profit_to_pyramid × current_lots`
- Higher-low pivot above current avg entry
- Price still above EMA5
- RSI > 45 (not bearish)
- Cooldown: 5 bars since last add

**Trail SL**:
- Activates after +1R profit
- Method 1: floor of last 3 completed candle lows (ratchet up only)
- Method 2: higher-low detection (strongest signal)

**Exits**:
- Trail SL hit
- EOD at 15:20 IST

### The Three Indices, The Three Lot Sizes (Jan 2026+)

| Index | Lot | Weekly expiry | Exchange | Notes |
|-------|-----|---------------|----------|-------|
| NIFTY | **65** | Tuesday | NSE (NFO) | Was 75 before Jan 2026 — NSE circular FAOP70616 |
| BANKNIFTY | **30** | Last Tuesday (monthly only) | NSE (NFO) | No weekly contracts |
| SENSEX | **20** | Thursday | BSE (BFO) | Different segment from NIFTY/BANKNIFTY |

When the user clicks NIFTY/BANKNIFTY/SENSEX in the UI, lot size and next expiry auto-populate. The expiry computation respects the post-Sep-2025 SEBI standardisation (NSE→Tuesday, BSE→Thursday).

### The KPI Wall

Above the desk, nine large dials:

| KPI | Formula |
|-----|---------|
| Total P&L | Sum of all leg P&Ls in INR |
| P&L (points) | Same in option points |
| **Capital Deployed** | avg_entry × total_lots × lot_size — premium paid |
| **ROI** | total_pnl / capital_deployed × 100 |
| **Initial Risk** | (entry₀ − sl₀) × lots₀ × lot_size — max loss on first entry |
| Peak Unrealized | Best mark-to-market gain |
| Final / Peak Lots | Where the position landed vs how big it got |
| Pyramids | Number of add-ons (0..5) |
| Avg Entry | Weighted-average fill price |

### The Live Mode (planned)

A "Go Live" button next to "Run Backtest" — when clicked, spawns a `PyramidExecutor` that monitors live candles, places real orders via `BrokerService.place_order(exchange="NFO"|"BFO", product_type="CARRYFORWARD")`, and persists to a `PyramidPosition` model. Same paper/live toggle as the rest of the system. Status broadcasts via `ws/pyramid/{position_id}/`.

_The pyramid room is where AlphaDesk admits that asymmetric returns require asymmetric position sizing. Most days you take a small loss or small win. The pyramid days are where the year is made — but only if the structure confirms every add._

---

## The Corridors (How Data Flows Between Floors)

### Corridor 1: Swing → Premarket (Floor 1 → Floor 2)

```
OKScanner.get_watchlist()
  → symbols in actionable cycle phases (WP/EC/BB)
    → PremarketScanner.scan(symbols)
      → top-10 scored StockSetup[]
```

_The swing scanner narrows the universe. The premarket scanner ranks what's left._

### Corridor 2: Premarket → Intraday (Floor 2 → Floor 3)

```
StockSetup[] with Camarilla pivots
  → IntradayMonitor watchlist
    → LevelMap.build() uses previous day OHLC from setup
      → Structure detectors know what levels to watch
```

_The premarket dossier arms the intraday detectors with levels before the first candle prints._

### Corridor 3: Signal → Vault → Broker (Floor 3 → Ground → Outside)

```
IntradaySignal (entry, sl, target, confidence)
  → validate_trade() [10 gates]
    → APPROVED → BrokerService.place_order()
      → Paper: PAPER-{uuid} instant fill
      → Live: Angel One SmartAPI real order
    → REJECTED → journal reason, move on
```

_Every signal must pass the vault guard. No exceptions. Not even "this one looks really good."_

### Corridor 4: The Audit Trail (runs through every floor)

```
Every LLM call → AuditLog (prompt, response, tokens, latency)
Every risk decision → AuditLog (approve/reject, details)
Every trade → TradeJournal (plan, execution, P&L)
Every straddle cycle → StraddlePosition.management_log (append-only)
```

_Nothing happens in this building without a paper trail. Every decision, even rejected ones, is recorded._

### Corridor 5: The RAG Loop (Floor 3 → Basement Archive → Floor 3)

```
Before planning a trade:
  retrieve_context(symbol)
    → Last 20 trades for this symbol (with win rate)
    → Last 5 portfolio trades
    → Active StrategyDoc rules
    → Portfolio snapshot (capital, daily loss)
  → Injected into Claude's prompt as context
```

_The AI learns from its own history. If it lost 3 times on HDFCBANK with the same setup, it sees that. "Don't repeat losing patterns."_

### Corridor 6: The Feedback Loop (every signal → Library → next month's planning)

```
Floor 3/5 (Trading Floor / Screener)
  signal fires
    ↓
  SignalLog.objects.create(...)        [non-blocking, every signal]
    ↓
  EOD: enrich_signals command
    fetches intraday candles
    → max_favorable_move, max_adverse_move, eod_price filled
    → outcome PENDING → EXPIRED (if no trade)
    → trade_journal FK linked (if there was a trade)
    ↓
  Floor 6.5 (Library) — Monthly Feedback Report
    capture_matrix per stock: market move % vs P&L captured
    signal_audit: by outcome, by source, by strategy
    rejections: what @RiskGuard blocked + would it have profited?
    lessons: AI insights from the month
    ↓
  Trader reads /monthly → adjusts thresholds, disables weak strategies, promotes strong ones
```

_This corridor is what turns AlphaDesk from a one-way trade pipe into a learning system. Every signal — taken or not — leaves a paper trail. At month-end, the trader sees not just "what I earned" but "what I left on the table" and "what gates blocked winners". The RAG loop (Corridor 5) is short-term memory; the feedback loop (Corridor 6) is long-term pattern recognition._

---

## The Basement — THE DATA PIPES (Angel One SmartAPI)

Below the vault, a machine room hums. `BrokerClient` — a process-wide singleton — manages all communication with the outside world.

### Rate Limiting (painted on the pipes)
- **0.4 seconds** minimum between any API call
- **5-second LTP cache** (don't ask the same price twice)
- **Batch 50 stocks** in a single `getMarketData` call
- **2-retry backoff** on candle fetches

### Key Tokens (engraved on the pipes)
| Instrument | Token | Exchange |
|-----------|-------|----------|
| NIFTY | 99926000 | NSE (NFO for options) |
| BANKNIFTY | 99926009 | NSE (NFO for options) |
| SENSEX | 99919000 | BSE (BFO for options) |
| India VIX | 99926017 | NSE |

### The Two Option Segments

A pivotal detail: NIFTY and BANKNIFTY options trade on **NSE's NFO segment**. SENSEX options trade on **BSE's BFO segment**. The symbol formats differ:

| Exchange | Format | Example |
|----------|--------|---------|
| NFO | `{NAME}{DDMMMYY}{STRIKE}{TYPE}` | `NIFTY13MAY2624200CE` |
| BFO | `{NAME}{YMMDD}{STRIKE}{TYPE}` | `SENSEX2650778000CE` (Y=26, M=5, DD=07) |

`ticker_service` indexes both segments under `_nfo_by_key`. Option lookup matches by **strike from the instrument's `strike` metadata field** (in paisa: 78000 → 7800000.0) and **expiry date from `expiry` metadata** — never by substring search on the symbol (which would false-positive across the BFO date encoding).

### Data Methods
- `ltpData` → single LTP (cached)
- `getMarketData(mode, tokens)` → batch OHLC/FULL/LTP (50 at a time)
- `getCandleData` → OHLCV candles (1m through 1d). For BSE options, pass `exchange="BFO"`.
- `optionGreek` → real IV/delta/gamma (market hours only)
- `rmsLimit` → real margin/capital from broker

---

## The Annex — THE V2 BACKEND WING (Multi-Tenant API)

Adjacent to the main tower, connected by a glass walkway, sits a newer building: the v2 backend. Same address, different architecture. Built for multi-tenant operation, async order execution, and clean app-by-app boundaries.

### The Floor Plan (`backend/apps/`)

| App | Responsibility |
|-----|----------------|
| `accounts`, `tenants` | Auth + tenant scoping; JWT carries `tenant_id` claim |
| `broker` | Broker connection records (paper / Angel One / Zerodha) |
| `portfolio` | Portfolios, positions, snapshots, **monthly feedback report** |
| `orders` | Order placement with outbox pattern, idempotency, async fills |
| `market_data` | Symbols, candles, market pulse, sector rotation, shortlist |
| `strategies` | Strategy registry + plugin discovery (entry-points) |
| `agents_core` | Agent run logs + WebSocket consumer for live event stream |
| `rag` | Retriever / Embedder / VectorStore interfaces + registry |
| `journals` | Journal entries linked to portfolio + agent_run + order |
| `notifications` | Alert delivery (in-app, email, Telegram, webhook) |
| `legacy` | Bridge endpoints into the existing `trading/` app — pyramid view, screener, straddle |
| `billing` | Plans, subscriptions, entitlements |
| `common` | Middleware, tenancy mixin, db_router, permission classes |

### The DB Router

A clever piece of plumbing: `apps.common.db_router.LegacyRouter` pins all `trading.*` ORM operations to the `legacy` SQLite alias, while the v2 stack runs on Postgres. Result: the React UI shows **real numbers from 700+ existing TradeJournal rows immediately**, without a schema migration. In dev, both DBs are the same SQLite file (single process). In prod, the legacy app runs on its own port (8001) and the v2 app runs on 8000 (Postgres).

### The Outbox Pattern

`POST /api/v1/orders/` doesn't talk to the broker directly. It writes an `Order` row + an `OutboxEvent` row in a single transaction (status=QUEUED), responds 202, and returns. A Celery worker picks up the OutboxEvent and routes the order through `BrokerService.place_order()`. Failures are retryable, and idempotency keys prevent duplicate fills.

### The WebSocket Bus

`/ws/ticks/` — live LTP fan-out. Frontend subscribes to symbols, backend filters by per-tenant subscription set.
`/ws/agent/{run_id}/` — live token stream from a running agent.
Auth is via JWT in the `Sec-WebSocket-Protocol` subprotocol header (the only way browsers can pass auth on WS).

---

## The Roof — THE PHILOSOPHY (Why This Building Exists)

Stand on the roof. Look down at the entire structure. The money-making thesis is simple:

### Edge 1: Structure at Levels (Equity)
> "Don't trade random breakouts. Trade **reversals and retests at confluent price levels** where multiple technical factors stack. The Sweet Spot filter ensures every trade happens at a meaningful price."

**Backtested edge**: Level Bounce — 70% win rate, +Rs 83,850/lot

### Edge 2: Regime Awareness (Equity)
> "The same setup has completely different odds depending on what kind of day it is. A breakout on a trending day (1.15x confidence) vs. a range day (0.5x confidence) — the math changes everything."

### Edge 3: Theta Decay (Options)
> "Sell time. Short straddles profit from the passage of time as long as NIFTY stays within a range. The 1.3x hard stop limits catastrophic loss. The 250-point shift re-centers when NIFTY drifts."

### Edge 4: Discipline Over Intelligence
> "The AI can think whatever it wants. @RiskGuard doesn't care. 10 deterministic gates. No trade exceeds 1% risk. No day exceeds 3% loss. The system survives bad days to capitalize on good ones."

### Edge 5: Self-Improving Memory
> "Every trade is journaled. Every decision is audited. The RAG loop feeds history back into the planner. The system remembers its mistakes. Over time, confidence calibration improves."

---

## Quick-Access Mental Anchors

When you need to recall something fast, picture:

| To Remember | Picture This |
|------------|--------------|
| Risk limits | The **red laser grid** around the vault: 1% per trade, 3% per day, 10% position, 3 max open |
| The risk guard | A **calculator with a stamp** — not AI, pure math, 10 gates |
| Level Bounce edge | A **rubber ball** bouncing off a thick line on a chart — 70% WR |
| Straddle hard stop | A **fire alarm** reading 1.3x — when it hits, you evacuate both legs |
| Sweet Spot filter | A **sieve** — signals without level confluence fall through and die |
| Regime multiplier | A **traffic light** — green trending day boosts breakdowns 1.15x, red range day kills breakouts to 0.5x |
| The 5-minute heartbeat | A **metronome** ticking every 5 minutes — scan, detect, filter, validate, execute |
| Theta decay | An **ice cube** melting — every minute that passes without NIFTY moving is money in your pocket |
| The 3 floors of decision | **Swing (weeks) → Premarket (day) → Intraday (minutes)** — telescope → binoculars → microscope |
| The audit trail | A **CCTV camera** in every room — nothing happens unrecorded |
| Pyramid sizing | A **stack of poker chips** doubling at each higher-low confirmation — not on hope, on structure |
| The Library (Floor 6.5) | A **leather-bound monthly ledger** — every signal recorded, every rejection reviewed in hindsight |
| BFO vs NFO | Two **different colored pipes** in the basement — orange for NFO (NIFTY/BANKNIFTY), blue for BFO (SENSEX) |
| The market-config seam | A single **brass plaque** on the wall: NIFTY=65, BANKNIFTY=30, SENSEX=20 — never written elsewhere |

---

## The Full Day Walk-Through

_07:00_ — You enter the building. Walk to Floor 1 (Observatory). The Oliver Kell wheel has rotated overnight. 5 new stars are in WP/EC/BB.

_08:30_ — Floor 2 (War Room). The premarket scanner scores 50 stocks. Top 10 get dossiers. Camarilla pivots computed. If LLM is enabled, Claude writes a morning brief.

_09:15_ — Market opens. You sprint to Floor 3 (Trading Floor).

_09:15-09:30_ — ORB forming. LevelMap builds. No signals yet (waiting for first candles to establish range).

_09:30_ — The 5-minute metronome starts ticking. Every tick: scan → detect → filter → validate → execute. Level Bounce is the primary hunter. Sweet Spot kills weak signals.

_09:30-15:15_ — Trades open. Trade Manager babysits: breakeven at 0.5R, partial at 1.0R, trail tightens at 1.5R. Time stop kills zombies at 10 candles.

_Meanwhile, Floor 4_ — Straddle lifecycle runs every 15-30 minutes. Theta melts the ice cube. If NIFTY drifts 250+ points, SHIFT_TO_ATM. If premium hits 1.3x sold, CLOSE_BOTH. @OptionsStrategist advises, @RiskGuard validates.

_Meanwhile, Floor 5_ — Screener watches 98 stocks for the 8 active strategies. Telegram alerts fire when patterns hit. Separate from the main trading pipeline — more of an early warning system.

_15:15_ — Intraday monitor stops. Trade Manager closes any remaining positions. Straddle monitor checks for expiry-day close.

_15:30_ — Daily review. P&L tallied. TradeJournal updated. PortfolioSnapshot saved. AuditLog complete.

_15:35_ — `enrich_signals` runs. The Librarian (Floor 6.5) walks the day's SignalLog entries, fetches intraday candles, and stamps each with what the market actually did. Skipped signals get marked EXPIRED with their max_favorable_move recorded. Risk-rejected signals get linked to their TradeJournal entry. The Library now has a complete record of every opportunity — taken or not.

_Evening_ — Swing scanner runs. The wheel rotates. Tomorrow's universe narrows.

_End of month_ — You climb to the Library (Floor 6.5). Open `/monthly` in the React UI. The Capture Rate Matrix tells you which stocks moved without you. The Risk Rejections card tells you what @RiskGuard blocked that would have made money. The lessons panel suggests which strategies to lean into and which to disable. You adjust thresholds. The system gets a little smarter.

_You take the elevator down, walk past the vault. The daily loss is within 3%. The capital is intact. @RiskGuard nods. You leave the building. Come back tomorrow._

---

_"The purpose of this building is not to be right. It is to survive being wrong — and to compound the days when you are right."_
