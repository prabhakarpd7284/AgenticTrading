# The Cascade — AlphaDesk Trading Framework

> How an experienced Indian-equity intraday trader narrows down the universe
> of ~2,000 listed stocks into 2–3 trade candidates — and how AlphaDesk's
> AI desk mirrors that workflow, screen-by-screen, gate-by-gate.

---

## Why a framework at all?

Most "AI trading" tools jump straight to "give me a signal on `RELIANCE`." That is
backwards. Profitable discretionary traders never start with a symbol — they start
with the **regime**, confirm it with **context**, find the **sector that's flowing**,
shortlist **leaders** in that sector, and only then look for a **setup**. Most
days, the correct answer at Stage 1 is "sit on your hands." That is a **feature**,
not a bug.

AlphaDesk's job is to execute that discipline mechanically. The Cascade is the
ordered pipeline that gets us from "markets just opened" to "here are the 1–2
positions the desk will take, with size and stops." Each stage is a hard gate —
if Stage N fails, Stages N+1…6 don't run.

---

## The six stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1  REGIME      →  Is this a day to trade at all?                        │
│  2  CONTEXT     →  What's driving flows? (FX, commodities, rates)        │
│  3  ROTATION    →  Which sectors are leading / lagging today?            │
│  4  SHORTLIST   →  Who are the leaders in those sectors?                 │
│  5  SETUP       →  Is there a clean, high-R:R entry on the shortlist?    │
│  6  EXECUTE     →  Size, place, manage, journal.                         │
└─────────────────────────────────────────────────────────────────────────┘
            ▲                                                    │
            └────────────────── feedback loop ───────────────────┘
                    (each fill updates regime awareness)
```

Every stage maps to one UI screen, one backend service, one or more audit-logged
agent gates.

| # | Stage | UI screen | Backend | Primary agent gate |
|---|-------|-----------|---------|--------------------|
| 1 | REGIME    | `/pulse` *(index)*        | `market_data/pulse_service.py` | `tradeable` flag feeds every agent |
| 2 | CONTEXT   | `/pulse` *(same screen)*   | `pulse_service.py` + news feed | Regime classifier uses FX + commodities |
| 3 | ROTATION  | `/pulse#sectors` → `/rotation` | `market_data/rotation_service.py` | Hot sectors feed Stage 4 shortlist |
| 4 | SHORTLIST | `/shortlist`              | `market_data/shortlist_service.py` | Liquidity + ATR + F&O screens, confluence score |
| 5 | SETUP     | `/setup/<symbol>`         | `market_data/setup_service.py` + `trading/agents/planner.py` | Deterministic plan + 10-criterion @RiskGuard X-ray |
| 6 | EXECUTE   | `/positions`, `/agents`   | `trading/services/broker_service.py` | `@RiskGuard` (9-criterion gate) |

Stages 5 and 6 already exist in the codebase as the directional / straddle
LangGraph workflows. The Cascade pivot adds Stages 1–4 as the *pre-trade funnel*
the agents read before they get to plan anything.

---

## Stage 1 — REGIME

**Question:** Is today a day to put risk on?

A discretionary trader glances at 4 numbers in the first 60 seconds after the
open:
1. **India VIX** — are options priced for panic, calm, or sleep?
2. **NIFTY gap** — is the tape running or fading the overnight?
3. **S&P 500 / SGX Nifty** — what's the global risk appetite?
4. **USDINR** — is the rupee telling a capital-flow story?

These four roll up into **three classifiers**, all deterministic:

| Classifier | Inputs | Outputs |
|------------|--------|---------|
| `vol`         | INDIAVIX last print | complacent (<12) · low (<16) · normal (<20) · elevated (<28) · high (<35) · extreme (≥35) |
| `trend`       | NIFTY change % vs prev close | up (>+0.6%) · down (<-0.6%) · range |
| `global_tone` | S&P 500 change % | risk_on (>+0.4%) · risk_off (<-0.4%) · neutral |

The classifier then collapses those three into a **`tradeable`** boolean and a
one-line summary. Hard rules (matching the existing `risk_engine.py` /
straddle graph gates):

- `vol = extreme` → **halt everything**, no fresh entries, manage only.
- `vol = high` AND `|gap| > 1.5%` → first 30 min **read-only**.
- `vol > 20` → `@OptionsStrategist` cannot sell fresh straddles.
- `vol > 35` → `@OptionsStrategist` cannot touch options at all.

**Code:** `backend/apps/market_data/services/pulse_service.py::classify_regime`.
**UI:** `frontend/src/features/market-pulse/MarketPulsePage.tsx` → `<RegimeBanner>`.

---

## Stage 2 — CONTEXT

**Question:** What's driving the flows I'm seeing in Stage 1?

Same screen, additional tiles. India doesn't trade in a vacuum:

- **Crude & Brent** — up big → OMCs, airlines, paints hurt; ONGC, Reliance E&P win.
- **Natural gas** — specific to GAIL, Petronet, city-gas names.
- **Gold & Silver** — risk-off barometer; jewellers rotate.
- **Copper** — real-economy read; Hindalco, Vedanta.
- **USDINR & DXY** — dollar strength = IT tailwind + commodity drag.
- **US10Y** — duration risk; metals, real-estate, rate-sensitives.

Context doesn't gate trading directly — but it tells the trader *which sectors*
to expect to lead (Stage 3). The classifier uses it only as input to the
`global_tone` label.

**Code + UI:** same as Stage 1.

---

## Stage 3 — ROTATION

**Question:** Given the regime + context, which sectors are leading today?

A sector heatmap of the 11 NSE sector indices, ranked by % move, with a
drill-in that shows the top 3 leaders + bottom 3 laggards inside each
sector (drawn from a curated subset of NIFTY50 constituents). The pulse
page heatmap tiles deep-link to `/rotation#<sector-key>`, which scrolls
straight to that sector's card.

Currently surfaced per sector:

- **Rank + % move** — matches the pulse heatmap order.
- **Leaders / Laggards** — top/bottom movers inside curated constituents.
- **Breadth** — up/down/flat counts across the sector's NIFTY50 names.

Planned follow-ups:

- Relative strength vs NIFTY over 1d / 5d / 20d.
- Volume surge detection per sector.

**Code:** `backend/apps/market_data/services/rotation_service.py`,
endpoint `GET /api/v1/market-data/rotation/` (60s cache).
**UI:** `frontend/src/features/rotation/RotationPage.tsx` at `/rotation`.

---

## Stage 4 — SHORTLIST

**Question:** Inside the hot sectors, who are the tradeable leaders?

Hard filters (non-negotiable — these mirror `@RiskGuard` criteria):

- F&O eligible (needed for options leg + for cleaner shorts)
- ATR(14) / price ≥ `SHORTLIST_MIN_ATR_PCT` (default 1.0% — framework
  quotes 1.5%, we accept 1.0% so low-vol days still produce a watchlist)
- Avg daily turnover ≥ `SHORTLIST_MIN_TURNOVER_CR` (default ₹10 cr)

Soft screens feed a **confluence score** (0-100), surfaced as reasons pills:

- **Sector rank** — linear decay across top-4 sectors (up to 25 pts)
- **Sector leader** — top-3 constituent flagged by Stage 3 (+15)
- **Day-move alignment** — aligned with sector and meaningful (|>0.5%|) → +15;
  against-sector → −10 penalty
- **ATR tier** — 10/15/20 pts at 1.0% / 1.5% / 2.5%+
- **Relative volume** — 5 pts at 1.2×, 10 pts at 1.5×
- **52-week position** — near-breakout (>0.95) or near-breakdown (<0.05) → +10

Output: a watchlist of up to 15 names ordered by score, plus a separate
`filtered_out` list showing why rejections missed the hard filters (audit
trail, not silently dropped).

**Code:** `backend/apps/market_data/services/shortlist_service.py`,
endpoint `GET /api/v1/market-data/shortlist/` (300s cache).
**UI:** `frontend/src/features/shortlist/ShortlistPage.tsx` at `/shortlist`.
Reuses `build_rotation()` output so there's no duplicate data pipe —
Stage 3 is the single source of truth for hot sectors and leaders.

---

## Stage 5 — SETUP

**Question:** For a given shortlisted symbol, is there a trade setup right now,
and would it actually clear the desk's risk gate?

There are two complementary surfaces:

1. **`/setup/<symbol>` — "would this clear?" preview.** Synchronous, no LLM.
   The shortlist deep-links here. The backend (`market_data/setup_service.py`)
   pulls the last 50×5m candles, builds a deterministic ATR-anchored plan
   (1.5×ATR stop, 3×ATR target, 1%-of-capital sizing), and runs the
   production `validate_trade` for the authoritative verdict. To make the
   page useful even when the gate **rejects**, a per-criterion helper runs
   each of the ten gates *independently* — the UI shows the full X-ray, not
   just the first failure. This is what the operator clicks before deciding
   whether to even fire the agent. *Code:* `setup_service.py`,
   `apps/market_data/api/views.py::SetupPreviewView`. *UI:*
   `frontend/src/features/setup/SetupPage.tsx`.
2. **`@DirectionalTrader` agent (`trading/agents/planner.py`) — full plan.**
   When the operator wants the LLM read (intraday candles, VWAP/EMA
   confluence, volume profile, RAG context from past setups), they fire an
   agent run from `/agents`. The agent's plan still goes through the same
   `validate_trade` gate — meaning the preview at `/setup/<symbol>` is a
   faithful prediction of what an agent run would face.

**Plan geometry (deterministic, no LLM)**

| Parameter | Formula |
|-----------|---------|
| Entry      | LTP (or last close) |
| Stop loss  | entry ∓ 1.5 × ATR(14, 5m) |
| Target     | entry ± 3.0 × ATR(14, 5m)  *(R:R = 2.0)* |
| Quantity   | floor(min(1% × capital ÷ risk_per_share, 10% × capital ÷ entry)) |
| Confidence | 0.6 (just above the 0.55 floor; baseline) |

**The 10-criterion @RiskGuard breakdown** the page renders:

| # | Criterion | Source of truth |
|---|-----------|-----------------|
| 0 | Cascade regime gate (vol + tradeable flag) | `risk_engine._check_regime` |
| 1 | Plan well-formed (fields, positive numbers) | `validate_trade` step 1 |
| 2 | Stop on the correct side of entry | step 2 |
| 3 | Target on the correct side of entry | step 3 |
| 4 | Risk ≤ 1% of capital | step 4 |
| 5 | Daily loss within 3% (incl. headroom check) | step 5 |
| 6 | Position notional ≤ 10% of capital | step 6 |
| 7 | R:R ≥ 1.5 | step 7 |
| 8 | Plan confidence ≥ 0.55 | step 8 |
| 9 | Open positions < 3 | step 9 |

The constants (`MAX_RISK_PER_TRADE_PCT`, `MIN_RISK_REWARD_RATIO`, …) are
imported from `trading/services/risk_engine.py` directly — change them
there and the preview UI updates automatically.

**Code:** `backend/apps/market_data/services/setup_service.py`,
`backend/apps/market_data/api/views.py::SetupPreviewView`,
`trading/services/risk_engine.py::validate_trade` (overall verdict).
**UI:** `frontend/src/features/setup/SetupPage.tsx`, deep-linked from
`/shortlist`.

---

## Stage 6 — EXECUTE *(exists)*

**Question:** Can I put this on, and how?

`@RiskGuard` is the last gate. No execution goes around it. 9 criteria:

1. Capital available
2. Per-trade risk ≤ `MAX_RISK_PER_TRADE_PCT` (default 1%)
3. Daily loss not breached (`MAX_DAILY_LOSS_PCT`, default 3%)
4. Position size ≤ `MAX_POSITION_SIZE_PCT` (default 10%)
5. No conflicting open position
6. Liquidity (avg turnover threshold)
7. No imminent corporate action
8. Trading mode check (paper vs live)
9. Regime gate (Stage 1 `tradeable` flag + vol tier)

The 9th criterion is the newly-added Cascade coupling — `@RiskGuard` now reads
the pulse cache and rejects if the regime hasn't cleared.

**Code:** `trading/services/risk_engine.py`, `trading/services/broker_service.py`.
**UI:** `/positions`, `/agents`.

---

## The feedback loop

Every fill updates three things:

1. **Journal** (`TradeJournal`, `StraddlePosition`) — the permanent record.
2. **Regime awareness** — if we take 2 losses in an hour in a "tradeable"
   regime, downgrade the `tradeable` flag to read-only for the rest of the
   session. This is a live tripwire, not a post-mortem.
3. **Portfolio** (`PortfolioSnapshot`) — updates the daily loss ceiling used in
   criterion 3 above.

This loop is why the pulse screen auto-refreshes at 15s — so the operator sees
the *same* state the agent gates read.

### Monthly view — slow-loop companion to the live tripwires

The three loops above keep intraday gates honest. The monthly view
(`/monthly`) is the *slow* loop — the end-of-day / end-of-week dashboard
that answers *"did this month actually compound, and where did the money
come from?"*. Three-level drill:

```
Month  →  Asset class (Cash | F&O | Commodity)  →  Underlying roll-up  →  Legs
```

Each underlying row surfaces the four metric families the operator needs
to decide whether to keep doing what they're doing:

- **Capital** — deployed and notional exposure (margin vs. cash)
- **Risk geometry** — aggregate target-₹ and stop-₹ across open legs
- **Realised quality** — win rate, avg R:R, trade count
- **Duration** — days held, % of trading days in position (was the book
  actually on risk, or parked?)

A 12-month YTD strip sits on top as a navigator — click any bar to jump
to a past month. UI lives at `frontend/src/features/monthly/MonthlyPage.tsx`
and drives off `frontend/src/lib/monthly.ts`, which today returns mock
fixtures gated by `MONTHLY_USE_MOCK = true`. Flipping that constant (or
swapping it for a Vite env flag) swings the same hook onto the live
`GET /api/v1/monthly/` endpoint — no component change needed. All mock
positions set `paper_mode: true` and the page surfaces a prominent
PAPER badge so the operator never confuses the preview for real fills.

---

## What this framework is not

- It is **not** a backtesting engine. Backtests validate Stage 5 setups in
  isolation; The Cascade is the *live pipeline* that gates whether to run a
  setup at all.
- It is **not** a signal feed. There is no "buy this stock" output. The
  framework narrows the universe; humans or agents then place a trade.
- It is **not** a replacement for discretion. Each stage exposes the data; the
  gates are deterministic but the size, stop, and management still involve
  judgement (agents or human).

---

## Where we are today (2026-04-20)

| Stage | Status |
|-------|--------|
| 1 REGIME | ✅ Backend + UI live |
| 2 CONTEXT | ✅ Same screen; news feed TODO |
| 3 ROTATION | ✅ Backend + `/rotation` drill-in live; RS + volume follow-up |
| 4 SHORTLIST | ✅ Backend + `/shortlist` UI live; reuses Stage 3 output |
| 5 SETUP | ✅ `/setup/<symbol>` preview live; LLM agent run via `/agents` |
| 6 EXECUTE | ✅ `@RiskGuard` 10-criterion gate incl. Cascade regime coupling |

Next in the queue:

1. Add Stage 3 relative-strength and volume-surge dimensions.
2. Enrich Stage 4 soft screens: VWAP / 20 EMA proximity, corporate-action
   blackout window (hard filter).
3. Wire the pulse classifier's auto-downgrade tripwire (2 losses/hour in a
   "tradeable" regime → flip to read-only).
4. Stage 5 next pass: plug `@DirectionalTrader` LLM output into the same
   preview page (currently shows the deterministic baseline plan only).
5. Stage 6 next pass: per-row "click to simulate" from the setup page that
   kicks an agent run on the shown plan — still paper by default.
6. Monthly view: the aggregation service design is locked in
   `docs/adr/ADR-0006-truthful-ledger-and-broker-integration.md` — a
   postback-driven, polling-reconciled, snapshot-timestamped ledger behind
   a single `BrokerAdapter` protocol. Implementation ships in seven phases;
   P1–P4 (~11 days) flip `MONTHLY_USE_MOCK` to `false` on real paper data.
   P5 adds the `AngelBrokerAdapter` so a tenant `broker_mode='live'` flip
   lights up live trading without code change.

---

## Glossary (for when the agents read this doc)

- **@DataAnalyst** — pulls market data, no decisions.
- **@DirectionalTrader** — LLM-backed Stage 5 for equity.
- **@OptionsStrategist** — LLM-backed Stage 5/6 for short straddles.
- **@RiskGuard** — deterministic Stage 6 gate, 9 criteria, last line of defence.
- **@PortfolioTracker** — read-only during planning, updated after fills.
