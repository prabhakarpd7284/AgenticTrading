# AgenticTrading — Claude Code Context

## What This Is

AlphaDesk — an AI-driven Indian stock market trading system. Three workflows, two interfaces, one risk gate.

**Workflows:**
- **Directional Trading** — BUY/SELL equity intraday positions (NSE cash market)
- **Options Straddle** — Short straddle lifecycle management (NFO options, NIFTY/BANKNIFTY)
- **Pyramid Options** — Aggressive momentum pyramiding on intraday options (NIFTY / BANKNIFTY / SENSEX)

**Interfaces:**
- **Streamlit dashboard** (`dashboard.py`) — original operator console, talks directly to legacy DB
- **React SPA** (`frontend/`) — trader UI; talks to v2 backend (port 8000) + legacy bridge (port 8001)

**Stack:** Django 5 · LangGraph · Angel One SmartAPI · Celery · Postgres (v2) + SQLite (legacy)
**Capital default:** 500,000 INR · **Mode default:** paper · **Broker:** Angel One

---

## Repo Layout

```
AgenticTrading/
├── trading/                  Legacy app — strategies, broker, risk engine, models
│   ├── agents/               @DirectionalTrader (LLM)
│   ├── graph/                LangGraph for equity workflow
│   ├── services/             data_service, broker_service, risk_engine, ticker_service
│   ├── options/straddle/     @OptionsStrategist + lifecycle graph
│   ├── pyramid/              Pyramid options strategy (deterministic, no LLM)
│   ├── screener/             Live intraday opportunity detector
│   ├── swing/                Oliver Kell daily/weekly cycle scanner
│   ├── basket/               Premarket basket builder
│   ├── backtester/           Generic v2 backtest engine
│   ├── rag/                  RAG retriever for trade context
│   ├── models.py             TradeJournal, StraddlePosition, AuditLog, SignalLog,
│   │                         WatchlistEntry, PortfolioSnapshot, SystemControl
│   └── management/commands/  CLI entry points
│
├── backend/                  v2 multi-tenant Django stack (Postgres + Redis)
│   └── apps/                 accounts, tenants, broker, portfolio, orders,
│                             market_data, strategies, agents_core, rag, journals,
│                             notifications, legacy (bridge), billing, common
│
├── frontend/                 React + Vite SPA
│   └── src/
│       ├── features/         pulse, rotation, shortlist, setup, dashboard,
│       │                     positions, monthly, agents, strategies, backtester,
│       │                     pyramid, swing-scanner, brokers, auth
│       └── lib/              api, ws, market-config, market-pulse, monthly, utils
│
├── dashboard.py              Streamlit operator console
├── dashboard_utils/          Streamlit components, candle cache, market scanner
└── docs/                     ADRs, architecture, MIND_PALACE, strategy specs
```

---

## AI Virtual Team

| Agent | Role | Files | LLM? |
|-------|------|-------|------|
| **@DataAnalyst** | Fetches and enriches market data from Angel One | `trading/services/data_service.py`, `trading/options/data_service.py` | No |
| **@DirectionalTrader** | Plans BUY/SELL equity trades from intraday structure | `trading/agents/planner.py`, `trading/graph/trading_graph.py` | Yes (Claude) |
| **@OptionsStrategist** | Manages straddle positions — HOLD/CLOSE/HEDGE/ROLL | `trading/options/straddle/graph.py`, `trading/options/straddle/prompts.py` | Yes (Claude) |
| **@RiskGuard** | Deterministic risk validation — no LLM, no exceptions | `trading/services/risk_engine.py` | Never |
| **@PortfolioTracker** | Tracks capital, P&L, daily loss, open positions | `trading/rag/retriever.py`, `trading/models.py:PortfolioSnapshot` | No |
| **@PyramidOperator** | Pure-Python pyramid strategy (no LLM) | `trading/pyramid/strategy.py` | Never |

### Agent Interaction Rules
- **@RiskGuard** is the last gate before every execution. It cannot be bypassed.
- **@DataAnalyst** runs first in every workflow. No trading decision without live data.
- **@DirectionalTrader**, **@OptionsStrategist**, and **@PyramidOperator** never share state — separate graphs/engines.
- **@PortfolioTracker** is read-only during planning; updated only after execution.

---

## How to Run

### One-time setup
```bash
# Legacy + Streamlit
python -m venv .venv && source .venv/bin/activate
pip install -e .
python manage.py migrate
python manage.py run_trading_agent --seed-strategies
python manage.py run_trading_agent --init-portfolio 500000

# v2 backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
python manage.py migrate

# Frontend
cd frontend && npm install
```

### Streamlit dashboard
```bash
streamlit run dashboard.py
```

### React UI + v2 backend (full stack)
```bash
# Terminal 1: v2 backend
cd backend && python manage.py runserver 0.0.0.0:8000

# Terminal 2: legacy bridge
python manage.py runserver 0.0.0.0:8001

# Terminal 3: Celery worker (for async order execution)
cd backend && celery -A config worker -Q default,agents,orders -l info

# Terminal 4: frontend
cd frontend && npm run dev
```

Or `docker compose -f docker-compose.local.yml up` for the full stack in one command.

### Directional Trading (Equity)
```bash
python manage.py run_trading_agent "Plan a BUY trade for HDFCBANK"
python manage.py run_trading_agent --show-journal
```

### Straddle Management (Options)
```bash
python manage.py manage_straddle --register --underlying NIFTY \
  --strike 24200 --expiry 2026-05-13 \
  --ce-symbol NIFTY13MAY2624200CE --ce-token 41762 --ce-sell 394.85 \
  --pe-symbol NIFTY13MAY2624200PE --pe-token 41763 --pe-sell 138.35 --lots 1

python manage.py manage_straddle --analyze --position 1
python manage.py manage_straddle --status --position 1
python manage.py manage_straddle --execute CLOSE_BOTH --position 1
python manage.py manage_straddle --list
```

### Pyramid Options
```bash
python manage.py run_pyramid --strike 24200 --type CE --underlying NIFTY
```
Or via the React UI at `/pyramid` — backtest with config inputs, then "Go Live (Paper)".

### Live Screener
```bash
python manage.py run_screener                       # live
python manage.py run_screener --telegram            # with alerts
python manage.py run_screener --backtest --from 2026-04-01 --to 2026-04-30
```

### Oliver Kell Swing Scanner
```bash
python manage.py run_ok_scanner --actionable-only --telegram
```

### EOD Signal Enrichment (feeds the monthly feedback report)
```bash
python manage.py enrich_signals          # today
python manage.py enrich_signals --all    # backfill all unrenriched
```

### Full trading day (premarket → screener → straddle monitor → review)
```bash
python manage.py run_trading_day
```

---

## Architecture

```
Directional Workflow (LangGraph):
  fetch_data → retrieve_context → planner(@DirectionalTrader) → risk(@RiskGuard) → execute → journal

Straddle Workflow (LangGraph):
  fetch_market_data → analyze_position → generate_action(@OptionsStrategist) → validate(@RiskGuard) → execute → journal

Pyramid Workflow (deterministic loop):
  fetch_candles → run_pyramid (entry/pyramid/trail/exit) → place_order(s) → persist

Screener Workflow (event loop):
  tick → CandleStore → IndicatorEngine → 8 strategies → Signal → SignalLog (persist) → Telegram

Feedback Loop:
  every signal fired (any source) → SignalLog
  EOD: enrich_signals fills max_favorable/adverse + outcome
  Monthly: /monthly aggregates capture rate, signal audit, rejections, analytics, benchmark, lessons
```

All workflows:
- Use Angel One SmartAPI (paper or live via `TRADING_MODE`)
- Log every decision to `AuditLog` (non-blocking)
- Save outcomes to the appropriate model
- Are validated deterministically before execution

---

## Key Invariants (Never Break These)

1. **@RiskGuard is always the last gate** before execution. LLM cannot bypass it.
2. **`TRADING_MODE=paper` by default** — set to `live` explicitly and deliberately.
3. **Never trust LLM for position sizing** — always computed deterministically (% of capital).
4. **Journal every decision** — rejected trades are also recorded.
5. **Expiry-day positions** — close before 3:15 PM IST. Gamma risk explodes after.
6. **Audit log is non-blocking** — logging failures never stop trading flow.
7. **Short straddle hard stop** — combined premium > 1.3× sold → close immediately.
8. **Index metadata has one source** — `frontend/src/lib/market-config.ts` for the UI; lot sizes/expiry days/tokens are NEVER hardcoded elsewhere.
9. **SignalLog is non-blocking** — every signal is persisted in a try/except so logging failures don't stop the screener.
10. **Pyramid trail SL only ratchets up** — never relaxes.

---

## Index Metadata (Jan 2026+)

| Index | Lot | Weekly expiry | Exchange | Angel One token |
|-------|-----|---------------|----------|-----------------|
| NIFTY | **65** | Tuesday | NSE (NFO) | 99926000 |
| BANKNIFTY | **30** | Last Tuesday (monthly only) | NSE (NFO) | 99926009 |
| SENSEX | **20** | Thursday | BSE (BFO) | 99919000 |

NIFTY lot dropped from 75 → 65 in Jan 2026 (NSE circular FAOP70616). NSE expiries moved to Tuesday in Sep 2025 (SEBI standardisation); BSE expiries moved to Thursday. BANKNIFTY no longer has weekly contracts.

**ticker_service** indexes both NFO and BFO instruments. Option lookup matches by **strike** (in paisa: 78000 → 7800000.0) and **expiry date metadata** — never by substring search on the symbol (BSE encoding can collide with strike digits).

---

## File Map (Trading Core)

```
trading/
├── agents/planner.py              @DirectionalTrader — dual-mode Claude (CLI/API)
├── graph/
│   ├── state.py                   TradingState, TradePlan, RiskResult, ExecutionResult
│   └── trading_graph.py           Equity LangGraph workflow (6 nodes)
├── services/
│   ├── data_service.py            @DataAnalyst — equity OHLCV from Angel One + intraday cache
│   ├── ticker_service.py          Symbol master (NSE + NFO + BFO), option lookup
│   ├── risk_engine.py             @RiskGuard — 10-criteria deterministic validator
│   ├── broker_service.py          Order execution (paper + live, NFO + BFO + NSE)
│   └── backtester.py              Historical candle replay (legacy)
├── options/
│   ├── data_service.py            NFO/BFO options LTP + VIX + index candles
│   └── straddle/
│       ├── state.py               StraddleState, StraddleAction, StraddleAnalysis
│       ├── prompts.py             @OptionsStrategist system prompt
│       ├── analyzer.py            P&L, delta, market phase, scenarios (pure Python)
│       └── graph.py               Straddle LangGraph workflow
├── pyramid/
│   ├── strategy.py                Pyramid engine (entry/pyramid/trail/exit, KPIs)
│   └── telegram.py                Pyramid alerts
├── screener/
│   ├── engine.py                  Live opportunity detector + dedup + cooldowns
│   ├── signals.py                 Signal dataclass + persist() to SignalLog
│   ├── strategies.py              8 active intraday strategies
│   └── tick_stream.py             WebSocket + REST polling fallback
├── swing/
│   ├── ok_cycles.py               Oliver Kell phase detector
│   ├── ok_scanner.py              NIFTY 100 daily/weekly scanner
│   └── ok_intraday.py             5m intraday variant
├── basket/                        Premarket basket builder (mood, manager, executor)
├── backtester/                    v2 generic backtest engine (engine, entry, exits, sizing)
├── rag/retriever.py               @PortfolioTracker — RAG context for planner
├── utils/
│   ├── indicators.py              EMA/RSI/BB/MACD/ATR/VWAP/WMA
│   ├── time_utils.py              Market hours, session phase, last_trading_day
│   ├── expiry_utils.py            Next expiry resolution (NSE Tue, BSE Thu)
│   └── candle_cache.py            (in dashboard_utils/) Daily → weekly/monthly aggregator
├── models.py                      TradeJournal, StraddlePosition, PortfolioSnapshot,
│                                  AuditLog, SignalLog, WatchlistEntry, SystemControl
└── management/commands/
    ├── run_trading_agent.py       CLI: equity directional
    ├── run_trading_day.py         CLI: full day orchestration
    ├── manage_straddle.py         CLI: straddle lifecycle
    ├── run_pyramid.py             CLI: pyramid backtest/execute
    ├── run_screener.py            CLI: live intraday screener (persists to SignalLog)
    ├── run_ok_scanner.py          CLI: Oliver Kell daily scan (persists to SignalLog)
    ├── enrich_signals.py          CLI: EOD signal outcome enrichment
    └── run_morning_basket.py      CLI: premarket basket
```

---

## File Map (v2 Backend — relevant for monthly report + orders)

```
backend/apps/portfolio/
  ├── models.py                    Portfolio, Position, PortfolioSnapshot
  ├── services/
  │   └── monthly_report.py        Aggregation: month groups, YTD, capture matrix,
  │                                signal audit, rejections, equity curve, analytics,
  │                                benchmark vs NIFTY50, AI lessons
  └── api/
      ├── views.py                 PortfolioViewSet, PositionViewSet, MonthlyReportView
      └── urls.py                  GET /api/v1/portfolios/monthly/

backend/apps/orders/
  ├── models.py                    Order (with status QUEUED/SENT/OPEN/FILLED/...)
  ├── services/place_order.py      Outbox pattern: validate → enqueue
  └── api/views.py                 POST /api/v1/orders/ → 202 Accepted

backend/apps/legacy/api/views.py   Bridge endpoints into trading/ — pyramid view,
                                    screener view, straddle view, etc.
```

---

## File Map (Frontend)

```
frontend/src/
├── app/
│   ├── router.tsx                 React Router with all routes
│   └── guards.tsx                 RequireAuth wrapper
├── features/
│   ├── pulse/                     Market Pulse (Cascade Stages 1+2)
│   ├── rotation/                  Sector rotation (Stage 3)
│   ├── shortlist/                 Tradeable shortlist (Stage 4)
│   ├── setup/                     Per-stock setup preview (Stage 5)
│   ├── dashboard/                 Capital + equity curve + AI activity
│   ├── positions/                 Live positions with WebSocket LTP
│   ├── monthly/                   Monthly Feedback Report
│   ├── pyramid/                   Pyramid strategy UI (live ticker, KPIs, charts)
│   ├── swing-scanner/             Oliver Kell scanner
│   ├── agents/                    Agent run console
│   ├── strategies/                Strategy library
│   ├── backtester/                Backtest UI
│   ├── broker/                    Angel One linking
│   └── auth/                      Login, signup, onboarding
├── lib/
│   ├── api.ts                     Axios client + JWT refresh + paginated unwrap
│   ├── ws.ts                      WebSocket helper with subprotocol JWT auth
│   ├── market-config.ts           ⭐ Single source of truth for index metadata
│   ├── market-pulse.ts            Pulse hook (used by /pulse and /pyramid ticker)
│   ├── monthly.ts                 Monthly report types + hook + mock data
│   └── utils.ts                   fmtInr, fmtPct, clsPnl, fmtRel, cn
└── components/ui/                 Card, Button, Badge, KPI, DataTable, Tabs, etc.
```

---

## Environment Variables

```bash
# Broker (root .env)
SMARTAPI_KEY=...
SMARTAPI_USERNAME=...
SMARTAPI_PASSWORD=...
SMARTAPI_TOTP_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Trading mode
TRADING_MODE=paper        # paper | live
PLANNER_MODE=cli          # cli (Max plan, free) | api (Anthropic credits)
LLM_MODEL=claude-sonnet-4-6

# Risk limits
DEFAULT_CAPITAL=500000
MAX_RISK_PER_TRADE_PCT=1.0
MAX_DAILY_LOSS_PCT=3.0
MAX_POSITION_SIZE_PCT=10.0

# v2 backend (backend/.env)
DJANGO_SECRET_KEY=...
DATABASE_URL=postgres://alphadesk:alphadesk@localhost:5432/alphadesk
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=...

# Frontend (frontend/.env.local)
VITE_API_URL=http://localhost:8000/api/v1
VITE_LEGACY_API_URL=http://localhost:8001/api/v1
VITE_WS_URL=ws://localhost:8000
```

All `.env*` are gitignored. Use `.env.example` files as templates. **Never commit secrets.**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Django 5.1 (legacy + v2) |
| Agent orchestration | LangGraph 0.6+ |
| LLM | Claude (Anthropic) — CLI or API |
| Broker | Angel One SmartAPI (`SmartApi` package) |
| Database | SQLite (legacy) · PostgreSQL (v2) |
| Cache / Queue | Redis (v2) |
| Async tasks | Celery |
| Real-time | Django Channels |
| API | DRF (v2) |
| Frontend | Vite · React 18 · TypeScript · TanStack Query · Zustand · Tailwind |
| Charting | Recharts · TradingView lightweight-charts |
| Streamlit dashboard | Streamlit + Pandas |
| Logging | structlog (v2) · Logzero (legacy) · Sentry |
| Deployment | Docker · ECS Fargate · Amplify · CloudWatch · Terraform |

---

## Documentation Index

| Document | What it covers |
|----------|---------------|
| `README.md` | Quick start + tech overview |
| `docs/MIND_PALACE.md` | Spatial mental map of the entire system |
| `docs/architecture/ARCHITECTURE.md` | Formal architecture spec |
| `docs/architecture/BACKEND_STRUCTURE.md` | v2 Django app layout |
| `docs/api/WEBSOCKETS.md` | WebSocket channel spec |
| `docs/adr/` | Architecture Decision Records |
| `docs/strategies/` | Strategy playbooks |
| `docs/SCREENER_SPEC.md` | Live screener spec |
| `TRADING_FRAMEWORK.md` | The Cascade — 6-stage pre-trade framework |
| `DELIVERY_MANIFEST.md` | What ships in this repo |
