# AgenticTrading — Claude Code Context

## What This Is

AlphaDesk — an AI-driven Indian stock market trading system. Three workflows, one interface, one risk gate.

**Workflows:**
- **Directional Trading** — BUY/SELL equity intraday positions (NSE cash market)
- **Options Straddle** — Short straddle lifecycle management (NFO options, NIFTY/BANKNIFTY)
- **Pyramid Options** — Aggressive momentum pyramiding on intraday options (NIFTY / BANKNIFTY / SENSEX)

**Interface:**
- **React SPA** (`frontend/`) — trader UI; talks to the single Django ASGI process on `:8000` (REST + WebSocket + Celery)

Phase 6 (Apr 2026) retired the legacy `:8001` bridge and SQLite DB — everything is multi-tenant Postgres on `:8000`. Phase 8 (May 2026) removed the Streamlit dashboard (`dashboard.py` / `dashboard_utils/`).

**Stack:** Django 5 · LangGraph · Angel One SmartAPI · Celery · Postgres (`:5436`) · Redis · React/Vite
**Capital default:** 500,000 INR · **Mode default:** paper · **Broker:** Angel One

---

## Repo Layout

```
AgenticTrading/
├── backend/                  v2 Django ASGI (REST + WS + Celery, multi-tenant Postgres)
│   ├── apps/                 accounts, tenants, market_data, strategies, agents_core,
│   │                         rag, events (audit + journals merged in Phase 4a),
│   │                         trading (portfolio + orders + trades merged in Phase 4c),
│   │                         notifications, billing, system, common
│   ├── plugins/              Strategy + broker plugins (entry-point registered)
│   │   ├── strategy_screener/        Live intraday screener
│   │   ├── strategy_directional/     Equity directional
│   │   ├── strategy_short_straddle/  Short straddle
│   │   ├── strategy_pyramid/         Pyramid options
│   │   ├── strategy_swing/           Oliver Kell cycle scanner
│   │   ├── strategy_basket/          Premarket basket builder
│   │   ├── strategy_backtest/        Generic backtester (compat shim)
│   │   ├── broker_angel/             Angel One SmartAPI adapter
│   │   └── broker_zerodha/           Zerodha Kite adapter (stub)
│   ├── config/               Django settings + URLs + ASGI/WSGI
│   └── tests/                pytest suite (99/99 passing)
│
├── trading/                  Legacy package (kept for CLI commands + screener internals
│                             like ticker_service.py, data_service.py, utils/, models.py
│                             which the plugins still import from). Multi-tenant Postgres
│                             access goes through backend/apps; this package is the older
│                             SQLite-era code that the plugins inherit from.
│
├── frontend/                 React + Vite SPA (port :5173)
│   └── src/
│       ├── features/         pulse, rotation, shortlist, setup, dashboard, positions,
│       │                     monthly, agents, strategies, backtester, pyramid,
│       │                     swing-scanner, broker, auth
│       └── lib/              api, ws, market-config, market-pulse, monthly, utils
│
├── docker-compose.dev.yml    Postgres :5436 + Redis :6379 (everything else is native)
├── scripts/
│   ├── dev_up.sh             Boots web (Daphne), celery, vite
│   └── dev_down.sh           Stops web/celery/vite by pidfile
└── docs/                     ADRs, architecture, MIND_PALACE, strategy specs
```

---

## AI Virtual Team

| Agent | Role | Files | LLM? |
|-------|------|-------|------|
| **@DataAnalyst** | Fetches and enriches market data from Angel One | `trading/services/data_service.py`, `trading/options/data_service.py` | No |
| **@DirectionalTrader** | Plans BUY/SELL equity trades from intraday structure | `trading/agents/planner.py`, `trading/graph/trading_graph.py` | Yes (Claude) |
| **@OptionsStrategist** | Manages straddle positions — HOLD/CLOSE/HEDGE/ROLL | `trading/options/straddle/graph.py`, `trading/options/straddle/prompts.py` | Yes (Claude) |
| **@RiskGuard** | Deterministic risk validation — no LLM, no exceptions | `trading/services/risk_engine.py` (and `backend/apps/trading/services/risk_engine.py` for the canonical v2 10-criterion engine) | Never |
| **@PortfolioTracker** | Tracks capital, P&L, daily loss, open positions | `trading/rag/retriever.py`; portfolio rows live in `backend/apps/trading/models.py` (`Portfolio`, `PortfolioSnapshot`) | No |
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
# Backend (one Django process on :8000)
cd backend
uv sync                          # installs deps + all strategy plugins
uv pip install -e .              # registers entry-points (alphadesk.strategies/.brokers)
.venv/bin/python manage.py migrate
cd ..

# Frontend
cd frontend && npm install && cd ..
```

Copy env templates (gitignored):
```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

### Start the stack
```bash
# 1. Bring up Postgres :5436 + Redis :6379 (only services in Docker)
docker compose -f docker-compose.dev.yml up -d

# 2. Boot the native processes (web :8000 via Daphne, celery worker, vite :5173)
bash scripts/dev_up.sh
```

Tail logs at `logs/{web,celery,vite}.log`. Stop with `bash scripts/dev_down.sh` (leaves Docker containers running).

Verify:
- API docs: <http://localhost:8000/api/docs/>
- UI: <http://localhost:5173> — default dev login `admin@local.dev` / `devadmin`

### Backend tests
```bash
cd backend && .venv/bin/python -m pytest -q
# expect: 99 passed
```

### Frontend tests
```bash
cd frontend && pnpm test
```

### Directional Trading (Equity)
```bash
cd backend
.venv/bin/python manage.py run_trading_agent "Plan a BUY trade for HDFCBANK"
.venv/bin/python manage.py run_trading_agent --show-journal
```

### Straddle Management (Options)
```bash
.venv/bin/python manage.py manage_straddle --register --underlying NIFTY \
  --strike 24200 --expiry 2026-05-13 \
  --ce-symbol NIFTY13MAY2624200CE --ce-token 41762 --ce-sell 394.85 \
  --pe-symbol NIFTY13MAY2624200PE --pe-token 41763 --pe-sell 138.35 --lots 1

.venv/bin/python manage.py manage_straddle --analyze --position 1
.venv/bin/python manage.py manage_straddle --status --position 1
.venv/bin/python manage.py manage_straddle --execute CLOSE_BOTH --position 1
.venv/bin/python manage.py manage_straddle --list
```

### Pyramid Options
```bash
.venv/bin/python manage.py run_pyramid --strike 24200 --type CE --underlying NIFTY
```
Or via the React UI at `/pyramid` — backtest with config inputs, then "Go Live (Paper)".

### Live Screener
```bash
.venv/bin/python manage.py run_screener                       # live
.venv/bin/python manage.py run_screener --telegram            # with alerts
.venv/bin/python manage.py run_screener --backtest --from 2026-04-01 --to 2026-04-30
```

### Oliver Kell Swing Scanner
```bash
.venv/bin/python manage.py run_ok_scanner --actionable-only --telegram
```

### EOD Signal Enrichment (feeds the monthly feedback report)
```bash
.venv/bin/python manage.py enrich_signals          # today
.venv/bin/python manage.py enrich_signals --all    # backfill all unenriched
```

### Full trading day (premarket → screener → straddle monitor → review)
```bash
.venv/bin/python manage.py run_trading_day
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
  tick → CandleStore → IndicatorEngine → 8 strategies → Signal → events.Event (persist) → Telegram

Feedback Loop:
  every signal fired (any source) → events.Event (type=signal.fired)
  EOD: enrich_signals fills max_favorable/adverse + outcome
  Monthly: /monthly aggregates capture rate, signal audit, rejections, analytics, benchmark, lessons
```

All workflows:
- Use Angel One SmartAPI (paper or live via `TRADING_MODE`)
- Log every decision to `events.Event` via `apps.events.services.event_writer.emit()` (non-blocking)
- Save outcomes to the appropriate model
- Are validated deterministically before execution

---

## Key Invariants (Never Break These)

1. **@RiskGuard is always the last gate** before execution. LLM cannot bypass it.
2. **`TRADING_MODE=paper` by default** — set to `live` explicitly and deliberately.
3. **Never trust LLM for position sizing** — always computed deterministically (% of capital).
4. **Journal every decision** — rejected trades are also recorded.
5. **Expiry-day positions** — close before 3:15 PM IST. Gamma risk explodes after.
6. **Event log is non-blocking** — emit failures never stop trading flow (use `event_writer.emit()`, never raw `Event.objects.create()`).
7. **Short straddle hard stop** — combined premium > 1.3× sold → close immediately.
8. **Index metadata has one source** — `frontend/src/lib/market-config.ts` for the UI; lot sizes/expiry days/tokens are NEVER hardcoded elsewhere.
9. **Signal persistence is non-blocking** — every signal is written in a try/except so logging failures don't stop the screener.
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
│   ├── risk_engine.py             @RiskGuard — deterministic validator (legacy variant)
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
├── swing/
│   ├── ok_cycles.py               Oliver Kell phase detector
│   ├── ok_scanner.py              NIFTY 100 daily/weekly scanner
│   └── ok_intraday.py             5m intraday variant
├── basket/                        Premarket basket builder (mood, manager, executor)
├── rag/retriever.py               @PortfolioTracker — RAG context for planner
├── utils/
│   ├── indicators.py              EMA/RSI/BB/MACD/ATR/VWAP/WMA
│   ├── time_utils.py              Market hours, session phase, last_trading_day
│   └── expiry_utils.py            Next expiry resolution (NSE Tue, BSE Thu)
├── models.py                      Legacy SQLite-era models (TradeJournal, StraddlePosition,
│                                  AuditLog, SignalLog, WatchlistEntry, SystemControl) —
│                                  superseded in v2 by apps.events.Event, apps.trading.Trade,
│                                  apps.system.SystemControl. Don't import from here in
│                                  v2 code paths.
└── management/commands/
    ├── run_trading_agent.py       CLI: equity directional
    ├── run_trading_day.py         CLI: full day orchestration
    ├── manage_straddle.py         CLI: straddle lifecycle
    ├── run_pyramid.py             CLI: pyramid backtest/execute
    ├── run_screener.py            CLI: live intraday screener
    ├── run_ok_scanner.py          CLI: Oliver Kell daily scan
    ├── enrich_signals.py          CLI: EOD signal outcome enrichment
    └── run_morning_basket.py      CLI: premarket basket
```

---

## File Map (v2 Backend)

```
backend/apps/
├── accounts/                       User + auth (JWT)
├── tenants/                        Tenant + Membership (multi-tenant)
├── trading/                        Portfolio + Position + Trade + Order (merged Phase 4c)
│   ├── models.py                   Portfolio, Position, Trade, Order, OptionsPosition, ...
│   ├── services/
│   │   ├── monthly_report.py       Aggregation: month groups, YTD, capture matrix,
│   │   │                           signal audit, rejections, equity curve, analytics,
│   │   │                           benchmark vs NIFTY50, AI lessons
│   │   ├── place_order.py          Outbox pattern: validate → enqueue
│   │   └── risk_engine.py          Canonical 10-criterion RiskEngine (paper + live)
│   └── api/                        PortfolioViewSet, OrderViewSet, MonthlyReportView,
│                                   GET /api/v1/portfolios/monthly/,
│                                   POST /api/v1/orders/ → 202 Accepted
├── market_data/                    Broker integration (absorbed Phase 4b)
│                                   — singleton client, candle cache, symbol master
├── events/                         Unified event log (audit + journals merged Phase 4a)
│   ├── models.py                   Event (TenantModel, append-only) — replaces journals,
│   │                               AuditEvent, AuditLog, straddle management_log
│   ├── services/event_writer.py    emit() — single non-blocking write path
│   └── api/                        Event firehose, journal-compatible endpoints
├── system/                         SystemControl (kill switch, AI pause) + TraderNote
├── strategies/                     Signal model + strategy registry
├── agents_core/                    AgentRun + workflow framework
├── rag/                            pgvector-backed RAG
├── notifications/                  Email + Telegram dispatch
├── billing/                        Subscriptions
└── common/                         TenantModel base, middleware, WS auth
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

> Note: `frontend/src/` should never contain `*.js` source files — only `.ts` / `.tsx`. Stray
> `.js` files are stale `tsc`-emitted output and shadow the real sources at test time
> (Vite resolves `.js` before `.tsx`). If you see them, delete them.

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
DATABASE_URL=postgres://alphadesk:alphadesk@localhost:5436/alphadesk
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=...

# Frontend (frontend/.env.local)
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000
```

All `.env*` are gitignored. Use `.env.example` files as templates. **Never commit secrets.**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Django 5.1 |
| Agent orchestration | LangGraph 0.6+ |
| LLM | Claude (Anthropic) — CLI or API |
| Broker | Angel One SmartAPI (`SmartApi` package) |
| Database | PostgreSQL (pgvector for RAG) |
| Cache / Queue | Redis |
| Async tasks | Celery |
| Real-time | Django Channels (Daphne) |
| API | DRF |
| Frontend | Vite · React 18 · TypeScript · TanStack Query · Zustand · Tailwind |
| Charting | Recharts · TradingView lightweight-charts |
| Logging | structlog · Sentry |
| Plugin loading | Python entry-points (`importlib.metadata`) |
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
