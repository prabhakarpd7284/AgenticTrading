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

**Stack:** Django 5 · LangGraph · Angel One SmartAPI · Celery · Postgres (`:5444`) · Redis · React/Vite
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
├── docker-compose.dev.yml    Postgres :5444 + Redis :6380 (everything else is native)
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
# 1. Bring up Postgres :5444 + Redis :6380 (only services in Docker)
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
DATABASE_URL=postgres://alphadesk:alphadesk@localhost:5444/alphadesk
REDIS_URL=redis://localhost:6380/0
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

---

## Roadmap: Mutual Fund Module (added 2026-10-15)

**Scope lock, read this before touching anything below.** In scope right
now: the existing AlphaDesk stock-trading engine (personal use, prove
profit before selling — unchanged), plus a **mutual fund data collector and
advisor**, for personal use on the working tenant
(`prabhakarpd7284@gmail.com` — see `.claude/skills/alphadesk-dev-stack`, NOT
`admin@local.dev`). **Explicitly out of scope right now:** FDs, bonds,
commodities, a cross-asset financial planner, tax filing, a conversational
chat persona, anything aimed at a second/paying user, and any SEBI RA/IA or
AMFI ARN registration work. Don't let scope creep back into any of those
while working the phases below — they're sequenced on purpose.

**Status as of 2026-10-17 (read this first):** MF-0 through MF-3 are
*built and committed* on branch `abhinav/redesign-v2` (`35985cc`), with 89
passing tests in `backend/tests/mutual_funds/`. What is **not** done: nothing
has run against real holdings (no real CAS PDF yet — CAS import and the
advisor are verified on synthetic fixtures only), the broker/tenant
setup (MF-0b and MF-0's last item), and the MF-4 owner decisions. Don't
rebuild anything marked `[x]` below — start from the unticked items. Check
`HANDOFF_MUTUAL_FUNDS.md` at the repo root for the detail behind each item.

New machine? `backend/.env` is gitignored — recreate it from
`backend/.env.example` with `DATABASE_URL` on port **5444**, run `uv sync`
(`casparser` was added, `uv.lock` not yet regenerated), start Docker, then
`migrate` and `pull_amfi_nav`. The NAV data lives in the local Postgres
volume, not in git.

### Phase MF-0 — Get the environment actually running (blocking everything else)

**Status:** DONE on the original dev machine (2026-10-15) except the broker
item. Run the first four steps again on any fresh machine — a native Claude
Code session has real shell access, so run `docker compose ... up -d`
yourself rather than suggesting it.

- [x] Run `docker compose -f docker-compose.dev.yml up -d` directly (don't
      just suggest the command — actually run it)
- [x] Run `docker compose -f docker-compose.dev.yml ps`; confirm Postgres
      `:5444` and Redis `:6380` both show `running`/`healthy`. If a
      container is unhealthy or restarting, check its logs
      (`docker compose -f docker-compose.dev.yml logs <service>`) before
      moving on — don't proceed to migrate against a half-up DB
- [x] `cd backend && .venv/bin/python manage.py migrate` — confirm
      `mutual_funds_scheme` / `mutual_funds_nav` / `mutual_funds_holding` /
      `mutual_funds_transaction` tables exist
- [x] `.venv/bin/python manage.py pull_amfi_nav` — confirm non-zero scheme
      + NAV row counts. Actual: **14,376 schemes / 14,376 NAV rows** (the
      "~16k+" estimate was high; this is what AMFI serves today). First run
      exposed two bugs, both fixed: `MFScheme.option` was `varchar(30)` but
      AMFI values reach 84 chars (now 100, migration `0002`), and the NAV
      file was decoded as latin-1 instead of UTF-8.
- [ ] Verify a broker link is `status: active` on the *right* tenant —
      wrong tenant or no active link means the rest of AlphaDesk is
      silently trading on a synthesized price feed (see the dev-stack
      skill's "Check 2"). **Open:** on the original machine's DB the only
      users were `istpduckymomo@gmail.com` and `smoke@alphadesk.local`
      (no `prabhakarpd7284@gmail.com`, no tenant `29e91b00…`), and the only
      broker link was a `paper` link — so the system was on synthesized
      prices. Blocked on MF-0b / tenant setup.

**Acceptance:** `pull_amfi_nav` succeeds with real counts; a Django shell
query (`MFScheme.objects.count()`, `MFNav.objects.count()`) confirms it.
*Met.* (Field is `MFNav.nav_date`, not `date`.)

### Phase MF-0b — Connect the real Zerodha account (do this now — account already exists)

The owner already has a funded Zerodha trading account, so this replaces
the paper-broker POC path for the stock-trading side. Confirmed by reading
the actual code before writing this:

- `backend/plugins/broker_zerodha/adapter.py` already has a real, complete
  `ZerodhaAdapter` — not a stub. It implements all 5 required methods
  (`authenticate`, `health_check`, `fetch_positions`, `fetch_holdings`,
  `fetch_margin`) via the `kiteconnect` SDK, which is already declared in
  `pyproject.toml` (`kiteconnect>=3.9`) — nothing new to install beyond
  `uv sync`.
- It does **not** implement order placement or quotes/options-chain — only
  position/holdings/margin read access right now. That matters for cost:
  Zerodha's **free "Personal" API plan** covers exactly this (order, GTT,
  alert, margin, portfolio read access, no subscription fee) — the paid
  Connect plan (₹500/app/month) is only needed once real-time/historical
  **market data** or order placement is added. So this can be wired up at
  zero cost today.
- The adapter's docstring is explicit: Kite's `access_token` expires daily
  (~6 AM IST) — there is **no existing daily-login helper** for any broker
  in this repo yet (checked: no `*_login.py` management command exists
  for angel_one or zerodha). Angel One avoids this via TOTP; Zerodha does
  not. Building this helper is new work, not a gap to just "turn on."

Steps:
1. [ ] Register a Kite Connect app at developers.kite.trade using the
       owner's existing Zerodha login (Personal plan, free) — gives an
       `api_key` + `api_secret`.
2. [ ] Add `KITE_API_KEY` / `KITE_API_SECRET` to `backend/.env` (new vars —
       not present yet; follow the existing `ANGEL_*` naming convention in
       that file if one exists, otherwise just `KITE_API_KEY=` /
       `KITE_API_SECRET=`).
3. [ ] Write a small daily-login helper (new — doesn't exist yet), e.g.
       `apps.market_data.management.commands.zerodha_login`: print
       `KiteConnect(api_key=...).login_url()`, prompt for the
       `request_token` from the redirect URL after the owner logs in in a
       browser, call `kite.generate_session(request_token, api_secret=...)`
       to get `access_token`.
4. [ ] POST that `{api_key, api_secret, access_token}` to
       `/api/v1/brokers/zerodha/connect/` (existing endpoint,
       `BrokerLinkViewSet.connect` in `apps/market_data/api/broker_views.py`
       — tests auth via `authenticate()` before saving, so bad creds never
       hit the DB).
5. [ ] Confirm the resulting `BrokerLink` shows `status: ACTIVE` on the
       correct tenant (`prabhakarpd7284@gmail.com`, per the dev-stack
       skill) — this is what satisfies Phase MF-0's last checklist item
       and stops the synthesized-price fallback for the equity side.
6. [ ] Remember this is a **daily** step, not one-time setup — the
       `access_token` goes stale every morning and step 3–4 must be
       repeated each trading day until this is automated.

**Acceptance:** `GET /api/v1/brokers/{id}/diagnose/` (or the positions
action) returns real position/holdings/margin data from the owner's actual
Zerodha account, not synthesized data.

### Phase MF-1 — Daily collector hardening
**Status:** built and tested (2026-10-17). Beat has not been observed firing.
- [x] Schedule `pull_amfi_nav` in Celery beat — `mf-nav-pull-night` (23:45
      IST) and `mf-nav-pull-morning` (07:30 IST), daily, idempotent; task is
      `apps.mutual_funds.tasks.pull_amfi_nav` (also runs the stale sweep)
- [x] Historical backfill: `manage.py backfill_amfi_history --scheme <code>
      [--years N]`. AMFI can only filter by **AMC**, not scheme, so each
      AMC-year is a ~15 MB download — pass the schemes you hold, never the
      whole catalog. Verified on scheme `148921`: 494 rows, 2 years, 13 s.
- [x] Stale flag: `manage.py check_stale_navs` (>3 trading days). Flags
      5,628 of 14,376 schemes today, mostly matured/closed funds still in
      AMFI's file (some last priced 2008) — **noisy**; MF-3 should only
      surface stale schemes the owner actually holds.

**Acceptance:** a beat entry exists and fires; manually triggering backfill
on one scheme produces >1 year of NAV history. *Backfill: met. Beat entry
exists and loads; "fires" is unverified until a worker + beat run.*

### Phase MF-2 — Real holdings via CAS import
**Status:** built; **UNVERIFIED against a real statement** — tested only on
synthetic fixtures (34 tests). Run the first real CAS with `--dry-run`.
- [x] Parser for a CAMS/KFin Consolidated Account Statement PDF
      (password = PAN+DOB), via the `casparser` package (added to
      `pyproject.toml`; `uv.lock` not regenerated — run `uv sync`)
- [x] Map each row to `MFScheme` by ISIN first, then the AMFI scheme code on
      the statement; exact AMC+normalized-name match only when the
      statement has no ISIN. Unmatched rows are skipped and reported.
- [x] Upsert into `MFHolding` + `MFTransaction`, tagged `source="cas"`;
      idempotent on re-import
- [x] Command: `import_cas_statement --file <path> --tenant <slug|uuid>
      --portfolio <name|uuid> [--dry-run]`; password from `$CAS_PASSWORD`,
      `--password`, or a prompt (never logged). Needs a `trading.Portfolio`
      to attach holdings to — none exists for MF yet, create/pick one.

Known gaps: units/amounts stored as absolute values (direction is
`txn_type`); closed-out schemes stored with 0 units (filter `units > 0`);
tax/segregation/gift lines skipped (count printed); NSDL/CDSL demat
statements rejected; no events emitted; `MFTransaction` has no DB unique
key (dedupe is in code).

**Acceptance:** running the command against a real CAS PDF produces
`MFHolding` rows that match the actual folio list on eyeball check against
the statement. *NOT met — needs a real CAS PDF (request from CAMS/KFintech).*

### Phase MF-3 — MF Advisor (personal use only — not sellable as-is)
**Status:** built; **never run on real holdings** (26 tests on fixtures,
SQLite). Run: `manage.py run_mf_advisor --tenant <..> [--portfolio <..>]
[--target-file target.json] [--assumed-drag-pct 0.75]`.
- [x] `apps/mutual_funds/services/advisor.py`, separate from collection code
- [x] **Direct vs Regular detection** — finds the direct twin (AMC +
      normalized name + option). The drag is an **ESTIMATE** from a
      configurable assumed spread (default 0.75%) because AMFI has no
      expense ratios; growth options also get an observed return gap when
      both plans share ≥180 days of NAV.
- [x] **Category allocation view** vs a target JSON the owner provides
      (keys must equal the AMFI category text inside the parentheses,
      case-insensitive — check against `MFScheme.category`)
- [x] **Overlap detection** at category+AMC level; always returns an
      explicit "stock-level overlap not computed" open question
- [x] **SIP continuity check** — infers cadence from ≥3 SIPs; flags "possibly
      lapsed" when >5 days overdue (a deliberately stopped SIP still flags)
- [x] Findings go through `event_writer.emit()` (wrapped, non-blocking).
      **Caveat:** `Event.Type` has no advisor member, so they use
      `workflow.step.completed` with `step_name="mf_advisor.<kind>"`. A
      dedicated `mf.advisor.finding` type needs an enum change + migration
      in `events` — undecided.

**Acceptance:** run against the owner's real CAS-imported holdings; produces
at least one direct-vs-regular flag and one allocation summary the owner
hand-verifies against one fund. *NOT met — blocked on MF-2 real CAS data.*

### Phase MF-4 — Housekeeping decisions (parallel track, doesn't block MF-0..3)
**Status:** investigated read-only (2026-10-17); **no decisions made, nothing
deleted.** Evidence and recommendations are in
`docs/MF4_HOUSEKEEPING_FINDINGS.md`. Items below stay unticked until the
owner confirms. Headlines: the six old app dirs hold only `__pycache__`
(recommend remove); both swing scanners are live (keep both); `db.sqlite3`
still has 2 straddle / 8 journal / 58 watchlist rows (archive after
confirming); fyers/vertical_spread/scalp are live (keep). Two things worth
acting on: v1 `enrich_signals` (16:00) may mark same-day `swing_v2_*`
signals `EXPIRED` (from reading code, not reproduced), and the Docker
images copy only `backend/` while `prod.py` never installs the legacy
`trading/` package, so daily scan/enrichment tasks would likely fail in
prod (not tested in a container).
- [ ] Confirm `audit`, `journals`, `broker`, `orders`, `portfolio`, `legacy`
      (unregistered in `LOCAL_APPS`, per `grep` confirmation on 2026-10-13)
      have no remaining plain-Python imports, then remove them
- [ ] Resolve `swing_scanner` vs `swing_scanner_v2` — both are currently
      registered live; ask Prabhakar which one should stay before deleting
      either
- [ ] Decide fate of `db.sqlite3` (README says Phase 6 dropped it; it's
      still on disk), the `fyers` broker plugin (registered but
      undocumented — active or dead?), and the StockEdge pipeline's terms
      of service before relying on it in anything that gets sold
- [ ] Fix the doc drift: `docs/MIND_PALACE.md` / `docs/architecture/
      ARCHITECTURE.md` still describe the old dual-Django/Streamlit design;
      CLAUDE.md/README themselves are also wrong about ports/dev-login per
      the dev-stack skill — reconcile all three against what's actually
      running

**Acceptance:** each item has an explicit keep/remove/defer decision
recorded, not just noticed again.

### Phase MF-5 (future — do not start before MF-0 through MF-3 are proven)
A light conversational persona over AlphaDesk's existing trading data
(positions, signals, risk status) plus the MF advisor's output. Motivated
by the owner missing trading opportunities from being busy — not a general
chatbot, a thin front-end on data that already exists. Design work not
started; revisit once MF-3 has run against real data for a while.
