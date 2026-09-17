# AlphaDesk — System Architecture

Status: **v2** (2026-05) — implemented (in dev), prod target
Owner: Platform Eng
Audience: engineering, infra, security, investor due-diligence

> **Reading order:** This is the formal spec. For the spatial mental model see `docs/MIND_PALACE.md`. For onboarding setup see `README.md`. For per-app details see `docs/architecture/BACKEND_STRUCTURE.md`.

---

## 1. Context diagram

```
                   ┌──────────────────────────────────────────────┐
                   │                 End Users                     │
                   │ Retail trader · Advisor RM · Prop-desk trader │
                   └─────────────────┬────────────────────────────┘
                                     │  HTTPS / WSS
                                     ▼
            ┌───────────────────────────────────────────────┐
            │     AWS Amplify (Vite/React) — static + CDN    │
            └─────────────────┬─────────────────────────────┘
                              │ REST + WebSocket
                              ▼
┌──────────────────────────────────────────────────────────────┐
│           API Edge — ALB + WAF + Cognito authorizer           │
└──┬──────────────────────────────────────┬────────────────────┘
   │                                      │
   ▼                                      ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  v2 Django REST + WS    │     │  Legacy bridge (Django)  │
│  (ECS Fargate, port 8k) │     │  (ECS Fargate, port 8001)│
│  Postgres + Redis       │     │  SQLite (read-mostly)    │
│  Multi-tenant, JWT      │     │  TradeJournal, AuditLog, │
│  Outbox + Celery worker │     │  SignalLog, Straddle,    │
└─┬───────────────────────┘     │  Pyramid view, screener  │
  │                              └────────────┬────────────┘
  │ DB router (apps.common.db_router)         │
  └─────────────────┬───────────────────────────┘
                    ▼
         ┌────────────────────────┐
         │ Angel One / BSE feeds  │
         │ Telegram alerts        │
         └────────────────────────┘
```

The system runs as **two cooperating Django processes** in production. The v2 stack owns auth, tenants, orders, positions, and the monthly report. The legacy bridge owns 700+ rows of TradeJournal data, the screener engine, the straddle/pyramid management commands, and direct broker connectivity. The frontend talks to both — v2 by default, legacy via `/api/v1/legacy/*` (proxied in dev, separate ALB target group in prod).

## 2. Logical components

### 2.1 Frontend (`frontend/`)
- Vite + React 18 + TypeScript
- Tailwind + shadcn/ui for design system
- TanStack Query (server state), Zustand (UI state), React Router
- Recharts + TradingView `lightweight-charts` for market visuals
- WebSocket client with auto-reconnect, subscribe/unsubscribe channels

### 2.2 Backend (`backend/`) — Django 5 apps

| App | Responsibility |
|-----|----------------|
| `accounts` | Users, passwords, MFA, sessions, JWT obtain/refresh with tenant_id claim |
| `tenants` | Organization / workspace isolation (row-level via TenantModel mixin) |
| `broker` | Broker account links, credential vault, token refresh |
| `market_data` | Symbol master, candles, **MarketPulseView**, **SectorRotationView**, **ShortlistView** |
| `portfolio` | Portfolios, Positions, Snapshots, **MonthlyReportView** (capture matrix, signal audit, rejections, equity curve, analytics, benchmark, lessons) |
| `orders` | Order model, outbox + idempotency, async fills via Celery |
| `strategies` | Strategy definitions, plugin discovery via entry-points |
| `agents_core` | AgentRun, AgentStep, WebSocket consumer for live token streams |
| `rag` | Retriever/Embedder/VectorStore interfaces + registry, pgvector backend |
| `journals` | JournalEntry linked to portfolio + agent_run + order |
| `notifications` | Email, Telegram, in-app, webhooks |
| `billing` | Plan / Subscription / entitlements, Razorpay |
| `legacy` | **Bridge** to legacy `trading/` app — pyramid view, screener view, straddle endpoints |
| `common` | Middleware (tenant resolution), TenantModel base, db_router (legacy alias) |

### 2.2b Legacy app (`trading/`) — read-mostly + strategy engine

The legacy Django app is preserved verbatim. It owns:
- `models.py` — TradeJournal, StraddlePosition, AuditLog, **SignalLog** (NEW), WatchlistEntry, PortfolioSnapshot, SystemControl
- `services/` — **broker_service** (paper + live, NSE/NFO/BFO), **risk_engine** (10-criteria), **ticker_service** (symbol master with NFO + BFO indexing), **data_service** (candles + intraday cache)
- `agents/planner.py` — @DirectionalTrader (LLM)
- `graph/trading_graph.py` — Equity LangGraph workflow
- `options/straddle/` — Short straddle lifecycle (LangGraph)
- **`pyramid/`** (NEW) — Pure-Python pyramid options strategy
- **`screener/`** (NEW) — Live intraday opportunity detector
- **`swing/`** (NEW) — Oliver Kell daily/weekly cycle scanner
- **`basket/`** (NEW) — Premarket basket builder
- **`backtester/`** (NEW) — Generic v2 backtest engine

The DB router (`apps.common.db_router.LegacyRouter`) pins `trading.*` ORM ops to the `legacy` SQLite alias when configured. In dev it's a single SQLite file; in prod each Django process runs against its own DB.

### 2.3 Agentic RAG plugin system
See `ADR-0002`. Strategies and retrievers are loaded from entry-points. A strategy
ships as a package implementing `AlphaStrategy`, a retriever ships as `Retriever`.
Core never imports a strategy directly.

### 2.4 Real-time layer
Channels + Redis pub/sub + ASGI. See `ADR-0003`. Channel groups:
- `tenant.{id}.ticks` — market data fan-out (per-tenant subscription set)
- `tenant.{id}.pnl`   — P&L updates (1s throttle)
- `tenant.{id}.agent.{run_id}` — streamed tokens from a running agent
- `pyramid.{position_id}` — pyramid execution status (planned)

WebSocket auth: JWT in `Sec-WebSocket-Protocol` subprotocol header (the only browser-supported way to pass auth on WS).

### 2.5 Execution layer
Saga + Outbox. See `ADR-0004`. Orders never go directly to the broker from a web request —
they go through the Outbox → Celery worker → Broker adapter, so failures are retryable
and auditable.

## 3. Data model highlights

### v2 (Postgres, multi-tenant)

```
User ──owns──▶ Membership ──of──▶ Tenant ──has──▶ Portfolio
                                      │             │
                                      ├─▶ BrokerLink
                                      ├─▶ Strategy ──▶ Backtest
                                      ├─▶ Position (with exit_price, realized_pnl, exchange)
                                      ├─▶ Order ──▶ OutboxEvent
                                      ├─▶ AgentRun ──▶ AgentStep
                                      └─▶ JournalEntry, AuditEvent
```

All tenant-owned tables have `tenant_id` indexed via `TenantModel`; DRF permission class enforces `queryset.filter(tenant=request.tenant)`.

### Legacy (SQLite, single-user)

```
TradeJournal ──┐
              ├──▶ AuditLog (LLM prompts, risk decisions, executions)
SignalLog ────┤    [NEW] Every screener/scanner signal — outcome tracked
              │
StraddlePosition (CE + PE legs, management_log JSONField)
WatchlistEntry (premarket scans + outcome: WATCHING/TRIGGERED/TRADED/SKIPPED)
PortfolioSnapshot (point-in-time capital + day_pnl)
SystemControl (PAUSE_AI flag, FORCE_CLOSE_ALL, etc.)
```

### The Feedback Loop (cross-cutting)

```
Screener/Scanner fires → SignalLog row (PENDING)
                            ↓
                 enrich_signals (EOD job)
                  → fetches candles
                  → max_favorable_move + max_adverse_move + eod_price
                  → outcome: TRADED / REJECTED / SKIPPED / EXPIRED
                  → trade_journal FK linked
                            ↓
              MonthlyReportView aggregates:
                  • Capture rate per stock
                  • Signal audit (by source, by strategy)
                  • Rejection review (with hindsight)
                  • Equity curve + drawdown
                  • Time-of-day / day-of-week / sector
                  • Benchmark vs NIFTY50
                  • AI lessons
```

## 4. Request lifecycles

### 4.1 Agent run (plan a trade)
1. User clicks "Plan NIFTY50 breakout".
2. FE POSTs `/api/agent-runs/` → returns `run_id`; FE opens WS `tenant.{t}.agent.{run_id}`.
3. Backend enqueues Celery task `agents.execute_run(run_id)`.
4. Worker loads the agent graph (Directional or Straddle) from the registry.
5. Each node publishes its output to the WS channel as it completes.
6. Final node writes `AgentRun` + `TradePlan` rows; RiskGuard validates; result sent.
7. If user accepts, FE POSTs `/api/orders/` → Outbox → worker → broker.

### 4.2 Live tick stream
1. Market data worker holds the broker WS connection, normalizes ticks.
2. Publishes to Redis channel `ticks.{exchange}.{token}`.
3. Fanout worker matches subscribers (per-tenant) and publishes to
   `tenant.{t}.ticks` — FE sockets receive only their tenant's subset.

### 4.3 Monthly feedback report
1. User opens `/monthly` in the React UI.
2. FE GETs `/api/v1/portfolios/monthly/` (optionally `?month=2026-04`).
3. `MonthlyReportView` calls `build_monthly_report(tenant, portfolio, month)`.
4. Service queries Position (v2 Postgres) + TradeJournal/SignalLog/AuditLog (legacy SQLite) and aggregates into a single payload: month groups, YTD, capture matrix, signal audit, rejections, equity curve, analytics, benchmark vs NIFTY50, and rule-based lessons.
5. Cached 120s in Django cache (memory in dev, Redis in prod).
6. Mock-mode toggle on the page lets the trader compare against synthetic baseline data.

### 4.4 Pyramid backtest (current) / live (planned)
1. User configures strike, type, underlying, capital, risk %, etc. on `/pyramid`.
2. Backtest path: FE GETs `/api/v1/legacy/pyramid/` — backend resolves the option symbol via `ticker_service.get_nfo_options()`, fetches 5-min candles from Angel One (NFO or BFO), runs `run_pyramid_with_chart_data()`, returns chart-ready candles + entries + trail SL + KPIs.
3. Live path (planned): FE POSTs `/api/v1/legacy/pyramid/live/` → spawns a `PyramidExecutor` in a daemon thread → executor monitors live candles, calls `BrokerService.place_order()` for entries/pyramids/exits, persists to `PyramidPosition`, broadcasts status via `pyramid.{position_id}` channel group.

## 5. Security

- **AuthN**: email+password (Argon2) + optional TOTP MFA; SSO (SAML/OIDC) for B2B.
- **AuthZ**: RBAC with roles (`owner`, `admin`, `trader`, `viewer`, `rm`, `client`).
- **Tenant isolation**: middleware resolves tenant from JWT; RLS-style filter everywhere.
- **Broker creds**: stored in AWS Secrets Manager, never in DB; referenced by ARN.
- **PII**: minimized; encrypted at rest (RDS KMS); exported PAN/Aadhaar never stored.
- **Transport**: TLS 1.3, HSTS, WAF on ALB.
- **Audit**: every privileged action → `AuditEvent` row + structured JSON log → CloudWatch.
- **Compliance**: SEBI compliance flags for advisory vs. execution; advisory tier
  never places orders. IP-based and geo-based rate limits.

## 6. Reliability & SLOs

| SLO | Target |
|-----|--------|
| API availability | 99.9% monthly |
| WS tick latency (p95, broker→browser) | < 600ms |
| Agent-run start latency (enqueue→first token) | < 3s |
| Order place roundtrip (paper) | < 250ms |
| Order place roundtrip (live Angel One) | broker-bounded, surfaced as metric |

Patterns:
- Circuit breaker on each broker adapter.
- Idempotency keys on `POST /orders`.
- Graceful degradation: if RAG vector store is down, agent falls back to no-context mode.
- Kill-switch: ops can flip `TRADING_MODE=halt` via SSM param; worker drains and refuses new executions.

## 7. Observability

- **Metrics**: CloudWatch + Prometheus (via ADOT) — request rate, error rate, saturation per service; custom metrics for agent-run count, order reject rate, RiskGuard block rate.
- **Logs**: structured JSON via `structlog`, shipped to CloudWatch Logs, mirrored to S3.
- **Traces**: OpenTelemetry; trace covers REST → worker → broker adapter.
- **Error tracking**: Sentry (frontend + backend).
- **Dashboards**: Grafana (on ECS or Grafana Cloud) with golden-signals per service.

## 8. Capacity targets (v1)

| Dimension | Target |
|-----------|--------|
| Concurrent tenants | 10,000 |
| Concurrent WS connections | 50,000 |
| Ticks ingested / sec | 20,000 |
| Agent runs / day | 250,000 |
| Orders / day (paper+live) | 100,000 |

Sizing assumption: 2× `c6i.large` ECS tasks for REST, 2× for WS, 4× workers, 1× beat,
`db.r6g.xlarge` Postgres, `cache.t4g.medium` Redis. Tripled in prod (multi-AZ).

## 9. Migration plan from current repo

| Phase | Change | Status (2026-05) |
|-------|--------|------------------|
| 1 | New `backend/` tree stood up in parallel. Old `trading/`, `dashboard.py` untouched. | ✅ Done |
| 2 | Legacy bridge (`apps.legacy`) exposes existing data + screener/pyramid views via DRF. | ✅ Done |
| 3 | New DRF endpoints (auth, portfolio, monthly, market_data) + WS live; old CLI commands kept functional. | ✅ Done |
| 4 | Frontend consumes new API. Both v2 (port 8000) and legacy bridge (port 8001) served. | ✅ Done — React UI parity with Streamlit for trader-facing views |
| 5 | Streamlit deprecated for trader use; kept for ops/debugging. | 🟡 In progress |
| 6 | Migrate strategy engines (`pyramid/`, `screener/`, `swing/`) into `backend/apps/strategies/` plugins. | ⏳ Planned |
| 7 | Delete legacy bridge after 30-day no-regression freeze. | ⏳ Planned |

The current shape is **dual-Django** (v2 + legacy). This is intentional and stable — it lets us ship the React UI immediately on top of 700+ existing TradeJournal rows without forcing a schema rewrite.

## 10. Index metadata + exchange routing (added 2026-05)

A subtle but important architectural choice: index options trade on two different exchanges with different symbol formats. The system handles this in three places:

| Concern | Source of truth | Notes |
|---------|----------------|-------|
| Lot sizes, expiry weekdays, tokens | `frontend/src/lib/market-config.ts` | Frontend imports `INDICES`, `getLotSize`, `getExpiryWeekday`. Mirror constants exist in `trading/utils/expiry_utils.py` for backend fallback. |
| NFO vs BFO indexing | `trading/services/ticker_service.py` | `_nfo_by_key` indexes both NFO + BFO instruments. `get_nfo_options()` matches by **strike in paisa** (78000 → 7800000.0) and **expiry date metadata**, not by substring search on the symbol (which would false-positive across BSE date encoding). |
| Order placement exchange | `trading/services/broker_service.py` + caller | Caller passes `exchange="NFO"` for NIFTY/BANKNIFTY, `exchange="BFO"` for SENSEX, `exchange="NSE"` for equity. `product_type="CARRYFORWARD"` for options. |

**Symbol formats:**
- NFO: `{NAME}{DDMMMYY}{STRIKE}{TYPE}` — e.g. `NIFTY13MAY2624200CE`
- BFO: `{NAME}{YMMDD}{STRIKE}{TYPE}` — e.g. `SENSEX2650778000CE` (Y=26, M=5, DD=07)

**Lot sizes (Jan 2026+, NSE circular FAOP70616):** NIFTY=65, BANKNIFTY=30, SENSEX=20.
**Expiry days (post Sep-2025 SEBI standardisation):** NSE=Tuesday, BSE=Thursday. BANKNIFTY no longer has weekly contracts (last Tuesday of month only).

## 11. Open questions / risks

- **Broker WS fan-out cost at 10k tenants** — may need Kinesis Data Streams in front.
- **pgvector vs. dedicated store (Qdrant/Weaviate)** — starting with pgvector to reduce ops; reassess at 5M embeddings.
- **LLM cost control** — per-tenant token budget + aggressive caching of prompts.
- **Regulatory posture** — decide whether we register as SEBI Investment Advisor (IA) or stay execution-only platform. Affects copy in UI (no "recommendations", use "analyses").
- **Dual-Django prod deployment** — running v2 (Postgres) alongside legacy bridge (SQLite) on separate ECS tasks works, but doubles the infra footprint. Migration phase 6 collapses this once strategies move into v2 plugins.
- **SignalLog volume** — ~50 signals/day × 250 trading days = ~12k rows/year. Fine for SQLite. If we move to live multi-tenant, partition by tenant + month.
- **Pyramid live execution** — currently CLI-only via `manage.py run_pyramid`. The UI "Go Live" wiring (background thread executor + WebSocket status broadcast) is planned but not yet implemented.
