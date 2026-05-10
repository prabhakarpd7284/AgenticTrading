# AlphaDesk — System Architecture

Status: **Draft v1** (2026-04)
Owner: Platform Eng
Audience: engineering, infra, security, investor due-diligence

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
└──────────────┬─────────────────────────────┬──────────────────┘
               │                             │
               ▼                             ▼
   ┌─────────────────────┐        ┌────────────────────────┐
   │ Django REST (Uvicorn)│◀──────▶│ Django Channels (WS)   │
   │  ECS Fargate task    │        │ ECS Fargate task       │
   └─────────┬────────────┘        └──────────┬─────────────┘
             │                                │
             ▼                                │
   ┌────────────────────┐                     │
   │ Celery workers     │◀────── Redis ──────▶│
   │ + beat (ECS)       │                     │
   └─┬──────────────────┘                     │
     │                                        │
     ▼                                        │
 ┌───────────────┐   ┌────────────┐   ┌──────────────────┐
 │ RDS Postgres  │   │ S3 (audit, │   │ Angel One / Zerodha │
 │ (pgvector)    │   │  exports)  │   │ Kite / Fyers APIs   │
 └───────────────┘   └────────────┘   └──────────────────┘
```

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
| `accounts` | Users, passwords, MFA, sessions |
| `tenants` | Organization / workspace isolation (row-level) |
| `broker` | Broker account links, credential vault, token refresh |
| `market_data` | Symbol master, LTP, candles, VIX, options chain cache |
| `portfolio` | Holdings, capital, snapshots, P&L |
| `orders` | Order DSL, validation, execution, outbox pattern |
| `strategies` | Strategy definitions (DSL + Python plugins), backtests |
| `agents_core` | Agent base classes, registry, LangGraph builder |
| `rag` | Retriever/Embedder/VectorStore interfaces + registry |
| `journals` | Immutable decision journal |
| `audit` | Security audit log (non-blocking) |
| `notifications` | Email, Telegram, in-app, webhooks |
| `billing` | Stripe / Razorpay, plan enforcement, entitlements |

### 2.3 Agentic RAG plugin system
See `ADR-0002`. Strategies and retrievers are loaded from entry-points. A strategy
ships as a package implementing `AlphaStrategy`, a retriever ships as `Retriever`.
Core never imports a strategy directly.

### 2.4 Real-time layer
Channels + Redis pub/sub + ASGI. See `ADR-0003`. Three channel groups:
- `tenant.{id}.ticks` — market data fan-out
- `tenant.{id}.pnl`   — P&L updates (1s throttle)
- `tenant.{id}.agent.{run_id}` — streamed tokens from a running agent

### 2.5 Execution layer
Saga + Outbox. See `ADR-0004`. Orders never go directly to the broker from a web request —
they go through the Outbox → Celery worker → Broker adapter, so failures are retryable
and auditable.

## 3. Data model highlights

```
User ──owns──▶ Membership ──of──▶ Tenant ──has──▶ Portfolio
                                      │             │
                                      ├─▶ BrokerLink
                                      ├─▶ Strategy ──▶ Backtest
                                      ├─▶ Position ──▶ Trade ──▶ Journal
                                      ├─▶ AgentRun ──▶ AgentStep
                                      └─▶ AuditEvent
```

All tenant-owned tables have `tenant_id` indexed; DRF permission class enforces
`queryset.filter(tenant_id=request.tenant_id)`.

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

| Phase | Change | Breakage |
|-------|--------|----------|
| 1 | New `backend/` tree stood up in parallel. Old `trading/`, `config/` untouched. | None |
| 2 | Shared domain code (agents, risk_engine, analyzer) moved under `backend/apps/…` and imported from old code via shim. | None |
| 3 | New DRF endpoints + WS live; old CLI commands kept functional. | None |
| 4 | Frontend consumes new API; old dashboard.py deprecated. | Dashboard UI |
| 5 | Delete old tree after 30-day freeze. | Removed scripts |

## 10. Open questions / risks

- **Broker WS fan-out cost at 10k tenants** — may need Kinesis Data Streams in front.
- **pgvector vs. dedicated store (Qdrant/Weaviate)** — starting with pgvector to reduce ops; reassess at 5M embeddings.
- **LLM cost control** — per-tenant token budget + aggressive caching of prompts.
- **Regulatory posture** — decide whether we register as SEBI Investment Advisor (IA) or stay execution-only platform. Affects copy in UI (no "recommendations", use "analyses").
