# AlphaDesk — delivery manifest

Everything produced in this build, grouped by area. Each file lives under the repo root shown.

## 1. Product + design docs

- `docs/product/PRODUCT_VISION.md` — problem, personas (retail SaaS, wealth advisor, prop desk), JTBD, pricing, success metrics
- `docs/architecture/ARCHITECTURE.md` — context diagram, logical components, data model, request lifecycles, security, SLOs
- `docs/architecture/BACKEND_STRUCTURE.md` — layered per-app structure, tenancy enforcement, API style, migration plan
- `docs/infra/DEPLOYMENT_COMPARISON.md` — 4-option pros/cons with cost table, stage-gated recommendation
- `docs/api/openapi.yaml` — 24 REST paths + 7 schemas
- `docs/api/WEBSOCKETS.md` — channel reference
- `docs/adr/ADR-0001` through `ADR-0005` — tenancy, plugin framework, real-time layer, order saga, deployment target

## 2. Backend (Django 5 + DRF + Channels + Celery)

Root: `backend/`

- `config/settings/{base,dev,prod,test}.py` — environment-split settings
- `config/{urls,asgi,wsgi,celery,channels_router}.py` — all four entry points wired
- `apps/common/` — `TenantModel`, `TenantMiddleware`, `TenantScoped` permission, `problem_json_handler` (RFC 7807), DI container, structured logging, event bus
- `apps/agents_core/` — **plugin framework**: `PluginRegistry`, domain Protocols (`Strategy`, `Retriever`, `BrokerAdapter`), orchestrator `run_agent`, `AgentRunConsumer` (WS), REST views
- `apps/rag/` — **pluggable RAG**: `DefaultRAGRouter` (fan-out, dedup, rerank), retrievers for portfolio / journal / news, pgvector store, Voyage embedder with hash fallback
- `apps/orders/` — **outbox + saga**: `DeterministicRiskGuard` (9 criteria, last gate), `PlaceOrder` use-case, `process_outbox` beat task with retries + DLQ
- `apps/portfolio/`, `apps/broker/`, `apps/market_data/`, `apps/accounts/`, `apps/tenants/`, `apps/journals/`, `apps/strategies/`, `apps/billing/`, `apps/audit/`, `apps/notifications/`
- `plugins/strategy_directional/` — LangGraph equity flow (5 nodes)
- `plugins/strategy_short_straddle/` — LangGraph options flow (6 nodes)
- `plugins/broker_angel/`, `plugins/broker_zerodha/` — broker adapter stubs
- `pyproject.toml` — 3 entry-point groups (`alphadesk.strategies`, `.retrievers`, `.brokers`), all 8 endpoints verified to resolve

## 3. Frontend (React 18 + Vite + TS + Tailwind)

Root: `frontend/`

- `src/lib/api.ts` — axios with JWT interceptor + auto-refresh
- `src/lib/ws.ts` — auto-reconnecting WebSocket (ping/pong + backoff)
- `src/stores/auth.ts` — Zustand persist store
- `src/features/` — auth, onboarding, dashboard (live equity curve), positions (live LTP), agents (streaming console), strategies, backtester, broker linking
- `vite.config.ts` — proxies `/api` and `/ws` to `:8000`
- `package.json` — 16 runtime deps, 13 dev deps; scripts: `dev`, `build`, `typecheck`, `lint`, `test`

## 4. Containers

Root: `docker/` + `docker-compose*.yml`

- `docker/backend.Dockerfile` — multi-stage python:3.11-slim, gunicorn default CMD
- `docker/worker.Dockerfile` — Celery worker with queues `default,agents,orders,backtests`
- `docker/frontend.Dockerfile` — node:20 build → nginx:1.27 serve
- `docker-compose.yml` — local dev (postgres, redis, backend daphne, worker, beat, frontend)
- `docker-compose.prod.yml` — Stage-1 overlay (pulls from ECR, splits gunicorn + daphne into separate services)
- `.env.example` — variables expected by the prod overlay
- `infra/nginx/frontend.conf` — SPA fallback + gzip + far-future cache
- `infra/ec2/cloud-init.yaml` — bootstraps Docker + systemd unit on a fresh Ubuntu EC2

## 5. Terraform IaC (Stage-2 AWS target)

Root: `infra/terraform/`

Modules — network, alb, ecs_service, rds, redis, amplify, ecr, secrets, state_backend.

Envs — `envs/dev/` and `envs/prod/` with their own backend and tfvars.

- `modules/network` — VPC, 2 AZ public + private subnets, IGW, NAT
- `modules/alb` — public ALB with `/api/*` and `/ws/*` target groups, sticky sessions on WS
- `modules/ecs_service` — Fargate cluster, 4 services (api, ws, worker, beat) with autoscaling on the API
- `modules/rds` — Postgres 16 with pgvector param group, single-AZ in dev, Multi-AZ in prod
- `modules/redis` — ElastiCache 7, 1 node in dev, replicated in prod
- `modules/amplify` — SPA hosting with GitHub token from Secrets Manager
- `modules/ecr` — immutable repos for backend + worker, 30-image lifecycle
- `modules/secrets` — 8 Secrets Manager entries (Django secret, DB URL, Redis URL, Anthropic, SmartAPI quartet)
- `modules/state_backend` — S3 + DynamoDB for remote state (bootstrap once)

## 6. CI/CD

Root: `.github/`

- `workflows/backend-ci.yml` — ruff + black + mypy + pytest + Django check/migrate; on main: build & push to ECR, ECS force-new-deployment, smoke test
- `workflows/frontend-ci.yml` — typecheck + lint + build; on main: build & push frontend image (Stage-1 fallback; Amplify auto-builds Stage-2)
- `workflows/terraform.yml` — `fmt --check`, `init`, `validate`, `plan` per env on PR with plan commented back; manual `apply` via `workflow_dispatch`
- `workflows/deploy-ec2.yml` — Stage-1 SSH deploy (pull latest ECR images, `docker compose up -d`, smoke test)
- `workflows/security.yml` — CodeQL (python + TS), pip-audit, npm audit, Trivy
- `dependabot.yml` — weekly updates for pip / npm / actions / docker / terraform
- `PULL_REQUEST_TEMPLATE.md` + `CODEOWNERS`

## 7. Verification results

| Check | Status |
|---|---|
| All 6 YAML workflows parse clean | pass |
| `docker-compose*.yml` + cloud-init YAML parse clean | pass |
| `openapi.yaml` — 24 paths, 7 schemas | pass |
| `frontend/package.json`, `tsconfig.json` parse clean | pass |
| All 11 Terraform files pass brace/paren balance scan | pass |
| All 8 plugin entry-points (strategies, retrievers, brokers) resolve to real Python symbols | pass |
| `typecheck` script referenced by CI added to `package.json` | fixed |
| Single-line HCL block bodies with comma-separators converted to multi-line | fixed |

## Counts

| Area | Files | Lines |
|---|---:|---:|
| Docs | 12 | 1,978 |
| Backend | 159 | 3,162 |
| Frontend | 31 | 1,224 |
| Docker + Compose + EC2 bootstrap | 8 | 304 |
| Terraform | 16 | 1,370 |
| GitHub Actions + metadata | 8 | 581 |
| **Total** | **234** | **8,619** |

## Stage-gated rollout recap

- **Stage 1 (now → first 50 users)** — one `t3.medium` EC2 running `docker-compose.prod.yml`, deployed by `deploy-ec2.yml`. ~$40 / mo. Good enough to run paper-trading and one live account.
- **Stage 2 (growth)** — flip to the Terraform stack: Amplify + ALB + ECS Fargate + RDS + ElastiCache. ~$250 / mo at low scale. `terraform.yml` handles plan/apply.
- **Stage 3 (multi-region / HA)** — swap ECS for EKS, add read replicas and a second region. Out of scope for this scaffold but the module layout already separates concerns for that cut.
