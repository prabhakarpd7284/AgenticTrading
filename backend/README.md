# AlphaDesk Backend

Django 5 + DRF + Channels + Celery. Multi-tenant, plugin-based.

One Django ASGI process on port 8000 hosts the REST API, the WebSocket
consumers, and the in-process broker WS holder. Postgres + Redis run in
Docker; everything else (backend, Celery worker, frontend) runs native.

## Local dev

```bash
# ── 1. Infra (Postgres :5436, Redis :6379) ──
docker compose -f ../docker-compose.dev.yml up -d

# ── 2. One-time backend setup ──
cd backend
cp .env.example .env                       # defaults work for local dev
uv sync                                    # installs deps + the 7 strategy plugins
uv pip install -e .                        # registers entry-points (alphadesk.strategies, .brokers)
.venv/bin/python manage.py migrate

# ── 3. Run the three processes ──
# Option A: one shot
bash ../scripts/dev_up.sh                  # web (:8000) + celery + vite (:5173)
# Option B: separately
DJANGO_SETTINGS_MODULE=config.settings.dev .venv/bin/python manage.py runserver 0.0.0.0:8000
DJANGO_SETTINGS_MODULE=config.settings.dev .venv/bin/celery -A config worker -l info
# (frontend lives in ../frontend; `npm run dev`)
```

Stop everything: `bash ../scripts/dev_down.sh` (leaves the docker containers running).

OpenAPI docs: <http://localhost:8000/api/docs/>

## Tests

```bash
.venv/bin/python -m pytest -q             # expect: 99 passed
```

The test runner spins up a fresh test DB from migrations on every run,
which doubles as the canonical "fresh install" verification.

## App layout (13 apps, post-Phase-4)

| App | What it owns |
|---|---|
| `accounts` · `tenants` · `system` · `billing` | Identity, tenancy, kill-switch flags, plans |
| `common` | Tenancy mixin, middleware, pagination, structured logging |
| `market_data` | Symbol / Candle / BrokerLink (Phase 4b) + pulse / rotation / shortlist services |
| `trading` (Phase 4c) | Portfolio + Position + PortfolioSnapshot + Order + OutboxEvent + Trade + OptionsPosition + OptionsLeg — the whole money triangle in one app |
| `agents_core` | Workflow runtime (AgentRun, AgentStep, plugin registry) |
| `strategies` | StrategyInstance, Backtest, Signal, WatchlistEntry |
| `rag` | KnowledgeDoc + Embedding (pgvector) |
| `events` (Phase 4a) | Unified Event log — absorbed audit + journals |
| `notifications` | Alert + Telegram / Email / Slack dispatchers |
| `legacy` | URL bridge — read-only views over the v2 tables at `/api/v1/legacy/*`, kept until the frontend migrates to `/api/v1/{trades,events,signals,…}/` |

## Key endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/auth/token/` | JWT obtain (with tenant_id claim) |
| `GET`  | `/api/v1/market-data/pulse/` | Market pulse (Cascade Stages 1+2) |
| `GET`  | `/api/v1/market-data/rotation/` | Sector rotation (Stage 3) |
| `GET`  | `/api/v1/market-data/shortlist/` | Shortlist (Stage 4) |
| `GET`  | `/api/v1/portfolios/monthly/` | Monthly feedback report |
| `POST` | `/api/v1/orders/` | Place an order (outbox + async) |
| `GET`  | `/api/v1/legacy/pyramid/` | Pyramid backtest (bridge) |
| `WS`   | `/ws/events/` | System-wide event firehose (per tenant) |
| `WS`   | `/ws/runs/{run_id}/` | Per-workflow timeline |
| `WS`   | `/ws/pnl/` | Live portfolio P&L |

## Strategy plugins

Each strategy is a standalone Python package under `backend/plugins/`,
registered via the `alphadesk.strategies` entry-point. The seven that
ship today:

| Plugin | Entry-point name |
|---|---|
| `plugins/strategy_directional` | `directional` |
| `plugins/strategy_short_straddle` | `short_straddle` |
| `plugins/strategy_pyramid` | `pyramid` |
| `plugins/strategy_screener` | `intraday_screener` |
| `plugins/strategy_swing` | `swing_scanner` |
| `plugins/strategy_basket` | `premarket_basket` |
| `plugins/strategy_backtest` | `backtest` |

### Adding one

1. Create `plugins/strategy_<name>/` with `__init__.py`, `plugin.py`
   exposing a class implementing the `Strategy` protocol
   (`schema()` + `build_graph(ctx)`), and any internal modules.
2. Register it in `pyproject.toml`:
   ```toml
   [project.entry-points."alphadesk.strategies"]
   my_strategy = "plugins.strategy_<name>:MyStrategy"
   ```
3. `uv pip install -e .` to re-register the entry-point.
4. Restart Django — the registry picks it up on `apps.ready()`.

Same pattern for brokers (`alphadesk.brokers`) and RAG retrievers
(`alphadesk.retrievers`).
