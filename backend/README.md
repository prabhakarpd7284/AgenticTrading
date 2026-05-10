# AlphaDesk Backend (v2)

Django 5 + DRF + Channels + Celery. Multi-tenant, plugin-based agentic RAG.

This is the **v2 backend** — runs on port 8000 with Postgres + Redis. The legacy `trading/` app continues to run on port 8001 (SQLite) and is bridged via `apps.legacy`. See the project root `README.md` for the full picture.

## Local dev

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
cp .env.example .env
# Edit .env — set DATABASE_URL, REDIS_URL, DJANGO_SECRET_KEY, ANTHROPIC_API_KEY
python manage.py migrate
python manage.py createsuperuser

# API + WS (use uvicorn for ASGI in prod, runserver is fine for dev)
python manage.py runserver 0.0.0.0:8000

# Celery worker (in another shell) — required for async order execution + agent runs
celery -A config worker -Q default,agents,orders,backtests -l info
celery -A config beat -l info
```

OpenAPI docs: http://localhost:8000/api/docs/

## Key endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/auth/token/` | JWT obtain (with tenant_id claim) |
| `GET`  | `/api/v1/market-data/pulse/` | Market pulse (Cascade Stages 1+2) |
| `GET`  | `/api/v1/market-data/rotation/` | Sector rotation (Stage 3) |
| `GET`  | `/api/v1/market-data/shortlist/` | Shortlist (Stage 4) |
| `GET`  | `/api/v1/portfolios/monthly/` | **Monthly feedback report** |
| `POST` | `/api/v1/orders/` | Place an order (outbox + async) |
| `GET`  | `/api/v1/legacy/pyramid/` | Pyramid backtest (bridges to legacy `trading/pyramid/`) |
| `WS`   | `/ws/ticks/` | Live LTP fan-out |
| `WS`   | `/ws/agent/{run_id}/` | Live agent token stream |

## Layout

See `docs/architecture/BACKEND_STRUCTURE.md` for per-app details.

## Adding a strategy

1. Create `plugins/strategy_myname/` with a class implementing `Strategy`.
2. Register it in `pyproject.toml` under `[project.entry-points."alphadesk.strategies"]`.
3. Reinstall: `pip install -e .`.
4. Hit `GET /api/v1/agents/catalog/` — your strategy appears.
5. `POST /api/v1/agents/runs/` with `strategy_name=myname` to run it.

## Adding a broker

Same pattern, entry-point `alphadesk.brokers`.

## Tests

```bash
pytest
```
