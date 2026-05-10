# AlphaDesk Backend

Django 5 + DRF + Channels + Celery. Multi-tenant, plugin-based agentic RAG.

## Local dev

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser
# API + WS
uvicorn config.asgi:application --reload --port 8000
# Worker (in another shell)
celery -A config worker -Q default,agents,orders,backtests -l info
celery -A config beat -l info
```

OpenAPI docs: http://localhost:8000/api/docs/

## Layout

See `docs/architecture/BACKEND_STRUCTURE.md`.

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
