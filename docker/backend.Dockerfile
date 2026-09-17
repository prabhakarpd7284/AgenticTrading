# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- deps layer (single source of truth: pyproject + uv.lock) ---
# The image is built from the committed lockfile so it can never drift from
# pyproject. The old hand-maintained requirements/*.txt silently omitted the
# broker SDKs (smartapi-python, fyers-apiv3, kiteconnect), yfinance and OTel —
# the running container could not place a single order.
RUN pip install --no-cache-dir uv
COPY backend/pyproject.toml backend/uv.lock /app/
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /app/requirements.lock.txt \
 && pip install --no-cache-dir -r /app/requirements.lock.txt "gunicorn>=22.0"

# --- app layer ---
COPY backend /app
# Install the project itself so the entry-point-driven plugin registries
# (alphadesk.strategies / .brokers) actually populate at boot. Without this the
# registries are empty — no strategies, no brokers, no trading.
RUN pip install --no-cache-dir --no-deps -e .
RUN python manage.py collectstatic --noinput --settings=config.settings.prod || true

ENV DJANGO_SETTINGS_MODULE=config.settings.prod
EXPOSE 8000

# Two entrypoints: gunicorn (REST) and daphne (ASGI/WS).
# The compose/ECS layer picks which CMD to run.
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "60"]
