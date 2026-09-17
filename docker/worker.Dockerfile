# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Built from the committed lockfile (pyproject + uv.lock) — see
# docker/backend.Dockerfile for why. The worker needs the broker SDKs to place
# orders out of the outbox and the project install to register plugins.
RUN pip install --no-cache-dir uv
COPY backend/pyproject.toml backend/uv.lock /app/
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /app/requirements.lock.txt \
 && pip install --no-cache-dir -r /app/requirements.lock.txt
COPY backend /app
RUN pip install --no-cache-dir --no-deps -e .

ENV DJANGO_SETTINGS_MODULE=config.settings.prod
CMD ["celery", "-A", "config", "worker", "-Q", "default,agents,orders,backtests", "-l", "info"]
