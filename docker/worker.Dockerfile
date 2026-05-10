# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements /app/requirements
RUN pip install -r requirements/prod.txt
COPY backend /app

ENV DJANGO_SETTINGS_MODULE=config.settings.prod
CMD ["celery", "-A", "config", "worker", "-Q", "default,agents,orders,backtests", "-l", "info"]
