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

# --- deps layer ---
COPY backend/requirements /app/requirements
RUN pip install -r requirements/prod.txt

# --- app layer ---
COPY backend /app
RUN python manage.py collectstatic --noinput --settings=config.settings.prod || true

ENV DJANGO_SETTINGS_MODULE=config.settings.prod
EXPOSE 8000

# Two entrypoints: gunicorn (REST) and daphne (ASGI/WS).
# The compose/ECS layer picks which CMD to run.
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "60"]
