"""Dev settings — single-process v2 stack on :8000.

Topology (post redesign-v2):
  * :8000  config.settings.dev  → v2 Django (REST + WS + Channels)
  * :5173  Vite dev server       → React UI (proxies /api → :8000)
  * Postgres + Redis run in Docker (docker-compose.dev.yml)

The legacy SQLite database, the separate :8001 process, and the
LegacyRouter have all been retired — every endpoint now reads from
the v2 Postgres tables. See docs/MIGRATION.md for the data lift history.
"""
from pathlib import Path

from .base import *  # noqa: F401,F403
from .base import BASE_DIR, INSTALLED_APPS, env

DEBUG = True

# ---------------------------------------------------------------------------
# Database — single Postgres alias. The legacy SQLite + LegacyRouter were
# retired once `apps.trading`, `apps.events`, `apps.strategies.Signal`,
# `apps.strategies.WatchlistEntry`, `apps.rag.KnowledgeDoc`, and
# `apps.system.*` absorbed all legacy data via `manage.py migrate_legacy`.
# ---------------------------------------------------------------------------
DATABASES = {
    "default": env.db_url(
        "DATABASE_URL",
        # Postgres in Docker (docker-compose.dev.yml). If DATABASE_URL is
        # unset and Postgres isn't running, Django will fail loudly at
        # connect time — which is what we want, no SQLite fallback.
        default="postgres://alphadesk:alphadesk@localhost:5436/alphadesk",
    ),
}

# ---------------------------------------------------------------------------
# Keep the `trading` package importable for non-Django code (pyramid /
# screener strategy engines still live there until Phase 3 plugin moves).
# It is intentionally NOT in INSTALLED_APPS — its Django models are not
# managed by this stack any more.
# ---------------------------------------------------------------------------
import sys as _sys

_REPO_ROOT = Path(BASE_DIR).parent  # AgenticTrading/
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# `trading` is the legacy package — installed with a custom AppConfig
# (label="trading_legacy", no models) so the dev-ops console can
# discover the operator CLIs under trading/management/commands/.
INSTALLED_APPS = INSTALLED_APPS + [
    "trading.apps.TradingLegacyConfig",
]

# The legacy trading/migrations/ directory references the bare app label
# 'trading' (which is now owned by apps.trading); skip them for the
# relabeled "trading_legacy" app — its AppConfig owns no models anyway.
MIGRATION_MODULES = {"trading_legacy": None}

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
INTERNAL_IPS = ["127.0.0.1"]

# ---------------------------------------------------------------------------
# Redis / Celery — pin to the dev docker host (docker-compose.dev.yml maps
# 6380:6379). base.py defaults REDIS_URL to :6379, which on this machine is a
# SEPARATE project's Redis (bull:* / analytics:* keys) — AlphaDesk must NEVER
# touch it. dev.py overriding only DATABASE_URL (above) previously left Celery
# pointing at base.py's :6379 default whenever backend/.env wasn't loaded.
# ---------------------------------------------------------------------------
REDIS_URL = env("REDIS_URL", default="redis://localhost:6380/0")
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default=REDIS_URL)
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default=REDIS_URL)

# ---------------------------------------------------------------------------
# Channels — in-memory layer in dev unless USE_REDIS=1. When Redis channels
# ARE enabled, rebuild CHANNEL_LAYERS from the pinned REDIS_URL above: base.py
# built its CHANNEL_LAYERS from the old :6379 default at import time, so we
# must rebuild here or USE_REDIS=1 would still route channels to :6379.
# ---------------------------------------------------------------------------
if env.bool("USE_REDIS", default=False):
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [REDIS_URL]},
        },
    }
else:
    CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }

# Broaden CSRF/CORS for the local Vite origin so the SPA on :5173 can POST.
CSRF_TRUSTED_ORIGINS = list(set((locals().get("CSRF_TRUSTED_ORIGINS") or []) + [
    "http://localhost:5173", "http://127.0.0.1:5173",
]))
