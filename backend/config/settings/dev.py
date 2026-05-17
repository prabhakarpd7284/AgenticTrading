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

# `apps.legacy` is the URL bridge — it has no models, just views. Keep
# it in INSTALLED_APPS so /api/v1/legacy/* keeps resolving while the
# frontend hasn't been migrated to /api/v1/{trades,events,...} yet.
#
# `trading` is the legacy package — installed with a custom AppConfig
# (label="trading_legacy", no models) so the dev-ops console can
# discover the operator CLIs under trading/management/commands/.
INSTALLED_APPS = INSTALLED_APPS + [
    "apps.legacy",
    "trading.apps.TradingLegacyConfig",
]

# The legacy trading/migrations/ directory references the bare app label
# 'trading' (which is now owned by apps.trading); skip them for the
# relabeled "trading_legacy" app — its AppConfig owns no models anyway.
MIGRATION_MODULES = {"trading_legacy": None}

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
INTERNAL_IPS = ["127.0.0.1"]

# ---------------------------------------------------------------------------
# Channels — in-memory layer in dev unless USE_REDIS=1 or REDIS_URL is set.
# ---------------------------------------------------------------------------
if not env.bool("USE_REDIS", default=False):
    CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
    }

# Broaden CSRF/CORS for the local Vite origin so the SPA on :5173 can POST.
CSRF_TRUSTED_ORIGINS = list(set((locals().get("CSRF_TRUSTED_ORIGINS") or []) + [
    "http://localhost:5173", "http://127.0.0.1:5173",
]))
