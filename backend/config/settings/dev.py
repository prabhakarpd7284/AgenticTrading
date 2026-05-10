"""Dev settings — v2 stack on :8000.

The legacy `trading` sqlite bridge now runs in its own Django process on
:8001 with `config.settings.legacy` (see that file).  This profile is
the v2 stack only — Postgres-backed by default via DATABASE_URL, with a
sqlite fallback for zero-config development.

Layout:
  * :8000  config.settings.dev     → v2 apps on Postgres (or sqlite default)
  * :8001  config.settings.legacy  → /api/v1/legacy/ on AgenticTrading/db.sqlite3
  * Frontend (Vite :5173) proxies /api/v1/legacy/* to :8001 and the rest
    to :8000 — see frontend/vite.config.ts.

We still register the `trading` + `apps.legacy` apps here so that (a) the
legacy URL namespace resolves when unit tests import it, and (b) if you
ever want to collapse back to a single process, setting
LEGACY_IN_PROCESS=1 falls back to the old multi-DB router.
"""
from pathlib import Path

from .base import *  # noqa: F401,F403
from .base import BASE_DIR, INSTALLED_APPS, env

DEBUG = True

# ---------------------------------------------------------------------------
# Database — v2 on DATABASE_URL (Postgres), sqlite fallback for zero-config.
# A second `legacy` alias points at the seeded sqlite at the repo root so v2
# services that read `trading.*` models (e.g. portfolio.MonthlyReport pulls
# from SignalLog/AuditLog/TradeJournal) succeed.  The LegacyRouter pins
# `trading.*` queries to the legacy alias; everything else stays on default.
# Without this, /api/v1/portfolios/monthly/ 500s with `relation
# "trading_signallog" does not exist`.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(BASE_DIR).parent                  # AgenticTrading/
_LEGACY_SQLITE = _REPO_ROOT / "db.sqlite3"

DATABASES = {
    "default": env.db_url(
        "DATABASE_URL",
        default=f"sqlite:///{_LEGACY_SQLITE}",
    ),
    "legacy": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(_LEGACY_SQLITE),
    },
}
DATABASE_ROUTERS = ["apps.common.db_router.LegacyRouter"]

# ---------------------------------------------------------------------------
# Make `trading` + `apps.legacy` importable so the URL conf + tests resolve
# even when this process isn't serving /api/v1/legacy/ itself.
# ---------------------------------------------------------------------------
import sys as _sys

if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

INSTALLED_APPS = INSTALLED_APPS + [
    "trading",
    "apps.legacy",
]

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
