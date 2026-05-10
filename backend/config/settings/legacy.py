"""Legacy-only settings profile.

Used for the dedicated Django process on :8001 that serves the
`/api/v1/legacy/` bridge backed by the seeded `db.sqlite3` at the repo
root. This keeps the legacy bridge isolated from the v2 Postgres stack
on :8000 — no database routers, no cross-DB joins, no surprises when
DATABASE_URL is pointed at Postgres for the main server.

URL conf is narrowed to just the endpoints the sqlite file can answer
for: /api/v1/legacy/*, /api/v1/auth/*, /api/v1/health/. Everything else
(portfolios, strategies, orders, agents) lives on :8000.

Run: python manage.py runserver 0.0.0.0:8001 --settings=config.settings.legacy
"""
from pathlib import Path

from .base import *  # noqa: F401,F403
from .base import BASE_DIR, INSTALLED_APPS, env

DEBUG = True

# ---------------------------------------------------------------------------
# Two DBs on this process:
#   default — v2 Postgres (DATABASE_URL), so JWT auth can look up the user
#             and tenant created on :8000.  Without this, every request 500s
#             with "no such table: accounts_user" because the legacy sqlite
#             only holds the `trading_*` tables.
#   legacy  — the seeded sqlite at the repo root, read-only source for the
#             `trading` app (TradeJournal, StraddlePosition, AuditLog, ...).
# The LegacyRouter in apps.common.db_router pins `trading.*` to the legacy
# alias; everything else (accounts, tenants, sessions) stays on default.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(BASE_DIR).parent
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
# Put the repo root on sys.path so `import trading` resolves, and install
# the legacy apps.  The base v2 apps come along for the ride because they
# own auth/tenant/permissions code the legacy views depend on.
# ---------------------------------------------------------------------------
import sys as _sys

if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

INSTALLED_APPS = INSTALLED_APPS + [
    "trading",
    "apps.legacy",
]

# ---------------------------------------------------------------------------
# Trim URL conf to just the legacy + auth + health surface.
# ---------------------------------------------------------------------------
ROOT_URLCONF = "config.urls_legacy"

# In-memory channels layer — we don't need WS on the legacy process.
CHANNEL_LAYERS = {
    "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
}

# Broaden CORS/CSRF for the Vite dev origin.
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
CSRF_TRUSTED_ORIGINS = list(set((locals().get("CSRF_TRUSTED_ORIGINS") or []) + [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]))
