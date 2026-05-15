"""Test settings.

The legacy `trading` Django app at repo root was installed here so the
legacy-bridge views (which once did `from trading.models import
TradeJournal/StraddlePosition/...`) could resolve. Phase 6 dropped those
imports; Phase 4c merged orders+portfolio+trades into `apps.trading`,
which collides with the legacy app's label — so the legacy app is now
unloaded for tests too. The repo root stays on sys.path because a few
helper imports (`trading.options.data_service`, `trading.utils.*`) still
live there until they get a permanent home.
"""
from pathlib import Path

from .base import *  # noqa: F401,F403
from .base import BASE_DIR, INSTALLED_APPS

# ---------------------------------------------------------------------------
# Put the repo root on sys.path so `from trading.options.data_service import …`
# (used by apps.legacy.api.views' option-token helpers) resolves.
# ---------------------------------------------------------------------------
import sys as _sys

_REPO_ROOT = Path(BASE_DIR).parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# Only the URL-bridge app is installed alongside the v2 apps.
INSTALLED_APPS = INSTALLED_APPS + [
    "apps.legacy",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
CHANNEL_LAYERS = {
    "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
}
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
