"""Test settings.

Mirrors `dev.py`'s install of the legacy `trading` app and repo-root sys.path
tweak so that `apps.legacy.api.views` (which does `from trading.models import
TradeJournal` / `StraddlePosition` / ...) can actually run under pytest.

Without this, every legacy-bridge test lands in the `_with_legacy`
`ImportError` branch and returns 503 "legacy_dependency_missing", which
isn't what those tests are trying to exercise — they want to lock in the
shape of the real responses.
"""
from pathlib import Path

from .base import *  # noqa: F401,F403
from .base import BASE_DIR, INSTALLED_APPS

# ---------------------------------------------------------------------------
# Put the repo root on sys.path so `import trading` resolves.
# ---------------------------------------------------------------------------
import sys as _sys

_REPO_ROOT = Path(BASE_DIR).parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Install legacy apps so their models migrate into the in-memory test DB.
# ---------------------------------------------------------------------------
INSTALLED_APPS = INSTALLED_APPS + [
    "trading",
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
