"""Test settings.

The repo root stays on sys.path because the v2 legacy_compat_views and a
few of the still-living service helpers (`trading.options.data_service`,
`trading.utils.*`) import from `trading.*`. The legacy `trading` Django
app itself is wired in via `trading.apps.TradingLegacyConfig` in dev.py
for ops-console command discovery; tests don't need it (commands are
spawned as subprocesses, not loaded into the test process).
"""
from pathlib import Path

from .base import *  # noqa: F401,F403
from .base import BASE_DIR

# ---------------------------------------------------------------------------
# Put the repo root on sys.path so `from trading.options.data_service import …`
# (used by legacy_compat_views' option-token helpers) resolves.
# ---------------------------------------------------------------------------
import sys as _sys

_REPO_ROOT = Path(BASE_DIR).parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

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
