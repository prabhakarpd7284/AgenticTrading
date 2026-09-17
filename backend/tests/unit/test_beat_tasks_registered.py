"""Every scheduled task must actually exist in the worker registry.

`autodiscover_tasks()` imports each app's `tasks` *package*, not its submodules.
A task defined in `apps/x/tasks/y.py` therefore never registers unless it is
re-exported from `apps/x/tasks/__init__.py`. When that is forgotten the beat
entry still fires on schedule and the worker silently drops it — no error, no
log, the job just never runs.

`apps/trading/tasks/__init__.py` carries a comment warning about precisely this,
which means it has bitten before. This test makes the whole schedule fail loudly
instead of silently.
"""
from __future__ import annotations

from django.conf import settings

from config.celery import app


def test_every_beat_entry_points_at_a_registered_task():
    app.loader.import_default_modules()
    registered = set(app.tasks.keys())

    missing = {
        name: entry["task"]
        for name, entry in settings.CELERY_BEAT_SCHEDULE.items()
        if entry["task"] not in registered
    }

    assert not missing, (
        "Beat entries reference unregistered tasks — these will silently "
        "no-op on every fire. Re-export them from the app's "
        f"tasks/__init__.py: {missing}"
    )
