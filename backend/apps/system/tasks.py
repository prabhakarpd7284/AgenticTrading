"""Celery tasks for apps.system — the run-receipt schedule watcher."""
from __future__ import annotations

from celery import shared_task


@shared_task(name="apps.system.tasks.watch_run_schedule", ignore_result=True)
def watch_run_schedule() -> dict:
    """Every 10 min: write ``not_run`` receipts (+ alert) for scheduled jobs
    that missed their window, and fail receipts stuck in RUNNING."""
    from apps.system.services.run_schedule import watch

    return watch()
