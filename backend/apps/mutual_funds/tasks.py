"""Celery tasks for the mutual fund collector (scheduled in CELERY_BEAT_SCHEDULE)."""
from __future__ import annotations

import structlog
from celery import shared_task
from django.core.management import call_command

log = structlog.get_logger(__name__)


@shared_task(name="apps.mutual_funds.tasks.pull_amfi_nav", ignore_result=True)
def pull_amfi_nav() -> None:
    """Daily AMFI NAV pull, then a stale-scheme sweep. Idempotent — safe to
    fire twice a day (late-evening publish + morning catch-up)."""
    call_command("pull_amfi_nav")
    call_command("check_stale_navs", limit=0)
    log.info("mf_daily_pull.done")


@shared_task(name="apps.mutual_funds.tasks.compute_mf_metrics_weekly", ignore_result=True)
def compute_mf_metrics_weekly() -> None:
    """Weekly screener refresh: top up benchmark TRI, then recompute metric
    snapshots as of the catalog NAV date. NAV history itself is appended
    daily by pull_amfi_nav."""
    try:
        call_command("pull_mf_benchmarks", incremental=True)
    except Exception as exc:  # noqa: BLE001 — metrics still run without fresh TRI
        log.warning("mf_benchmarks.weekly_failed", error=str(exc))
    call_command("compute_mf_metrics")
    log.info("mf_metrics_weekly.done")
