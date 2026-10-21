"""Celery task for the 21:00 IST nightly scan (CELERY_BEAT_SCHEDULE)."""
from __future__ import annotations

from datetime import timedelta

import structlog
from celery import shared_task

log = structlog.get_logger(__name__)


@shared_task(name="apps.news.tasks.run_nightly_scan", ignore_result=True,
             time_limit=2 * 3600, soft_time_limit=2 * 3600 - 300)
def run_nightly_scan(force: bool = False) -> dict:
    """Fires every evening; runs only when TOMORROW is an NSE trading day
    (Mon–Thu + Sunday nights, holiday-aware)."""
    from django.utils import timezone

    from apps.market_data.services.market_calendar import is_trading_day

    today = timezone.localdate()
    if not force and not is_trading_day(today + timedelta(days=1)):
        log.info("nightly_scan.skipped", reason="next day is not a trading day")
        return {"skipped": "next day is not a trading day"}

    from apps.news.services.nightly_scan import run_nightly_scan as run

    scan = run(session_date=today, telegram=True, refresh=True)
    log.info("nightly_scan.done", for_date=str(scan.for_date), status=scan.status,
             secs=scan.timings.get("total"))
    return {"for_date": scan.for_date.isoformat(), "status": scan.status}
