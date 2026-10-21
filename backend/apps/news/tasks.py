"""Celery task for the 21:00 IST nightly scan (CELERY_BEAT_SCHEDULE)."""
from __future__ import annotations

from datetime import timedelta

import structlog
from celery import shared_task

log = structlog.get_logger(__name__)

TASK_NAME = "apps.news.tasks.run_nightly_scan"


@shared_task(name=TASK_NAME, ignore_result=True,
             time_limit=2 * 3600, soft_time_limit=2 * 3600 - 300)
def run_nightly_scan(force: bool = False) -> dict:
    """Fires every evening; runs only when TOMORROW is an NSE trading day
    (Mon–Thu + Sunday nights, holiday-aware). Every fire writes a RunReceipt."""
    from django.utils import timezone

    from apps.market_data.services.market_calendar import is_trading_day
    from apps.system.services.run_receipts import infer_trigger, tracked_run

    with tracked_run("scan", "nightly_scan", {"force": force}, trigger=infer_trigger({"force": force}),
                     task_name=TASK_NAME, primary_metric="rows",
                     max_seconds=2 * 3600 + 600) as receipt:
        today = timezone.localdate()
        if not force and not is_trading_day(today + timedelta(days=1)):
            log.info("nightly_scan.skipped", reason="next day is not a trading day")
            receipt.skip("next day is not a trading day")
            return {"skipped": "next day is not a trading day"}

        from apps.news.services.nightly_scan import run_nightly_scan as run

        scan = run(session_date=today, telegram=True, refresh=True)
        try:
            describe_scan(receipt, scan)
        except Exception as e:  # noqa: BLE001 — receipts never break the job
            log.warning("nightly_scan.receipt_failed", error=str(e))
        log.info("nightly_scan.done", for_date=str(scan.for_date), status=scan.status,
                 secs=scan.timings.get("total"))
        return {"for_date": scan.for_date.isoformat(), "status": scan.status}


def describe_scan(receipt, scan) -> None:
    """Outputs + checks for a finished NightlyScan."""
    from apps.market_data.constants import NIFTY_50_SYMBOLS
    from apps.system.services import run_checks as rc

    rows = list(scan.rows.values_list("data_date", "allowed"))
    summary = scan.summary or {}
    allowed: dict[str, int] = {}
    for _, a in rows:
        allowed[a] = allowed.get(a, 0) + 1
    receipt.set_detail("nightly_scan", scan.id)
    receipt.link("Nightly scan", "/nightly-scan")
    receipt.output(
        for_date=scan.for_date.isoformat(), session_date=scan.session_date.isoformat(),
        scan_status=scan.status, rows=len(rows), allowed=allowed,
        headlines_total=summary.get("headlines_total"),
        headlines_scored=summary.get("headlines_scored"),
        telegram_sent=scan.telegram_sent,
    )
    receipt.check(rc.output_count("rows_vs_universe", len(rows), expected=len(NIFTY_50_SYMBOLS),
                                  label="symbols scanned"))
    stale = sum(1 for d, _ in rows if d is None or d < scan.session_date)
    if rows and stale == len(rows):
        receipt.check(rc.make("levels_fresh", rc.FAIL,
                              f"all {stale} symbols' levels predate session {scan.session_date}"))
    elif stale:
        receipt.check(rc.make("levels_fresh", rc.WARN,
                              f"{stale} of {len(rows)} symbols' levels predate session {scan.session_date}"))
    elif rows:
        receipt.check(rc.make("levels_fresh", rc.PASS, f"levels from session {scan.session_date}"))
    receipt.check(rc.broker_live(required=False))
    sources = scan.sources or {}
    sent = sources.get("claude_sentiment") or {}
    if sent.get("failed_batches"):
        receipt.swallowed_error(f"{sent['failed_batches']} sentiment batch(es) failed",
                                count=int(sent["failed_batches"]))
    failed_sources = [k for k, v in sources.items() if isinstance(v, dict) and v.get("ok") is False]
    for k in failed_sources:
        receipt.warn(f"source degraded: {k}")
    if not scan.telegram_sent:
        err = summary.get("telegram_error") or "not sent"
        receipt.warn(f"Telegram summary: {err}")
    receipt.set_provenance(pivot_source=summary.get("pivot_source"), session_date=str(scan.session_date))
    if scan.status == "partial":
        receipt.mark_partial("scan degraded: " + ", ".join(summary.get("degraded") or []))
    elif scan.status == "failed":
        receipt.check(rc.make("scan_status", rc.FAIL, scan.error or "scan failed"))
