"""Failed / missed run alerts (UX redesign plan §8.4: "failed or missed jobs push an alert").

Goes through the same Telegram path the nightly scan already uses
(``plugins.strategy_screener.telegram.TelegramAlertService``, synchronous
send) and, for tenant-scoped receipts, the Event log via
``event_writer.emit()``. Everything here is non-blocking: it logs and
returns on any failure. ``settings.RUN_RECEIPTS_TELEGRAM = False`` turns
the Telegram push off (tests set this).
"""
from __future__ import annotations

import html

import structlog

log = structlog.get_logger(__name__)


def format_alert(receipt) -> str:
    from zoneinfo import ZoneInfo

    ist = ZoneInfo("Asia/Kolkata")
    if receipt.status == "not_run":
        when = receipt.expected_at.astimezone(ist).strftime("%d %b %H:%M") if receipt.expected_at else "?"
        head = f"⏰ <b>Job did not run</b>: {html.escape(receipt.name)}"
        body = f"Scheduled {when} IST ({html.escape(receipt.schedule_key)}) — no run recorded in its window."
    else:
        when = receipt.started_at.astimezone(ist).strftime("%d %b %H:%M")
        head = f"❌ <b>Job failed</b>: {html.escape(receipt.name)}"
        body = f"Started {when} IST · {html.escape((receipt.error or '')[:600])}"
    fails = [c for c in (receipt.checks or []) if c.get("status") == "fail"]
    lines = [head, body]
    for c in fails[:4]:
        lines.append(f"• {html.escape(c.get('name', ''))}: {html.escape(str(c.get('message', ''))[:200])}")
    lines.append("<i>AlphaDesk → Runs for the full receipt.</i>")
    return "\n".join(lines)[:3900]


def _send_telegram(text: str) -> bool:
    from django.conf import settings

    if not getattr(settings, "RUN_RECEIPTS_TELEGRAM", True):
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv()
        from plugins.strategy_screener.telegram import TelegramAlertService

        svc = TelegramAlertService()
        if not svc.is_configured:
            return False
        before = svc.total_errors
        svc._do_send(text, "HTML")
        return svc.total_errors == before
    except Exception as e:  # noqa: BLE001
        log.warning("run_alert.telegram_failed", error=str(e))
        return False


def alert_receipt(receipt) -> None:
    """Push one alert per receipt (idempotent via ``alerted_at``)."""
    try:
        from django.utils import timezone

        if receipt.alerted_at is not None:
            return
        sent = _send_telegram(format_alert(receipt))
        if receipt.tenant_id:
            from apps.events.services.event_writer import emit

            emit(tenant=receipt.tenant, type="workflow.step.failed", severity="error",
                 step_name=f"run_receipt.{receipt.status}",
                 text=f"{receipt.name} {receipt.status}: {(receipt.error or '')[:200]}",
                 payload={"receipt_id": str(receipt.id), "name": receipt.name,
                          "status": receipt.status})
        receipt.alerted_at = timezone.now()
        type(receipt).objects.filter(pk=receipt.pk).update(alerted_at=receipt.alerted_at)
        log.info("run_alert.sent", name=receipt.name, status=receipt.status, telegram=sent)
    except Exception as e:  # noqa: BLE001
        log.warning("run_alert.failed", name=getattr(receipt, "name", "?"), error=str(e))
