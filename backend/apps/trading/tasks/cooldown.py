"""EOD losing-streak cooldown refresh (TRADING_RULES.md §1)."""
from __future__ import annotations

import structlog
from celery import shared_task

log = structlog.get_logger(__name__)


TASK_NAME = "apps.trading.tasks.cooldown.refresh_cooldowns"


@shared_task(name=TASK_NAME, ignore_result=True)
def refresh_cooldowns(trigger: str | None = None) -> dict:
    """Persist each trading portfolio's cooldown status for the NEXT trading
    day, after the session's trades are closed and derived. (The order path
    re-evaluates from trades on its own; this keeps the UI/Telegram row fresh.)
    Writes a RunReceipt; ``trigger`` only labels it."""
    from django.utils import timezone

    from apps.market_data.services.market_calendar import next_trading_day
    from apps.system.services import run_checks as rc
    from apps.system.services.run_receipts import infer_trigger, tracked_run
    from apps.trading.services.cooldown import refresh_cooldown, trading_portfolios

    with tracked_run("job", "cooldown_refresh", {}, trigger=infer_trigger({"trigger": trigger}),
                     task_name=TASK_NAME, primary_metric="portfolios", max_seconds=1800) as run:
        as_of = next_trading_day(timezone.localdate())
        out = {}
        for p in trading_portfolios():
            try:
                out[str(p.id)] = refresh_cooldown(p, as_of).status
                run.unit_ok()
            except Exception as e:  # noqa: BLE001
                log.warning("cooldown.refresh_failed", portfolio=str(p.id), error=str(e))
                run.unit_failed(error=f"portfolio {p.id}: {e}")
        log.info("cooldown.refreshed", as_of=str(as_of), portfolios=out)
        try:
            statuses: dict[str, int] = {}
            for s in out.values():
                statuses[s] = statuses.get(s, 0) + 1
            run.output(as_of=as_of.isoformat(), portfolios=len(out), statuses=statuses)
            run.check(rc.output_count("portfolios_refreshed", len(out), minimum=1,
                                      label="trading portfolios refreshed", severity=rc.WARN))
            run.link("Cooldown status", "/nightly-scan")
        except Exception as e:  # noqa: BLE001
            log.warning("cooldown.receipt_failed", error=str(e))
        return out
