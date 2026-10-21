"""EOD losing-streak cooldown refresh (TRADING_RULES.md §1)."""
from __future__ import annotations

import structlog
from celery import shared_task

log = structlog.get_logger(__name__)


@shared_task(name="apps.trading.tasks.cooldown.refresh_cooldowns", ignore_result=True)
def refresh_cooldowns() -> dict:
    """Persist each trading portfolio's cooldown status for the NEXT trading
    day, after the session's trades are closed and derived. (The order path
    re-evaluates from trades on its own; this keeps the UI/Telegram row fresh.)"""
    from django.utils import timezone

    from apps.market_data.services.market_calendar import next_trading_day
    from apps.trading.services.cooldown import refresh_cooldown, trading_portfolios

    as_of = next_trading_day(timezone.localdate())
    out = {}
    for p in trading_portfolios():
        try:
            out[str(p.id)] = refresh_cooldown(p, as_of).status
        except Exception as e:  # noqa: BLE001
            log.warning("cooldown.refresh_failed", portfolio=str(p.id), error=str(e))
    log.info("cooldown.refreshed", as_of=str(as_of), portfolios=out)
    return out
