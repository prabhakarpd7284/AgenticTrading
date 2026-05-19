"""Periodic refresh of auto-kind watchlists.

Scheduled by celery beat (see config.settings.base.CELERY_BEAT_SCHEDULE).
Walks every non-MANUAL TradingViewWatchlist and dispatches to its resolver,
writing the new symbol list + timestamp back. Per-row errors are logged
and skipped so one bad row doesn't abort the whole sweep.

The 5-minute default cadence is conservative: SIGNAL_RANK / SOURCE_HOT
benefit from sub-minute freshness during active hours, but at the cost of
a more aggressive beat schedule. 5 minutes keeps the periodic load
negligible while still catching most operator-visible state changes.
The /refresh/ endpoint provides on-demand sync re-resolve for the cases
where staleness matters.
"""
from __future__ import annotations

import structlog
from celery import shared_task

from apps.notifications.models import TradingViewWatchlist
from apps.notifications.services.watchlist_resolvers import refresh_watchlist

log = structlog.get_logger()


@shared_task(name="apps.notifications.tasks.watchlists.refresh_auto_watchlists")
def refresh_auto_watchlists() -> dict:
    """Refresh every non-MANUAL watchlist. Returns a summary for telemetry."""
    refreshed = 0
    errors = 0
    skipped = 0

    qs = TradingViewWatchlist.objects.exclude(
        kind=TradingViewWatchlist.Kind.MANUAL,
    ).select_related("tenant")

    for wl in qs.iterator(chunk_size=100):
        try:
            count = refresh_watchlist(wl)
            refreshed += 1
            log.debug(
                "watchlist.refresh.ok",
                wl_id=str(wl.id), kind=wl.kind, symbols=count,
            )
        except Exception:  # noqa: BLE001
            errors += 1
            log.exception(
                "watchlist.refresh.failed",
                wl_id=str(wl.id), kind=wl.kind,
            )

    summary = {"refreshed": refreshed, "errors": errors, "skipped": skipped}
    log.info("watchlist.refresh.cycle", **summary)
    return summary
