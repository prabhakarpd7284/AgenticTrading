"""Periodic broker-snapshot refresh.

Three tasks:

* ``refresh_broker_positions`` — beat-scheduled fan-out. Picks every
  ACTIVE BrokerLink and dispatches a per-link refresh. Cadence varies
  with session phase (see schedule_cadence_seconds).

* ``refresh_link_snapshot(link_id)`` — does the actual broker call for
  a single link. Idempotent; safe to invoke ad-hoc from the shell.

* ``prune_old_snapshots`` — daily housekeeping; keeps the most recent
  N snapshots per link so the table doesn't grow unbounded.
"""
from __future__ import annotations

import logging

from celery import shared_task
from django.utils import timezone

from apps.market_data.models import BrokerLink, BrokerPositionSnapshot

logger = logging.getLogger(__name__)

# Cap retained snapshots per link. Tuned so the UI's freshness indicator
# can show recent fetched_at history without TTL-pruning load.
SNAPSHOT_RETENTION_PER_LINK = 50


def schedule_cadence_seconds(now=None) -> int:
    """Refresh cadence — tighter during market hours, looser off-hours.

    Mirrors trading.utils.time_utils.is_market_open without importing it
    here (avoids the legacy `trading` package import cost in the worker).
    """
    if now is None:
        now = timezone.localtime()
    if now.weekday() >= 5:
        return 600
    t = now.time()
    from datetime import time as _t
    if _t(9, 15) <= t <= _t(15, 30):
        return 30
    return 300


@shared_task
def refresh_broker_positions() -> dict:
    """Fan out per-link refreshes. Safe to invoke standalone; the beat
    schedule calls this every 30s during market hours.

    Skip rules (in order):
      1. DISABLED links — never polled.
      2. EXPIRED OAuth brokers — need explicit /oauth/reauth/; polling
         just hammers the broker with failing auth probes.
      3. ERRORED links within the back-off window (default 2 min since
         the last attempt) — avoids hammering a broker that just rate-
         limited us ("Access denied because of exceeding access rate").
      4. ACTIVE links whose latest successful snapshot is < FRESH_WINDOW
         old — typically the user just clicked Refresh in the UI and
         the data is already cached. Re-fetching now wastes broker
         quota for an answer we already have.
    """
    from datetime import timedelta

    DAILY_TOKEN_BROKERS = {"zerodha", "fyers"}
    ERRORED_BACKOFF = timedelta(seconds=120)
    # If a snapshot is newer than this, skip the next beat tick — usually
    # means a UI-triggered refresh just landed. Should be a bit less
    # than the beat cadence (30s) so we don't go > 1 tick without data.
    FRESH_WINDOW = timedelta(seconds=20)

    now = timezone.now()
    qs = BrokerLink.objects.exclude(status=BrokerLink.Status.DISABLED)
    qs = qs.exclude(
        status=BrokerLink.Status.EXPIRED,
        broker_name__in=DAILY_TOKEN_BROKERS,
    )
    dispatched = 0
    skipped_backoff = 0
    skipped_fresh = 0
    for link in qs:
        last_snap_ts = link.snapshots.values_list("fetched_at", flat=True).first()
        if link.status == BrokerLink.Status.ERRORED:
            last_attempt = last_snap_ts or link.last_refreshed_at
            if last_attempt and (now - last_attempt) < ERRORED_BACKOFF:
                skipped_backoff += 1
                continue
        elif link.status == BrokerLink.Status.ACTIVE:
            if last_snap_ts and (now - last_snap_ts) < FRESH_WINDOW:
                skipped_fresh += 1
                continue
        refresh_link_snapshot.delay(str(link.id))
        dispatched += 1
    return {
        "links_dispatched": dispatched,
        "skipped_backoff": skipped_backoff,
        "skipped_fresh": skipped_fresh,
        "ts": now.isoformat(),
    }


@shared_task
def refresh_link_snapshot(link_id: str) -> dict:
    """Take one snapshot for one link. Returns a small dict for logging."""
    from apps.market_data.api.broker_views import _take_snapshot
    try:
        link = BrokerLink.objects.get(id=link_id)
    except BrokerLink.DoesNotExist:
        return {"link_id": link_id, "error": "not found"}

    snap = _take_snapshot(link)
    return {
        "link_id": link_id,
        "broker": link.broker_name,
        "ok": snap.ok,
        "positions": len(snap.positions or []),
        "holdings": len(snap.holdings or []),
        "error": snap.error,
    }


@shared_task
def prune_old_snapshots() -> dict:
    """Trim each link's snapshot history to the last SNAPSHOT_RETENTION_PER_LINK rows."""
    pruned = 0
    for link in BrokerLink.objects.all():
        keep_ids = list(
            link.snapshots.order_by("-fetched_at")
            .values_list("id", flat=True)[:SNAPSHOT_RETENTION_PER_LINK]
        )
        deleted, _ = link.snapshots.exclude(id__in=keep_ids).delete()
        pruned += deleted
    return {"pruned": pruned}
