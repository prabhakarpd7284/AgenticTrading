"""Cache warmers for the slow market-data builders.

Pulse and Rotation are expensive on a cold cache (10-17s) because each rebuild
hits the broker for VIX + NIFTY + 22 sector quotes. Both have short TTLs
(30s pulse, 60s rotation) since the dashboard polls them frequently.

Beat keeps the cache continuously warm so the user-facing GET is always a
sub-100ms cache hit. We force=True at a cadence slightly under each TTL.
"""
from __future__ import annotations

import structlog
from celery import shared_task

log = structlog.get_logger()


@shared_task
def warm_pulse() -> dict:
    from apps.market_data.services.pulse_service import build_pulse
    import time
    t0 = time.time()
    build_pulse(force=True)
    ms = int((time.time() - t0) * 1000)
    log.info("warmer.pulse", ms=ms)
    return {"builder": "pulse", "ms": ms}


@shared_task
def warm_rotation() -> dict:
    from apps.market_data.services.rotation_service import build_rotation
    import time
    t0 = time.time()
    build_rotation(force=True)
    ms = int((time.time() - t0) * 1000)
    log.info("warmer.rotation", ms=ms)
    return {"builder": "rotation", "ms": ms}
