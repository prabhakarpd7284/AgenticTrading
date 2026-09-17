import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

# ──────────────────────────────────────────────────────────────────────────
# Allow sync Django ORM from inside an async event loop.
#
# Agent tasks run `asyncio.run(graph.ainvoke(...))` inside the Celery worker
# process. Every LangGraph node is `async def`, so any Django ORM call
# emitted from a node (AgentStep.objects.create in publisher.emit, Candle
# lookups in DefaultMarketData.ltp/candles, BrokerLink queries in
# options_chain, Event writes in the journal port) hits Django's
# `async_unsafe()` guard and raises SynchronousOnlyOperation. Every such
# raise is caught by an outer try/except so the run keeps going, but the
# AgentStep row never persists and the live WS stream stays silent —
# resulting in "succeeded with only seq=1 init on the UI".
#
# Setting `DJANGO_ALLOW_ASYNC_UNSAFE=true` tells Django "I know I'm
# calling sync ORM from an async context; trust me, it won't deadlock".
# Safe here because: (1) every ORM call in the agent path is a single-row
# insert or a 1-row filter — sub-millisecond — so the event loop barely
# stalls; (2) the Daphne process (which DOES host real concurrent async
# consumers) is unaffected — this env var is set in the Celery process
# only; (3) the proper fix (rewriting every ORM call as
# `sync_to_async(...)`) is a longer refactor that adds zero correctness
# benefit for this code path.
#
# If/when we move to a fully async ORM (Django 5.x has async-native APIs),
# this can come out.
# ──────────────────────────────────────────────────────────────────────────
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

app = Celery("alphadesk")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self) -> None:
    print(f"Request: {self.request!r}")


# ──────────────────────────────────────────────────────────────────────────
# v2 swing scanner — periodic tasks + beat schedule.
#
# Kept here (not in settings) so the wiring is self-contained and additive:
# defining the tasks in this always-imported module registers them, and
# `on_after_finalize` MERGES the schedule into whatever the project already
# has — it never clobbers an existing CELERY_BEAT_SCHEDULE. v1 tasks/commands
# are untouched.
#
# Tier cadence: small=15m (intraday-ish), medium=1h, long=daily; enrichment
# walks each open swing across its time-stop window once a day after close.
# ──────────────────────────────────────────────────────────────────────────
# soft_time_limit (raises inside the task → clean exit + lock release) and a
# hard time_limit bound a scan that would otherwise run for hours. With the
# day-range fetch fix a full scan is well under a minute, so 25m/30m is huge
# headroom — purely a backstop against another runaway like 2026-06-24.
@app.task(ignore_result=True, soft_time_limit=1500, time_limit=1800)
def run_swing_v2(tier: str = "medium") -> None:
    from django.core.management import call_command

    # Redis advisory lock (NX + TTL) keyed by tier: with CELERY_TASK_ACKS_LATE
    # a worker crash REDELIVERS the message, which used to relaunch an
    # already-running 11h scan from scratch. The lock makes the task
    # non-re-entrant per tier; the TTL frees it if a worker dies holding it.
    from trading.services.data_service import _get_redis_for_throttle

    lock_key = f"alphadesk:swing_v2:lock:{tier}"
    r = None
    try:
        r = _get_redis_for_throttle()
    except Exception:
        r = None
    if r is not None:
        try:
            if not r.set(lock_key, "1", nx=True, ex=1800):
                logger.info("run_swing_v2(%s) skipped — already running", tier)
                return
        except Exception:
            r = None  # lock unavailable → run without it rather than skip silently
    try:
        call_command("run_ok_scanner_v2", tier=tier, actionable_only=True)
    finally:
        if r is not None:
            try:
                r.delete(lock_key)
            except Exception:
                pass


@app.task(ignore_result=True, soft_time_limit=900, time_limit=1200)
def enrich_swing_v2() -> None:
    from django.core.management import call_command
    call_command("enrich_signals_v2")


@app.on_after_finalize.connect
def _register_swing_v2_beat(sender, **_kwargs) -> None:
    from celery.schedules import crontab
    # Times are IST (Celery uses settings.TIME_ZONE). NSE session 09:15–15:30.
    sender.add_periodic_task(
        crontab(minute="*/30", hour="9-15"),
        run_swing_v2.s("small"), name="swing_v2_small",
    )
    sender.add_periodic_task(
        crontab(minute=5, hour="10,12,14"),
        run_swing_v2.s("medium"), name="swing_v2_medium",
    )
    sender.add_periodic_task(
        crontab(minute=45, hour=15),
        run_swing_v2.s("long"), name="swing_v2_long",
    )
    sender.add_periodic_task(
        crontab(minute=0, hour=16),
        enrich_swing_v2.s(), name="enrich_swing_v2",
    )
