"""Celery tasks for the mutual fund collector (scheduled in CELERY_BEAT_SCHEDULE).

Each fire writes a RunReceipt (apps.system) with counts + checks. ``trigger``
is only for the receipt ("manual" when rerun from the Runs page).
"""
from __future__ import annotations

import structlog
from celery import shared_task
from django.core.management import call_command

log = structlog.get_logger(__name__)

PULL_TASK = "apps.mutual_funds.tasks.pull_amfi_nav"
METRICS_TASK = "apps.mutual_funds.tasks.compute_mf_metrics_weekly"


@shared_task(name=PULL_TASK, ignore_result=True)
def pull_amfi_nav(trigger: str | None = None) -> None:
    """Daily AMFI NAV pull, then a stale-scheme sweep. Idempotent — safe to
    fire twice a day (late-evening publish + morning catch-up)."""
    from apps.mutual_funds.management.commands.check_stale_navs import Command as StaleCmd
    from apps.mutual_funds.management.commands.pull_amfi_nav import Command as PullCmd
    from apps.system.services.run_receipts import infer_trigger, tracked_run

    with tracked_run("import", "pull_amfi_nav", {}, trigger=infer_trigger({"trigger": trigger}),
                     task_name=PULL_TASK, primary_metric="navs_created", max_seconds=3600) as run:
        pull = PullCmd()
        call_command(pull)
        stale = StaleCmd()
        call_command(stale, limit=0)
        try:
            describe_pull(run, getattr(pull, "stats", {}), getattr(stale, "stats", {}))
        except Exception as e:  # noqa: BLE001 — receipts never break the job
            log.warning("mf_daily_pull.receipt_failed", error=str(e))
        log.info("mf_daily_pull.done")


def _expected_nav_date(now=None):
    """Latest business day AMFI should have published by now (~23:00 IST)."""
    from django.utils import timezone

    from apps.market_data.services.market_calendar import is_trading_day
    from apps.system.services.run_checks import IST, previous_trading_day

    now = (now or timezone.now()).astimezone(IST)
    today = now.date()
    if is_trading_day(today) and now.hour >= 23:
        return today
    return previous_trading_day(today)


def describe_pull(run, stats: dict, stale_stats: dict) -> None:
    from apps.mutual_funds.services.nav_backfill import catalog_date
    from apps.system.services import run_checks as rc

    run.output(**{k: v for k, v in stats.items() if k != "error"}, **stale_stats)
    run.link("Mutual funds", "/mutual-funds")
    if stats.get("error"):
        # The command logs and returns on a fetch/parse failure — without
        # this check the job would look like a clean success.
        run.check(rc.make("amfi_fetch", rc.FAIL, stats["error"]))
        return
    run.check(rc.output_count("rows_parsed", stats.get("rows_parsed"), minimum=1000,
                              label="scheme/NAV rows parsed"))
    have = catalog_date()
    want = _expected_nav_date()
    run.output(catalog_nav_date=have.isoformat() if have else None, expected_nav_date=want.isoformat())
    if have is None:
        run.check(rc.make("nav_fresh", rc.FAIL, "no NAV rows stored"))
    else:
        behind = rc.trading_days_between(have, want)
        status = rc.PASS if behind == 0 else (rc.WARN if behind == 1 else rc.FAIL)
        run.check(rc.make("nav_fresh", status,
                          f"catalog NAV date {have} (expected {want}; {behind} business day(s) behind)",
                          sessions_behind=behind))
    run.set_provenance(source="AMFI NAVAll.txt")


@shared_task(name=METRICS_TASK, ignore_result=True)
def compute_mf_metrics_weekly(trigger: str | None = None) -> None:
    """Weekly screener refresh: top up benchmark TRI, then recompute metric
    snapshots as of the catalog NAV date. NAV history itself is appended
    daily by pull_amfi_nav."""
    from apps.mutual_funds.management.commands.compute_mf_metrics import Command as MetricsCmd
    from apps.system.services import run_checks as rc
    from apps.system.services.run_receipts import infer_trigger, tracked_run

    with tracked_run("job", "compute_mf_metrics_weekly", {}, trigger=infer_trigger({"trigger": trigger}),
                     task_name=METRICS_TASK, primary_metric="snapshots", max_seconds=3 * 3600) as run:
        try:
            call_command("pull_mf_benchmarks", incremental=True)
            run.output(benchmarks_refreshed=True)
        except Exception as exc:  # noqa: BLE001 — metrics still run without fresh TRI
            log.warning("mf_benchmarks.weekly_failed", error=str(exc))
            run.swallowed_error(f"pull_mf_benchmarks: {exc}")
            run.output(benchmarks_refreshed=False)
        cmd = MetricsCmd()
        call_command(cmd)
        try:
            stats = getattr(cmd, "stats", {})
            run.output(**stats)
            run.check(rc.rows_written("metric snapshots", stats.get("snapshots")))
            run.link("MF screener", "/mutual-funds")
        except Exception as e:  # noqa: BLE001
            log.warning("mf_metrics_weekly.receipt_failed", error=str(e))
        log.info("mf_metrics_weekly.done")
