"""Run receipts (UX redesign R1): tracked_run, checks, schedule registry, watcher."""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from apps.system.models import RunReceipt
from apps.system.services import run_checks as rc
from apps.system.services import run_schedule
from apps.system.services.run_receipts import current_run, tracked_run, tracked_task

IST = ZoneInfo("Asia/Kolkata")


# ─── tracked_run ─────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_success_receipt_records_outputs_checks_and_links():
    with tracked_run("job", "demo", {"x": 1}, task_name="t.demo") as run:
        run.output(rows=5)
        run.check(rc.rows_written("rows", 5))
        run.link("Somewhere", "/somewhere")
        assert current_run() is run
    r = RunReceipt.objects.get()
    assert r.status == "success"
    assert r.inputs == {"x": 1}
    assert r.outputs["rows"] == 5
    assert r.outputs["links"] == [{"label": "Somewhere", "href": "/somewhere"}]
    assert r.finished_at and r.duration_seconds is not None
    assert {c["name"] for c in r.checks} >= {"rows_written", "duration_anomaly"}
    assert r.tenant_id is None


@pytest.mark.django_db
def test_exception_propagates_and_receipt_is_failed():
    with pytest.raises(ValueError, match="bad input"):
        with tracked_run("job", "boom"):
            raise ValueError("bad input")
    r = RunReceipt.objects.get()
    assert r.status == "failed"
    assert "ValueError: bad input" in r.error


@pytest.mark.django_db
def test_failed_check_makes_run_failed_and_alerts_once():
    with mock.patch("apps.system.services.run_alerts._send_telegram", return_value=True) as tg:
        with tracked_run("job", "zero") as run:
            run.check(rc.rows_written("rows", 0))
    r = RunReceipt.objects.get()
    assert r.status == "failed"
    assert "rows_written" in r.error
    assert r.alerted_at is not None
    tg.assert_called_once()


@pytest.mark.django_db
def test_units_failed_gives_partial_and_counts_swallowed_errors():
    with tracked_run("job", "units") as run:
        run.unit_ok(3)
        run.unit_failed(error="one broke")
    r = RunReceipt.objects.get()
    assert r.status == "partial"
    sw = next(c for c in r.checks if c["name"] == "swallowed_errors")
    assert sw["status"] == "warn" and sw["count"] == 1


@pytest.mark.django_db
def test_receipt_write_failure_never_breaks_the_job():
    with mock.patch.object(RunReceipt.objects, "create", side_effect=RuntimeError("db down")):
        with tracked_run("job", "resilient") as run:
            run.output(rows=1)
            value = 42
    assert value == 42
    assert RunReceipt.objects.count() == 0


@pytest.mark.django_db
def test_current_run_outside_tracking_is_a_noop():
    h = current_run()
    h.output(a=1)
    h.check(rc.make("x", rc.FAIL, "ignored"))
    assert RunReceipt.objects.count() == 0


@pytest.mark.django_db
def test_tracked_task_decorator_skipped_and_trigger():
    @tracked_task("job", "deco", task_name="t.deco")
    def task(force=False):
        return {"skipped": "not a trading day"} if not force else {"ok": True}

    assert task() == {"skipped": "not a trading day"}
    assert task(force=True) == {"ok": True}
    skipped, ran = RunReceipt.objects.order_by("started_at")
    assert skipped.status == "skipped" and skipped.trigger == "beat"
    assert skipped.outputs["skipped"] == "not a trading day"
    assert ran.status == "success" and ran.trigger == "manual"


@pytest.mark.django_db
def test_zero_output_anomaly_against_history():
    for _ in range(5):
        with tracked_run("job", "scan", primary_metric="signals") as run:
            run.output(signals=4)
    with tracked_run("job", "scan", primary_metric="signals") as run:
        run.output(signals=0)
    last = RunReceipt.objects.order_by("-started_at").first()
    chk = next(c for c in last.checks if c["name"] == "zero_output_anomaly")
    assert chk["status"] == "warn" and chk["median"] == 4
    assert last.status == "success"   # an anomaly warns; it doesn't fail the run


# ─── checks library ─────────────────────────────────────────────────────

def test_output_count_and_duration_checks():
    assert rc.output_count("n", 50, expected=50)["status"] == "pass"
    assert rc.output_count("n", 30, expected=50)["status"] == "warn"
    assert rc.output_count("n", 0, expected=50)["status"] == "fail"
    assert rc.duration_anomaly(500, [10, 12, 11])["status"] == "warn"
    assert rc.duration_anomaly(11, [10, 12, 11])["status"] == "pass"
    assert rc.duration_anomaly(11, [10])["status"] == "info"


@pytest.mark.django_db
def test_broker_live_ignores_paper_and_inactive_links(tenant, owner):
    from tests.factories import BrokerLinkFactory

    BrokerLinkFactory(tenant=tenant, owner=owner, broker_name="paper", status="active")
    BrokerLinkFactory(tenant=tenant, owner=owner, broker_name="zerodha", status="expired")
    assert rc.broker_live(tenant)["status"] == "fail"
    assert rc.broker_live(tenant, required=False)["status"] == "warn"
    BrokerLinkFactory(tenant=tenant, owner=owner, broker_name="angel_one", status="active")
    assert rc.broker_live(tenant)["status"] == "pass"


@pytest.mark.django_db
def test_candle_freshness_sessions_behind():
    from apps.market_data.models import Candle, Symbol

    sym = Symbol.objects.create(exchange="NSE", token="1", tradingsymbol="X")
    # Wed 2026-10-14 16:00 IST → expected session is 14 Oct.
    now = datetime(2026, 10, 14, 16, 0, tzinfo=IST)
    assert rc.candle_freshness("1d", now=now)["status"] == "fail"   # none stored
    Candle.objects.create(symbol=sym, interval="1d", t=datetime(2026, 10, 13, tzinfo=IST),
                          o=Decimal(1), h=Decimal(1), l=Decimal(1), c=Decimal(1), v=1)
    assert rc.candle_freshness("1d", now=now)["status"] == "warn"   # 1 session behind
    Candle.objects.create(symbol=sym, interval="1d", t=datetime(2026, 10, 14, tzinfo=IST),
                          o=Decimal(1), h=Decimal(1), l=Decimal(1), c=Decimal(1), v=1)
    assert rc.candle_freshness("1d", now=now)["status"] == "pass"


# ─── schedule registry + watcher ────────────────────────────────────────

def _entry(key):
    return next(e for e in run_schedule.registry() if e.key == key)


def test_registry_built_from_beat_crontabs_only():
    keys = {e.key for e in run_schedule.registry()}
    assert {"nightly-scan", "mf-nav-pull-night", "mf-nav-pull-morning", "daily-swing-scan"} <= keys
    assert "process-order-outbox" not in keys          # interval heartbeat, not a job
    assert "watch-run-schedule" not in keys
    assert all(e.tracked for e in run_schedule.registry())


def test_due_times_respect_day_of_week():
    swing = _entry("daily-swing-scan")
    assert swing.due_times(datetime(2026, 10, 14).date()) == [datetime(2026, 10, 14, 8, 20, tzinfo=IST)]
    assert swing.due_times(datetime(2026, 10, 17).date()) == []   # Saturday
    assert len(_entry("nightly-scan").due_times(datetime(2026, 10, 17).date())) == 1


def _receipt(task, started, status="success", **kw):
    return RunReceipt.objects.create(kind="job", name=run_schedule.TRACKED_TASKS[task], task_name=task,
                                     status=status, started_at=started, finished_at=started, **kw)


@pytest.mark.django_db
def test_schedule_states_ok_missed_pending_and_untracked():
    day = datetime(2026, 10, 14).date()   # Wednesday
    now = datetime(2026, 10, 14, 22, 0, tzinfo=IST)
    # Tracking started this morning, before the 08:20 swing scan.
    _receipt("apps.trading.tasks.reconcile.reconcile_stale_intraday",
             datetime(2026, 10, 14, 8, 10, 30, tzinfo=IST))
    _receipt("apps.news.tasks.run_nightly_scan", datetime(2026, 10, 14, 21, 0, 2, tzinfo=IST))
    slots = {s["key"]: s for s in run_schedule.schedule_for_day(day, now=now)}
    assert slots["premarket-reconcile-stale-intraday"]["state"] == "ok"
    assert slots["nightly-scan"]["state"] == "ok"
    assert slots["daily-swing-scan"]["state"] == "missed"
    assert slots["mf-nav-pull-night"]["state"] == "pending"
    assert slots["mf-nav-pull-morning"]["state"] == "untracked"   # 07:30 predates tracking
    summary = run_schedule.summarize(list(slots.values()))
    assert summary["health"] == "red" and summary["missed"] >= 1


@pytest.mark.django_db
def test_watcher_writes_not_run_once_and_fails_stuck_runs():
    now = datetime(2026, 10, 14, 10, 0, tzinfo=IST)
    _receipt("apps.trading.tasks.reconcile.reconcile_stale_intraday",
             datetime(2026, 10, 14, 8, 10, 30, tzinfo=IST))
    stuck = _receipt("apps.news.tasks.run_nightly_scan", now - timedelta(hours=5), status="running",
                     max_seconds=3600)
    stuck.finished_at = None
    stuck.save()
    with mock.patch("apps.system.services.run_alerts._send_telegram", return_value=False):
        first = run_schedule.watch(now=now)
        second = run_schedule.watch(now=now)
    assert first["not_run_created"] >= 1 and first["stuck_failed"] == 1
    assert second == {"not_run_created": 0, "stuck_failed": 0}
    nr = RunReceipt.objects.get(status="not_run", schedule_key="daily-swing-scan")
    assert nr.trigger == "watcher" and nr.alerted_at is not None
    stuck.refresh_from_db()
    assert stuck.status == "failed"
    slots = {s["key"]: s for s in run_schedule.schedule_for_day(now.date(), now=now)}
    assert slots["daily-swing-scan"]["state"] == "failed"


# ─── wrapped jobs ────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_amfi_pull_fetch_failure_is_a_failed_receipt(monkeypatch):
    from apps.mutual_funds.tasks import pull_amfi_nav

    def boom():
        raise RuntimeError("AMFI unreachable")

    monkeypatch.setattr("apps.mutual_funds.management.commands.pull_amfi_nav.fetch_nav_text", boom)
    pull_amfi_nav()   # the command swallows the error — the receipt must not
    r = RunReceipt.objects.get(name="pull_amfi_nav")
    assert r.status == "failed"
    assert any(c["name"] == "amfi_fetch" and c["status"] == "fail" for c in r.checks)
    assert r.task_name == "apps.mutual_funds.tasks.pull_amfi_nav"


@pytest.mark.django_db
def test_cooldown_refresh_writes_receipt(paper_portfolio):
    from apps.trading.tasks.cooldown import refresh_cooldowns

    refresh_cooldowns(trigger="manual")
    r = RunReceipt.objects.get(name="cooldown_refresh")
    assert r.trigger == "manual"
    assert r.outputs["portfolios"] >= 1
    assert r.status in ("success", "partial")


@pytest.mark.django_db
def test_nightly_scan_skip_writes_skipped_receipt(monkeypatch):
    from apps.news import tasks

    monkeypatch.setattr("apps.market_data.services.market_calendar.is_trading_day", lambda d=None: False)
    assert tasks.run_nightly_scan() == {"skipped": "next day is not a trading day"}
    r = RunReceipt.objects.get(name="nightly_scan")
    assert r.status == "skipped" and r.task_name == tasks.TASK_NAME


@pytest.mark.django_db
def test_pipeline_task_links_pipeline_run(monkeypatch):
    from apps.strategies.tasks import daily_pipeline
    from apps.system.models import PipelineRun

    monkeypatch.setattr("django.core.management.call_command", lambda *a, **k: None)
    daily_pipeline.run_eod_enrichment(force=True)
    r = RunReceipt.objects.get(name="pipeline.eod_enrichment")
    pr = PipelineRun.objects.get()
    assert r.detail_kind == "pipeline_run" and r.detail_id == str(pr.id)
    assert r.trigger == "manual" and r.outputs["mode"] == "today"
    # No real broker link in the test DB → the broker check fails the receipt.
    assert any(c["name"] == "broker_live" and c["status"] == "fail" for c in r.checks)
    assert r.status == "failed"
    assert pr.status == "success"   # the detail row is untouched by receipt logic
