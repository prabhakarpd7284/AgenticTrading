"""R2 run work: after-close capture jobs, schedule registry additions (swing v2
+ capture), stuck-run sweeper, and AgentRun / Backtest receipts."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest import mock
from zoneinfo import ZoneInfo

import pytest
from django.utils import timezone

from apps.market_data.models import Candle, Symbol
from apps.system.models import PipelineRun, RunReceipt
from apps.system.services import run_schedule

IST = ZoneInfo("Asia/Kolkata")
SESSION = date(2026, 10, 21)          # a Wednesday, NSE trading day
PREV = date(2026, 10, 19)             # 2026-10-20 is an NSE holiday (Dussehra)
NO_TELEGRAM = mock.patch("apps.system.services.run_alerts._send_telegram", return_value=False)


@pytest.fixture(autouse=True)
def _no_leaked_receipts(request):
    """Async WS tests (test_scalp_sim_ws) create AgentRuns on a separate
    connection that commits, so their mirrored receipts outlive the test.
    Start every receipt test from an empty table (rolled back afterwards)."""
    if "django_db" in request.keywords or "db" in request.fixturenames:
        request.getfixturevalue("db")
        RunReceipt.objects.all().delete()


def _bars(d: date, n: int) -> list[list]:
    start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
    return [[(start + timedelta(minutes=i)).isoformat(), 1, 2, 0.5, 1.5, 10] for i in range(n)]


class FakeBroker:
    def __init__(self, bars_by_token):
        self.bars_by_token = bars_by_token
        self.calls = []

    def fetch_candles(self, token, start, end, interval, exchange=None, wait_on_breaker=False):
        self.calls.append((token, start, end, interval, wait_on_breaker))
        return self.bars_by_token.get(token, [])


def _capture_env(bars_by_token):
    nifty = Symbol.objects.create(exchange="NSE", token="99926000", tradingsymbol="NIFTY", segment="INDEX")
    rel = Symbol.objects.create(exchange="NSE", token="2885", tradingsymbol="RELIANCE", segment="EQ")
    rows = {"NIFTY": nifty, "RELIANCE": rel}
    broker = FakeBroker(bars_by_token)
    patches = [
        mock.patch("apps.market_data.management.commands.backfill_candles.resolve_targets",
                   return_value=([("NIFTY", "NSE", "99926000"), ("RELIANCE", "NSE", "2885")], ["GHOST"])),
        mock.patch("apps.market_data.management.commands.backfill_candles.ensure_symbol_row",
                   side_effect=lambda name, ex, tok: rows[name]),
        mock.patch("trading.services.data_service.BrokerClient.get_instance", return_value=broker),
    ]
    return patches, broker


# ─── after-close candle capture ──────────────────────────────────────────

@pytest.mark.django_db
def test_candle_capture_full_session_passes():
    from apps.market_data.tasks.eod_capture import capture_session_candles

    patches, broker = _capture_env({"99926000": _bars(SESSION, 375), "2885": _bars(SESSION, 375)})
    with patches[0], patches[1], patches[2]:
        capture_session_candles(session=SESSION.isoformat(), trigger="manual")
    r = RunReceipt.objects.get(name="eod_candle_capture")
    assert r.status == "success"
    assert r.outputs["bars_stored"] == 750
    assert r.outputs["angel_calls"] == 2
    bars = next(c for c in r.checks if c["name"] == "bars_per_symbol")
    assert bars["status"] == "pass"
    assert any("GHOST" in w for w in r.warnings)
    assert broker.calls[0][1:4] == (f"{SESSION} 09:15", f"{SESSION} 15:30", "ONE_MINUTE")
    assert all(call[4] for call in broker.calls)          # waits on the shared breaker
    assert Candle.objects.filter(interval="1m").count() == 750


@pytest.mark.django_db
def test_candle_capture_short_symbol_fails_receipt():
    from apps.market_data.tasks.eod_capture import capture_session_candles

    patches, _ = _capture_env({"99926000": _bars(SESSION, 375), "2885": _bars(SESSION, 120)})
    with patches[0], patches[1], patches[2], NO_TELEGRAM:
        capture_session_candles(session=SESSION.isoformat())
    r = RunReceipt.objects.get(name="eod_candle_capture")
    assert r.status == "failed"
    bars = next(c for c in r.checks if c["name"] == "bars_per_symbol")
    assert bars["status"] == "fail"
    assert "RELIANCE=120" in bars["message"]


@pytest.mark.django_db
def test_candle_capture_skips_non_trading_day(monkeypatch):
    from apps.market_data.tasks.eod_capture import capture_session_candles

    monkeypatch.setattr("apps.market_data.services.market_calendar.is_trading_day", lambda d=None: False)
    assert capture_session_candles() == {"skipped": "not a trading day"}
    assert RunReceipt.objects.get(name="eod_candle_capture").status == "skipped"


# ─── bhavcopy capture ────────────────────────────────────────────────────

def _bhav_rows(n):
    return [{"FinInstrmId": str(1000 + i), "TckrSymb": f"S{i}", "FinInstrmNm": f"S{i}", "SctySrs": "EQ",
             "OpnPric": "1", "HghPric": "2", "LwPric": "0.5", "ClsPric": "1.5", "TtlTradgVol": "100"}
            for i in range(n)]


@pytest.mark.django_db
def test_bhavcopy_not_published_is_partial_then_failed_on_retry():
    from apps.market_data.tasks.eod_capture import capture_bhavcopy

    with mock.patch("apps.market_data.management.commands.backfill_bhavcopy.download", return_value=None), \
            NO_TELEGRAM:
        capture_bhavcopy(attempt=1, session=SESSION.isoformat())
        capture_bhavcopy(attempt=2, session=SESSION.isoformat())
    first, retry = RunReceipt.objects.filter(name="eod_bhavcopy_capture").order_by("started_at")
    assert first.status == "partial"
    assert retry.status == "failed"
    assert "not published" in retry.error


@pytest.mark.django_db
def test_bhavcopy_row_count_compared_with_previous_session():
    from apps.market_data.management.commands.backfill_bhavcopy import write_day
    from apps.market_data.tasks.eod_capture import capture_bhavcopy

    write_day(PREV, _bhav_rows(100))
    with mock.patch("apps.market_data.management.commands.backfill_bhavcopy.download", return_value=b"zip"), \
            mock.patch("apps.market_data.management.commands.backfill_bhavcopy.parse",
                       return_value=_bhav_rows(98)):
        capture_bhavcopy(attempt=1, session=SESSION.isoformat())
    r = RunReceipt.objects.get(name="eod_bhavcopy_capture")
    assert r.status == "success"
    assert r.outputs["rows"] == 98
    assert r.outputs["previous_session"] == PREV.isoformat()
    check = next(c for c in r.checks if c["name"] == "bhavcopy_rows")
    assert check["status"] == "pass"

    # A second run (the 20:15 retry) finds the day already stored and re-checks it.
    with NO_TELEGRAM:
        capture_bhavcopy(attempt=2, session=SESSION.isoformat())
    retry = RunReceipt.objects.filter(name="eod_bhavcopy_capture").order_by("-started_at").first()
    assert retry.outputs["mode"] == "already present"
    assert retry.status == "success"


def test_bhav_count_check_thresholds():
    from apps.market_data.services.session_capture import check_bhav_count

    assert check_bhav_count(2900, 2920, PREV)["status"] == "pass"
    assert check_bhav_count(2600, 2920, PREV)["status"] == "warn"
    assert check_bhav_count(1000, 2920, PREV)["status"] == "fail"
    assert check_bhav_count(0, 2920, PREV)["status"] == "fail"
    assert check_bhav_count(2900, 0, None)["status"] == "info"


# ─── schedule registry ──────────────────────────────────────────────────

def _entry(key):
    return next(e for e in run_schedule.registry() if e.key == key)


def test_registry_includes_capture_jobs_and_swing_v2():
    keys = {e.key for e in run_schedule.registry()}
    assert {"eod-candle-capture", "eod-bhavcopy-capture", "eod-bhavcopy-capture-retry"} <= keys
    assert {"swing_v2_small", "swing_v2_medium", "swing_v2_long", "enrich_swing_v2"} <= keys
    assert all(e.tracked for e in run_schedule.registry())
    small = _entry("swing_v2_small")
    assert small.match_task == "config.celery.run_swing_v2[small]"
    assert small.receipt_name == "swing_v2.small"
    assert len(small.due_times(SESSION)) == 14                   # 09:00 … 15:30 every 30 min
    assert len(small.due_times(date(2026, 10, 24))) == 14        # weekends too (unchanged)
    assert _entry("eod-candle-capture").due_times(SESSION) == [datetime(2026, 10, 21, 15, 50, tzinfo=IST)]
    assert _entry("eod-candle-capture").due_times(date(2026, 10, 24)) == []


@pytest.mark.django_db
def test_swing_v2_tiers_match_their_own_slots():
    now = datetime(2026, 10, 22, 11, 0, tzinfo=IST)
    RunReceipt.objects.create(kind="scan", name="swing_v2.medium", task_name="config.celery.run_swing_v2[medium]",
                              status="success", started_at=datetime(2026, 10, 22, 10, 5, 3, tzinfo=IST))
    RunReceipt.objects.create(kind="scan", name="swing_v2.small", task_name="config.celery.run_swing_v2[small]",
                              status="success", started_at=datetime(2026, 10, 22, 9, 0, 2, tzinfo=IST))
    slots = {(s["key"], s["due_at"]): s for s in run_schedule.schedule_for_day(date(2026, 10, 22), now=now)}
    at = lambda h, m: datetime(2026, 10, 22, h, m, tzinfo=IST).isoformat()  # noqa: E731
    assert slots[("swing_v2_small", at(9, 0))]["state"] == "ok"
    assert slots[("swing_v2_medium", at(10, 5))]["state"] == "ok"
    # The 10:05 medium run must not satisfy the 10:00 small slot.
    assert slots[("swing_v2_small", at(10, 0))]["state"] == "missed"
    # Before the entry's tracking start, a slot is untracked — not a false "missed".
    old = {(s["key"], s["due_at"]): s for s in run_schedule.schedule_for_day(SESSION, now=now)}
    assert old[("swing_v2_small", datetime(2026, 10, 21, 10, 0, tzinfo=IST).isoformat())]["state"] == "untracked"


@pytest.mark.django_db
def test_run_swing_v2_writes_receipt(monkeypatch):
    from config.celery import run_swing_v2

    monkeypatch.setattr("django.core.management.call_command", lambda *a, **k: None)
    monkeypatch.setattr("trading.services.data_service._get_redis_for_throttle", lambda: None)
    run_swing_v2("small")
    r = RunReceipt.objects.get(name="swing_v2.small")
    assert r.task_name == "config.celery.run_swing_v2[small]"
    assert r.status == "success"
    assert r.outputs["signals"] == 0


# ─── stuck-run sweeper ──────────────────────────────────────────────────

@pytest.mark.django_db
def test_sweeper_fails_stuck_pipeline_runs_only():
    now = timezone.now()
    stuck = PipelineRun.objects.create(task="swing_scan")
    fresh = PipelineRun.objects.create(task="swing_scan")
    screener = PipelineRun.objects.create(task="screener_session")
    PipelineRun.objects.filter(id=stuck.id).update(started_at=now - timedelta(days=1))
    PipelineRun.objects.filter(id=screener.id).update(started_at=now - timedelta(hours=5))
    out = run_schedule.sweep_stuck_details(now)
    assert out["pipeline_runs_failed"] == 1
    stuck.refresh_from_db()
    fresh.refresh_from_db()
    screener.refresh_from_db()
    assert stuck.status == "failed"
    assert "Never finished" in stuck.error
    assert stuck.finished_at is not None
    assert fresh.status == "running"
    assert screener.status == "running"       # 5 h is inside a screener session's 7.5 h budget


# ─── AgentRun / Backtest receipts ───────────────────────────────────────

@pytest.mark.django_db
def test_failed_agent_run_becomes_failed_receipt_and_alerts(paper_portfolio, owner):
    from apps.agents_core.models import AgentRun

    run = AgentRun.objects.create(tenant=paper_portfolio.tenant, strategy_name="scalp", strategy_version="1",
                                  portfolio=paper_portfolio, triggered_by=owner)
    r = RunReceipt.objects.get(detail_kind="agent_run", detail_id=str(run.id))
    assert r.status == "queued"
    assert r.tenant_id == paper_portfolio.tenant_id
    with NO_TELEGRAM:
        run.status = AgentRun.Status.FAILED
        run.error = "boom"
        run.completed_at = timezone.now()
        run.save()
    r.refresh_from_db()
    assert r.status == "failed"
    assert r.kind == "agent"
    assert r.name == "agent.scalp"
    assert r.error == "boom"
    assert r.alerted_at is not None
    assert RunReceipt.objects.filter(detail_kind="agent_run").count() == 1


@pytest.mark.django_db
def test_expired_agent_runs_surface_as_failed_receipts(paper_portfolio, owner):
    from apps.agents_core.models import AgentRun
    from apps.agents_core.tasks.housekeeping import expire_runs

    run = AgentRun.objects.create(tenant=paper_portfolio.tenant, strategy_name="scalp", strategy_version="1",
                                  portfolio=paper_portfolio, triggered_by=owner, status="running")
    AgentRun.objects.filter(id=run.id).update(created_at=timezone.now() - timedelta(hours=1))
    with NO_TELEGRAM:
        assert expire_runs() == 1
    r = RunReceipt.objects.get(detail_kind="agent_run", detail_id=str(run.id))
    assert r.status == "failed"
    assert r.error.startswith("expired:")


@pytest.mark.django_db
def test_backtest_receipts_flag_placeholder_metrics_and_stuck_rows(paper_portfolio):
    from apps.strategies.models import Backtest, StrategyInstance

    tenant = paper_portfolio.tenant
    inst = StrategyInstance.objects.create(tenant=tenant, strategy_name="swing_scanner", strategy_version="1",
                                           name="OK swing")
    done = Backtest.objects.create(tenant=tenant, instance=inst, from_date=date(2026, 1, 1),
                                   to_date=date(2026, 6, 30), status="done",
                                   metrics={"sharpe": 0.0, "maxdd": 0.0, "trades": 0})
    r = RunReceipt.objects.get(detail_kind="backtest", detail_id=str(done.id))
    assert r.status == "success"
    assert r.name == "backtest.swing_scanner"
    assert next(c for c in r.checks if c["name"] == "real_metrics")["status"] == "warn"

    stuck = Backtest.objects.create(tenant=tenant, instance=inst, from_date=date(2026, 1, 1),
                                    to_date=date(2026, 6, 30), status="running")
    Backtest.objects.filter(id=stuck.id).update(updated_at=timezone.now() - timedelta(hours=8),
                                                created_at=timezone.now() - timedelta(hours=8))
    with NO_TELEGRAM:
        assert run_schedule.sweep_stuck_details()["backtests_failed"] == 1
    stuck.refresh_from_db()
    assert stuck.status == "failed"
    assert RunReceipt.objects.get(detail_kind="backtest", detail_id=str(stuck.id)).status == "failed"


def test_bars_check_message():
    from apps.market_data.services.session_capture import check_bars

    assert check_bars({"A": 375, "B": 361})["status"] == "pass"
    assert check_bars({})["status"] == "fail"
    assert check_bars({"A": 375, "B": 200})["short"] == {"B": 200}




@pytest.mark.django_db
def test_backdated_agent_receipts_do_not_turn_old_slots_into_missed(paper_portfolio, owner):
    """Regression (2026-10-21): mirroring a 3-day-old agent run pulled the
    watcher's "tracking since" back 3 days and every earlier slot became a
    false not_run + Telegram alert."""
    from apps.agents_core.models import AgentRun

    now = datetime(2026, 10, 22, 22, 0, tzinfo=IST)
    RunReceipt.objects.create(kind="job", name="nightly_scan", task_name="apps.news.tasks.run_nightly_scan",
                              status="success", started_at=datetime(2026, 10, 22, 21, 0, 5, tzinfo=IST))
    with NO_TELEGRAM:
        AgentRun.objects.create(tenant=paper_portfolio.tenant, strategy_name="scalp", strategy_version="1",
                                portfolio=paper_portfolio, triggered_by=owner, status="failed", error="expired",
                                started_at=datetime(2026, 10, 18, 12, 0, tzinfo=IST))
        out = run_schedule.watch(now=now)
    assert out["not_run_created"] == 0
    slots = {s["key"]: s for s in run_schedule.schedule_for_day(date(2026, 10, 22), now=now)}
    assert slots["daily-swing-scan"]["state"] == "untracked"


@pytest.mark.django_db
def test_watcher_backfills_missing_detail_receipts_without_alerting(paper_portfolio, owner):
    from apps.agents_core.models import AgentRun

    run = AgentRun.objects.create(tenant=paper_portfolio.tenant, strategy_name="scalp", strategy_version="1",
                                  portfolio=paper_portfolio, triggered_by=owner)
    AgentRun.objects.filter(id=run.id).update(status="failed", error="expired")   # bypasses signals
    RunReceipt.objects.filter(detail_kind="agent_run").delete()                      # pre-receipt row
    with mock.patch("apps.system.services.run_alerts._send_telegram") as tg:
        out = run_schedule.watch()
        again = run_schedule.watch()
    assert out["detail_receipts_created"] >= 1   # ≥: committed rows from async WS tests count too
    assert again["detail_receipts_created"] == 0
    tg.assert_not_called()
    r = RunReceipt.objects.get(detail_kind="agent_run", detail_id=str(run.id))
    assert r.status == "failed"
    assert r.alerted_at is not None
