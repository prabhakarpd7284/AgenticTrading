"""Expected-schedule registry (UX redesign plan §8.3).

Built from ``settings.CELERY_BEAT_SCHEDULE`` — every **crontab** entry is a
scheduled job with due times we can compute for any day. Interval entries
(the 1 s outbox poll, 60 s snapshots, …) are heartbeats, not jobs, and are
left out. Each entry gets a grace window; a slot whose window has passed
with no receipt for that task is *missed* — which is exactly how "beat
never fired" gets caught.

Slot states (UI colours):
  ok       green  — a success/skipped receipt in the window
  warn     amber  — partial, or success with warn checks
  failed   red    — failed receipt, or a ``not_run`` receipt
  missed   red    — window passed, no receipt (watcher turns it into not_run)
  running  blue   — receipt still running
  pending  grey   — not due yet / inside the window, nothing yet
  untracked grey  — job is not wrapped in tracked_run, or the slot predates
                    receipt tracking on this install
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Celery task name → receipt name, for every task wrapped in tracked_run.
TRACKED_TASKS: dict[str, str] = {
    "apps.news.tasks.run_nightly_scan": "nightly_scan",
    "apps.strategies.tasks.daily_pipeline.run_swing_scan": "pipeline.swing_scan",
    "apps.strategies.tasks.daily_pipeline.run_premarket_basket": "pipeline.premarket_basket",
    "apps.strategies.tasks.daily_pipeline.run_screener_session": "pipeline.screener_session",
    "apps.strategies.tasks.daily_pipeline.run_eod_enrichment": "pipeline.eod_enrichment",
    "apps.strategies.tasks.daily_pipeline.derive_intraday_trades": "pipeline.derive_trades",
    "apps.mutual_funds.tasks.pull_amfi_nav": "pull_amfi_nav",
    "apps.mutual_funds.tasks.compute_mf_metrics_weekly": "compute_mf_metrics_weekly",
    "apps.trading.tasks.cooldown.refresh_cooldowns": "cooldown_refresh",
    "apps.trading.tasks.reconcile.reconcile_stale_intraday": "reconcile_stale_intraday",
    "apps.market_data.tasks.eod_capture.capture_session_candles": "eod_candle_capture",
    "apps.market_data.tasks.eod_capture.capture_bhavcopy": "eod_bhavcopy_capture",
    # Swing v2 (config/celery.py): one Celery task serves three tiers, so
    # receipts carry a per-tier task_name (see swing_v2_receipt_task).
    "config.celery.run_swing_v2[small]": "swing_v2.small",
    "config.celery.run_swing_v2[medium]": "swing_v2.medium",
    "config.celery.run_swing_v2[long]": "swing_v2.long",
    "config.celery.enrich_swing_v2": "swing_v2.enrich",
}

# config/celery.py merges the swing-v2 periodic tasks into the same dict at
# app finalize time (add_periodic_task), so they appear in CELERY_BEAT_SCHEDULE
# only in processes where Celery finalized. Skip those runtime copies and add
# the entries from the module-level ``SWING_V2_BEAT`` declaration instead, so
# every process sees the same registry.
RUNTIME_ADDED_PREFIX = "config.celery."

# Per-entry presentation + window. Anything not listed gets the defaults.
DEFAULT_GRACE_MIN = 30
ENTRY_META: dict[str, dict] = {
    "premarket-reconcile-stale-intraday": {"label": "Premarket stale-intraday sweep"},
    "eod-reconcile-stale-intraday": {"label": "EOD stale-intraday sweep"},
    "daily-swing-scan": {"label": "Swing scan (Oliver Kell)"},
    "daily-premarket-basket": {"label": "Premarket basket"},
    "daily-screener-session": {"label": "Live screener session", "grace_min": 20},
    "daily-eod-enrichment": {"label": "EOD signal enrichment"},
    "daily-derive-trades": {"label": "Derive paper trades"},
    "mf-nav-pull-night": {"label": "AMFI NAV pull (night)", "grace_min": 45},
    "mf-nav-pull-morning": {"label": "AMFI NAV pull (morning)", "grace_min": 45},
    "mf-metrics-weekly": {"label": "MF metrics (weekly)", "grace_min": 60},
    "daily-cooldown-refresh": {"label": "Cooldown refresh"},
    "nightly-scan": {"label": "Nightly scan", "grace_min": 45},
    # "since": the day an entry started writing receipts. Earlier slots are
    # "untracked", not "missed" — the job did not exist / was not wrapped yet.
    "eod-candle-capture": {"label": "After-close 1m candle capture", "grace_min": 40, "since": "2026-10-22"},
    "eod-bhavcopy-capture": {"label": "Bhavcopy capture", "grace_min": 60, "since": "2026-10-22"},
    "eod-bhavcopy-capture-retry": {"label": "Bhavcopy capture (retry)", "grace_min": 40,
                                   "since": "2026-10-22"},
    "swing_v2_small": {"label": "Swing v2 scan (small tier)", "grace_min": 20, "since": "2026-10-22"},
    "swing_v2_medium": {"label": "Swing v2 scan (medium tier)", "since": "2026-10-22"},
    "swing_v2_long": {"label": "Swing v2 scan (long tier)", "since": "2026-10-22"},
    "enrich_swing_v2": {"label": "Swing v2 enrichment", "since": "2026-10-22"},
}

# Jobs the Runs page may re-trigger. Only idempotent, bounded, paper-safe
# tasks: the live screener session and trade derivation are excluded (long-
# running / creates trades) and stay on the /pipeline page.
RERUNNABLE: dict[str, dict] = {
    "nightly_scan": {"task": "apps.news.tasks.run_nightly_scan", "kwargs": {"force": True},
                     "note": "Idempotent per trading day; re-scores news with Claude and sends Telegram."},
    "pipeline.swing_scan": {"task": "apps.strategies.tasks.daily_pipeline.run_swing_scan",
                            "kwargs": {"force": True}, "note": "Scans NIFTY 100 (broker calls)."},
    "pipeline.premarket_basket": {"task": "apps.strategies.tasks.daily_pipeline.run_premarket_basket",
                                  "kwargs": {"force": True}, "note": "Scan only; never executes."},
    "pipeline.eod_enrichment": {"task": "apps.strategies.tasks.daily_pipeline.run_eod_enrichment",
                                "kwargs": {"force": True}, "note": "Enriches today's signals."},
    "pull_amfi_nav": {"task": "apps.mutual_funds.tasks.pull_amfi_nav",
                      "kwargs": {"trigger": "manual"}, "note": "Idempotent AMFI download (~1 min)."},
    "compute_mf_metrics_weekly": {"task": "apps.mutual_funds.tasks.compute_mf_metrics_weekly",
                                  "kwargs": {"trigger": "manual"},
                                  "note": "Heavy: several minutes of CPU/RAM."},
    "cooldown_refresh": {"task": "apps.trading.tasks.cooldown.refresh_cooldowns",
                         "kwargs": {"trigger": "manual"}, "note": "Cheap DB recompute."},
    "eod_candle_capture": {"task": "apps.market_data.tasks.eod_capture.capture_session_candles",
                           "kwargs": {"trigger": "manual"},
                           "note": "Latest closed session's 1m candles, NIFTY 50 + indices (~58 Angel calls)."},
    "eod_bhavcopy_capture": {"task": "apps.market_data.tasks.eod_capture.capture_bhavcopy",
                             "kwargs": {"trigger": "manual", "attempt": 2},
                             "note": "Latest closed session's NSE bhavcopy (one download)."},
}


@dataclass(frozen=True)
class ScheduleEntry:
    key: str
    task: str
    label: str
    schedule: object        # celery crontab
    grace: timedelta
    receipt_name: str | None
    receipt_task: str = ""  # RunReceipt.task_name to match (defaults to task)
    since: date | None = None   # first day this entry writes receipts

    @property
    def match_task(self) -> str:
        return self.receipt_task or self.task

    @property
    def tracked(self) -> bool:
        return self.receipt_name is not None

    def describe(self) -> str:
        s = self.schedule
        try:
            hours = sorted(s.hour)
            mins = sorted(s.minute)
            dows = sorted(s.day_of_week)
            times = ", ".join(f"{h:02d}:{m:02d}" for h in hours for m in mins) if len(hours) * len(mins) <= 4 \
                else f"{len(hours) * len(mins)}×/day"
            names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            if dows == list(range(7)):
                days = "daily"
            elif dows == [1, 2, 3, 4, 5]:
                days = "Mon–Fri"
            else:
                days = ",".join(names[d] for d in dows)
            return f"{days} · {times} IST"
        except Exception:  # noqa: BLE001
            return str(s)

    def due_times(self, day: date) -> list[datetime]:
        """Due datetimes (IST) of this entry on ``day``."""
        s = self.schedule
        celery_dow = (day.weekday() + 1) % 7   # celery: 0 = Sunday
        if celery_dow not in s.day_of_week or day.day not in s.day_of_month \
                or day.month not in s.month_of_year:
            return []
        return sorted(datetime.combine(day, time(h, m), tzinfo=IST)
                      for h in s.hour for m in s.minute)


def _entry(key: str, task: str, sched, receipt_task: str = "") -> ScheduleEntry:
    meta = ENTRY_META.get(key, {})
    match = receipt_task or task
    return ScheduleEntry(
        key=key, task=task, label=meta.get("label", key.replace("-", " ").replace("_", " ").capitalize()),
        schedule=sched, grace=timedelta(minutes=meta.get("grace_min", DEFAULT_GRACE_MIN)),
        receipt_name=TRACKED_TASKS.get(match), receipt_task=receipt_task,
        since=date.fromisoformat(meta["since"]) if meta.get("since") else None,
    )


def _swing_v2_entries() -> list[ScheduleEntry]:
    from celery.schedules import crontab

    try:
        from config.celery import SWING_V2_BEAT, swing_v2_receipt_task
    except Exception as e:  # noqa: BLE001
        log.warning("run_schedule.swing_v2_unavailable", error=str(e))
        return []
    out = []
    for key, (cron, task, args) in SWING_V2_BEAT.items():
        rt = swing_v2_receipt_task(args[0]) if task.endswith("run_swing_v2") and args else ""
        out.append(_entry(key, task, crontab(**cron), rt))
    return out


def registry() -> list[ScheduleEntry]:
    from celery.schedules import crontab
    from django.conf import settings

    out = []
    for key, entry in (getattr(settings, "CELERY_BEAT_SCHEDULE", {}) or {}).items():
        sched = entry.get("schedule")
        if not isinstance(sched, crontab):
            continue
        if entry.get("task", "").startswith(RUNTIME_ADDED_PREFIX):
            continue
        out.append(_entry(key, entry.get("task", ""), sched))
    out.extend(_swing_v2_entries())
    return out


def _tracking_since():
    """When scheduled-job receipts started. Only receipts that can match a
    schedule slot count (they carry a Celery ``task_name``): receipts mirrored
    from AgentRun / Backtest rows are backdated to when those runs happened
    and would otherwise pull this back to the oldest agent run, turning every
    earlier slot into a false "missed" (which happened once, 2026-10-21)."""
    from django.db.models import Min

    from apps.system.models import RunReceipt

    return (RunReceipt.objects.exclude(status=RunReceipt.Status.NOT_RUN).exclude(task_name="")
            .aggregate(m=Min("started_at"))["m"])


def _receipt_brief(r) -> dict:
    return {"id": str(r.id), "status": r.status, "started_at": r.started_at.isoformat(),
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "duration_seconds": r.duration_seconds, "trigger": r.trigger,
            "warn_checks": sum(1 for c in (r.checks or []) if c.get("status") == "warn"),
            "error": (r.error or "")[:300]}


def slot_state(entry: ScheduleEntry, due: datetime, *, now: datetime, tracking_since,
               receipts: list) -> dict:
    """State of one scheduled slot. ``receipts`` = candidate receipts for the task."""
    window_end = due + entry.grace
    base = {"key": entry.key, "label": entry.label, "task": entry.task,
            "receipt_name": entry.receipt_name, "due_at": due.isoformat(),
            "window_end": window_end.isoformat(), "schedule": entry.describe(),
            "rerunnable": entry.receipt_name in RERUNNABLE, "receipt": None}
    if not entry.tracked:
        return {**base, "state": "untracked", "reason": "job is not wrapped in run receipts yet"}

    lo = due - timedelta(minutes=2)
    not_run = next((r for r in receipts if r.status == "not_run" and r.schedule_key == entry.key
                    and r.expected_at == due), None)
    in_window = [r for r in receipts if r.status != "not_run" and lo <= r.started_at <= window_end]
    in_window.sort(key=lambda r: (r.trigger != "beat", r.started_at))
    r = in_window[0] if in_window else None

    if r is not None:
        brief = _receipt_brief(r)
        if r.status in ("success", "skipped"):
            state = "warn" if brief["warn_checks"] and r.status == "success" else "ok"
        elif r.status == "partial":
            state = "warn"
        elif r.status in ("running", "queued"):
            state = "running"
        else:
            state = "failed"
        return {**base, "state": state, "receipt": brief}
    if not_run is not None:
        return {**base, "state": "failed", "receipt": _receipt_brief(not_run),
                "reason": "did not run in its window"}
    if now < due:
        return {**base, "state": "pending", "reason": "not due yet"}
    if now <= window_end:
        return {**base, "state": "pending", "reason": "due — waiting for the run"}
    if tracking_since is None or due < tracking_since:
        return {**base, "state": "untracked", "reason": "before run receipts were enabled"}
    if entry.since is not None and due.astimezone(IST).date() < entry.since:
        return {**base, "state": "untracked", "reason": f"job tracked from {entry.since.isoformat()}"}
    return {**base, "state": "missed", "reason": "no run recorded in its window"}


def schedule_for_day(day: date | None = None, *, now: datetime | None = None) -> list[dict]:
    from django.utils import timezone

    from apps.system.models import RunReceipt

    now = now or timezone.now()
    day = day or now.astimezone(IST).date()
    entries = registry()
    start = datetime.combine(day, time(0), tzinfo=IST) - timedelta(minutes=5)
    end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=IST) + timedelta(hours=2)
    tasks = [e.match_task for e in entries if e.tracked]
    receipts = list(RunReceipt.objects.filter(task_name__in=tasks, tenant__isnull=True)
                    .filter(started_at__gte=start, started_at__lte=end)) + \
        list(RunReceipt.objects.filter(status=RunReceipt.Status.NOT_RUN,
                                       expected_at__gte=start, expected_at__lte=end))
    by_task: dict[str, list] = {}
    for r in receipts:
        by_task.setdefault(r.task_name, []).append(r)
    since = _tracking_since()
    slots = []
    for e in entries:
        for due in e.due_times(day):
            slots.append(slot_state(e, due, now=now, tracking_since=since,
                                    receipts=by_task.get(e.match_task, [])))
    slots.sort(key=lambda s: s["due_at"])
    return slots


def summarize(slots: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for s in slots:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    judged = sum(counts.get(k, 0) for k in ("ok", "warn", "failed", "missed", "running"))
    ok = counts.get("ok", 0) + counts.get("running", 0)
    bad = counts.get("failed", 0) + counts.get("missed", 0)
    health = "red" if bad else ("amber" if counts.get("warn") else ("green" if judged else "grey"))
    return {"ok": ok, "total": judged, "warn": counts.get("warn", 0), "failed": counts.get("failed", 0),
            "missed": counts.get("missed", 0), "pending": counts.get("pending", 0),
            "untracked": counts.get("untracked", 0), "health": health,
            "label": f"{ok} of {judged} scheduled jobs OK" if judged else "no scheduled jobs due yet"}


# ─── Watcher ────────────────────────────────────────────────────────────

def watch(now: datetime | None = None) -> dict:
    """Write ``not_run`` receipts for missed slots (today + yesterday) and
    fail receipts stuck in RUNNING past their watchdog. Alerts each once."""
    from django.db import IntegrityError, transaction
    from django.utils import timezone

    from apps.system.models import RunReceipt
    from apps.system.services.run_alerts import alert_receipt

    now = now or timezone.now()
    today = now.astimezone(IST).date()
    created, stuck = 0, 0
    entries = {e.key: e for e in registry()}
    for day in (today - timedelta(days=1), today):
        for slot in schedule_for_day(day, now=now):
            if slot["state"] != "missed":
                continue
            e = entries[slot["key"]]
            due = datetime.fromisoformat(slot["due_at"])
            try:
                with transaction.atomic():
                    r, was_created = RunReceipt.objects.get_or_create(
                        schedule_key=e.key, expected_at=due, status=RunReceipt.Status.NOT_RUN,
                        defaults={
                            "kind": "job", "name": e.receipt_name or e.key, "task_name": e.match_task,
                            "trigger": RunReceipt.Trigger.WATCHER, "started_at": due,
                            "finished_at": now, "duration_seconds": 0,
                            "error": f"Scheduled {due.astimezone(IST):%H:%M} IST; nothing ran by "
                                     f"{(due + e.grace).astimezone(IST):%H:%M}.",
                            "checks": [{"name": "ran_in_window", "status": "fail",
                                        "message": f"no run of {e.task} between {due.isoformat()} and "
                                                   f"{(due + e.grace).isoformat()}"}],
                        })
            except IntegrityError:
                continue
            if was_created:
                created += 1
                alert_receipt(r)

    for r in RunReceipt.objects.filter(status__in=[RunReceipt.Status.RUNNING, RunReceipt.Status.QUEUED]):
        if (now - r.started_at).total_seconds() > r.max_seconds:
            r.status = RunReceipt.Status.FAILED
            r.finished_at = now
            r.duration_seconds = round((now - r.started_at).total_seconds(), 1)
            r.error = (f"Still running after {r.max_seconds // 60} min — the worker was "
                       "probably killed before the run could finish.")
            r.checks = list(r.checks or []) + [{"name": "finished", "status": "fail",
                                                 "message": "run never reported completion"}]
            r.save(update_fields=["status", "finished_at", "duration_seconds", "error", "checks"])
            stuck += 1
            alert_receipt(r)
    swept = sweep_stuck_details(now)
    try:
        from apps.system.services.detail_receipts import ensure_detail_receipts

        swept["detail_receipts_created"] = ensure_detail_receipts()
    except Exception as e:  # noqa: BLE001
        log.warning("run_schedule.ensure_detail_receipts_failed", error=str(e))
    if created or stuck or any(swept.values()):
        log.info("run_schedule.watch", not_run_created=created, stuck_failed=stuck, **swept)
    return {"not_run_created": created, "stuck_failed": stuck, **swept}


# ─── Stuck detail rows (PipelineRun / Backtest) ─────────────────────────
# Expected worst-case durations. A row still open long after this was left
# behind by a worker that died mid-run (the laptop shut down, dev_down, OOM):
# nothing will ever finish it, and the /pipeline page shows it "running"
# forever. Generous margins: these only catch rows that are certainly dead.
PIPELINE_MAX_RUNTIME = {
    "screener_session": timedelta(hours=7, minutes=30),   # task hard limit is 7 h
}
PIPELINE_DEFAULT_MAX_RUNTIME = timedelta(hours=3)          # receipts allow 2 h
BACKTEST_MAX_RUNTIME = timedelta(hours=6)

STUCK_REASON = ("Never finished: still '{status}' {age} after it started (expected at most {limit}). "
                "The worker running it stopped mid-run (machine shut down / worker restarted) and "
                "nothing can resume it. Marked failed by the run sweeper.")


def _fmt_td(td: timedelta) -> str:
    hours = td.total_seconds() / 3600
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


def sweep_stuck_details(now: datetime | None = None) -> dict:
    """Mark PipelineRun / strategies.Backtest rows stuck far past their
    expected duration as failed, with the reason. Never raises."""
    from django.utils import timezone

    now = now or timezone.now()
    out = {"pipeline_runs_failed": 0, "backtests_failed": 0}
    try:
        from apps.system.models import PipelineRun

        for run in PipelineRun.objects.filter(status=PipelineRun.Status.RUNNING):
            limit = PIPELINE_MAX_RUNTIME.get(run.task, PIPELINE_DEFAULT_MAX_RUNTIME)
            age = now - run.started_at
            if age <= limit:
                continue
            run.status = PipelineRun.Status.FAILED
            run.finished_at = now
            run.error = STUCK_REASON.format(status="running", age=_fmt_td(age), limit=_fmt_td(limit))
            run.save(update_fields=["status", "finished_at", "error"])
            out["pipeline_runs_failed"] += 1
    except Exception as e:  # noqa: BLE001
        log.warning("run_schedule.sweep_pipeline_failed", error=str(e))
    try:
        from apps.strategies.models import Backtest

        cutoff = now - BACKTEST_MAX_RUNTIME
        for bt in Backtest.objects.filter(status__in=[Backtest.Status.QUEUED, Backtest.Status.RUNNING],
                                          updated_at__lt=cutoff):
            age = now - bt.created_at
            bt.error = STUCK_REASON.format(status=bt.status, age=_fmt_td(age), limit=_fmt_td(BACKTEST_MAX_RUNTIME))
            bt.status = Backtest.Status.FAILED
            bt.save(update_fields=["status", "error", "updated_at"])
            out["backtests_failed"] += 1
    except Exception as e:  # noqa: BLE001
        log.warning("run_schedule.sweep_backtests_failed", error=str(e))
    return out
