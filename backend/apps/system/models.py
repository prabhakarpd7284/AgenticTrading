"""Tenant-scoped system control flags + trader notes.

Replaces the legacy `trading.SystemControl` and `trading.TraderNote`
models with tenant-aware versions. SystemControl powers the kill switch,
AI pause/resume, force-close-all flags. TraderNote is per-symbol scratch
notes that appear on the setup page.
"""
from __future__ import annotations

import uuid

from django.db import models
from django.utils import timezone

from apps.common.tenancy import TenantModel


class SystemControl(TenantModel):
    """Generic key-value flag table. One row per (tenant, key).

    Common keys: `ai_trading_paused`, `kill_switch`, `force_close_at`.
    """

    id = models.BigAutoField(primary_key=True)
    key = models.CharField(max_length=50)
    value = models.JSONField(default=dict)

    class Meta:
        unique_together = ("tenant", "key")
        ordering = ["key"]

    def __str__(self) -> str:
        return f"{self.tenant_id}/{self.key} = {self.value}"


class TraderNote(TenantModel):
    """One persistent note per (tenant, symbol). Surfaces on /plan/setup/<symbol>."""

    id = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=30, db_index=True)
    note = models.TextField(blank=True, default="")

    class Meta:
        unique_together = ("tenant", "symbol")
        ordering = ["symbol"]

    def __str__(self) -> str:
        return f"Note: {self.symbol} ({len(self.note)} chars)"


class PipelineRun(models.Model):
    """One execution record for a daily-pipeline Celery task.

    The daily pipeline (swing scan / screener session / EOD enrichment)
    runs unattended via Celery beat. This table is its audit log — every
    run, beat-triggered or manual, writes a row so the /pipeline debugger
    page can show what ran, when, how it ended, and what it produced.

    Not tenant-scoped: the pipeline is a process-wide system operation
    (the screener resolves a default tenant for signal persistence), so
    this is a plain Model like ``market_data.Symbol`` / ``Candle``.
    """

    class Task(models.TextChoices):
        SWING_SCAN = "swing_scan", "Swing scan"
        SCREENER_SESSION = "screener_session", "Screener session"
        EOD_ENRICHMENT = "eod_enrichment", "EOD enrichment"

    class Status(models.TextChoices):
        RUNNING = "running"
        SUCCESS = "success"
        FAILED = "failed"

    class Trigger(models.TextChoices):
        BEAT = "beat"        # fired by the Celery beat schedule
        MANUAL = "manual"    # fired from the /pipeline debugger UI

    id = models.BigAutoField(primary_key=True)
    task = models.CharField(max_length=32, choices=Task.choices, db_index=True)
    status = models.CharField(max_length=12, choices=Status.choices,
                               default=Status.RUNNING)
    trigger = models.CharField(max_length=12, choices=Trigger.choices,
                                default=Trigger.BEAT)
    started_at = models.DateTimeField(auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    summary = models.JSONField(default=dict, blank=True,
                                help_text="Task result payload (counts, stats).")
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-started_at"]
        indexes = [models.Index(fields=["task", "-started_at"])]

    def __str__(self) -> str:
        return f"{self.task} [{self.status}] @ {self.started_at:%Y-%m-%d %H:%M}"


class RunReceipt(models.Model):
    """One receipt per execution of anything that runs (UX redesign plan §8).

    Answers "did it run, did it run correctly, and what did it produce?" in
    one uniform shape, whatever the job: a Celery beat task, a pipeline
    stage, a manual rerun, or a scheduled slot that never fired at all
    (``status=not_run``, written by the schedule watcher).

    Existing per-domain run tables (``PipelineRun``, ``NightlyScan``,
    ``AgentRun``, ``Backtest``) stay as the detail records; a receipt points
    at its detail row via ``detail_kind`` / ``detail_id``.

    ``tenant`` is nullable on purpose. Most scheduled jobs are process-wide
    (the AMFI NAV pull, MF metrics, the nightly scan, the daily pipeline and
    the cooldown sweep each run once for the whole install and write shared
    or cross-tenant rows) — pinning their receipt to an arbitrary tenant
    would be a lie. Those receipts carry ``tenant=NULL`` and only aggregate
    counts in ``outputs``; the API shows them to every authenticated member
    alongside that member's own tenant-scoped receipts, never another
    tenant's.
    """

    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        SUCCESS = "success"
        PARTIAL = "partial"
        FAILED = "failed"
        NOT_RUN = "not_run"     # scheduled slot passed with no execution
        SKIPPED = "skipped"     # ran, and correctly decided there was nothing to do

    class Trigger(models.TextChoices):
        BEAT = "beat"
        MANUAL = "manual"
        API = "api"
        WATCHER = "watcher"     # not_run receipts written by the schedule watcher

    TERMINAL = ("success", "partial", "failed", "not_run", "skipped")

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "tenants.Tenant", null=True, blank=True, on_delete=models.CASCADE,
        related_name="+", db_index=True,
        help_text="NULL for process-wide system jobs (see class docstring).",
    )
    kind = models.CharField(max_length=24, db_index=True,
                            help_text="job | pipeline | scan | import | backtest | agent | analysis")
    name = models.CharField(max_length=64, db_index=True,
                            help_text="Stable job key, e.g. 'nightly_scan', 'pull_amfi_nav'.")
    task_name = models.CharField(max_length=200, blank=True, default="",
                                 help_text="Celery task name — used to match beat schedule slots.")
    schedule_key = models.CharField(max_length=64, blank=True, default="",
                                    help_text="CELERY_BEAT_SCHEDULE entry (set on not_run receipts).")
    expected_at = models.DateTimeField(null=True, blank=True,
                                       help_text="Scheduled due time (not_run receipts).")
    trigger = models.CharField(max_length=12, choices=Trigger.choices, default=Trigger.BEAT)
    celery_task_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.RUNNING,
                              db_index=True)
    inputs = models.JSONField(default=dict, blank=True)
    outputs = models.JSONField(default=dict, blank=True,
                               help_text="Counts, ids and links the run produced.")
    warnings = models.JSONField(default=list, blank=True)
    checks = models.JSONField(default=list, blank=True,
                              help_text="[{name, status: pass|warn|fail|info, message, ...}]")
    data_provenance = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    max_seconds = models.IntegerField(
        default=3 * 3600,
        help_text="Watchdog: a receipt still RUNNING after this is marked failed "
                  "(a hard-killed worker never reaches the finaliser).",
    )
    detail_kind = models.CharField(max_length=32, blank=True, default="",
                                   help_text="e.g. 'pipeline_run', 'nightly_scan'")
    detail_id = models.CharField(max_length=64, blank=True, default="")
    alerted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-started_at"]
        indexes = [
            models.Index(fields=["name", "-started_at"]),
            models.Index(fields=["tenant", "-started_at"]),
            models.Index(fields=["task_name", "-started_at"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["schedule_key", "expected_at"],
                condition=models.Q(status="not_run"),
                name="uniq_not_run_per_schedule_slot",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.name} [{self.status}] @ {self.started_at:%Y-%m-%d %H:%M}"
