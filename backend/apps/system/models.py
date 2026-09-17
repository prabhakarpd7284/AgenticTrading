"""Tenant-scoped system control flags + trader notes.

Replaces the legacy `trading.SystemControl` and `trading.TraderNote`
models with tenant-aware versions. SystemControl powers the kill switch,
AI pause/resume, force-close-all flags. TraderNote is per-symbol scratch
notes that appear on the setup page.
"""
from __future__ import annotations

from django.db import models

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
