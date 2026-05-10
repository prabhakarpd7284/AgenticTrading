from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class AgentRun(TenantModel):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        SUCCEEDED = "succeeded"
        FAILED = "failed"
        CANCELLED = "cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    strategy_name = models.CharField(max_length=128, db_index=True)
    strategy_version = models.CharField(max_length=32)
    portfolio = models.ForeignKey("portfolio.Portfolio", on_delete=models.PROTECT)
    triggered_by = models.ForeignKey("accounts.User", on_delete=models.PROTECT)
    config = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    result = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "status", "-created_at"]),
            models.Index(fields=["tenant", "strategy_name", "-created_at"]),
        ]


class AgentStep(models.Model):
    id = models.BigAutoField(primary_key=True)
    run = models.ForeignKey(AgentRun, on_delete=models.CASCADE, related_name="steps")
    seq = models.PositiveIntegerField()
    node = models.CharField(max_length=64)
    event_type = models.CharField(max_length=16)
    payload = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["run", "seq"])]
        unique_together = [("run", "seq")]
