from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class StrategyInstance(TenantModel):
    """A tenant's instantiation of a catalog strategy (with their params)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    strategy_name = models.CharField(max_length=128)
    strategy_version = models.CharField(max_length=32)
    name = models.CharField(max_length=120)     # human label
    params = models.JSONField(default=dict)
    is_enabled = models.BooleanField(default=False)
    schedule = models.JSONField(default=dict, blank=True)   # cron-like


class Backtest(TenantModel):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        DONE = "done"
        FAILED = "failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    instance = models.ForeignKey(StrategyInstance, on_delete=models.CASCADE, related_name="backtests")
    from_date = models.DateField()
    to_date = models.DateField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    metrics = models.JSONField(default=dict, blank=True)
    equity_curve = models.JSONField(default=list, blank=True)
    error = models.TextField(blank=True)
