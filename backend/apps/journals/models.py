from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class JournalEntry(TenantModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey("portfolio.Portfolio", on_delete=models.PROTECT)
    agent_run = models.ForeignKey("agents_core.AgentRun", null=True, blank=True,
                                   on_delete=models.SET_NULL)
    order = models.ForeignKey("orders.Order", null=True, blank=True,
                               on_delete=models.SET_NULL)
    kind = models.CharField(max_length=32)  # plan | entry | exit | adjustment | rejection
    title = models.CharField(max_length=200)
    body = models.TextField()
    tags = models.JSONField(default=list, blank=True)
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["tenant", "-created_at"])]
