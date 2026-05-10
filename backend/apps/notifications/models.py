from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class Alert(TenantModel):
    class Severity(models.TextChoices):
        INFO = "info"
        WARN = "warn"
        CRIT = "crit"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey("accounts.User", null=True, blank=True, on_delete=models.SET_NULL)
    severity = models.CharField(max_length=8, choices=Severity.choices, default=Severity.INFO)
    title = models.CharField(max_length=200)
    body = models.TextField(blank=True)
    read_at = models.DateTimeField(null=True, blank=True)
