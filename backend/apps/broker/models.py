from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class BrokerLink(TenantModel):
    class Status(models.TextChoices):
        ACTIVE = "active"
        EXPIRED = "expired"
        DISABLED = "disabled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    broker_name = models.CharField(max_length=32)  # angel_one | zerodha | fyers
    owner = models.ForeignKey("accounts.User", on_delete=models.CASCADE)
    credential_arn = models.CharField(max_length=256, help_text="AWS Secrets Manager ARN")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE)
    last_refreshed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=["tenant", "owner"])]
