from __future__ import annotations

from django.db import models

from apps.common.tenancy import TenantModel


class AuditEvent(TenantModel):
    """Security-audit ledger. Writes are fire-and-forget (non-blocking)."""
    id = models.BigAutoField(primary_key=True)
    actor = models.ForeignKey("accounts.User", null=True, blank=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=64)      # e.g. order.place, broker.link
    target_type = models.CharField(max_length=64, blank=True)
    target_id = models.CharField(max_length=80, blank=True)
    ip = models.GenericIPAddressField(null=True, blank=True)
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["tenant", "-created_at"])]
