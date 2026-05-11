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
