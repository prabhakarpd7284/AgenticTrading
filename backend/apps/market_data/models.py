from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class BrokerLink(TenantModel):
    """A tenant's connection to an upstream broker (Angel One / Zerodha / …).

    Absorbed from the standalone `broker` app in Phase 4b — broker
    integration is part of the markets domain, not its own concern.
    """

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
        # Physical table keeps its original name — the model moved apps in
        # Phase 4b but the table didn't, so existing DBs need no DDL.
        db_table = "broker_brokerlink"
        indexes = [models.Index(fields=["tenant", "owner"])]


class Symbol(models.Model):
    id = models.BigAutoField(primary_key=True)
    exchange = models.CharField(max_length=8)    # NSE | BSE | NFO | MCX
    token = models.CharField(max_length=32)
    tradingsymbol = models.CharField(max_length=80)
    name = models.CharField(max_length=200, blank=True)
    segment = models.CharField(max_length=16, blank=True)
    lot_size = models.IntegerField(default=1)
    tick_size = models.DecimalField(max_digits=8, decimal_places=4, default=0.05)

    class Meta:
        unique_together = [("exchange", "token")]
        indexes = [models.Index(fields=["exchange", "tradingsymbol"])]


class Candle(models.Model):
    id = models.BigAutoField(primary_key=True)
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name="candles")
    interval = models.CharField(max_length=8)
    t = models.DateTimeField(db_index=True)
    o = models.DecimalField(max_digits=14, decimal_places=4)
    h = models.DecimalField(max_digits=14, decimal_places=4)
    l = models.DecimalField(max_digits=14, decimal_places=4)
    c = models.DecimalField(max_digits=14, decimal_places=4)
    v = models.BigIntegerField()

    class Meta:
        unique_together = [("symbol", "interval", "t")]
        indexes = [models.Index(fields=["symbol", "interval", "-t"])]
