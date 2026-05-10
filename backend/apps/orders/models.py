from __future__ import annotations

import uuid
from decimal import Decimal

from django.db import models

from apps.common.tenancy import TenantModel


class Order(TenantModel):
    class Status(models.TextChoices):
        QUEUED = "queued"
        SENT = "sent"
        OPEN = "open"
        FILLED = "filled"
        CANCELLED = "cancelled"
        REJECTED = "rejected"
        FAILED = "failed"

    class Side(models.TextChoices):
        BUY = "BUY"
        SELL = "SELL"

    class Product(models.TextChoices):
        INTRADAY = "INTRADAY"
        DELIVERY = "DELIVERY"
        CARRYFORWARD = "CARRYFORWARD"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey("portfolio.Portfolio", on_delete=models.PROTECT)
    created_by = models.ForeignKey("accounts.User", on_delete=models.PROTECT)
    broker_link = models.ForeignKey(
        "broker.BrokerLink", on_delete=models.SET_NULL, null=True, blank=True,
    )
    symbol = models.CharField(max_length=80)
    side = models.CharField(max_length=4, choices=Side.choices)
    qty = models.IntegerField()
    order_type = models.CharField(max_length=16, default="MARKET")  # MARKET | LIMIT | SL | SL-M
    product = models.CharField(max_length=16, choices=Product.choices, default=Product.INTRADAY)
    price = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    sl = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    tp = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    broker_order_id = models.CharField(max_length=80, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    idempotency_key = models.CharField(max_length=128, blank=True, db_index=True)
    origin = models.CharField(max_length=32, default="ui")   # ui | agent | strategy
    error = models.TextField(blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "status", "-created_at"]),
            models.Index(fields=["tenant", "portfolio", "-created_at"]),
        ]


class OutboxEvent(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        IN_FLIGHT = "in_flight"
        SUCCEEDED = "succeeded"
        FAILED = "failed"
        DLQ = "dlq"

    id = models.BigAutoField(primary_key=True)
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="events")
    payload = models.JSONField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    attempts = models.IntegerField(default=0)
    last_error = models.TextField(blank=True)
    next_run_at = models.DateTimeField(auto_now_add=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["status", "next_run_at"])]
