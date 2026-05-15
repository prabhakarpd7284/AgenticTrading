from __future__ import annotations

import uuid
from decimal import Decimal

from django.db import models

from apps.common.tenancy import TenantModel


class Portfolio(TenantModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=80, default="Default")
    capital = models.DecimalField(max_digits=16, decimal_places=2, default=Decimal("0"))
    used_capital = models.DecimalField(max_digits=16, decimal_places=2, default=Decimal("0"))
    realized_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=Decimal("0"))
    day_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=Decimal("0"))
    mode = models.CharField(max_length=10, default="paper")  # paper | live
    broker_link = models.ForeignKey(
        "market_data.BrokerLink", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="portfolios",
    )

    class Meta:
        indexes = [models.Index(fields=["tenant", "mode"])]


class Position(TenantModel):
    class Status(models.TextChoices):
        OPEN = "open"
        CLOSED = "closed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="positions")
    symbol = models.CharField(max_length=80)
    side = models.CharField(max_length=4)           # BUY | SELL
    qty = models.IntegerField()
    avg_price = models.DecimalField(max_digits=14, decimal_places=4)
    last_ltp = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    sl = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    tp = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    unrealized_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=Decimal("0"))
    exit_price = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    realized_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=Decimal("0"))
    exchange = models.CharField(max_length=8, default="NSE")  # NSE | NFO | MCX
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.OPEN)
    opened_at = models.DateTimeField(auto_now_add=True)
    closed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "portfolio", "status"]),
            models.Index(fields=["tenant", "symbol"]),
            models.Index(fields=["tenant", "portfolio", "opened_at"]),
        ]


class PortfolioSnapshot(TenantModel):
    id = models.BigAutoField(primary_key=True)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="snapshots")
    captured_at = models.DateTimeField(auto_now_add=True, db_index=True)
    equity = models.DecimalField(max_digits=16, decimal_places=2)
    day_pnl = models.DecimalField(max_digits=16, decimal_places=2)
    unrealized_pnl = models.DecimalField(max_digits=16, decimal_places=2)
    open_positions = models.IntegerField()
