from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class Plan(models.Model):
    class Code(models.TextChoices):
        PAPER = "paper"
        TRADER = "trader"
        PRO = "pro"
        ADVISOR = "advisor"
        DESK = "desk"

    id = models.BigAutoField(primary_key=True)
    code = models.CharField(max_length=16, choices=Code.choices, unique=True)
    name = models.CharField(max_length=80)
    monthly_inr = models.IntegerField(default=0)
    entitlements = models.JSONField(default=dict)


class Subscription(TenantModel):
    class Status(models.TextChoices):
        ACTIVE = "active"
        PAST_DUE = "past_due"
        CANCELED = "canceled"
        TRIALING = "trialing"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.TRIALING)
    current_period_start = models.DateTimeField(null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)
    provider = models.CharField(max_length=16, default="razorpay")
    provider_ref = models.CharField(max_length=128, blank=True)
