from __future__ import annotations

import uuid

from django.db import models


class Tenant(models.Model):
    class Kind(models.TextChoices):
        RETAIL = "retail", "Retail"
        ADVISOR = "advisor", "Advisor"
        PROP = "prop", "Prop desk"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    kind = models.CharField(max_length=16, choices=Kind.choices, default=Kind.RETAIL)
    slug = models.SlugField(max_length=80, unique=True)
    dedicated = models.BooleanField(default=False, help_text="Route to dedicated DB cluster")
    white_label_domain = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["kind"])]

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"


class Membership(models.Model):
    class Role(models.TextChoices):
        OWNER = "owner"
        ADMIN = "admin"
        TRADER = "trader"
        VIEWER = "viewer"
        RM = "rm"                 # relationship manager (advisor)
        CLIENT = "client"         # client-of-advisor

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey("accounts.User", on_delete=models.CASCADE, related_name="memberships")
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="memberships")
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.TRADER)
    is_active = models.BooleanField(default=True)
    invited_at = models.DateTimeField(auto_now_add=True)
    accepted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("user", "tenant")]
