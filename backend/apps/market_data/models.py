from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class BrokerLink(TenantModel):
    """A tenant's connection to an upstream broker (Angel One / Zerodha / Fyers).

    Each row is one broker account. A tenant can link multiple accounts
    (e.g. two Zerodha accounts + one Angel One) — flagged via `is_default`
    for the "primary" account used for new order placement.

    Credentials are stored two ways:
      * Local/dev: `credential_blob` — Fernet-encrypted JSON with API
        keys, secrets, tokens. Key derived from DJANGO_SECRET_KEY.
      * Prod: `credential_arn` — AWS Secrets Manager ARN (legacy path).
    """

    class Status(models.TextChoices):
        ACTIVE = "active"
        EXPIRED = "expired"
        DISABLED = "disabled"
        ERRORED = "errored"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    broker_name = models.CharField(max_length=32)  # angel_one | zerodha | fyers
    display_name = models.CharField(max_length=64, blank=True, default="")
    owner = models.ForeignKey("accounts.User", on_delete=models.CASCADE)

    credential_arn = models.CharField(max_length=256, blank=True, default="",
                                       help_text="AWS Secrets Manager ARN (prod)")
    credential_blob = models.BinaryField(null=True, blank=True,
                                          help_text="Fernet-encrypted credential JSON (dev/local)")
    credential_meta = models.JSONField(default=dict, blank=True,
                                        help_text="Non-secret broker metadata (account id, user code, etc.)")

    is_default = models.BooleanField(default=False,
                                      help_text="Primary account for new order placement")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE)
    last_refreshed_at = models.DateTimeField(null=True, blank=True)
    last_error = models.CharField(max_length=512, blank=True, default="")

    class Meta:
        db_table = "broker_brokerlink"
        indexes = [
            models.Index(fields=["tenant", "owner"]),
            models.Index(fields=["tenant", "status"]),
        ]

    def __str__(self) -> str:
        label = self.display_name or self.broker_name
        return f"{label} [{self.status}]"


class BrokerPositionSnapshot(TenantModel):
    """Cached positions/holdings/margin pulled from a broker.

    Written by the periodic Celery refresh task; read by the combined
    positions endpoint so the UI doesn't hit live broker APIs on every
    page load. One row per (link, fetched_at) — older rows TTL-pruned.
    """
    id = models.BigAutoField(primary_key=True)
    link = models.ForeignKey(BrokerLink, on_delete=models.CASCADE, related_name="snapshots")
    fetched_at = models.DateTimeField(db_index=True)
    positions = models.JSONField(default=list)
    holdings = models.JSONField(default=list)
    margin = models.JSONField(default=dict)
    ok = models.BooleanField(default=True)
    error = models.CharField(max_length=512, blank=True, default="")

    class Meta:
        ordering = ["-fetched_at"]
        indexes = [
            models.Index(fields=["tenant", "link", "-fetched_at"]),
        ]


class PendingOAuth(TenantModel):
    """Bridges /brokers/{name}/oauth/start/ → /brokers/{name}/oauth/callback/.

    OAuth-redirect-based broker flows (Zerodha Kite, Fyers v3) hop through
    the broker's login page, which means the callback request has no JWT.
    We persist the in-progress handshake keyed by a random ``state`` token
    embedded in the auth URL, then the callback uses it to recover who
    started the flow and what credentials it should pair the resulting
    access_token with.

    Rows are short-lived — TTL pruned after 10 minutes — so secrets aren't
    sitting around past the user's attention span.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    state = models.CharField(max_length=64, db_index=True, unique=True)
    broker_name = models.CharField(max_length=32)
    owner = models.ForeignKey("accounts.User", on_delete=models.CASCADE)
    display_name = models.CharField(max_length=64, blank=True, default="")
    # Encrypted blob carrying the half-creds (api_key + api_secret for Kite,
    # app_id + secret_key for Fyers) that the callback needs to complete the
    # token exchange. Same Fernet path as BrokerLink.credential_blob.
    handshake_blob = models.BinaryField()
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        indexes = [models.Index(fields=["state"])]
        ordering = ["-created_at"]


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
