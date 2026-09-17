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


# ---------------------------------------------------------------------------
# StockEdge advisory overlay (shared reference data — NOT tenant-scoped)
#
# StockEdge analytics (market breadth, sector rotation, FII/DII, F&O OI/PCR …)
# are ingested as an independent research/confirmation overlay. They never
# reach @RiskGuard or position sizing — purely advisory. Like Symbol/Candle,
# these are plain non-tenant reference tables shared read-only across tenants
# (see docs/integrations/STOCKEDGE_INTEGRATION.md §4).
# ---------------------------------------------------------------------------
class StockEdgeSnapshot(models.Model):
    """One captured StockEdge dataset for a given (dataset, as_of_date, exchange).

    Append-only-ish: re-pulls of the same day upsert in place (unique_together)
    so we keep exactly one canonical row per day/exchange while preserving the
    full captured payload in ``raw`` for replay/diffing against our own signals.
    """
    id = models.BigAutoField(primary_key=True)
    dataset = models.CharField(max_length=40, db_index=True,
                               help_text="market_breadth | sector_rotation | fii_dii | fno_oi | …")
    as_of_date = models.DateField(db_index=True)
    captured_at = models.DateTimeField(auto_now_add=True,
                                       help_text="First time this (dataset, day, exchange) was captured")
    refreshed_at = models.DateTimeField(auto_now=True,
                                        help_text="Last time this snapshot was re-pulled/updated in place")
    source_url = models.URLField(max_length=512, blank=True, default="")
    exchange = models.CharField(max_length=8, default="NSE")
    raw = models.JSONField(default=dict, help_text="Full captured/normalized payload")
    meta = models.JSONField(default=dict, blank=True,
                            help_text="Capture metadata (columns, unit, scan id, etc.)")

    class Meta:
        ordering = ["-as_of_date"]
        unique_together = [("dataset", "as_of_date", "exchange")]
        indexes = [models.Index(fields=["dataset", "-as_of_date"])]

    def __str__(self) -> str:
        return f"{self.dataset} {self.exchange} @ {self.as_of_date}"


class StockEdgeBreadthRow(models.Model):
    """Flattened market-breadth row — one index universe (Nifty 50 … Microcap 250).

    Values are percentages (0-100): the share of the index's constituents that
    satisfy each condition (RS positive, price above SMA20/50/100/200).
    """
    id = models.BigAutoField(primary_key=True)
    snapshot = models.ForeignKey(StockEdgeSnapshot, on_delete=models.CASCADE,
                                 related_name="breadth_rows")
    index_name = models.CharField(max_length=80)
    constituent_count = models.IntegerField(null=True, blank=True)
    rs_pos = models.FloatField(null=True, blank=True, help_text="% constituents with RS > 0")
    sma20 = models.FloatField(null=True, blank=True, help_text="% constituents above SMA20")
    sma50 = models.FloatField(null=True, blank=True, help_text="% constituents above SMA50")
    sma100 = models.FloatField(null=True, blank=True, help_text="% constituents above SMA100")
    sma200 = models.FloatField(null=True, blank=True, help_text="% constituents above SMA200")
    as_of_date = models.DateField(db_index=True)
    exchange = models.CharField(max_length=8, default="NSE")

    class Meta:
        indexes = [models.Index(fields=["as_of_date", "index_name"])]

    def __str__(self) -> str:
        return f"{self.index_name} {self.as_of_date} (breadth={self.breadth_score})"

    @property
    def breadth_score(self) -> float | None:
        """Mean of the available breadth percentages, ignoring missing values.

        Returns None when no component is present.
        """
        vals = [v for v in (self.rs_pos, self.sma20, self.sma50, self.sma100, self.sma200)
                if v is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 2)


class StockEdgeScanRow(models.Model):
    """One per-stock row from a StockEdge CSV export (scan / strategy / scores).

    Common stock columns are promoted to real fields for querying; everything
    dataset-specific (momentum scores + zones, strategy match type + criteria
    flags, …) is preserved in ``attrs`` so a single model serves every
    stock-list export. Shared, non-tenant reference data.
    """
    id = models.BigAutoField(primary_key=True)
    snapshot = models.ForeignKey(StockEdgeSnapshot, on_delete=models.CASCADE,
                                 related_name="scan_rows")
    symbol = models.CharField(max_length=40, db_index=True)
    name = models.CharField(max_length=160, blank=True, default="")
    sector = models.CharField(max_length=80, blank=True, default="")
    industry = models.CharField(max_length=160, blank=True, default="")
    ltp = models.FloatField(null=True, blank=True)
    change_pct = models.FloatField(null=True, blank=True)
    market_cap_cr = models.FloatField(null=True, blank=True, help_text="Market cap in Rs. crore")
    attrs = models.JSONField(default=dict, blank=True,
                             help_text="Dataset-specific fields (scores, zones, flags, type, …)")
    as_of_date = models.DateField(db_index=True)
    exchange = models.CharField(max_length=8, default="NSE")

    class Meta:
        indexes = [
            models.Index(fields=["as_of_date", "symbol"]),
            models.Index(fields=["snapshot", "symbol"]),
        ]

    def __str__(self) -> str:
        return f"{self.symbol} @ {self.as_of_date}"
