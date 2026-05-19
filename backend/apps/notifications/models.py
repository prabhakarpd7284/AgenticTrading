from __future__ import annotations

import secrets
import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class Alert(TenantModel):
    class Severity(models.TextChoices):
        INFO = "info"
        WARN = "warn"
        CRIT = "crit"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey("accounts.User", null=True, blank=True, on_delete=models.SET_NULL)
    severity = models.CharField(max_length=8, choices=Severity.choices, default=Severity.INFO)
    title = models.CharField(max_length=200)
    body = models.TextField(blank=True)
    read_at = models.DateTimeField(null=True, blank=True)


# ════════════════════════════════════════════════════════════════════════
# TradingView webhook integration
# ════════════════════════════════════════════════════════════════════════

def _make_webhook_secret() -> str:
    """URL-safe random token used as the auth-via-URL for the public
    /webhooks/tradingview/<secret>/ endpoint. 32 bytes → 43 chars of base64,
    unguessable in practice. Rotate via the API when leaked."""
    return secrets.token_urlsafe(32)


class TradingViewLink(TenantModel):
    """One TradingView webhook configuration per (tenant, owner) target.

    Each row represents one webhook URL the user has pasted into a TradingView
    alert. TradingView itself doesn't authenticate to us — the only thing
    proving the request came from the right account is the unguessable
    `webhook_secret` in the URL path.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        "accounts.User", on_delete=models.CASCADE,
        related_name="tradingview_links",
    )

    display_name = models.CharField(
        max_length=80, blank=True, default="",
        help_text="Operator-facing label, e.g. 'Strategy: VWAP breakout'.",
    )
    webhook_secret = models.CharField(
        max_length=64, unique=True, default=_make_webhook_secret,
        help_text="URL-path secret. Anyone with this can POST signals to this link.",
    )
    is_active = models.BooleanField(default=True)

    # ── Auto-fire workflow on receipt (opt-in, RiskGuard still gates) ──
    autofire_enabled = models.BooleanField(
        default=False,
        help_text="When true, qualifying alerts spawn an AgentRun (still goes through RiskGuard).",
    )
    default_strategy_name = models.CharField(
        max_length=64, blank=True, default="",
        help_text="Strategy from the catalog to fire (e.g. 'directional'). Required when autofire is on.",
    )
    portfolio = models.ForeignKey(
        "trading.Portfolio", null=True, blank=True,
        on_delete=models.PROTECT,
        help_text="Which portfolio the auto-fired run targets.",
    )
    allowed_actions = models.JSONField(
        default=list, blank=True,
        help_text="If non-empty, only alerts whose `action` is in this list auto-fire. E.g. ['BUY','SELL'].",
    )

    # ── Activity stats ──
    last_received_at = models.DateTimeField(null=True, blank=True, db_index=True)
    receive_count = models.PositiveIntegerField(default=0)
    last_error = models.CharField(max_length=512, blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "owner", "-created_at"]),
            models.Index(fields=["webhook_secret"]),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"TradingView[{self.display_name or 'unnamed'}] owner={self.owner_id}"

    def rotate_secret(self) -> None:
        self.webhook_secret = _make_webhook_secret()
        self.last_error = ""
        self.save(update_fields=["webhook_secret", "last_error", "updated_at"])


class TradingViewWatchlist(TenantModel):
    """A named list of symbols the operator cares about.

    Standalone — not bound to a specific TradingViewLink. The UI uses it as a
    soft filter ("show only signals for symbols in this watchlist") and a
    reference when configuring auto-fire allowlists. Will grow into
    rule-driven dynamic membership later (e.g. "all NIFTY 50 stocks where
    RSI<30") but starts as an explicit symbol list to keep the contract
    simple."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        "accounts.User", on_delete=models.CASCADE,
        related_name="tradingview_watchlists",
    )
    name = models.CharField(max_length=80)
    description = models.TextField(blank=True, default="")
    symbols = models.JSONField(
        default=list, blank=True,
        help_text="Uppercase symbols. Server normalises on save.",
    )

    class Meta:
        indexes = [models.Index(fields=["tenant", "owner", "-updated_at"])]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "owner", "name"],
                name="uniq_watchlist_name_per_owner",
            ),
        ]
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.name} ({len(self.symbols)} symbols)"

    def save(self, *args, **kwargs):
        # Normalise symbols — uppercased, trimmed, deduped while preserving
        # insertion order so the operator's intent isn't reshuffled.
        seen: set[str] = set()
        cleaned: list[str] = []
        for s in self.symbols or []:
            sym = str(s).upper().strip()
            if sym and sym not in seen:
                seen.add(sym)
                cleaned.append(sym)
        self.symbols = cleaned
        super().save(*args, **kwargs)


class TradingViewSignal(TenantModel):
    """One row per webhook payload received.

    Append-only audit log. Parsing failures don't block recording — the raw
    body is kept so the operator can diagnose a misformatted alert template.
    The parsed signal/workflow_run FKs are populated post-parse on success.
    """

    id = models.BigAutoField(primary_key=True)
    link = models.ForeignKey(
        TradingViewLink, on_delete=models.CASCADE,
        related_name="received_signals",
    )

    received_at = models.DateTimeField(auto_now_add=True, db_index=True)
    raw_payload = models.TextField(
        help_text="Verbatim request body — TradingView lets users template anything.",
    )
    parsed = models.JSONField(
        default=dict, blank=True,
        help_text="{symbol, action, price, strategy, comment, ...} after parsing.",
    )
    parse_error = models.TextField(blank=True, default="")

    # Cross-links populated when persistence + autofire succeed.
    signal = models.ForeignKey(
        "strategies.Signal", null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    workflow_run = models.ForeignKey(
        "agents_core.AgentRun", null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
        help_text="Set when autofire was enabled and the run was queued.",
    )

    class Meta:
        indexes = [
            models.Index(fields=["tenant", "-received_at"]),
            models.Index(fields=["link", "-received_at"]),
        ]
        ordering = ["-received_at"]
