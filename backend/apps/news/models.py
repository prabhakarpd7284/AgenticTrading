"""News store, corporate-event calendar and the nightly scan.

Shared reference data — NOT tenant-scoped (like ``market_data.Symbol`` /
``Candle``): headlines, NSE filings and the scan output are the same for
every tenant. Only the losing-streak cooldown (``trading.CooldownState``)
is per portfolio.

Headlines and links only — never full article text (TRADING_RULES.md §5.3).
"""
from __future__ import annotations

from django.db import models


class NewsItem(models.Model):
    class Kind(models.TextChoices):
        NEWS = "news"
        FILING = "filing"     # NSE corporate announcement
        EVENT = "event"

    class Materiality(models.TextChoices):
        RESULTS = "results"
        ORDER = "order"
        GUIDANCE = "guidance"
        RATING = "rating"
        REGULATORY = "regulatory"
        MANAGEMENT = "management"
        MACRO = "macro"
        OTHER = "other"
        NOISE = "noise"

    id = models.BigAutoField(primary_key=True)
    url = models.URLField(max_length=1000)
    url_hash = models.CharField(max_length=40, unique=True)
    title = models.CharField(max_length=500)
    title_hash = models.CharField(max_length=40, db_index=True)
    source = models.CharField(max_length=40, db_index=True,
                              help_text="Collector key: google_news, et_markets, nse_announcements …")
    publisher = models.CharField(max_length=120, blank=True, default="")
    kind = models.CharField(max_length=8, choices=Kind.choices, default=Kind.NEWS)
    is_market_wide = models.BooleanField(default=False)
    published_at = models.DateTimeField(db_index=True)
    fetched_at = models.DateTimeField(auto_now_add=True)
    raw = models.JSONField(default=dict, blank=True, help_text="Small source metadata (no article text)")

    # ── Sentiment (Claude CLI, or "rule" for boilerplate filings) ──
    sentiment_score = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    materiality = models.CharField(max_length=12, choices=Materiality.choices, blank=True, default="")
    confidence = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    sentiment_model = models.CharField(max_length=64, blank=True, default="")
    scored_at = models.DateTimeField(null=True, blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-published_at"]
        indexes = [models.Index(fields=["is_market_wide", "-published_at"])]

    def __str__(self) -> str:
        return f"[{self.source}] {self.title[:80]}"


class NewsItemSymbol(models.Model):
    """Which NSE symbols a headline is about (many-to-many by symbol text)."""
    id = models.BigAutoField(primary_key=True)
    item = models.ForeignKey(NewsItem, on_delete=models.CASCADE, related_name="mentions")
    symbol = models.CharField(max_length=32, db_index=True)

    class Meta:
        unique_together = [("item", "symbol")]


class CorporateEvent(models.Model):
    """Upcoming board meeting / results / corporate action (NSE calendar)."""
    id = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=32, db_index=True)
    company = models.CharField(max_length=200, blank=True, default="")
    event_date = models.DateField(db_index=True)
    purpose = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    source = models.CharField(max_length=40)
    fetched_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("symbol", "event_date", "purpose", "source")]
        ordering = ["event_date", "symbol"]


class NightlyScan(models.Model):
    """One scan per trading day it governs (idempotent per ``for_date``)."""

    class Status(models.TextChoices):
        RUNNING = "running"
        SUCCESS = "success"
        PARTIAL = "partial"     # finished, but a step/source degraded
        FAILED = "failed"

    id = models.BigAutoField(primary_key=True)
    for_date = models.DateField(unique=True, help_text="Trading day this scan's filter applies to")
    session_date = models.DateField(help_text="Last completed session the levels/news come from")
    status = models.CharField(max_length=8, choices=Status.choices, default=Status.RUNNING)
    started_at = models.DateTimeField()
    finished_at = models.DateTimeField(null=True, blank=True)
    timings = models.JSONField(default=dict, blank=True)
    sources = models.JSONField(default=dict, blank=True, help_text="Per-source status/counts")
    summary = models.JSONField(default=dict, blank=True)
    telegram_sent = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-for_date"]


class NightlyScanRow(models.Model):
    class Label(models.TextChoices):
        STRONG_POSITIVE = "strong_positive"
        POSITIVE = "positive"
        NEUTRAL = "neutral"
        NEGATIVE = "negative"
        STRONG_NEGATIVE = "strong_negative"
        UNSCORED = "unscored"

    id = models.BigAutoField(primary_key=True)
    scan = models.ForeignKey(NightlyScan, on_delete=models.CASCADE, related_name="rows")
    symbol = models.CharField(max_length=32, db_index=True)

    # Levels for the next session (from the last session's H/L/C)
    data_date = models.DateField(null=True, blank=True)
    levels = models.JSONField(default=dict, blank=True)
    cpr_width_pct = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    cpr_class = models.CharField(max_length=10, blank=True, default="")
    two_day_relationship = models.CharField(max_length=24, blank=True, default="")
    camarilla_relationship = models.CharField(max_length=24, blank=True, default="")
    bias = models.CharField(max_length=24, blank=True, default="")

    # News + sentiment rollup
    sentiment_score = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    sentiment_label = models.CharField(max_length=16, choices=Label.choices, default=Label.NEUTRAL)
    sentiment_weight = models.DecimalField(max_digits=7, decimal_places=3, null=True, blank=True)
    headline_count = models.IntegerField(default=0)
    scored_count = models.IntegerField(default=0)
    event_tomorrow = models.BooleanField(default=False)
    event_purpose = models.CharField(max_length=255, blank=True, default="")

    # Final verdict (TRADING_RULES.md §3)
    allow_long = models.BooleanField(default=True)
    allow_short = models.BooleanField(default=True)
    allowed = models.CharField(max_length=5, default="both")   # both | long | short | none
    reasons = models.JSONField(default=list, blank=True)
    headlines = models.JSONField(default=list, blank=True, help_text="Top headlines snapshot")

    class Meta:
        unique_together = [("scan", "symbol")]
        ordering = ["symbol"]
