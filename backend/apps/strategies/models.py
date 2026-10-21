from __future__ import annotations

import uuid

from django.db import models

from apps.common.tenancy import TenantModel


class StrategyInstance(TenantModel):
    """A tenant's strategy: a rule spec (Strategy Studio) or a built-in plugin
    template with its params. The definition itself lives in versioned
    ``StrategyVersion`` rows; editing creates version n+1 so every backtest
    points at the exact definition it tested (Studio plan §2.1)."""

    class Kind(models.TextChoices):
        SPEC = "spec"        # rule-builder StrategySpec (no code)
        PLUGIN = "plugin"    # built-in plugin template + params

    class Stage(models.TextChoices):
        DRAFT = "draft"
        BACKTESTED = "backtested"
        PAPER = "paper"
        LIVE = "live"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    strategy_name = models.CharField(max_length=128)          # plugin name, or "spec"
    strategy_version = models.CharField(max_length=32)        # plugin version, or spec schema version
    name = models.CharField(max_length=120)     # human label
    description = models.TextField(blank=True, default="")
    kind = models.CharField(max_length=8, choices=Kind.choices, default=Kind.PLUGIN)
    template_key = models.CharField(max_length=64, blank=True, default="")
    params = models.JSONField(default=dict)     # mirror of the current version's params/spec
    stage = models.CharField(max_length=12, choices=Stage.choices, default=Stage.DRAFT)
    current_version = models.PositiveIntegerField(default=1)
    archived_at = models.DateTimeField(null=True, blank=True)
    is_enabled = models.BooleanField(default=False)
    schedule = models.JSONField(default=dict, blank=True)   # cron-like

    class Meta:
        ordering = ["-updated_at"]


class StrategyVersion(TenantModel):
    """Immutable definition of one version of a StrategyInstance."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    instance = models.ForeignKey(StrategyInstance, on_delete=models.CASCADE, related_name="versions")
    version = models.PositiveIntegerField()
    spec = models.JSONField(default=dict, blank=True)      # kind=spec: the StrategySpec
    params = models.JSONField(default=dict, blank=True)    # kind=plugin: schema params
    spec_hash = models.CharField(max_length=64)
    created_by = models.ForeignKey("accounts.User", null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="+")
    note = models.CharField(max_length=240, blank=True, default="")

    class Meta:
        ordering = ["-version"]
        unique_together = [("instance", "version")]


class Backtest(TenantModel):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        DONE = "done"
        FAILED = "failed"
        CANCELLED = "cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    instance = models.ForeignKey(StrategyInstance, on_delete=models.CASCADE, related_name="backtests")
    version = models.ForeignKey(StrategyVersion, null=True, blank=True, on_delete=models.SET_NULL,
                                related_name="backtests")
    from_date = models.DateField()
    to_date = models.DateField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    config = models.JSONField(default=dict, blank=True)          # universe, capital, costs, split
    metrics = models.JSONField(default=dict, blank=True)         # headline (whole period)
    split_metrics = models.JSONField(default=dict, blank=True)   # {"is": {...}, "oos": {...}}
    data_provenance = models.JSONField(default=dict, blank=True)
    equity_curve = models.JSONField(default=list, blank=True)
    progress = models.PositiveSmallIntegerField(default=0)       # 0-100
    progress_note = models.CharField(max_length=200, blank=True, default="")
    cancel_requested = models.BooleanField(default=False)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]


class BacktestTrade(TenantModel):
    """One closed trade of a Backtest (net of costs and slippage)."""
    id = models.BigAutoField(primary_key=True)
    backtest = models.ForeignKey(Backtest, on_delete=models.CASCADE, related_name="trades")
    seq = models.PositiveIntegerField()
    symbol = models.CharField(max_length=40)
    side = models.CharField(max_length=5)                 # BUY | SHORT
    segment = models.CharField(max_length=3, default="is")  # is | oos
    entry_time = models.CharField(max_length=32)
    exit_time = models.CharField(max_length=32)
    entry_price = models.FloatField()
    exit_price = models.FloatField()
    stop = models.FloatField()
    target = models.FloatField(null=True, blank=True)
    qty = models.IntegerField()
    gross_pnl = models.FloatField()
    costs = models.FloatField()
    net_pnl = models.FloatField()
    r_multiple = models.FloatField()
    exit_reason = models.CharField(max_length=40)
    bars_held = models.IntegerField(default=0)
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["seq"]
        indexes = [models.Index(fields=["backtest", "seq"])]


# ── Feedback-loop models (formerly trading.SignalLog + trading.WatchlistEntry) ──

class Signal(TenantModel):
    """One row per signal fired by any screener/scanner/premarket detector.

    Replaces the legacy `trading.SignalLog`. Powers the monthly capture-rate
    report: "stock moved X% — how much of that did AlphaDesk capture?"
    """

    class Source(models.TextChoices):
        SCREENER    = "SCREENER"
        OK_SCANNER  = "OK_SCANNER"
        PREMARKET   = "PREMARKET"
        TRADINGVIEW = "TRADINGVIEW"
        STOCKEDGE   = "STOCKEDGE"      # StockEdge-derived (e.g. composite momentum)

    class Outcome(models.TextChoices):
        PENDING   = "PENDING"
        TRADED    = "TRADED"
        REJECTED  = "REJECTED"
        SKIPPED   = "SKIPPED"
        EXPIRED   = "EXPIRED"

    id = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=30, db_index=True)
    signal_date = models.DateField(db_index=True)
    signal_time = models.DateTimeField()
    source = models.CharField(max_length=16, choices=Source.choices)
    strategy = models.CharField(max_length=40)
    side = models.CharField(max_length=5)          # BUY | SELL
    entry_price = models.FloatField()
    stoploss = models.FloatField()
    target = models.FloatField()
    confidence = models.FloatField(default=0.0)
    risk_reward = models.FloatField(default=0.0)
    reasons = models.JSONField(default=list)
    indicators = models.JSONField(default=dict)

    # Outcome tracking
    outcome = models.CharField(max_length=10, choices=Outcome.choices, default=Outcome.PENDING)
    outcome_reason = models.TextField(blank=True, default="")
    trade = models.ForeignKey(
        "trading.Trade", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="signals",
    )

    # Post-hoc enrichment (EOD job fills these)
    eod_price = models.FloatField(null=True, blank=True)
    max_favorable_move = models.FloatField(null=True, blank=True)
    max_adverse_move = models.FloatField(null=True, blank=True)

    # Legacy lift shim
    legacy_signal_log_id = models.IntegerField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["-signal_time"]
        indexes = [
            models.Index(fields=["tenant", "signal_date", "symbol"]),
            models.Index(fields=["tenant", "source", "signal_date"]),
            models.Index(fields=["tenant", "outcome"]),
        ]

    def __str__(self) -> str:
        return f"[{self.source}] {self.side} {self.symbol} @ {self.entry_price:.2f} [{self.outcome}]"


class WatchlistEntry(TenantModel):
    """Daily premarket-scanner output. Each row is one symbol picked for today.
    Replaces the legacy `trading.WatchlistEntry` model with tenant-scoping.
    """

    class Outcome(models.TextChoices):
        WATCHING  = "WATCHING"
        TRIGGERED = "TRIGGERED"
        TRADED    = "TRADED"
        SKIPPED   = "SKIPPED"
        NO_SIGNAL = "NO_SIGNAL"

    id = models.BigAutoField(primary_key=True)
    symbol = models.CharField(max_length=30, db_index=True)
    scan_date = models.DateField(db_index=True)
    score = models.FloatField(help_text="Premarket scanner score 0-100")
    bias = models.CharField(max_length=10, default="NEUTRAL")
    setups = models.JSONField(default=list)

    # Previous-day context
    prev_high = models.FloatField(default=0.0)
    prev_low = models.FloatField(default=0.0)
    prev_close = models.FloatField(default=0.0)
    prev_atr = models.FloatField(default=0.0)

    # Today's context (updated during market hours)
    orb_high = models.FloatField(null=True, blank=True)
    orb_low = models.FloatField(null=True, blank=True)
    vwap = models.FloatField(null=True, blank=True)

    # Outcome
    outcome = models.CharField(max_length=12, choices=Outcome.choices, default=Outcome.WATCHING)
    triggered_setup = models.CharField(max_length=20, blank=True, default="")
    reason = models.TextField(blank=True, default="")

    # Legacy lift shim
    legacy_watchlist_id = models.IntegerField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["-scan_date", "-score"]
        unique_together = ("tenant", "symbol", "scan_date")
        indexes = [models.Index(fields=["tenant", "scan_date", "score"])]

    def __str__(self) -> str:
        return f"[{self.scan_date}] {self.symbol} score={self.score:.0f} {self.outcome}"
