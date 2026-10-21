"""Mutual fund data: a shared scheme/NAV catalog (not tenant-scoped — it's
the same AMFI data for everyone) plus tenant-scoped holdings/transactions
that sit alongside apps.trading.Portfolio, the same way equity Position
does. No advisory/recommendation logic lives here — see
apps/mutual_funds/services/advisor.py for that, kept separate so the
data-collection path never silently grows advice-giving side effects.
"""
from __future__ import annotations

from django.db import models

from apps.common.tenancy import TenantModel


class MFScheme(models.Model):
    """One row per AMFI scheme code. Shared reference data — same for every
    tenant, refreshed daily from the AMFI NAV file. Not a TenantModel."""

    scheme_code = models.CharField(max_length=20, primary_key=True)
    isin_growth = models.CharField(max_length=20, blank=True, default="")
    isin_div_reinvestment = models.CharField(max_length=20, blank=True, default="")
    amc_name = models.CharField(max_length=200, db_index=True)
    scheme_name = models.CharField(max_length=300)
    plan = models.CharField(max_length=20, blank=True, default="")  # "Direct Plan" / "Regular Plan"
    option = models.CharField(max_length=100, blank=True, default="")  # "Growth Option" / "IDCW Option"; AMFI's longest is 84 chars
    category = models.CharField(max_length=200, blank=True, default="")  # AMFI category header line
    is_direct = models.BooleanField(default=False)
    # False = no NAV in the recent past: matured / merged / wound up. Dead
    # schemes are kept (never deleted) so point-in-time backtests are not
    # survivorship-biased. Refreshed by the NAV-history backfill.
    is_active = models.BooleanField(default=True, db_index=True)
    # Where the catalog row came from: the AMFI daily file, or a history
    # source (mfapi.in) for a scheme that no longer appears in that file.
    catalog_source = models.CharField(max_length=20, blank=True, default="amfi")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "mutual_funds_scheme"
        indexes = [models.Index(fields=["amc_name", "scheme_name"])]

    def __str__(self) -> str:
        return f"{self.scheme_name} ({self.plan} {self.option})"


class MFNav(models.Model):
    """Daily NAV time series per scheme. One row per scheme per day."""

    scheme = models.ForeignKey(MFScheme, on_delete=models.CASCADE, related_name="navs")
    nav_date = models.DateField(db_index=True)
    nav = models.DecimalField(max_digits=14, decimal_places=4)

    class Meta:
        db_table = "mutual_funds_nav"
        constraints = [
            models.UniqueConstraint(fields=["scheme", "nav_date"], name="uniq_mf_nav_scheme_date")
        ]
        indexes = [models.Index(fields=["scheme", "-nav_date"])]

    def __str__(self) -> str:
        return f"{self.scheme_id} @ {self.nav_date} = {self.nav}"


class MFHolding(TenantModel):
    """A folio-level holding, same role in the MF world as apps.trading.Position
    plays for equity — deliberately kept in its own app rather than merged
    into trading.Position, since the lifecycle (folios, SIPs, CAS imports)
    is different enough to not share a table without forcing nullable fields
    on both sides."""

    SOURCE_MANUAL = "manual"
    SOURCE_CAS = "cas"
    SOURCE_CHOICES = [(SOURCE_MANUAL, "Manual entry"), (SOURCE_CAS, "CAS import")]

    portfolio = models.ForeignKey("trading.Portfolio", on_delete=models.CASCADE, related_name="mf_holdings")
    scheme = models.ForeignKey(MFScheme, on_delete=models.PROTECT, related_name="holdings")
    folio_number = models.CharField(max_length=50, blank=True, default="")
    units = models.DecimalField(max_digits=18, decimal_places=4, default=0)
    avg_cost_per_unit = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default=SOURCE_MANUAL)

    class Meta:
        db_table = "mutual_funds_holding"
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "portfolio", "scheme", "folio_number"],
                name="uniq_mf_holding_per_folio",
            )
        ]

    @property
    def invested_amount(self) -> float | None:
        if self.avg_cost_per_unit is None:
            return None
        return float(self.units * self.avg_cost_per_unit)

    def __str__(self) -> str:
        return f"{self.scheme.scheme_name} — {self.units} units (folio {self.folio_number or '—'})"


class MFTransaction(TenantModel):
    """Transaction-level history, when available (CAS import). Optional —
    a holding can exist with just units/avg_cost and no transaction rows."""

    BUY = "buy"
    SELL = "sell"
    SIP = "sip"
    SWITCH_IN = "switch_in"
    SWITCH_OUT = "switch_out"
    DIVIDEND = "dividend"
    TYPE_CHOICES = [
        (BUY, "Buy"), (SELL, "Sell"), (SIP, "SIP"),
        (SWITCH_IN, "Switch in"), (SWITCH_OUT, "Switch out"), (DIVIDEND, "Dividend"),
    ]

    holding = models.ForeignKey(MFHolding, on_delete=models.CASCADE, related_name="transactions")
    txn_type = models.CharField(max_length=12, choices=TYPE_CHOICES)
    txn_date = models.DateField()
    units = models.DecimalField(max_digits=18, decimal_places=4)
    nav = models.DecimalField(max_digits=14, decimal_places=4)
    amount = models.DecimalField(max_digits=16, decimal_places=2)

    class Meta:
        db_table = "mutual_funds_transaction"
        ordering = ["txn_date"]
        indexes = [models.Index(fields=["holding", "txn_date"])]

    def __str__(self) -> str:
        return f"{self.txn_type} {self.units} units @ {self.nav} on {self.txn_date}"


# ── Screener data (MF-S): shared catalog, NOT tenant-scoped ──────────────

class MFBackfillJob(models.Model):
    """One unit of resumable NAV-history work. For mfapi.in the unit is one
    scheme (``key`` = scheme code); for AMFI it would be ``<amc_id>:<year>``.
    Re-running a backfill skips ``done`` / ``empty`` jobs whose ``from_date``
    already covers the requested depth."""

    STATUS_PENDING = "pending"
    STATUS_DONE = "done"
    STATUS_EMPTY = "empty"      # source answered, but had no rows
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [(s, s) for s in (STATUS_PENDING, STATUS_DONE, STATUS_EMPTY, STATUS_FAILED)]

    source = models.CharField(max_length=20)
    key = models.CharField(max_length=40)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True)
    from_date = models.DateField(null=True, blank=True)   # requested depth
    attempts = models.PositiveIntegerField(default=0)
    rows_fetched = models.PositiveIntegerField(default=0)
    rows_inserted = models.PositiveIntegerField(default=0)
    bytes_downloaded = models.PositiveBigIntegerField(default=0)
    first_nav_date = models.DateField(null=True, blank=True)
    last_nav_date = models.DateField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "mutual_funds_backfill_job"
        constraints = [models.UniqueConstraint(fields=["source", "key"], name="uniq_mf_backfill_job")]

    def __str__(self) -> str:
        return f"{self.source}:{self.key} [{self.status}]"


class MFDataIssue(models.Model):
    """A data-quality finding on a scheme's NAV series. Metrics flag any
    window that overlaps an issue; nothing is ever interpolated."""

    KIND_GAP = "gap"                    # > N trading days without a NAV
    KIND_JUMP = "jump"                  # |1-day move| > threshold (split/bonus/error)
    KIND_NONPOSITIVE = "nonpositive"    # NAV <= 0
    KIND_SOURCE_MISMATCH = "source_mismatch"  # two sources disagree on one date
    KIND_CHOICES = [(k, k) for k in (KIND_GAP, KIND_JUMP, KIND_NONPOSITIVE, KIND_SOURCE_MISMATCH)]

    scheme = models.ForeignKey(MFScheme, on_delete=models.CASCADE, related_name="data_issues")
    kind = models.CharField(max_length=20, choices=KIND_CHOICES)
    start_date = models.DateField()
    end_date = models.DateField()
    detail = models.JSONField(default=dict, blank=True)
    detected_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "mutual_funds_data_issue"
        constraints = [
            models.UniqueConstraint(fields=["scheme", "kind", "start_date"], name="uniq_mf_data_issue")
        ]
        indexes = [models.Index(fields=["scheme", "kind"])]

    def __str__(self) -> str:
        return f"{self.scheme_id} {self.kind} {self.start_date}..{self.end_date}"


class MFBenchmarkNav(models.Model):
    """Daily index level for the few benchmarks that matter (Nifty TRI)."""

    index_name = models.CharField(max_length=60)      # e.g. "NIFTY 500"
    nav_date = models.DateField()
    value = models.DecimalField(max_digits=16, decimal_places=4)
    is_tri = models.BooleanField(default=True)        # False = price index (warn)
    source = models.CharField(max_length=40, default="niftyindices")

    class Meta:
        db_table = "mutual_funds_benchmark_nav"
        constraints = [
            models.UniqueConstraint(fields=["index_name", "nav_date"], name="uniq_mf_benchmark_nav")
        ]

    def __str__(self) -> str:
        return f"{self.index_name} @ {self.nav_date} = {self.value}"


class MFMetricSnapshot(models.Model):
    """Precomputed NAV-based metrics for one scheme as of one date, so the
    screener never recomputes on request. Every return here is NAV-based
    (Growth options only by the universe rule, so no dividends are lost).
    Null = not computable (too little history), never imputed."""

    scheme = models.ForeignKey(MFScheme, on_delete=models.CASCADE, related_name="metric_snapshots")
    as_of = models.DateField(db_index=True)
    category = models.CharField(max_length=120, db_index=True)   # canonical label at compute time
    is_active = models.BooleanField(default=True)

    first_nav_date = models.DateField(null=True, blank=True)
    last_nav_date = models.DateField(null=True, blank=True)
    history_years = models.FloatField(null=True, blank=True)
    nav_points = models.PositiveIntegerField(default=0)

    cagr_1y = models.FloatField(null=True, blank=True)
    cagr_3y = models.FloatField(null=True, blank=True)
    cagr_5y = models.FloatField(null=True, blank=True)
    cagr_since_inception = models.FloatField(null=True, blank=True)

    rolling_3y_median = models.FloatField(null=True, blank=True)
    rolling_3y_min = models.FloatField(null=True, blank=True)
    rolling_3y_pct_positive = models.FloatField(null=True, blank=True)
    rolling_3y_pct_above_cat_median = models.FloatField(null=True, blank=True)
    rolling_3y_windows = models.PositiveIntegerField(default=0)

    volatility = models.FloatField(null=True, blank=True)
    downside_deviation = models.FloatField(null=True, blank=True)
    max_drawdown = models.FloatField(null=True, blank=True)          # negative fraction
    max_drawdown_days = models.PositiveIntegerField(null=True, blank=True)   # peak -> trough
    recovery_days = models.PositiveIntegerField(null=True, blank=True)       # trough -> new high; null = not recovered
    worst_1y = models.FloatField(null=True, blank=True)

    sharpe = models.FloatField(null=True, blank=True)
    sortino = models.FloatField(null=True, blank=True)
    calmar = models.FloatField(null=True, blank=True)

    benchmark_name = models.CharField(max_length=60, blank=True, default="")
    excess_cagr_3y_vs_benchmark = models.FloatField(null=True, blank=True)
    beta = models.FloatField(null=True, blank=True)
    up_capture = models.FloatField(null=True, blank=True)
    down_capture = models.FloatField(null=True, blank=True)
    tracking_error = models.FloatField(null=True, blank=True)
    information_ratio = models.FloatField(null=True, blank=True)
    excess_cagr_3y_vs_category = models.FloatField(null=True, blank=True)

    risk_free_rate = models.FloatField(null=True, blank=True)
    quality_flags = models.JSONField(default=list, blank=True)
    computed_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "mutual_funds_metric_snapshot"
        constraints = [
            models.UniqueConstraint(fields=["scheme", "as_of"], name="uniq_mf_metric_snapshot")
        ]
        indexes = [models.Index(fields=["category", "as_of"])]

    def __str__(self) -> str:
        return f"{self.scheme_id} metrics @ {self.as_of}"
