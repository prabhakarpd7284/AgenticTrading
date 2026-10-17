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
