"""The trading domain — one app for everything money-related.

Phase 4c merged three apps into this one:
  - portfolio  → Portfolio, Position, PortfolioSnapshot
  - orders     → Order, OutboxEvent
  - trades     → Trade, OptionsPosition, OptionsLeg

The triangle (portfolio ⇄ orders ⇄ trades) was one domain pretending to
be three — co-locating the models kills the duplicate-state risk the
redesign exists to fight (e.g. the dead portfolio.Position sitting next
to the live trades.Trade).

Each model keeps its original db_table so existing databases needed no
DDL — the models changed apps, the tables did not.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

from django.db import models

from apps.common.tenancy import TenantModel


# ════════════════════════════════════════════════════════════════════
# Portfolio  (was apps.portfolio)
# ════════════════════════════════════════════════════════════════════

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
        db_table = "portfolio_portfolio"
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
        db_table = "portfolio_position"
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

    class Meta:
        db_table = "portfolio_portfoliosnapshot"


# ════════════════════════════════════════════════════════════════════
# Order  (was apps.orders)
# ════════════════════════════════════════════════════════════════════

class Order(TenantModel):
    class Status(models.TextChoices):
        QUEUED = "queued"
        SENT = "sent"
        OPEN = "open"
        FILLED = "filled"
        CANCELLED = "cancelled"
        REJECTED = "rejected"
        FAILED = "failed"

    class Side(models.TextChoices):
        BUY = "BUY"
        SELL = "SELL"

    class Product(models.TextChoices):
        INTRADAY = "INTRADAY"
        DELIVERY = "DELIVERY"
        CARRYFORWARD = "CARRYFORWARD"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.PROTECT)
    created_by = models.ForeignKey("accounts.User", on_delete=models.PROTECT)
    broker_link = models.ForeignKey(
        "market_data.BrokerLink", on_delete=models.SET_NULL, null=True, blank=True,
    )
    symbol = models.CharField(max_length=80)
    side = models.CharField(max_length=4, choices=Side.choices)
    qty = models.IntegerField()
    order_type = models.CharField(max_length=16, default="MARKET")  # MARKET | LIMIT | SL | SL-M
    product = models.CharField(max_length=16, choices=Product.choices, default=Product.INTRADAY)
    price = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    sl = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    tp = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    broker_order_id = models.CharField(max_length=80, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    idempotency_key = models.CharField(max_length=128, blank=True, db_index=True)
    origin = models.CharField(max_length=32, default="ui")   # ui | agent | strategy
    error = models.TextField(blank=True)

    class Meta:
        db_table = "orders_order"
        indexes = [
            models.Index(fields=["tenant", "status", "-created_at"]),
            models.Index(fields=["tenant", "portfolio", "-created_at"]),
        ]


class OutboxEvent(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        IN_FLIGHT = "in_flight"
        SUCCEEDED = "succeeded"
        FAILED = "failed"
        DLQ = "dlq"

    id = models.BigAutoField(primary_key=True)
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="events")
    payload = models.JSONField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    attempts = models.IntegerField(default=0)
    last_error = models.TextField(blank=True)
    next_run_at = models.DateTimeField(auto_now_add=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "orders_outboxevent"
        indexes = [models.Index(fields=["status", "next_run_at"])]


# ════════════════════════════════════════════════════════════════════
# OptionsPosition / OptionsLeg / Trade  (was apps.trades)
# ════════════════════════════════════════════════════════════════════

class OptionsPosition(TenantModel):
    """Multi-leg options aggregate. Generalizes the legacy StraddlePosition
    so we can model straddles, strangles, spreads, condors, and pyramid-
    options under one schema. Per-leg actions and rolls write Event rows
    rather than mutating a JSON management_log.
    """

    class PositionType(models.TextChoices):
        SHORT_STRADDLE   = "SHORT_STRADDLE"
        SHORT_STRANGLE   = "SHORT_STRANGLE"
        LONG_STRADDLE    = "LONG_STRADDLE"
        BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
        BEAR_PUT_SPREAD  = "BEAR_PUT_SPREAD"
        IRON_CONDOR      = "IRON_CONDOR"
        PYRAMID_OPTION   = "PYRAMID_OPTION"
        CUSTOM           = "CUSTOM"

    class Status(models.TextChoices):
        ACTIVE  = "ACTIVE"
        PARTIAL = "PARTIAL"   # one or more legs closed
        HEDGED  = "HEDGED"    # futures hedge added
        CLOSED  = "CLOSED"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(
        Portfolio, on_delete=models.PROTECT, related_name="options_positions",
    )
    strategy_run = models.ForeignKey(
        "agents_core.AgentRun", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="+",
    )

    position_type = models.CharField(max_length=24, choices=PositionType.choices)
    underlying    = models.CharField(max_length=20, db_index=True)
    expiry        = models.DateField()
    lot_size      = models.IntegerField(default=1)
    lots          = models.IntegerField(default=1)

    status = models.CharField(max_length=10, choices=Status.choices,
                               default=Status.ACTIVE, db_index=True)

    # ── Live state ──
    net_delta       = models.DecimalField(max_digits=8, decimal_places=4, default=0)
    current_pnl_inr = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    realized_pnl    = models.DecimalField(max_digits=16, decimal_places=2, default=0)

    # ── Timestamps ──
    opened_at  = models.DateTimeField(auto_now_add=False, null=True, blank=True)
    closed_at  = models.DateTimeField(null=True, blank=True)
    trade_date = models.DateField(db_index=True)

    # Legacy lift shim
    legacy_straddle_id = models.IntegerField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = "trades_optionsposition"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["tenant", "status"]),
            models.Index(fields=["tenant", "underlying", "expiry"]),
        ]

    def __str__(self) -> str:
        return f"{self.position_type} {self.underlying} {self.expiry} [{self.status}]"


class OptionsLeg(models.Model):
    """One leg of an OptionsPosition. Quantity is unsigned; direction
    is in leg_role (SHORT_CE means we sold a call; LONG_CE means we
    bought a call)."""

    class LegRole(models.TextChoices):
        SHORT_CE   = "SHORT_CE"
        SHORT_PE   = "SHORT_PE"
        LONG_CE    = "LONG_CE"
        LONG_PE    = "LONG_PE"
        HEDGE_FUT  = "HEDGE_FUT"

    id = models.BigAutoField(primary_key=True)
    position = models.ForeignKey(
        OptionsPosition, on_delete=models.CASCADE, related_name="legs",
    )
    leg_role = models.CharField(max_length=12, choices=LegRole.choices)
    symbol   = models.CharField(max_length=40)
    token    = models.CharField(max_length=20, blank=True, default="",
                                 help_text="Broker token (Angel One NFO/BFO)")
    strike   = models.IntegerField(null=True, blank=True)
    qty      = models.IntegerField(help_text="Absolute lots × lot_size; sign comes from leg_role")

    open_price    = models.DecimalField(max_digits=14, decimal_places=4)
    current_price = models.DecimalField(max_digits=14, decimal_places=4, default=0)
    closed_price  = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    closed_at     = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "trades_optionsleg"
        ordering = ["id"]
        indexes = [models.Index(fields=["position", "leg_role"])]

    def __str__(self) -> str:
        return f"{self.leg_role} {self.symbol} qty={self.qty} @ {self.open_price}"


class Trade(TenantModel):
    """One row per trade intent. Lifecycle: PLAN → APPROVED/REJECTED →
    QUEUED → SENT → PARTIAL/FILLED → CLOSED/CANCELLED/EXPIRED.

    The single source of truth for "a thing we wanted/want to be in the
    market on", spanning the full plan→execute→close lifecycle. It absorbs
    the legacy portfolio.Position (current state) and trading.TradeJournal
    (decision history) into one row per intent. State transitions are
    enforced in services/trade_lifecycle.py.
    """

    class Status(models.TextChoices):
        PLAN      = "PLAN"        # planner emitted the draft
        APPROVED  = "APPROVED"    # risk engine approved
        REJECTED  = "REJECTED"    # risk engine blocked
        QUEUED    = "QUEUED"      # in order outbox
        SENT      = "SENT"        # broker accepted
        PARTIAL   = "PARTIAL"     # partial fill
        FILLED    = "FILLED"      # fully filled
        CLOSED    = "CLOSED"      # position closed (SL/TGT/manual/EOD)
        CANCELLED = "CANCELLED"   # operator or broker cancel
        EXPIRED   = "EXPIRED"     # plan or order timeout

    class Side(models.TextChoices):
        BUY  = "BUY"
        SELL = "SELL"

    class Origin(models.TextChoices):
        WORKFLOW    = "workflow"     # produced by a StrategyRun
        MANUAL      = "manual"       # operator entered via UI
        API         = "api"          # external API trigger
        BROKER_SYNC = "broker_sync"  # backfilled from broker reconciliation

    class CloseReason(models.TextChoices):
        SL_HIT      = "SL_HIT"
        TARGET_HIT  = "TARGET_HIT"
        MANUAL      = "MANUAL"
        EOD         = "EOD"
        TRAIL       = "TRAIL"

    # ── Identity ──
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(
        Portfolio, on_delete=models.PROTECT, related_name="trades",
    )
    strategy_run = models.ForeignKey(
        "agents_core.AgentRun", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="trades", help_text="Workflow run that produced this trade",
    )

    symbol = models.CharField(max_length=80, db_index=True)
    exchange = models.CharField(max_length=8, default="NSE")  # NSE | NFO | BSE | BFO | MCX
    side = models.CharField(max_length=4, choices=Side.choices)
    product = models.CharField(max_length=16, default="INTRADAY",
                                help_text="INTRADAY | DELIVERY | CARRYFORWARD")

    # ── Plan economics (immutable after risk decision) ──
    entry_price = models.DecimalField(max_digits=14, decimal_places=4)
    stop_loss   = models.DecimalField(max_digits=14, decimal_places=4)
    target      = models.DecimalField(max_digits=14, decimal_places=4)
    quantity    = models.IntegerField()
    lot_size    = models.IntegerField(default=1)
    confidence  = models.DecimalField(max_digits=5, decimal_places=4, default=0)
    reasoning   = models.TextField(blank=True, default="",
                                    help_text="LLM rationale; empty for non-LLM workflows")

    # ── Risk gate ──
    risk_approved   = models.BooleanField(default=False)
    risk_reason     = models.CharField(max_length=255, blank=True, default="")
    risk_details    = models.JSONField(default=dict, blank=True,
                                        help_text="Full 10-criterion result for the audit")
    risk_decided_at = models.DateTimeField(null=True, blank=True)

    # ── Execution ──
    status = models.CharField(max_length=16, choices=Status.choices,
                               default=Status.PLAN, db_index=True)
    primary_order = models.ForeignKey(
        Order, null=True, blank=True, on_delete=models.SET_NULL, related_name="+",
    )
    fill_price    = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    fill_quantity = models.IntegerField(null=True, blank=True)
    filled_at     = models.DateTimeField(null=True, blank=True)

    # ── Exit ──
    exit_price    = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)
    exit_quantity = models.IntegerField(null=True, blank=True)
    closed_at     = models.DateTimeField(null=True, blank=True)
    close_reason  = models.CharField(max_length=32, choices=CloseReason.choices,
                                      blank=True, default="")

    # ── Performance ──
    realized_pnl   = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    unrealized_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    pnl_percent    = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True)
    last_ltp       = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)

    # ── Metadata ──
    origin     = models.CharField(max_length=16, choices=Origin.choices, default=Origin.WORKFLOW)
    trade_date = models.DateField(db_index=True)

    # Legacy lift shim — set during SQLite→Postgres migration, dropped after
    legacy_trade_journal_id = models.IntegerField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = "trades_trade"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["tenant", "status", "-created_at"]),
            models.Index(fields=["tenant", "portfolio", "status"]),
            models.Index(fields=["tenant", "symbol", "trade_date"]),
            models.Index(fields=["tenant", "strategy_run"]),
        ]

    def __str__(self) -> str:
        return f"{self.side} {self.quantity} {self.symbol} @ {self.entry_price} [{self.status}]"

    # ── Convenience ──
    @property
    def risk_amount(self) -> float:
        return float(abs(self.entry_price - self.stop_loss) * self.quantity)

    @property
    def reward_amount(self) -> float:
        return float(abs(self.target - self.entry_price) * self.quantity)

    @property
    def risk_reward_ratio(self) -> float:
        r = self.risk_amount
        return float(self.reward_amount) / r if r > 0 else 0.0
