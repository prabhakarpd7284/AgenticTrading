"""
Django models for the agentic trading system.

Tables:
    TradeJournal      - Every trade decision, rationale, and outcome
    StrategyDoc       - Trading rules, patterns, strategy documents
    PortfolioSnapshot - Point-in-time portfolio state for tracking
    AuditLog          - Every LLM prompt, response, and risk decision
    StraddlePosition  - Short straddle lifecycle (both CE + PE legs)
"""
from django.db import models


class TradeJournal(models.Model):
    """
    Records every trade the system plans/executes.
    Feeds back into RAG retriever for future decisions.
    """

    class Side(models.TextChoices):
        BUY = "BUY"
        SELL = "SELL"

    class Status(models.TextChoices):
        PENDING = "PENDING"       # initial state
        PLANNED = "PLANNED"       # planner generated it
        APPROVED = "APPROVED"     # risk engine approved
        REJECTED = "REJECTED"     # risk engine rejected
        EXECUTED = "EXECUTED"     # order placed
        FILLED = "FILLED"        # order filled
        PARTIAL = "PARTIAL"      # partially filled
        CANCELLED = "CANCELLED"  # order cancelled
        PAPER = "PAPER"          # paper trade (simulated)

    symbol = models.CharField(max_length=30, db_index=True)
    side = models.CharField(max_length=4, choices=Side.choices)
    entry_price = models.FloatField()
    stop_loss = models.FloatField()
    target = models.FloatField()
    quantity = models.IntegerField()
    reasoning = models.TextField(help_text="LLM rationale for the trade")
    confidence = models.FloatField(default=0.0, help_text="Planner confidence 0-1")

    # Execution fields
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    order_id = models.CharField(max_length=64, blank=True, default="")
    fill_price = models.FloatField(null=True, blank=True)
    fill_quantity = models.IntegerField(null=True, blank=True)

    # P&L (computed after exit)
    pnl = models.FloatField(null=True, blank=True, help_text="Realized P&L in INR")
    pnl_percent = models.FloatField(null=True, blank=True)

    # Risk engine feedback
    risk_approved = models.BooleanField(default=False)
    risk_reason = models.CharField(max_length=255, blank=True, default="")

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    trade_date = models.DateField(db_index=True, help_text="Trading session date")

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["symbol", "trade_date"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"{self.side} {self.quantity}x {self.symbol} @ {self.entry_price} [{self.status}]"

    @property
    def risk_amount(self) -> float:
        """Absolute risk in INR for this trade."""
        return abs(self.entry_price - self.stop_loss) * self.quantity

    @property
    def reward_amount(self) -> float:
        """Absolute reward in INR for this trade."""
        return abs(self.target - self.entry_price) * self.quantity

    @property
    def risk_reward_ratio(self) -> float:
        """Risk:Reward ratio. Higher is better."""
        risk = self.risk_amount
        if risk == 0:
            return 0.0
        return self.reward_amount / risk


class StrategyDoc(models.Model):
    """
    Trading strategy rules and documents.
    Injected as RAG context for the planner agent.
    """

    class Category(models.TextChoices):
        ENTRY = "ENTRY"         # entry signal / setup rules
        EXIT = "EXIT"           # exit / target / trailing stop rules
        RISK = "RISK"           # position sizing, risk limits
        SIZING = "SIZING"       # quantity / capital allocation rules
        FILTER = "FILTER"       # trade filter / avoid conditions
        GENERAL = "GENERAL"     # general trading wisdom / notes
        RULE = "RULE"           # explicit trading rule
        PATTERN = "PATTERN"     # chart pattern documentation
        STRATEGY = "STRATEGY"   # full strategy document

    title = models.CharField(max_length=200)
    content = models.TextField(help_text="Strategy text — injected into planner prompt")
    category = models.CharField(max_length=12, choices=Category.choices, default=Category.RULE)
    is_active = models.BooleanField(default=True, help_text="Inactive docs are excluded from RAG")

    # For future pgvector upgrade
    # embedding = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["category", "title"]

    def __str__(self):
        return f"[{self.category}] {self.title}"


class PortfolioSnapshot(models.Model):
    """
    Point-in-time snapshot of portfolio state.
    Used by risk engine for position sizing and daily loss tracking.
    """
    capital = models.FloatField(help_text="Total capital in INR")
    invested = models.FloatField(default=0.0, help_text="Currently invested amount")
    available_cash = models.FloatField(help_text="Available cash for trading")
    daily_pnl = models.FloatField(default=0.0, help_text="P&L for the day so far")
    total_pnl = models.FloatField(default=0.0, help_text="Cumulative P&L across all sessions")
    daily_loss = models.FloatField(default=0.0, help_text="Total losses for the day (positive number)")
    open_positions = models.IntegerField(default=0)
    open_positions_count = models.IntegerField(default=0, help_text="Alias for open_positions for compatibility")
    snapshot_date = models.DateField(db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-snapshot_date"]
        get_latest_by = "snapshot_date"

    def __str__(self):
        return f"Portfolio {self.snapshot_date}: {self.capital} INR, P&L: {self.daily_pnl}"


class AuditLog(models.Model):
    """
    Audit trail for every LLM interaction and risk decision.

    Every prompt to Claude, every model response, every risk decision.
    This is non-negotiable for a system that touches capital.
    """

    class EventType(models.TextChoices):
        PLANNER_REQUEST = "PLANNER_REQ"    # prompt sent to LLM
        PLANNER_RESPONSE = "PLANNER_RES"   # structured response from LLM
        PLANNER_ERROR = "PLANNER_ERR"      # LLM call failed
        RISK_APPROVED = "RISK_APPROVE"     # risk engine approved
        RISK_REJECTED = "RISK_REJECT"      # risk engine rejected
        EXECUTION = "EXECUTION"            # order placed (paper or live)
        RECONCILIATION = "RECONCILE"       # position reconciliation event

    event_type = models.CharField(max_length=16, choices=EventType.choices, db_index=True)
    symbol = models.CharField(max_length=30, blank=True, default="", db_index=True)

    # LLM audit fields
    prompt = models.TextField(blank=True, default="", help_text="Full prompt sent to LLM")
    response = models.TextField(blank=True, default="", help_text="Full LLM response (raw)")
    model_name = models.CharField(max_length=64, blank=True, default="")
    tokens_used = models.IntegerField(null=True, blank=True)
    latency_ms = models.IntegerField(null=True, blank=True, help_text="LLM call latency in milliseconds")

    # Risk audit fields
    risk_details = models.JSONField(null=True, blank=True, help_text="Full risk validation details")

    # Execution audit fields
    execution_details = models.JSONField(null=True, blank=True)

    # Linkage
    trade_journal = models.ForeignKey(
        TradeJournal, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="audit_logs",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["event_type", "created_at"]),
            models.Index(fields=["symbol", "created_at"]),
        ]

    def __str__(self):
        return f"[{self.event_type}] {self.symbol} @ {self.created_at.strftime('%H:%M:%S')}"


class StraddlePosition(models.Model):
    """
    Tracks a live short straddle — both CE and PE legs independently.

    A short straddle sells one call and one put at the same strike.
    This model records the full lifecycle: sell → manage → close.
    Management actions are appended to management_log (JSON array).
    """

    class Status(models.TextChoices):
        ACTIVE  = "ACTIVE"   # both legs open
        PARTIAL = "PARTIAL"  # one leg closed
        HEDGED  = "HEDGED"   # futures hedge added
        CLOSED  = "CLOSED"   # both legs closed

    class ActionTaken(models.TextChoices):
        NONE            = "NONE"
        HOLD            = "HOLD"
        CLOSE_BOTH      = "CLOSE_BOTH"
        CLOSE_CE        = "CLOSE_CE"
        CLOSE_PE        = "CLOSE_PE"
        HEDGE_FUTURES   = "HEDGE_FUTURES"
        ROLL            = "ROLL"

    # ── Position setup ──
    underlying      = models.CharField(max_length=20, default="NIFTY", db_index=True)
    strike          = models.IntegerField(help_text="Strike price e.g. 24200")
    expiry          = models.DateField(help_text="Option expiry date")
    lot_size        = models.IntegerField(default=65, help_text="NIFTY lot size")
    lots            = models.IntegerField(default=1)

    # ── CE leg (short call) ──
    ce_symbol       = models.CharField(max_length=40, help_text="e.g. NIFTY10MAR2624200CE")
    ce_token        = models.CharField(max_length=20, help_text="Angel One NFO token")
    ce_sell_price   = models.FloatField(help_text="Premium received when CE was sold")
    ce_current_price= models.FloatField(default=0.0)

    # ── PE leg (short put) ──
    pe_symbol       = models.CharField(max_length=40, help_text="e.g. NIFTY10MAR2624200PE")
    pe_token        = models.CharField(max_length=20, help_text="Angel One NFO token")
    pe_sell_price   = models.FloatField(help_text="Premium received when PE was sold")
    pe_current_price= models.FloatField(default=0.0)

    # ── Live state (updated each cycle) ──
    net_delta       = models.FloatField(default=0.0)
    current_pnl_inr = models.FloatField(default=0.0)
    status          = models.CharField(max_length=10, choices=Status.choices, default=Status.ACTIVE)
    action_taken    = models.CharField(
        max_length=20, choices=ActionTaken.choices, default=ActionTaken.NONE
    )

    # ── Management history ──
    management_log  = models.JSONField(
        default=list, blank=True,
        help_text="List of management events: [{time, action, nifty, pnl_inr, note, executed}]",
    )

    # ── Timestamps ──
    opened_at       = models.DateTimeField(auto_now_add=True)
    closed_at       = models.DateTimeField(null=True, blank=True)
    last_updated    = models.DateTimeField(auto_now=True)
    trade_date      = models.DateField(db_index=True)

    class Meta:
        ordering = ["-opened_at"]
        indexes  = [
            models.Index(fields=["underlying", "strike", "expiry"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return (
            f"Short Straddle {self.underlying} {self.strike} "
            f"[{self.expiry}] [{self.status}] P&L: {self.current_pnl_inr:+,.0f} INR"
        )

    @property
    def total_premium_sold(self) -> float:
        return (self.ce_sell_price + self.pe_sell_price) * self.lot_size * self.lots

    @property
    def combined_sell_pts(self) -> float:
        return self.ce_sell_price + self.pe_sell_price

    @property
    def combined_current_pts(self) -> float:
        return self.ce_current_price + self.pe_current_price
