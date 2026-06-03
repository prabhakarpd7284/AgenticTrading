"""Unified event log.

Replaces journals.JournalEntry, audit.AuditEvent, the legacy trading.AuditLog,
and StraddlePosition.management_log JSON arrays — every state change in the
system emits one Event row. The Now activity feed, workflow timelines,
monthly rejection list, security audit, and journal search all read from
this one stream.

Event types are a closed taxonomy (see Type below). New types are added by
updating the choices, not by inventing free-form strings.
"""
from __future__ import annotations

from django.db import models

from apps.common.tenancy import TenantModel


class Event(TenantModel):
    """One row per state change. Append-only.

    Writes are best-effort: emitting an Event MUST NOT block trading.
    Use `apps.events.services.event_writer.emit()` which catches and logs.
    """

    class Type(models.TextChoices):
        # Workflow lifecycle
        WORKFLOW_STARTED   = "workflow.started"
        WORKFLOW_STEP_STARTED   = "workflow.step.started"
        WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
        WORKFLOW_STEP_FAILED    = "workflow.step.failed"
        WORKFLOW_COMPLETED = "workflow.completed"
        WORKFLOW_CANCELLED = "workflow.cancelled"

        # LLM (Claude) calls
        LLM_REQUEST  = "llm.request"
        LLM_RESPONSE = "llm.response"
        LLM_ERROR    = "llm.error"

        # Risk engine
        RISK_APPROVED = "risk.approved"
        RISK_REJECTED = "risk.rejected"

        # Order lifecycle
        ORDER_QUEUED    = "order.queued"
        ORDER_SENT      = "order.sent"
        ORDER_FILLED    = "order.filled"
        ORDER_PARTIAL   = "order.partial"
        ORDER_CANCELLED = "order.cancelled"
        ORDER_REJECTED  = "order.rejected"
        ORDER_FAILED    = "order.failed"

        # Trade aggregate
        TRADE_PLANNED   = "trade.planned"
        TRADE_OPENED    = "trade.opened"
        TRADE_CLOSED    = "trade.closed"
        TRADE_SL_HIT    = "trade.sl_hit"
        TRADE_TGT_HIT   = "trade.target_hit"
        TRADE_TRAILED   = "trade.trailed"
        TRADE_FEEDBACK  = "trade.feedback"   # trader 👍/👎 + note on a derived trade

        # Straddle / options
        STRADDLE_OPENED     = "straddle.opened"
        STRADDLE_LEG_ROLLED = "straddle.leg_rolled"
        STRADDLE_HEDGED     = "straddle.hedged"
        STRADDLE_CLOSED     = "straddle.closed"

        # Pyramid
        PYRAMID_ADD   = "pyramid.add"
        PYRAMID_EXIT  = "pyramid.exit"

        # Signal feedback ledger
        SIGNAL_FIRED    = "signal.fired"
        SIGNAL_EXPIRED  = "signal.expired"
        SIGNAL_ENRICHED = "signal.enriched"

        # System / security
        SYSTEM_AI_PAUSED   = "system.ai_paused"
        SYSTEM_AI_RESUMED  = "system.ai_resumed"
        SYSTEM_KILL_SWITCH = "system.kill_switch"
        BROKER_LINKED      = "broker.linked"
        BROKER_EXPIRED     = "broker.expired"

    class Severity(models.TextChoices):
        INFO  = "info"
        WARN  = "warn"
        ERROR = "error"

    class ActorKind(models.TextChoices):
        USER         = "user"
        WORKFLOW     = "workflow"
        RISK_ENGINE  = "risk_engine"
        BROKER       = "broker"
        SYSTEM       = "system"

    # Identity
    id = models.BigAutoField(primary_key=True)
    ts = models.DateTimeField(db_index=True, help_text="Event time (may differ from row created_at)")
    type = models.CharField(max_length=40, choices=Type.choices, db_index=True)
    severity = models.CharField(max_length=8, choices=Severity.choices, default=Severity.INFO)

    # Who
    actor_kind = models.CharField(max_length=16, choices=ActorKind.choices, default=ActorKind.WORKFLOW)
    actor_user = models.ForeignKey(
        "accounts.User", null=True, blank=True, on_delete=models.SET_NULL, related_name="+",
    )

    # Workflow context
    workflow_run = models.ForeignKey(
        "agents_core.AgentRun", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="events",
    )
    step_name = models.CharField(max_length=64, blank=True, default="")

    # Domain links (nullable — only set when relevant)
    # `trade_id` and `signal_id` are soft UUID refs to dodge FK ordering pain
    # during the unification migration; `options_position` is a hard FK since
    # OptionsPosition is in the same app boundary as Trade.
    trade_id = models.UUIDField(null=True, blank=True, db_index=True,
                                 help_text="Soft FK to trades.Trade.id")
    options_position = models.ForeignKey(
        "trading.OptionsPosition", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="events",
    )
    order = models.ForeignKey(
        "trading.Order", null=True, blank=True, on_delete=models.SET_NULL, related_name="+",
    )
    signal_id = models.BigIntegerField(null=True, blank=True, db_index=True,
                                        help_text="Soft FK to strategies.Signal.id")

    # Body
    payload = models.JSONField(default=dict, blank=True,
                                help_text="Type-specific structured body (prompt, risk breakdown, etc.)")
    text = models.TextField(blank=True, default="",
                             help_text="Human-readable line for the timeline UI")

    # Request context (web events only)
    ip = models.GenericIPAddressField(null=True, blank=True)
    request_id = models.CharField(max_length=64, blank=True, default="")

    class Meta:
        ordering = ["-ts"]
        indexes = [
            models.Index(fields=["tenant", "-ts"]),
            models.Index(fields=["tenant", "type", "-ts"]),
            models.Index(fields=["tenant", "workflow_run", "-ts"]),
            models.Index(fields=["tenant", "trade_id", "-ts"]),
        ]

    def __str__(self) -> str:
        return f"[{self.ts:%Y-%m-%d %H:%M:%S}] {self.type} {self.text[:60]}"
