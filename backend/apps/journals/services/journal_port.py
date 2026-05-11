"""JournalPort adapter — used by the agent framework.

Post redesign-v2, this adapter writes to `apps.events.Event` rather
than the old `apps.journals.JournalEntry`. The JournalEntry model is
kept around (with zero rows) for back-compat with any external code
that may still query it; new writes flow into the unified Event log.
"""
from __future__ import annotations

from apps.events.models import Event
from apps.events.services.event_writer import emit


class JournalAdapter:
    """Adapter implementing the `JournalPort` protocol from
    apps.agents_core.domain.contracts. Plugin nodes call
    `ctx.journal.record(entry)` and we forward that into the unified
    Event log.
    """

    # Map common plugin journal kinds → Event types
    _KIND_TO_EVENT_TYPE = {
        "plan":       Event.Type.TRADE_PLANNED,
        "entry":      Event.Type.TRADE_OPENED,
        "exit":       Event.Type.TRADE_CLOSED,
        "adjustment": Event.Type.TRADE_TRAILED,
        "rejection":  Event.Type.RISK_REJECTED,
    }

    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

    def record(self, entry: dict) -> None:
        # Lazy lookup so plugins can pass any kind; unknown kinds default
        # to a generic workflow info event so nothing is silently dropped.
        kind = entry.get("kind", "plan")
        event_type = self._KIND_TO_EVENT_TYPE.get(kind, Event.Type.WORKFLOW_COMPLETED)

        # Resolve tenant lazily — tenant_id may be a UUID or model instance
        from apps.tenants.models import Tenant
        tenant = (
            entry.get("tenant")
            or Tenant.objects.filter(id=self.tenant_id).first()
        )

        emit(
            tenant=tenant,
            type=event_type,
            text=entry.get("title", "") or entry.get("kind", ""),
            payload={
                "body": entry.get("body", ""),
                "tags": entry.get("tags", []),
                "meta": entry.get("meta", {}),
                "kind": kind,
                "portfolio_id": str(entry.get("portfolio_id"))
                                  if entry.get("portfolio_id") else None,
                "agent_run_id": str(entry.get("agent_run_id"))
                                  if entry.get("agent_run_id") else None,
                "order_id": str(entry.get("order_id"))
                               if entry.get("order_id") else None,
            },
        )
