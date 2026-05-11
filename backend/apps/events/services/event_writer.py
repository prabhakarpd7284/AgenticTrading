"""Single emit point for the unified Event log.

Every state change in the system flows through `emit()`. Failures are
swallowed with structlog warning — emitting an Event MUST NOT block
trading flow or any other write path.

This is the only public API for the events app's domain code.
"""
from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

import structlog
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.utils import timezone

from apps.events.models import Event

log = structlog.get_logger(__name__)


def emit(
    *,
    tenant,
    type: str,
    text: str = "",
    severity: str = "info",
    actor_kind: str = "workflow",
    actor_user=None,
    workflow_run=None,
    step_name: str = "",
    trade_id: UUID | None = None,
    order=None,
    signal_id: int | None = None,
    payload: dict[str, Any] | None = None,
    ts=None,
    ip: str | None = None,
    request_id: str = "",
    broadcast: bool = True,
) -> Event | None:
    """Append one row to the Event log + (optionally) broadcast via Channels.

    Returns the Event on success, None on failure. Never raises.

    `broadcast=True` (default) publishes to two channel groups:
      - `events.{tenant_id}`  — system-wide firehose (drives Now activity feed)
      - `runs.{workflow_run_id}` — per-run timeline (only if workflow_run set)
    """
    try:
        event = Event.objects.create(
            tenant=tenant,
            ts=ts or timezone.now(),
            type=type,
            severity=severity,
            actor_kind=actor_kind,
            actor_user=actor_user,
            workflow_run=workflow_run,
            step_name=step_name,
            trade_id=trade_id,
            order=order,
            signal_id=signal_id,
            payload=payload or {},
            text=text,
            ip=ip,
            request_id=request_id,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("event.persist_failed", type=type, error=str(e))
        return None

    if broadcast:
        _broadcast(event)
    return event


def _broadcast(event: Event) -> None:
    """Publish the event to Channels groups. Non-blocking on failure."""
    layer = get_channel_layer()
    if layer is None:
        return

    body = {
        "type": "event.message",
        "id": event.id,
        "ts": event.ts.isoformat(),
        "event_type": event.type,
        "severity": event.severity,
        "actor_kind": event.actor_kind,
        "workflow_run": str(event.workflow_run_id) if event.workflow_run_id else None,
        "step_name": event.step_name,
        "trade_id": str(event.trade_id) if event.trade_id else None,
        "order_id": str(event.order_id) if event.order_id else None,
        "signal_id": event.signal_id,
        "text": event.text,
        "payload": event.payload,
    }

    try:
        # System-wide firehose for the Now activity feed
        async_to_sync(layer.group_send)(f"events.{event.tenant_id}", body)
        # Per-run timeline (only if this event belongs to a workflow run)
        if event.workflow_run_id:
            async_to_sync(layer.group_send)(f"runs.{event.workflow_run_id}", body)
    except Exception as e:  # noqa: BLE001
        log.warning("event.broadcast_failed", event_id=event.id, error=str(e))


def request_id_for(req) -> str:
    """Stable per-request UUID for grouping events from the same HTTP call."""
    rid = getattr(req, "_event_request_id", None)
    if rid:
        return rid
    rid = uuid.uuid4().hex[:16]
    setattr(req, "_event_request_id", rid)
    return rid
