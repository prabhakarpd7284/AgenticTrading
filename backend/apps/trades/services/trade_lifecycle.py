"""Trade state machine.

Every transition writes one events.Event row. Invalid transitions raise
TradeTransitionError. The state diagram:

    PLAN ─approve──► APPROVED ─queue──► QUEUED ─sent──► SENT ─fill──► PARTIAL/FILLED
       │                                                              │
       └─reject──► REJECTED                                            └─close──► CLOSED
                                              SENT ─cancel──► CANCELLED
                                              PARTIAL ─cancel──► CANCELLED
                                              PLAN/APPROVED ─expire──► EXPIRED
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from django.db import transaction
from django.utils import timezone

from apps.events.models import Event
from apps.events.services.event_writer import emit
from apps.trades.models import Trade

log = structlog.get_logger(__name__)


class TradeTransitionError(Exception):
    """Raised on an invalid state transition (e.g. CLOSED → SENT)."""


# Allowed transitions: from_status → set of to_status
_TRANSITIONS = {
    Trade.Status.PLAN:      {Trade.Status.APPROVED, Trade.Status.REJECTED, Trade.Status.EXPIRED},
    Trade.Status.APPROVED:  {Trade.Status.QUEUED, Trade.Status.CANCELLED, Trade.Status.EXPIRED},
    Trade.Status.QUEUED:    {Trade.Status.SENT, Trade.Status.CANCELLED, Trade.Status.REJECTED},
    Trade.Status.SENT:      {Trade.Status.PARTIAL, Trade.Status.FILLED, Trade.Status.CANCELLED,
                              Trade.Status.REJECTED},
    Trade.Status.PARTIAL:   {Trade.Status.FILLED, Trade.Status.CANCELLED, Trade.Status.CLOSED},
    Trade.Status.FILLED:    {Trade.Status.CLOSED},
    # Terminal states
    Trade.Status.CLOSED:    set(),
    Trade.Status.REJECTED:  set(),
    Trade.Status.CANCELLED: set(),
    Trade.Status.EXPIRED:   set(),
}


def _check_transition(from_status: str, to_status: str) -> None:
    allowed = _TRANSITIONS.get(from_status, set())
    if to_status not in allowed:
        raise TradeTransitionError(
            f"Invalid Trade transition {from_status} → {to_status}. "
            f"Allowed from {from_status}: {sorted(allowed) or '(terminal)'}"
        )


@transaction.atomic
def transition(
    trade: Trade,
    to_status: str,
    *,
    event_type: str,
    text: str = "",
    payload: dict[str, Any] | None = None,
    workflow_run=None,
    **trade_updates,
) -> Trade:
    """Atomically move the Trade to a new status and emit an Event row.

    Use `trade_updates` to set additional fields in the same transaction
    (e.g. fill_price, fill_quantity, realized_pnl, closed_at).
    """
    from_status = trade.status
    _check_transition(from_status, to_status)

    trade.status = to_status
    for k, v in trade_updates.items():
        setattr(trade, k, v)
    trade.save()

    emit(
        tenant=trade.tenant,
        type=event_type,
        text=text or f"trade {trade.id} {from_status} → {to_status}",
        actor_kind=Event.ActorKind.WORKFLOW if workflow_run else Event.ActorKind.SYSTEM,
        workflow_run=workflow_run,
        trade_id=trade.id,
        payload={"from": from_status, "to": to_status, **(payload or {})},
    )
    return trade


# ── Convenience wrappers (one per event type for readability) ──

def approve(trade: Trade, *, reason: str = "", risk_details: dict | None = None,
            workflow_run=None) -> Trade:
    return transition(
        trade, Trade.Status.APPROVED,
        event_type=Event.Type.RISK_APPROVED,
        text=f"risk approved {trade.symbol} {trade.side} {trade.quantity}",
        payload={"reason": reason},
        workflow_run=workflow_run,
        risk_approved=True,
        risk_reason=reason,
        risk_details=risk_details or {},
        risk_decided_at=timezone.now(),
    )


def reject(trade: Trade, *, reason: str, risk_details: dict | None = None,
           workflow_run=None) -> Trade:
    return transition(
        trade, Trade.Status.REJECTED,
        event_type=Event.Type.RISK_REJECTED,
        text=f"risk rejected: {reason}",
        payload={"reason": reason, "criteria": (risk_details or {}).get("criteria", {})},
        workflow_run=workflow_run,
        risk_approved=False,
        risk_reason=reason,
        risk_details=risk_details or {},
        risk_decided_at=timezone.now(),
    )


def fill(trade: Trade, *, price: float, qty: int, fully_filled: bool = True,
         workflow_run=None) -> Trade:
    next_status = Trade.Status.FILLED if fully_filled else Trade.Status.PARTIAL
    return transition(
        trade, next_status,
        event_type=Event.Type.ORDER_FILLED if fully_filled else Event.Type.ORDER_PARTIAL,
        text=f"{'fully' if fully_filled else 'partially'} filled {qty} @ {price}",
        payload={"price": price, "qty": qty},
        workflow_run=workflow_run,
        fill_price=price,
        fill_quantity=qty,
        filled_at=timezone.now() if fully_filled else None,
    )


def close(trade: Trade, *, price: float, qty: int, reason: str,
          realized_pnl: float, workflow_run=None) -> Trade:
    return transition(
        trade, Trade.Status.CLOSED,
        event_type=Event.Type.TRADE_CLOSED,
        text=f"closed {qty} @ {price} ({reason}) pnl={realized_pnl}",
        payload={"price": price, "qty": qty, "reason": reason, "pnl": realized_pnl},
        workflow_run=workflow_run,
        exit_price=price,
        exit_quantity=qty,
        close_reason=reason,
        realized_pnl=realized_pnl,
        closed_at=timezone.now(),
    )


def cancel(trade: Trade, *, reason: str = "operator_cancel", workflow_run=None) -> Trade:
    return transition(
        trade, Trade.Status.CANCELLED,
        event_type=Event.Type.ORDER_CANCELLED,
        text=f"cancelled: {reason}",
        payload={"reason": reason},
        workflow_run=workflow_run,
    )


def expire(trade: Trade, *, reason: str = "timeout", workflow_run=None) -> Trade:
    return transition(
        trade, Trade.Status.EXPIRED,
        event_type=Event.Type.WORKFLOW_CANCELLED,
        text=f"expired: {reason}",
        payload={"reason": reason},
        workflow_run=workflow_run,
    )
