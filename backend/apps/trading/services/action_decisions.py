"""Action Board decisions: approve / skip / snooze / expire (UX redesign R3).

Approve never shortcuts the order path: it re-runs the dry-run gates with the
planner's own confidence (``PlaceOrder`` scores automated orders at a fixed
0.55, so without this re-check a low-confidence idea would sail through the
confidence criterion), then submits through ``PlaceOrder`` with
``origin="strategy"`` — kill switch → AI pause → trading-rules gate
(cooldown, sentiment) → notional backstop → RiskEngine (the last gate) →
outbox → paper fill. PAPER only in this phase: a live portfolio is refused.

Every decision — including a rejection by the order path — is written to the
event log through ``event_writer.emit()`` (non-blocking).
"""
from __future__ import annotations

from datetime import datetime

import structlog
from django.db import transaction
from django.utils import timezone

log = structlog.get_logger(__name__)

DECIDE_ROLES = ("owner", "admin", "trader")
OPEN = ("proposed", "snoozed")


class ActionConflict(Exception):
    """The card cannot take this decision in its current state (HTTP 409)."""


def can_decide(user, tenant) -> bool:
    if user is None or getattr(user, "is_anonymous", True) or tenant is None:
        return False
    from apps.tenants.models import Membership

    return Membership.objects.filter(user=user, tenant=tenant, is_active=True,
                                     role__in=DECIDE_ROLES).exists()


def _emit(card, *, type_: str, step: str, text: str, user=None, severity: str = "info",
          extra: dict | None = None, order=None) -> None:
    from apps.events.services.event_writer import emit

    emit(tenant=card.tenant, type=type_, severity=severity,
         actor_kind="user" if user is not None else "system", actor_user=user,
         step_name=step, text=text[:500], order=order,
         payload={"action_id": str(card.id), "date": card.date.isoformat(), "symbol": card.symbol,
                  "side": card.side, "strategy": card.strategy, "status": card.status,
                  "qty": card.suggested_qty, "note": card.decision_note, **(extra or {})})


def refresh_statuses(qs, now: datetime | None = None) -> int:
    """Lazy lifecycle: snoozes that ran out go back to proposed; open cards
    past ``expires_at`` (15:00 IST) expire. Returns how many changed."""
    from apps.trading.models import TodayAction

    now = now or timezone.now()
    changed = 0
    for card in qs.filter(status=TodayAction.Status.SNOOZED, snoozed_until__lte=now,
                          expires_at__gt=now):
        card.status = TodayAction.Status.PROPOSED
        card.save(update_fields=["status", "updated_at"])
        changed += 1
    for card in qs.filter(status__in=OPEN, expires_at__lte=now):
        card.status = TodayAction.Status.EXPIRED
        card.save(update_fields=["status", "updated_at"])
        _emit(card, type_="workflow.step.completed", step="action_board.expire",
              text=f"Action Board card expired undecided: {card.side} {card.symbol}")
        changed += 1
    return changed


def _lock(card):
    from apps.trading.models import TodayAction

    return TodayAction.objects.select_for_update().get(pk=card.pk)


def _require_open(card, now) -> None:
    from apps.trading.models import TodayAction

    if card.status == TodayAction.Status.SNOOZED and card.snoozed_until and card.snoozed_until <= now:
        card.status = TodayAction.Status.PROPOSED
    if card.status not in OPEN:
        raise ActionConflict(f"card is {card.status} — only proposed or snoozed cards can be decided")
    if card.expires_at <= now:
        raise ActionConflict("card has expired (cards expire at 15:00 IST)")


def skip(card, user, reason: str = "", now: datetime | None = None):
    from apps.trading.models import TodayAction

    now = now or timezone.now()
    with transaction.atomic():
        card = _lock(card)
        _require_open(card, now)
        card.status = TodayAction.Status.SKIPPED
        card.decided_at, card.decided_by = now, user
        card.decision_note = (reason or "").strip()[:2000]
        card.save(update_fields=["status", "decided_at", "decided_by", "decision_note", "updated_at"])
    _emit(card, type_="workflow.step.completed", step="action_board.skip", user=user,
          text=f"Skipped {card.side} {card.symbol}" + (f": {card.decision_note}" if card.decision_note else ""))
    return card


def snooze(card, user, until: datetime, now: datetime | None = None):
    from apps.trading.models import TodayAction

    now = now or timezone.now()
    if until <= now:
        raise ActionConflict("snooze time must be in the future")
    with transaction.atomic():
        card = _lock(card)
        _require_open(card, now)
        if until >= card.expires_at:
            raise ActionConflict("snooze must end before the card expires (15:00 IST)")
        card.status = TodayAction.Status.SNOOZED
        card.snoozed_until = until
        card.save(update_fields=["status", "snoozed_until", "updated_at"])
    _emit(card, type_="workflow.step.completed", step="action_board.snooze", user=user,
          text=f"Snoozed {card.side} {card.symbol} until {timezone.localtime(until):%H:%M}",
          extra={"snoozed_until": until.isoformat()})
    return card


def approve(card, user, note: str = "", now: datetime | None = None):
    """Approve → PlaceOrder (paper). The card ends ``executed`` (order accepted)
    or ``rejected`` (with the gate's reason). Raises ActionConflict only for
    state problems (already decided, expired, wrong day, live portfolio)."""
    from apps.common.exceptions import RiskRejected
    from apps.trading.domain.entities import OrderDraft
    from apps.trading.models import TodayAction
    from apps.trading.services.action_board import dry_run_gates
    from apps.trading.services.cooldown import evaluate_for_portfolio
    from apps.trading.services.place_order import PlaceOrder

    now = now or timezone.now()
    with transaction.atomic():
        card = _lock(card)
        _require_open(card, now)
        if card.date != timezone.localtime(now).date():
            raise ActionConflict("only today's cards can be approved")
        portfolio = card.portfolio
        if portfolio.mode != "paper":
            raise ActionConflict("Action Board is PAPER-only in this phase — portfolio is "
                                 f"{portfolio.mode}; refusing to place a live order")
        if card.shadow:
            raise ActionConflict("shadow card (recorded during a cooldown) — not executable")
        card.status = TodayAction.Status.APPROVED
        card.decided_at, card.decided_by = now, user
        card.decision_note = (note or "").strip()[:2000]
        card.save(update_fields=["status", "decided_at", "decided_by", "decision_note", "updated_at"])

    entry, stop, target = float(card.entry_price), float(card.stoploss), float(card.target)
    try:
        cd = evaluate_for_portfolio(portfolio, card.date)
    except Exception:  # noqa: BLE001 — PlaceOrder evaluates the cooldown itself too
        cd = None
    mult = float(cd.risk_multiplier) if cd is not None and not cd.blocks_entries else 1.0
    pre = dry_run_gates(portfolio, symbol=card.symbol, side=card.side, entry=entry, stop=stop,
                        target=target, qty=card.suggested_qty, confidence=float(card.confidence or 0),
                        risk_multiplier=mult, day=card.date, cooldown=cd)

    def reject(reason: str, stage: str):
        card.status = TodayAction.Status.REJECTED
        card.decision_note = (f"{card.decision_note}\n" if card.decision_note else "") + f"Rejected: {reason}"
        card.risk_check = {**(card.risk_check or {}), "at_approve": pre, "rejected_at": stage}
        card.save(update_fields=["status", "decision_note", "risk_check", "updated_at"])
        _emit(card, type_="risk.rejected", step="action_board.approve", user=user, severity="warn",
              text=f"Approve rejected for {card.side} {card.suggested_qty} {card.symbol}: {reason}",
              extra={"stage": stage, "reason": reason})
        return card

    if not pre["approved"]:
        return reject(pre["reason"], "pre_check")

    draft = OrderDraft(symbol=card.symbol, side=card.side, qty=card.suggested_qty, order_type="LIMIT",
                       product="INTRADAY", price=entry, sl=stop, tp=target, origin="strategy")
    try:
        result = PlaceOrder().execute(tenant=card.tenant, user=user, portfolio=portfolio, draft=draft,
                                      idempotency_key=f"action-{card.id}")
    except RiskRejected as e:
        return reject(str(e), "place_order")
    except Exception as e:  # noqa: BLE001 — never leave a card stuck in "approved"
        log.exception("action_board.approve_failed", action_id=str(card.id))
        return reject(f"order path error: {type(e).__name__}: {e}", "error")

    from apps.trading.models import Order

    order = Order.objects.filter(id=result.id).first()
    card.status = TodayAction.Status.EXECUTED
    card.order = order
    card.risk_check = {**(card.risk_check or {}), "at_approve": pre}
    card.save(update_fields=["status", "order", "risk_check", "updated_at"])
    _emit(card, type_="order.queued", step="action_board.approve", user=user, order=order,
          text=f"Approved {card.side} {card.suggested_qty} {card.symbol} LIMIT {entry:.2f} → paper order "
               f"{result.id} ({result.status})",
          extra={"order_id": result.id, "order_status": result.status})
    return card
