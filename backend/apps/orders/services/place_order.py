"""Use-case: place an order. Uses Outbox pattern — returns HTTP 202 + order id."""
from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction

from apps.agents_core.domain.contracts import RiskPort
from apps.common.exceptions import RiskRejected
from apps.orders.domain.entities import OrderDraft, OrderResult
from apps.orders.models import Order, OutboxEvent


@dataclass
class PlaceOrder:
    risk: RiskPort

    @transaction.atomic
    def execute(self, tenant, user, portfolio, draft: OrderDraft,
                idempotency_key: str = "") -> OrderResult:
        # Idempotency dedupe (best-effort — full impl uses Redis lock)
        if idempotency_key:
            existing = Order.objects.filter(
                tenant=tenant, idempotency_key=idempotency_key,
            ).first()
            if existing:
                return OrderResult(id=str(existing.id), status=existing.status)

        decision = self.risk.validate({**draft.model_dump(), "portfolio_id": portfolio.id})
        if not decision.approved:
            raise RiskRejected(decision.reason)

        order = Order.objects.create(
            tenant=tenant,
            portfolio=portfolio,
            created_by=user,
            broker_link=portfolio.broker_link,
            symbol=draft.symbol,
            side=draft.side,
            qty=draft.qty,
            order_type=draft.order_type,
            product=draft.product,
            price=draft.price,
            sl=draft.sl,
            tp=draft.tp,
            status=Order.Status.QUEUED,
            idempotency_key=idempotency_key,
            origin=draft.origin,
        )
        OutboxEvent.objects.create(
            order=order,
            payload=draft.model_dump(),
        )
        return OrderResult(id=str(order.id), status=order.status)
