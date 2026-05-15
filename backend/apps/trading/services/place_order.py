"""Use-case: place an order. Uses Outbox pattern — returns HTTP 202 + order id.

Risk validation goes through the canonical RiskEngine in
apps.trading.services.risk_engine (10-criterion gate ported from the
legacy trading.services.risk_engine).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from django.db import transaction

from apps.common.exceptions import RiskRejected
from apps.trading.domain.entities import OrderDraft, OrderResult
from apps.trading.models import Order, OutboxEvent
from apps.trading.services.risk_engine import RiskEngine, TradeDraft


@dataclass
class PlaceOrder:
    risk: RiskEngine = field(default_factory=RiskEngine)

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

        # Convert OrderDraft → TradeDraft for the canonical risk engine.
        trade_draft = TradeDraft(
            symbol=draft.symbol,
            side=draft.side,
            entry_price=float(draft.price or 0),
            stop_loss=float(draft.sl or 0),
            target=float(draft.tp or 0),
            quantity=int(draft.qty),
            confidence=0.55,  # Manual / API orders default to threshold; agents pass real value
        )
        decision = self.risk.validate(trade_draft, portfolio_id=portfolio.id)
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
