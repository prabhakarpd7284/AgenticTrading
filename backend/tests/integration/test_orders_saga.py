"""Outbox saga behaviour — idempotency, retries, DLQ.

These are integration tests because the saga is inherently about wiring:
  Order ↔ OutboxEvent ↔ broker registry ↔ status-machine transitions.

We register a paper broker stub in the registry whose `place()` can be
configured per test (happy-path / raise / count calls).
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest
from django.utils import timezone

from apps.orders.models import Order, OutboxEvent
from apps.orders.services.order_saga import MAX_ATTEMPTS, OrderSaga
from tests.factories import OrderFactory, OutboxEventFactory, PortfolioFactory


pytestmark = pytest.mark.django_db


class _HappyBroker:
    def __init__(self):
        self.calls: list[dict] = []

    def place(self, payload):  # noqa: D401
        self.calls.append(payload)

        class Ack:
            id = "BROKER-ACK-1"

        return Ack()


class _FailingBroker:
    """Always raises — lets us exercise the retry + DLQ path."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    def place(self, _payload):  # noqa: D401
        self.calls += 1
        raise self.exc


def _with_broker(broker):
    """Patch broker_registry.get to return the given broker stub."""
    return patch(
        "apps.orders.services.order_saga.broker_registry.get",
        return_value=broker,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_saga_transitions_to_sent_on_broker_ack(tenant, paper_portfolio, owner):
    order = OrderFactory(
        tenant=tenant,
        portfolio=paper_portfolio,
        created_by=owner,
        status=Order.Status.QUEUED,
    )
    event = OutboxEventFactory(order=order)
    broker = _HappyBroker()

    with _with_broker(broker):
        OrderSaga().handle(event)

    event.refresh_from_db()
    order.refresh_from_db()
    assert event.status == OutboxEvent.Status.SUCCEEDED
    assert order.status == Order.Status.SENT
    assert order.broker_order_id == "BROKER-ACK-1"
    assert event.attempts == 1
    assert len(broker.calls) == 1


# ---------------------------------------------------------------------------
# Retry path — failure but under MAX_ATTEMPTS
# ---------------------------------------------------------------------------
def test_saga_reschedules_with_exponential_backoff_on_failure(
    tenant, paper_portfolio, owner,
):
    order = OrderFactory(
        tenant=tenant, portfolio=paper_portfolio, created_by=owner,
    )
    event = OutboxEventFactory(order=order, attempts=0)
    broker = _FailingBroker(RuntimeError("broker 503"))

    before = timezone.now()
    with _with_broker(broker):
        OrderSaga().handle(event)

    event.refresh_from_db()
    order.refresh_from_db()
    assert event.status == OutboxEvent.Status.PENDING
    assert event.attempts == 1
    assert event.last_error == "broker 503"
    # next_run_at = now + 2 seconds on first failure (2 ** 1).
    delta = (event.next_run_at - before).total_seconds()
    assert 1.5 <= delta <= 3.5
    # Order is not marked FAILED yet — only at DLQ.
    assert order.status == Order.Status.QUEUED


# ---------------------------------------------------------------------------
# DLQ path — exhausted attempts
# ---------------------------------------------------------------------------
def test_saga_sends_to_dlq_after_max_attempts(tenant, paper_portfolio, owner):
    order = OrderFactory(
        tenant=tenant, portfolio=paper_portfolio, created_by=owner,
    )
    event = OutboxEventFactory(
        order=order,
        attempts=MAX_ATTEMPTS - 1,  # this call will be the Nth attempt
    )
    broker = _FailingBroker(RuntimeError("broker gone"))

    with _with_broker(broker):
        OrderSaga().handle(event)

    event.refresh_from_db()
    order.refresh_from_db()
    assert event.status == OutboxEvent.Status.DLQ
    assert event.attempts == MAX_ATTEMPTS
    assert order.status == Order.Status.FAILED
    assert order.error == "broker gone"


# ---------------------------------------------------------------------------
# Idempotency — two outbox rows for the same order still call the broker
# each time, but Order.idempotency_key guarantees PlaceOrder dedupes at the
# entry point. We prove the entry-point dedupe in this test.
# ---------------------------------------------------------------------------
def test_place_order_dedupes_on_idempotency_key(tenant, paper_portfolio, owner):
    from apps.orders.domain.entities import OrderDraft
    from apps.orders.services.place_order import PlaceOrder

    class _AlwaysApprove:
        # PlaceOrder calls `risk.validate(trade_draft, portfolio_id=...)`
        # since the Phase 2 RiskEngine consolidation — accept the kwarg.
        def validate(self, _draft, **_kwargs):
            from apps.agents_core.domain.contracts import RiskDecision
            return RiskDecision(approved=True)

    draft = OrderDraft(
        symbol="HDFCBANK", side="BUY", qty=5,
        order_type="MARKET", product="INTRADAY",
        price=1_600.0,
    )
    uc = PlaceOrder(risk=_AlwaysApprove())
    key = "client-abc-123"

    first = uc.execute(tenant, owner, paper_portfolio, draft, idempotency_key=key)
    second = uc.execute(tenant, owner, paper_portfolio, draft, idempotency_key=key)

    assert first.id == second.id
    assert Order.objects.filter(
        tenant=tenant, idempotency_key=key,
    ).count() == 1
    assert OutboxEvent.objects.count() == 1
