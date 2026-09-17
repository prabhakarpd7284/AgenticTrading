"""Order-path safety guards — kill switch, live gate, notional caps, idempotency.

Covers the Phase-2 order cluster (#12 sizing, #13 live gate, #14 kill switch,
#15 idempotency constraint, #16 fail-loud live placement).
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest
from django.conf import settings
from django.db import IntegrityError, transaction
from django.test import override_settings

from apps.agents_core.domain.contracts import RiskDecision
from apps.common.exceptions import RiskRejected
from apps.system.services.flags import set_flag
from apps.trading.domain.entities import OrderDraft
from apps.trading.models import Order, OutboxEvent
from apps.trading.services.order_saga import OrderSaga
from apps.trading.services.place_order import PlaceOrder
from tests.factories import BrokerLinkFactory, OrderFactory, OutboxEventFactory, PortfolioFactory

pytestmark = pytest.mark.django_db


class _AlwaysApprove:
    def validate(self, _draft, **_kwargs):
        return RiskDecision(approved=True)


class _HappyBroker:
    def __init__(self):
        self.calls: list = []

    def place(self, payload):
        self.calls.append(payload)

        class Ack:
            id = "BROKER-ACK-1"

        return Ack()


def _with_broker(broker):
    return patch(
        "apps.trading.services.order_saga.broker_registry.get", return_value=broker
    )


def _draft(**over):
    base = dict(
        symbol="HDFCBANK", side="BUY", qty=5, order_type="MARKET",
        product="INTRADAY", price=1_600.0, sl=1_580.0, tp=1_660.0,
    )
    base.update(over)
    return OrderDraft(**base)


def _live_portfolio(tenant, owner):
    link = BrokerLinkFactory(tenant=tenant, owner=owner, broker_name="angel_one")
    pf = PortfolioFactory(
        tenant=tenant, mode="live", broker_link=link, capital=Decimal("500000"),
    )
    return pf, link


# ── #14 kill switch ────────────────────────────────────────────────────────
def test_place_order_blocked_by_kill_switch(tenant, paper_portfolio, owner):
    set_flag("kill_switch", True, tenant.id)
    uc = PlaceOrder(risk=_AlwaysApprove())
    with pytest.raises(RiskRejected, match="kill switch"):
        uc.execute(tenant, owner, paper_portfolio, _draft(), idempotency_key="k1")
    assert Order.objects.count() == 0


def test_saga_holds_event_when_kill_switch_on(tenant, paper_portfolio, owner):
    set_flag("kill_switch", True, tenant.id)
    order = OrderFactory(tenant=tenant, portfolio=paper_portfolio, created_by=owner,
                         status=Order.Status.QUEUED)
    event = OutboxEventFactory(order=order, status=OutboxEvent.Status.PENDING)
    broker = _HappyBroker()
    with _with_broker(broker):
        OrderSaga().handle(event)
    assert broker.calls == []                       # never placed during a halt
    event.refresh_from_db()
    assert event.status == OutboxEvent.Status.PENDING   # held, not failed
    assert event.attempts == 0                          # no retry attempt consumed
    assert "kill switch" in event.last_error


# ── #12 sizing backstops ────────────────────────────────────────────────────
def test_place_order_rejects_qty_over_absolute_cap(tenant, paper_portfolio, owner):
    uc = PlaceOrder(risk=_AlwaysApprove())
    huge = int(settings.ALPHADESK["MAX_ORDER_QTY"]) + 1
    with pytest.raises(RiskRejected, match="exceeds the absolute cap"):
        uc.execute(tenant, owner, paper_portfolio, _draft(qty=huge), idempotency_key="q1")
    assert Order.objects.count() == 0


def test_place_order_rejects_excessive_notional(tenant, paper_portfolio, owner):
    # qty under the qty-cap but price*qty over the absolute notional cap.
    uc = PlaceOrder(risk=_AlwaysApprove())
    with pytest.raises(RiskRejected, match="notional"):
        uc.execute(tenant, owner, paper_portfolio,
                   _draft(qty=20_000, price=1_000.0), idempotency_key="n1")
    assert Order.objects.count() == 0


# ── #13 live gate ───────────────────────────────────────────────────────────
def test_place_order_rejects_live_when_mode_not_live(tenant, owner):
    live_pf, _ = _live_portfolio(tenant, owner)
    uc = PlaceOrder(risk=_AlwaysApprove())
    with pytest.raises(RiskRejected, match="Live trading is disabled"):
        uc.execute(tenant, owner, live_pf, _draft(), idempotency_key="L1")
    assert Order.objects.count() == 0


@override_settings(TRADING_MODE="live")
def test_place_order_rejects_live_without_tenant_flag(tenant, owner):
    live_pf, _ = _live_portfolio(tenant, owner)  # flag NOT set
    uc = PlaceOrder(risk=_AlwaysApprove())
    with pytest.raises(RiskRejected, match="not enabled for this tenant"):
        uc.execute(tenant, owner, live_pf, _draft(), idempotency_key="L2")


@override_settings(TRADING_MODE="live")
def test_place_order_rejects_live_without_capable_broker(tenant, owner):
    live_pf, _ = _live_portfolio(tenant, owner)  # angel_one has no place()
    set_flag("live_trading_enabled", True, tenant.id)
    uc = PlaceOrder(risk=_AlwaysApprove())
    with pytest.raises(RiskRejected, match="no order-placement capability"):
        uc.execute(tenant, owner, live_pf, _draft(), idempotency_key="L3")


def test_saga_fails_live_order_when_trading_disabled(tenant, owner):
    live_pf, link = _live_portfolio(tenant, owner)
    order = OrderFactory(tenant=tenant, portfolio=live_pf, created_by=owner,
                         broker_link=link, status=Order.Status.QUEUED)
    event = OutboxEventFactory(order=order)
    broker = _HappyBroker()
    with _with_broker(broker):
        OrderSaga().handle(event)
    assert broker.calls == []
    order.refresh_from_db(); event.refresh_from_db()
    assert order.status == Order.Status.FAILED
    assert event.status == OutboxEvent.Status.FAILED
    assert "live trading disabled" in order.error


# ── #16 fail loud when the broker cannot place ──────────────────────────────
@override_settings(TRADING_MODE="live")
def test_saga_fails_loud_when_broker_cannot_place(tenant, owner):
    set_flag("live_trading_enabled", True, tenant.id)
    live_pf, link = _live_portfolio(tenant, owner)
    order = OrderFactory(tenant=tenant, portfolio=live_pf, created_by=owner,
                         broker_link=link, status=Order.Status.QUEUED)
    event = OutboxEventFactory(order=order)

    class _NoPlace:
        def place(self, _payload):
            raise NotImplementedError("angel_one adapter does not support place()")

    with _with_broker(_NoPlace()):
        OrderSaga().handle(event)
    order.refresh_from_db(); event.refresh_from_db()
    assert order.status == Order.Status.FAILED
    assert event.status == OutboxEvent.Status.FAILED   # terminal, NOT dlq (non-retryable)
    assert "cannot place" in order.error


# ── #15 idempotency unique constraint ───────────────────────────────────────
def test_duplicate_non_empty_idempotency_key_is_rejected(tenant, paper_portfolio, owner):
    Order.objects.create(
        tenant=tenant, portfolio=paper_portfolio, created_by=owner,
        symbol="X", side="BUY", qty=1, idempotency_key="dup",
    )
    with pytest.raises(IntegrityError), transaction.atomic():
        Order.objects.create(
            tenant=tenant, portfolio=paper_portfolio, created_by=owner,
            symbol="X", side="BUY", qty=1, idempotency_key="dup",
        )


def test_empty_idempotency_key_is_not_unique(tenant, paper_portfolio, owner):
    # The partial constraint excludes empty keys — header-less orders coexist.
    for _ in range(3):
        Order.objects.create(
            tenant=tenant, portfolio=paper_portfolio, created_by=owner,
            symbol="X", side="BUY", qty=1, idempotency_key="",
        )
    assert Order.objects.filter(idempotency_key="").count() == 3
