"""Factory-boy factories.

Each factory defaults to a valid, saveable instance. Test code overrides
the attributes it cares about and inherits the rest.

Design goals:
  - Every tenant-scoped factory takes a `tenant=` kwarg (no magic globals).
  - Numeric defaults line up with the paper-mode persona in CLAUDE.md:
    500,000 INR capital, no open positions.
  - No network calls, no broker hits, no Anthropic calls — factories are
    pure Django ORM plus Python primitives.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import factory
from factory.django import DjangoModelFactory

from apps.accounts.models import User
from apps.broker.models import BrokerLink
from apps.orders.models import Order, OutboxEvent
from apps.portfolio.models import Portfolio, Position
from apps.tenants.models import Membership, Tenant


class TenantFactory(DjangoModelFactory):
    class Meta:
        model = Tenant

    id = factory.LazyFunction(uuid.uuid4)
    name = factory.Sequence(lambda n: f"Tenant {n}")
    slug = factory.Sequence(lambda n: f"tenant-{n}")
    kind = Tenant.Kind.RETAIL


class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
        django_get_or_create = ("email",)

    email = factory.Sequence(lambda n: f"user{n}@alphadesk.test")
    full_name = factory.Sequence(lambda n: f"User {n}")
    is_active = True

    @factory.post_generation
    def password(self, create, extracted, **kwargs):
        self.set_password(extracted or "x" * 14)
        if create:
            self.save()


class MembershipFactory(DjangoModelFactory):
    class Meta:
        model = Membership

    user = factory.SubFactory(UserFactory)
    tenant = factory.SubFactory(TenantFactory)
    role = Membership.Role.OWNER
    is_active = True


class BrokerLinkFactory(DjangoModelFactory):
    class Meta:
        model = BrokerLink

    tenant = factory.SubFactory(TenantFactory)
    owner = factory.SubFactory(UserFactory)
    broker_name = "paper"
    status = "linked"


class PortfolioFactory(DjangoModelFactory):
    class Meta:
        model = Portfolio

    tenant = factory.SubFactory(TenantFactory)
    name = "Default"
    capital = Decimal("500000.00")
    used_capital = Decimal("0.00")
    realized_pnl = Decimal("0.00")
    day_pnl = Decimal("0.00")
    mode = "paper"


class PositionFactory(DjangoModelFactory):
    class Meta:
        model = Position

    tenant = factory.SubFactory(TenantFactory)
    portfolio = factory.SubFactory(PortfolioFactory)
    symbol = "HDFCBANK"
    side = "BUY"
    qty = 10
    avg_price = Decimal("1600.0000")


class OrderFactory(DjangoModelFactory):
    class Meta:
        model = Order

    tenant = factory.SubFactory(TenantFactory)
    portfolio = factory.SubFactory(PortfolioFactory)
    created_by = factory.SubFactory(UserFactory)
    symbol = "HDFCBANK"
    side = "BUY"
    qty = 5
    order_type = "MARKET"
    product = "INTRADAY"
    price = Decimal("1600.0000")
    status = Order.Status.QUEUED
    origin = "ui"


class OutboxEventFactory(DjangoModelFactory):
    class Meta:
        model = OutboxEvent

    order = factory.SubFactory(OrderFactory)
    payload = factory.LazyAttribute(
        lambda o: {
            "symbol": o.order.symbol,
            "side": o.order.side,
            "qty": o.order.qty,
            "order_type": o.order.order_type,
            "product": o.order.product,
            "price": float(o.order.price or 0),
        }
    )
    status = OutboxEvent.Status.PENDING
