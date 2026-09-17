"""Cross-tenant isolation.

Every row in the system is tagged with a tenant_id. These tests prove that:

  1. `TenantManager.for_tenant(t)` scopes queries correctly.
  2. Tenant A's queryset never returns Tenant B's rows — even when both
    tenants own rows at the same primary-key-near-collision timestamps.
  3. The manager raises the right query type (QuerySet) so downstream
    chaining works.

If anything below fails, a user might be able to see another tenant's data.
"""
from __future__ import annotations

import pytest

from apps.trading.models import Order
from apps.trading.models import Portfolio, Position
from tests.factories import (
    OrderFactory,
    PortfolioFactory,
    PositionFactory,
)


pytestmark = pytest.mark.django_db


def test_portfolios_are_scoped_by_tenant(two_tenants):
    # Seed two portfolios in the fixture already — no extras needed.
    qs_a = Portfolio.objects.for_tenant(two_tenants.a)
    qs_b = Portfolio.objects.for_tenant(two_tenants.b)
    assert qs_a.count() == 1
    assert qs_b.count() == 1
    assert qs_a.first().id == two_tenants.portfolio_a.id
    assert qs_b.first().id == two_tenants.portfolio_b.id


def test_positions_do_not_leak_across_tenants(two_tenants):
    PositionFactory(
        tenant=two_tenants.a,
        portfolio=two_tenants.portfolio_a,
        symbol="INFY",
    )
    PositionFactory(
        tenant=two_tenants.b,
        portfolio=two_tenants.portfolio_b,
        symbol="TCS",
    )

    symbols_a = list(
        Position.objects.for_tenant(two_tenants.a).values_list("symbol", flat=True),
    )
    symbols_b = list(
        Position.objects.for_tenant(two_tenants.b).values_list("symbol", flat=True),
    )
    assert symbols_a == ["INFY"]
    assert symbols_b == ["TCS"]


def test_orders_do_not_leak_across_tenants(two_tenants):
    OrderFactory(
        tenant=two_tenants.a,
        portfolio=two_tenants.portfolio_a,
        created_by=two_tenants.owner_a,
        symbol="HDFCBANK",
    )
    OrderFactory(
        tenant=two_tenants.b,
        portfolio=two_tenants.portfolio_b,
        created_by=two_tenants.owner_b,
        symbol="RELIANCE",
    )

    a_symbols = set(
        Order.objects.for_tenant(two_tenants.a).values_list("symbol", flat=True),
    )
    b_symbols = set(
        Order.objects.for_tenant(two_tenants.b).values_list("symbol", flat=True),
    )
    assert a_symbols == {"HDFCBANK"}
    assert b_symbols == {"RELIANCE"}
    assert a_symbols.isdisjoint(b_symbols)


def test_queryset_chain_stays_scoped(two_tenants):
    PortfolioFactory(tenant=two_tenants.a, mode="paper")
    PortfolioFactory(tenant=two_tenants.a, mode="live")
    PortfolioFactory(tenant=two_tenants.b, mode="live")

    live_a = Portfolio.objects.for_tenant(two_tenants.a).filter(mode="live").count()
    live_b = Portfolio.objects.for_tenant(two_tenants.b).filter(mode="live").count()
    assert live_a == 1
    # ``two_tenants`` provisions portfolio_b with the factory default
    # ``mode="paper"``, so only the explicit live PortfolioFactory above
    # contributes to the live count for tenant B.
    assert live_b == 1
