"""Shared pytest fixtures for the backend test suite.

Pytest collection rules:
  - Anything that touches the DB is auto-decorated with @pytest.mark.django_db
    via the `db` fixture.
  - The `two_tenants` fixture provides a pair of unrelated tenants — used by
    tenancy isolation tests so we never accidentally write a leak-the-row bug.
  - `paper_portfolio` returns a fully wired Portfolio with 500k INR capital,
    the canonical AlphaDesk paper-mode persona.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from rest_framework.test import APIClient

from tests.factories import (
    BrokerLinkFactory,
    MembershipFactory,
    PortfolioFactory,
    TenantFactory,
    UserFactory,
)


# ---------------------------------------------------------------------------
# Single tenant — most common case
# ---------------------------------------------------------------------------
@pytest.fixture
def tenant(db):
    return TenantFactory()


@pytest.fixture
def owner(db, tenant):
    user = UserFactory()
    MembershipFactory(user=user, tenant=tenant, role="owner")
    return user


@pytest.fixture
def broker_link(db, tenant):
    return BrokerLinkFactory(tenant=tenant)


@pytest.fixture
def paper_portfolio(db, tenant, broker_link):
    """Canonical paper-mode persona — 500,000 INR, no day P&L, no positions."""
    return PortfolioFactory(
        tenant=tenant,
        broker_link=broker_link,
        capital=Decimal("500000.00"),
        day_pnl=Decimal("0.00"),
        mode="paper",
    )


# ---------------------------------------------------------------------------
# Two tenants — for cross-tenant isolation tests
# ---------------------------------------------------------------------------
@dataclass
class TenantPair:
    a: object
    b: object
    owner_a: object
    owner_b: object
    portfolio_a: object
    portfolio_b: object


@pytest.fixture
def two_tenants(db) -> TenantPair:
    a, b = TenantFactory(), TenantFactory()
    oa, ob = UserFactory(), UserFactory()
    MembershipFactory(user=oa, tenant=a, role="owner")
    MembershipFactory(user=ob, tenant=b, role="owner")
    pa = PortfolioFactory(tenant=a)
    pb = PortfolioFactory(tenant=b)
    return TenantPair(a=a, b=b, owner_a=oa, owner_b=ob, portfolio_a=pa, portfolio_b=pb)


# ---------------------------------------------------------------------------
# DRF APIClient
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def auth_client(api_client, owner):
    """Authenticated client carrying a *real* JWT with the tenant_id claim.

    `force_authenticate` alone sets request.user but skips the JWT path, so
    `TenantMiddleware` never resolves `request.tenant` — fine for reads
    (filters just return empty) but writes to TenantModel tables hit a
    NOT NULL tenant_id violation. Minting a token with the tenant claim
    keeps the middleware happy for both.
    """
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims

    access = RefreshToken.for_user(owner).access_token
    _attach_tenant_claims(access, owner)
    api_client.force_authenticate(user=owner)
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return api_client
