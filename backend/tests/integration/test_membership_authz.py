"""Membership write authorization — blocks cross-tenant escalation (#4)."""

from __future__ import annotations

import pytest

from apps.tenants.models import Membership
from tests.factories import MembershipFactory, UserFactory

pytestmark = pytest.mark.django_db
URL = "/api/v1/tenants/memberships/"


def test_membership_create_cannot_target_another_tenant(auth_client, two_tenants):
    """A client-supplied `tenant` is ignored — membership lands in the caller's
    own tenant, never a foreign one. (Was: POST {tenant: victim, role: owner}.)"""
    victim = two_tenants.a
    user = UserFactory()
    auth_client.post(
        URL,
        {"user": user.id, "tenant": str(victim.id), "role": "owner", "is_active": True},
        format="json",
    )
    assert not Membership.objects.filter(tenant=victim, user=user).exists()


def test_membership_write_requires_owner_or_admin(api_client, tenant):
    """A viewer cannot create memberships (no in-tenant self-escalation)."""
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims

    viewer = UserFactory()
    MembershipFactory(user=viewer, tenant=tenant, role="viewer")
    access = RefreshToken.for_user(viewer).access_token
    _attach_tenant_claims(access, viewer)
    api_client.force_authenticate(user=viewer)
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")

    resp = api_client.post(URL, {"user": UserFactory().id, "role": "owner"}, format="json")
    assert resp.status_code == 403
    # only the viewer's own membership exists — no new owner row created
    assert not Membership.objects.filter(tenant=tenant).exclude(user=viewer).exists()
