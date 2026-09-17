"""Tests for the tenant-aware JWT issuance + personal-tenant bootstrap.

These guarantee:
*   Signing up auto-provisions a personal tenant + owner membership so the
    first access token already carries a valid ``tenant_id`` claim.
*   ``/auth/token/`` returns ``{access, refresh, tenant: {id, name, slug, role}}``.
*   A token obtained via the custom flow lets the user hit tenant-scoped
    endpoints that previously returned 403 (``/portfolios/``, ``/agents/runs/``).
*   Back-filling personal tenants for legacy users is idempotent.
"""
from __future__ import annotations

import pytest
from django.urls import reverse
from rest_framework import status

from apps.accounts.models import User
from apps.accounts.services.tenant_bootstrap import ensure_tenant
from apps.tenants.models import Membership, Tenant


pytestmark = pytest.mark.django_db


# --------------------------------------------------------------------------
# Bootstrap helper
# --------------------------------------------------------------------------
def test_ensure_tenant_is_idempotent(db):
    user = User.objects.create_user(email="idemp@x.io", password="hunter2hunter2")
    t1 = ensure_tenant(user)
    t2 = ensure_tenant(user)
    assert t1.id == t2.id
    assert Membership.objects.filter(user=user).count() == 1


def test_ensure_tenant_generates_unique_slug(db):
    # Two users with colliding email-prefix shouldn't break on slug uniqueness
    u1 = User.objects.create_user(email="trader@org-a.io", password="hunter2hunter2")
    u2 = User.objects.create_user(email="trader@org-b.io", password="hunter2hunter2")
    t1 = ensure_tenant(u1)
    t2 = ensure_tenant(u2)
    assert t1.slug != t2.slug


# --------------------------------------------------------------------------
# Signup + JWT flow
# --------------------------------------------------------------------------
def test_signup_autocreates_personal_tenant(api_client):
    resp = api_client.post(
        "/api/v1/auth/auth/signup/",
        {"email": "new@alphadesk.io", "full_name": "New Dev", "password": "longpassword123"},
        format="json",
    )
    assert resp.status_code == status.HTTP_201_CREATED, resp.content
    user = User.objects.get(email="new@alphadesk.io")
    membership = Membership.objects.filter(user=user, is_active=True).first()
    assert membership is not None
    assert membership.role == Membership.Role.OWNER


def test_token_obtain_carries_tenant_claim(api_client):
    user = User.objects.create_user(
        email="claim@alphadesk.io", password="longpassword123",
        full_name="Claim Bearer",
    )
    tenant = ensure_tenant(user)
    resp = api_client.post(
        reverse("token_obtain_pair"),
        {"email": "claim@alphadesk.io", "password": "longpassword123"},
        format="json",
    )
    assert resp.status_code == 200, resp.content
    body = resp.json()
    assert {"access", "refresh", "tenant"} <= body.keys()
    assert body["tenant"]["id"] == str(tenant.id)

    # Decode the access token without signature verification
    import jwt as pyjwt
    payload = pyjwt.decode(body["access"], options={"verify_signature": False})
    assert payload.get("tenant_id") == str(tenant.id)
    assert payload.get("role") == "owner"


def test_token_unlocks_tenant_scoped_endpoints(api_client):
    """The root cause of the user's 403s — verify the fix end-to-end."""
    User.objects.create_user(email="gate@alphadesk.io", password="longpassword123")
    tok = api_client.post(
        reverse("token_obtain_pair"),
        {"email": "gate@alphadesk.io", "password": "longpassword123"},
        format="json",
    ).json()["access"]

    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {tok}")
    for path in ("/api/v1/portfolios/", "/api/v1/agents/runs/"):
        r = api_client.get(path)
        assert r.status_code == 200, f"{path} → {r.status_code}: {r.content[:200]!r}"


def test_sibling_router_paths_are_not_shadowed_by_catchall_viewset(api_client):
    """Regression guard — ``router.register("", TenantViewSet)`` used to
    emit a detail pattern ``^(?P<pk>[^/.]+)/$`` that swallowed sibling
    routes like ``/memberships/`` (capturing ``pk="memberships"``).  The
    fix mounts the sibling routers under explicit prefixes.  Same gotcha
    affected ``apps.trading.api.urls`` — check both here.
    """
    User.objects.create_user(email="shadow@alphadesk.io", password="longpassword123")
    tok = api_client.post(
        reverse("token_obtain_pair"),
        {"email": "shadow@alphadesk.io", "password": "longpassword123"},
        format="json",
    ).json()["access"]
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {tok}")
    for path in (
        "/api/v1/tenants/memberships/",
        "/api/v1/portfolios/positions/",
        "/api/v1/portfolios/snapshots/",
    ):
        r = api_client.get(path)
        # If the catchall shadowed the sibling this would be a 404.
        assert r.status_code == 200, f"{path} → {r.status_code}: {r.content[:200]!r}"


# --------------------------------------------------------------------------
# Backfill command
# --------------------------------------------------------------------------
def test_backfill_is_idempotent(db):
    from django.core.management import call_command
    # Create a user without going through signup (simulating legacy data)
    user = User.objects.create_user(email="legacy@x.io", password="longpassword123")
    assert Membership.objects.filter(user=user).count() == 0

    call_command("backfill_personal_tenants")
    call_command("backfill_personal_tenants")          # second run is a no-op

    assert Membership.objects.filter(user=user, is_active=True).count() == 1
    assert Tenant.objects.filter(memberships__user=user).count() == 1
