"""Tests for the Channels JWT auth middleware.

Why these exist: the original AuthMiddlewareStack reads Django session cookies
which we don't issue, so every WebSocket connection closed with code 4401
before reaching the consumer.  JWTAuthMiddleware fixes this by pulling the
token from the WebSocket subprotocol.

These tests use Channels' ``ApplicationCommunicator`` directly so they don't
require a running server.
"""
from __future__ import annotations

import pytest
from django.contrib.auth.models import AnonymousUser

from apps.accounts.models import User
from apps.accounts.services.tenant_bootstrap import ensure_tenant
from apps.common.ws_auth import JWTAuthMiddleware


pytestmark = pytest.mark.django_db


async def _run(scope, inner_hits):
    async def inner(scope, receive, send):
        inner_hits.append(scope)

    mw = JWTAuthMiddleware(inner)
    await mw(scope, lambda: None, lambda m: None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_middleware_rejects_anonymous(db):
    """No token → AnonymousUser + tenant=None; consumer will close(4401)."""
    hits: list = []
    await _run({"type": "websocket", "subprotocols": [], "query_string": b"", "headers": []}, hits)
    scope = hits[0]
    assert isinstance(scope["user"], AnonymousUser)
    assert scope["tenant"] is None


@pytest.mark.asyncio
async def test_middleware_populates_user_and_tenant_from_subprotocol(db):
    user = await _create_user("ws@x.io")
    tenant = await _create_tenant(user)
    token = await _mint_token(user, tenant)
    hits: list = []
    await _run({
        "type": "websocket",
        "subprotocols": ["jwt", token],
        "query_string": b"",
        "headers": [],
    }, hits)
    scope = hits[0]
    assert scope["user"].id == user.id
    assert str(scope["tenant"].id) == str(tenant.id)


@pytest.mark.asyncio
async def test_middleware_falls_back_to_query_string(db):
    user = await _create_user("ws2@x.io")
    tenant = await _create_tenant(user)
    token = await _mint_token(user, tenant)
    hits: list = []
    await _run({
        "type": "websocket",
        "subprotocols": [],
        "query_string": f"token={token}".encode(),
        "headers": [],
    }, hits)
    scope = hits[0]
    assert scope["user"].id == user.id


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
from asgiref.sync import sync_to_async


@sync_to_async
def _create_user(email: str) -> User:
    return User.objects.create_user(email=email, password="longpassword123")


@sync_to_async
def _create_tenant(user):
    return ensure_tenant(user)


@sync_to_async
def _mint_token(user, tenant) -> str:
    from apps.accounts.api.jwt import _attach_tenant_claims
    from rest_framework_simplejwt.tokens import AccessToken
    token = AccessToken.for_user(user)
    _attach_tenant_claims(token, user, tenant_id=str(tenant.id))
    return str(token)
