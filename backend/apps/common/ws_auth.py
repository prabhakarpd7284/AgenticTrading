"""Channels auth middleware that authenticates by JWT.

The default ``AuthMiddlewareStack`` reads Django session cookies, which we
don't issue — our SPA lives on a separate origin and authenticates with
short-lived JWTs held in memory. So the stock middleware always yields
``AnonymousUser`` and every consumer rejects the connection with code 4401.

This middleware looks for the JWT in (in order):

1. The ``Sec-WebSocket-Protocol`` header (two protocols: ``jwt`` followed by
   the raw token). This is the most ergonomic option for browsers because
   it survives the WebSocket handshake untouched — unlike ``Authorization``,
   which browsers refuse to attach to WebSocket upgrades.

2. The ``token`` query parameter, for native clients that can't use
   subprotocols (e.g. websocat for debugging).

On success, the token is validated, the user is resolved, and their tenant
(from the ``tenant_id`` claim) is hydrated into ``scope``. Consumers then
only have to look at ``scope["user"]`` / ``scope["tenant"]``.
"""
from __future__ import annotations

import logging
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser

log = logging.getLogger(__name__)


@database_sync_to_async
def _resolve(token_str: str):
    """Validate the access token and return (user, tenant) — no exceptions."""
    try:
        from rest_framework_simplejwt.tokens import AccessToken
        from apps.accounts.models import User
        from apps.tenants.models import Tenant

        token = AccessToken(token_str)
        user = User.objects.filter(id=token.get("user_id"), is_active=True).first()
        tenant = None
        tid = token.get("tenant_id")
        if tid:
            tenant = Tenant.objects.filter(id=tid).first()
        return user, tenant
    except Exception as exc:  # noqa: BLE001 — never crash the handshake
        log.debug("ws jwt auth failed: %s", exc)
        return None, None


def _extract_token(scope) -> str | None:
    # 1. Subprotocols: clients send ["jwt", "<token>"]
    for proto in scope.get("subprotocols") or []:
        if proto and proto != "jwt":
            return proto
    # 2. Query string (?token=...)
    qs = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
    if "token" in qs and qs["token"]:
        return qs["token"][0]
    # 3. Authorization header (allowed by RFC 6455 but rejected by browsers —
    #    still useful for server-to-server tests)
    for name, value in scope.get("headers") or []:
        if name == b"authorization":
            raw = value.decode("latin-1")
            if raw.startswith("Bearer "):
                return raw.removeprefix("Bearer ")
    return None


class JWTAuthMiddleware(BaseMiddleware):
    """Populates scope['user'] and scope['tenant'] from a JWT access token."""

    async def __call__(self, scope, receive, send):
        token = _extract_token(scope)
        user, tenant = (None, None)
        if token:
            user, tenant = await _resolve(token)
        scope["user"] = user or AnonymousUser()
        scope["tenant"] = tenant
        return await super().__call__(scope, receive, send)


def JWTAuthMiddlewareStack(inner):
    """Convenience wrapper mirroring channels.auth.AuthMiddlewareStack."""
    return JWTAuthMiddleware(inner)
