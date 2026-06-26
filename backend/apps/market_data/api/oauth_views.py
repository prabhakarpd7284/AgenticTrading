"""OAuth-redirect connect flow for Zerodha Kite + Fyers v3.

These brokers don't accept long-lived API tokens — operators must redirect
through the broker's hosted login page once per day. The flow:

  1. UI calls   POST /api/v1/brokers/{name}/oauth/start/
                body: {app_id, secret_key, display_name, meta}
     Backend:   persists a PendingOAuth row keyed by a random `state`,
                returns the broker login URL with that state embedded.

  2. UI:        window.location = login_url (or window.open for popup)
                User authenticates at the broker.

  3. Broker:    redirects to GET /api/v1/brokers/{name}/oauth/callback/
                ?auth_code=... &state=... (Fyers)
                ?request_token=... &state=... (Zerodha)

  4. Backend:   matches state → PendingOAuth → completes the token
                exchange → upserts the BrokerLink → bounces the browser
                back to the SPA at /broker?connected={name}.

Angel One bypasses this — it uses TOTP and doesn't need a redirect flow.
"""
from __future__ import annotations

import logging
import secrets
from datetime import timedelta

from django.http import HttpResponseRedirect, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from apps.market_data.api.broker_views import _take_snapshot
from apps.market_data.models import BrokerLink, PendingOAuth
from apps.market_data.services.crypto import (
    decrypt_credentials, encrypt_credentials,
)
from rest_framework.generics import get_object_or_404

logger = logging.getLogger(__name__)

PENDING_TTL_MINUTES = 30


# ─── Per-broker glue ──────────────────────────────────────────────────

def _zerodha_login_url(handshake: dict, state: str) -> str:
    from kiteconnect import KiteConnect  # type: ignore
    kite = KiteConnect(api_key=handshake["api_key"])
    # Kite's login_url() doesn't take `state` directly; we append manually
    # so the callback can correlate. Zerodha echoes query params back on
    # the redirect.
    base = kite.login_url()
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}state={state}"


def _zerodha_exchange(handshake: dict, request_params: dict) -> dict:
    """Return the credentials dict to persist on success."""
    from kiteconnect import KiteConnect  # type: ignore
    request_token = request_params.get("request_token")
    if not request_token:
        raise ValueError("missing request_token in callback")
    kite = KiteConnect(api_key=handshake["api_key"])
    data = kite.generate_session(request_token, api_secret=handshake["api_secret"])
    return {
        "api_key": handshake["api_key"],
        "api_secret": handshake["api_secret"],
        "access_token": data["access_token"],
    }, {"user_id": data.get("user_id", "")}


def _fyers_login_url(handshake: dict, state: str, redirect_uri: str) -> str:
    from fyers_apiv3.fyersModel import SessionModel  # type: ignore
    session = SessionModel(
        client_id=handshake["app_id"],
        secret_key=handshake["secret_key"],
        redirect_uri=redirect_uri,
        response_type="code",
        state=state,
        grant_type="authorization_code",
    )
    return session.generate_authcode()


def _fyers_exchange(handshake: dict, request_params: dict, redirect_uri: str) -> tuple[dict, dict]:
    from fyers_apiv3.fyersModel import SessionModel  # type: ignore
    auth_code = request_params.get("auth_code") or request_params.get("code")
    if not auth_code:
        raise ValueError("missing auth_code in callback")
    session = SessionModel(
        client_id=handshake["app_id"],
        secret_key=handshake["secret_key"],
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    resp = session.generate_token()
    if not isinstance(resp, dict) or resp.get("s") != "ok":
        raise ValueError(f"fyers token exchange failed: {resp}")
    return {
        "app_id": handshake["app_id"],
        "secret_key": handshake["secret_key"],
        "access_token": resp["access_token"],
    }, {}


# ─── Helpers ──────────────────────────────────────────────────────────

OAUTH_BROKERS = {"zerodha", "fyers"}


def _callback_redirect_uri(request: Request, broker_name: str) -> str:
    """Absolute redirect URI registered in the broker app portal.

    Preference order:
      1. settings.BROKER_OAUTH_REDIRECT_HOST  — explicit override (prod URL)
      2. Referer header origin                — the SPA origin (localhost:5173)
      3. request.get_host()                   — daphne's own host

    Defaulting to the SPA origin means the browser-visible callback URL
    matches what Vite's dev server proxies to daphne — the operator only
    has to register one URL in the broker's API portal.
    """
    from urllib.parse import urlparse

    from django.conf import settings
    override = getattr(settings, "BROKER_OAUTH_REDIRECT_HOST", None)
    if override:
        base = override.rstrip("/")
    else:
        referer = request.META.get("HTTP_REFERER") or ""
        if referer:
            p = urlparse(referer)
            base = f"{p.scheme}://{p.netloc}"
        else:
            scheme = "https" if request.is_secure() else "http"
            base = f"{scheme}://{request.get_host()}"
    return f"{base}/api/v1/brokers/{broker_name}/oauth/callback/"


def _spa_redirect_target(request: Request, broker_name: str, ok: bool, msg: str = "") -> str:
    """Where to bounce the browser after the broker callback finishes."""
    spa_origin = request.META.get("HTTP_REFERER", "")
    # Strip any path so we land on /broker no matter what.
    from urllib.parse import urlparse
    if spa_origin:
        p = urlparse(spa_origin)
        spa_base = f"{p.scheme}://{p.netloc}"
    else:
        # Vite dev server default. Override with FRONTEND_ORIGIN env in prod.
        from django.conf import settings
        spa_base = getattr(settings, "FRONTEND_ORIGIN", "http://localhost:5173")
    qs = f"connected={broker_name}" if ok else f"error={broker_name}&reason={msg[:80]}"
    return f"{spa_base}/broker?{qs}"


def _prune_old_pending():
    cutoff = timezone.now() - timedelta(minutes=PENDING_TTL_MINUTES)
    PendingOAuth.objects.filter(created_at__lt=cutoff).delete()


# ─── Endpoints ────────────────────────────────────────────────────────

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def oauth_start(request: Request, broker_name: str):
    """Kick off the OAuth handshake. Returns the broker login URL."""
    if broker_name not in OAUTH_BROKERS:
        return Response(
            {"detail": f"{broker_name} is not OAuth-based; use /connect/ instead."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    handshake = request.data.get("handshake") or {}
    meta = request.data.get("meta") or {}
    display_name = request.data.get("display_name") or ""
    if not isinstance(handshake, dict) or not handshake:
        return Response({"detail": "handshake payload required"}, status=400)

    _prune_old_pending()
    state = secrets.token_urlsafe(32)
    PendingOAuth.objects.create(
        tenant=request.tenant,
        owner=request.user,
        broker_name=broker_name,
        display_name=display_name,
        state=state,
        handshake_blob=encrypt_credentials(handshake),
        meta=meta,
    )

    redirect_uri = _callback_redirect_uri(request, broker_name)
    try:
        if broker_name == "zerodha":
            login_url = _zerodha_login_url(handshake, state)
        else:  # fyers
            login_url = _fyers_login_url(handshake, state, redirect_uri)
    except Exception as e:
        logger.exception("oauth.start.failed broker=%s err=%s", broker_name, e)
        return Response({"detail": f"Could not build login URL: {e}"}, status=502)

    return Response({
        "login_url": login_url,
        "state": state,
        "redirect_uri": redirect_uri,
        "expires_in_seconds": PENDING_TTL_MINUTES * 60,
    })


@csrf_exempt
@api_view(["GET"])
@permission_classes([AllowAny])
def oauth_callback(request: Request, broker_name: str):
    """Public endpoint hit by the broker's redirect.

    Security: a returned ``state`` must match a pending handshake exactly — a
    provided-but-unknown state is treated as forged/expired and rejected (it
    never falls back to another tenant's pending row). Some brokers (Kite) strip
    the state param on redirect; only then do we fall back, and ONLY when a
    single unresolved handshake exists for this broker system-wide — if multiple
    logins are in flight we refuse, so a callback can never bind to the wrong
    tenant under concurrency.
    """
    state = request.query_params.get("state")
    pending = None
    if state:
        # Exact match only, scoped to the broker. A provided-but-unknown state
        # is forged/expired — fail rather than silently grab someone else's row.
        pending = PendingOAuth.objects.filter(
            state=state, broker_name=broker_name
        ).first()
        if pending is None:
            return _error_redirect(request, broker_name, "invalid or expired state")
    else:
        # No state echoed back (Kite strips it). Fall back ONLY when the choice
        # is unambiguous — exactly one pending handshake for this broker. More
        # than one ⇒ concurrent logins ⇒ refuse rather than risk cross-tenant
        # token binding.
        candidates = list(
            PendingOAuth.objects.filter(broker_name=broker_name).order_by("-created_at")[:2]
        )
        if len(candidates) > 1:
            return _error_redirect(
                request, broker_name,
                "ambiguous callback — another login is in progress; please retry",
            )
        pending = candidates[0] if candidates else None
    if not pending:
        return _error_redirect(request, broker_name, "no pending handshake found")

    if timezone.now() - pending.created_at > timedelta(minutes=PENDING_TTL_MINUTES):
        pending.delete()
        return _error_redirect(request, broker_name, "handshake expired — try again")

    try:
        handshake = decrypt_credentials(pending.handshake_blob)
        redirect_uri = _callback_redirect_uri(request, broker_name)
        if broker_name == "zerodha":
            creds, extra_meta = _zerodha_exchange(handshake, request.query_params)
        else:  # fyers
            creds, extra_meta = _fyers_exchange(handshake, request.query_params, redirect_uri)
    except Exception as e:
        logger.exception("oauth.callback.exchange_failed broker=%s err=%s", broker_name, e)
        pending.delete()
        return _error_redirect(request, broker_name, str(e)[:120])

    merged_meta = {**(pending.meta or {}), **extra_meta}
    link, _ = BrokerLink.objects.update_or_create(
        tenant=pending.tenant,
        broker_name=broker_name,
        owner=pending.owner,
        display_name=pending.display_name,
        defaults={
            "credential_blob": encrypt_credentials(creds),
            "credential_meta": merged_meta,
            "status": BrokerLink.Status.ACTIVE,
            "last_error": "",
            "last_refreshed_at": timezone.now(),
        },
    )
    pending.delete()

    try:
        _take_snapshot(link)
    except Exception:
        pass  # snapshot failure shouldn't fail the connect

    return HttpResponseRedirect(_spa_redirect_target(request, broker_name, ok=True))


def _error_redirect(request, broker_name, msg):
    return HttpResponseRedirect(_spa_redirect_target(request, broker_name, ok=False, msg=msg))


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def oauth_reauth(request: Request, pk: str):
    """Daily re-login for OAuth brokers (Zerodha, Fyers).

    The original connect flow stored the long-lived handshake half
    (api_key+api_secret / app_id+secret_key) inside ``credential_blob``
    alongside the daily ``access_token``. When the daily token expires we
    can re-use that handshake to start a fresh OAuth flow — the operator
    just clicks "Re-login" instead of re-pasting their API keys every
    morning.
    """
    link = get_object_or_404(
        BrokerLink, pk=pk, tenant=request.tenant,
    )
    if link.broker_name not in OAUTH_BROKERS:
        return Response(
            {"detail": f"{link.broker_name} does not use OAuth re-login."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        stored = decrypt_credentials(link.credential_blob)
    except Exception as e:
        return Response(
            {"detail": f"Could not read stored credentials: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if link.broker_name == "zerodha":
        handshake = {
            "api_key": stored.get("api_key", ""),
            "api_secret": stored.get("api_secret", ""),
        }
    else:  # fyers
        handshake = {
            "app_id": stored.get("app_id", ""),
            "secret_key": stored.get("secret_key", ""),
        }
    if not all(handshake.values()):
        return Response(
            {"detail": "Missing handshake fields in stored credentials — please unlink and reconnect."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    _prune_old_pending()
    state = secrets.token_urlsafe(32)
    PendingOAuth.objects.create(
        tenant=request.tenant,
        owner=request.user,
        broker_name=link.broker_name,
        display_name=link.display_name,
        state=state,
        handshake_blob=encrypt_credentials(handshake),
        meta=link.credential_meta or {},
    )

    redirect_uri = _callback_redirect_uri(request, link.broker_name)
    try:
        if link.broker_name == "zerodha":
            login_url = _zerodha_login_url(handshake, state)
        else:  # fyers
            login_url = _fyers_login_url(handshake, state, redirect_uri)
    except Exception as e:
        logger.exception("oauth.reauth.failed broker=%s err=%s", link.broker_name, e)
        return Response(
            {"detail": f"Could not build login URL: {e}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    return Response({
        "login_url": login_url,
        "state": state,
        "redirect_uri": redirect_uri,
        "expires_in_seconds": PENDING_TTL_MINUTES * 60,
    })


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def oauth_pending(request: Request):
    """List in-flight OAuth handshakes for the current user (debug aid)."""
    _prune_old_pending()
    rows = PendingOAuth.objects.filter(
        tenant=request.tenant, owner=request.user,
    ).values("id", "broker_name", "display_name", "created_at")
    return JsonResponse({"pending": list(rows)})
