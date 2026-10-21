"""Daily Fyers login: auth_code -> access_token -> BrokerLink.

Fyers only accepts https redirect URLs, so the local SPA callback
(http://localhost:5173/...) can't be registered. Instead register Fyers' own
page, https://trade.fyers.in/api-login/redirect-uri/index.html, and paste the
URL it lands on back here. Fyers tokens expire daily, so this runs once per
trading day::

    .venv/bin/python manage.py fyers_login --tenant <slug|uuid>
    # 1. open the printed login URL, log in with your Fyers account
    # 2. paste the `auth_code` (or the whole redirect URL) at the prompt

    .venv/bin/python manage.py fyers_login --tenant <slug> --auth-code <code>
    .venv/bin/python manage.py fyers_login --tenant <slug> --dry-run   # don't save

Reads ``FYERS_APP_ID`` / ``FYERS_SECRET_KEY`` / ``FYERS_REDIRECT_URI`` from
settings (backend/.env). Saving goes through the same ``connect_broker()``
path as ``zerodha_login``: ``authenticate()`` is probed before anything is
saved, and the credentials are stored Fernet-encrypted. Secrets and tokens are
never printed or logged.
"""
from __future__ import annotations

import secrets
from urllib.parse import parse_qs, urlparse

import structlog
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.market_data.api.oauth_views import _fyers_exchange, _fyers_login_url
from apps.market_data.management.commands.zerodha_login import (
    resolve_owner, resolve_tenant,
)
from apps.market_data.services.broker_connect import (
    BrokerConnectError, connect_broker,
)

log = structlog.get_logger(__name__)

BROKER = "fyers"


def extract_auth_code(raw: str) -> str:
    """Accept either the bare auth_code or the full Fyers redirect URL."""
    raw = (raw or "").strip()
    if "auth_code=" in raw:
        vals = parse_qs(urlparse(raw).query).get("auth_code") or \
            parse_qs(raw.split("?", 1)[-1]).get("auth_code")
        raw = (vals or [""])[0].strip()
    return raw


class Command(BaseCommand):
    help = "Exchange a Fyers auth_code for the daily access_token and save the Fyers BrokerLink."

    def add_arguments(self, parser):
        parser.add_argument("--tenant", default=None, help="Tenant slug or UUID")
        parser.add_argument("--user", default=None,
                            help="Member email to own the link (default: the tenant's single owner)")
        parser.add_argument("--auth-code", default=None,
                            help="auth_code from the redirect URL (or the whole URL). Prompted if omitted.")
        parser.add_argument("--display-name", default="",
                            help="BrokerLink display name (same name = same link, updated daily)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Exchange + verify the token but do not save the BrokerLink")

    def handle(self, *args, **o):
        app_id = getattr(settings, "FYERS_APP_ID", "") or ""
        secret_key = getattr(settings, "FYERS_SECRET_KEY", "") or ""
        redirect_uri = getattr(settings, "FYERS_REDIRECT_URI", "") or ""
        if not app_id or not secret_key:
            raise CommandError("FYERS_APP_ID / FYERS_SECRET_KEY are not set in backend/.env.")

        tenant = resolve_tenant(o["tenant"])
        owner = resolve_owner(tenant, o["user"])

        try:
            import fyers_apiv3  # noqa: F401
        except ImportError:
            raise CommandError("fyers-apiv3 is not installed — run uv sync in backend/.")

        handshake = {"app_id": app_id, "secret_key": secret_key}
        self.stdout.write("Log in to Fyers here, then copy auth_code from the redirect URL:")
        self.stdout.write(f"  {_fyers_login_url(handshake, secrets.token_urlsafe(16), redirect_uri)}")

        raw = o["auth_code"]
        if not raw:
            try:
                raw = input("auth_code (or full redirect URL): ")
            except EOFError:
                raw = ""
        auth_code = extract_auth_code(raw)
        if not auth_code:
            raise CommandError("No auth_code given.")

        try:
            credentials, meta = _fyers_exchange(handshake, {"auth_code": auth_code}, redirect_uri)
        except Exception as e:
            log.warning("fyers_login.exchange_failed", tenant=tenant.slug, error=type(e).__name__)
            raise CommandError(f"Fyers rejected the auth_code ({type(e).__name__}). "
                               "Codes are single-use and expire in minutes — log in again.")

        if o["dry_run"]:
            from apps.market_data.adapters.factory import get_adapter_class
            cls = get_adapter_class(BROKER)
            ok = bool(cls and cls(credentials=credentials, meta=meta).authenticate())
            log.info("fyers_login.dry_run", tenant=tenant.slug, auth_ok=ok)
            self.stdout.write(self.style.WARNING(
                f"[dry-run] token exchanged, authenticate()={ok}; BrokerLink NOT saved."
            ))
            return

        try:
            link = connect_broker(
                tenant=tenant, owner=owner, broker_name=BROKER,
                credentials=credentials, meta=meta, display_name=o["display_name"],
            )
        except BrokerConnectError as e:
            log.warning("fyers_login.connect_failed", tenant=tenant.slug, error=str(e)[:200])
            raise CommandError(f"BrokerLink not saved: {e}")

        log.info("fyers_login.connected", tenant=tenant.slug, link_id=str(link.id))
        self.stdout.write(self.style.SUCCESS(
            f"Fyers linked: tenant={tenant.slug} link={link.id} status={link.status}. "
            "Re-run each trading morning (token expires daily)."
        ))
