"""Daily Zerodha Kite login: request_token -> access_token -> BrokerLink.

Kite access tokens expire every morning (~6 AM IST), so this runs once per
trading day::

    .venv/bin/python manage.py zerodha_login --tenant <slug|uuid>
    # 1. open the printed login URL, log in with your Zerodha account
    # 2. paste the `request_token` (or the whole redirect URL) at the prompt

    .venv/bin/python manage.py zerodha_login --tenant <slug> --request-token <tok>
    .venv/bin/python manage.py zerodha_login --tenant <slug> --dry-run   # don't save

Reads ``KITE_API_KEY`` / ``KITE_API_SECRET`` from settings (backend/.env).
The exchanged credentials go through the same ``connect_broker()`` path as
``POST /api/v1/brokers/zerodha/connect/``: ``authenticate()`` is probed
before anything is saved, and the credentials are stored Fernet-encrypted.
Secrets and tokens are never printed or logged.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import structlog
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.market_data.services.broker_connect import (
    BrokerConnectError, connect_broker,
)
from apps.tenants.models import Membership, Tenant

log = structlog.get_logger(__name__)

BROKER = "zerodha"


def resolve_tenant(ref: str | None) -> Tenant:
    if ref:
        qs = Tenant.objects.filter(slug=ref)
        if not qs.exists():
            try:
                qs = Tenant.objects.filter(id=ref)
                tenant = qs.first()
            except Exception:  # not a UUID
                tenant = None
        else:
            tenant = qs.first()
        if tenant is None:
            raise CommandError(f"No tenant with id/slug '{ref}'.")
        return tenant
    tenants = list(Tenant.objects.all()[:10])
    if len(tenants) == 1:
        return tenants[0]
    listing = "\n".join(f"  {t.slug}  ({t.id})  {t.name}" for t in tenants) or "  (none)"
    raise CommandError(f"--tenant is required when there is not exactly one tenant. Choose:\n{listing}")


def resolve_owner(tenant: Tenant, email: str | None):
    qs = Membership.objects.filter(tenant=tenant, is_active=True).select_related("user")
    if email:
        m = qs.filter(user__email__iexact=email).first()
        if m is None:
            raise CommandError(f"No active member '{email}' in tenant '{tenant.slug}'.")
        return m.user
    owners = list(qs.filter(role=Membership.Role.OWNER)[:2])
    if len(owners) == 1:
        return owners[0].user
    raise CommandError(
        f"Tenant '{tenant.slug}' has {len(owners) or 'no'} owner(s); pass --user <email>."
    )


def extract_request_token(raw: str) -> str:
    """Accept either the bare token or the full Kite redirect URL."""
    raw = (raw or "").strip()
    if "request_token=" in raw:
        vals = parse_qs(urlparse(raw).query).get("request_token") or \
            parse_qs(raw.split("?", 1)[-1]).get("request_token")
        raw = (vals or [""])[0].strip()
    return raw


class Command(BaseCommand):
    help = "Exchange a Kite request_token for the daily access_token and save the Zerodha BrokerLink."

    def add_arguments(self, parser):
        parser.add_argument("--tenant", default=None, help="Tenant slug or UUID")
        parser.add_argument("--user", default=None,
                            help="Member email to own the link (default: the tenant's single owner)")
        parser.add_argument("--request-token", default=None,
                            help="request_token from the redirect URL (or the whole URL). Prompted if omitted.")
        parser.add_argument("--display-name", default="",
                            help="BrokerLink display name (same name = same link, updated daily)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Exchange + verify the token but do not save the BrokerLink")

    def handle(self, *args, **o):
        api_key = getattr(settings, "KITE_API_KEY", "") or ""
        api_secret = getattr(settings, "KITE_API_SECRET", "") or ""
        if not api_key or not api_secret:
            raise CommandError("KITE_API_KEY / KITE_API_SECRET are not set in backend/.env.")

        tenant = resolve_tenant(o["tenant"])
        owner = resolve_owner(tenant, o["user"])

        try:
            from kiteconnect import KiteConnect  # type: ignore
        except ImportError:
            raise CommandError("kiteconnect is not installed — run uv sync in backend/.")

        kite = KiteConnect(api_key=api_key)
        self.stdout.write("Log in to Zerodha here, then copy request_token from the redirect URL:")
        self.stdout.write(f"  {kite.login_url()}")

        raw = o["request_token"]
        if not raw:
            try:
                raw = input("request_token (or full redirect URL): ")
            except EOFError:
                raw = ""
        request_token = extract_request_token(raw)
        if not request_token:
            raise CommandError("No request_token given.")

        try:
            session = kite.generate_session(request_token, api_secret=api_secret)
        except Exception as e:
            # Kite's error text doesn't echo the token/secret; keep it short anyway.
            log.warning("zerodha_login.exchange_failed", tenant=tenant.slug, error=type(e).__name__)
            raise CommandError(f"Kite rejected the request_token ({type(e).__name__}: {str(e)[:200]}). "
                               "Tokens are single-use and expire in minutes — log in again.")
        access_token = (session or {}).get("access_token")
        if not access_token:
            raise CommandError("Kite returned no access_token.")

        credentials = {"api_key": api_key, "api_secret": api_secret, "access_token": access_token}
        meta = {k: session[k] for k in ("user_id", "user_name") if session.get(k)}

        if o["dry_run"]:
            from apps.market_data.adapters.factory import get_adapter_class
            cls = get_adapter_class(BROKER)
            ok = bool(cls and cls(credentials=credentials, meta=meta).authenticate())
            log.info("zerodha_login.dry_run", tenant=tenant.slug, auth_ok=ok)
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
            log.warning("zerodha_login.connect_failed", tenant=tenant.slug, error=str(e)[:200])
            raise CommandError(f"BrokerLink not saved: {e}")

        log.info("zerodha_login.connected", tenant=tenant.slug, link_id=str(link.id),
                 user_id=meta.get("user_id", ""))
        self.stdout.write(self.style.SUCCESS(
            f"Zerodha linked: tenant={tenant.slug} link={link.id} status={link.status} "
            f"user_id={meta.get('user_id', '?')}. Re-run tomorrow morning (token expires ~6 AM IST)."
        ))
