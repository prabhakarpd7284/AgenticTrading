"""Cross-cutting HTTP middleware — request-id, tenant resolution, structlog context."""
from __future__ import annotations

import uuid
from typing import Callable

import structlog
from django.http import HttpRequest, HttpResponse

log = structlog.get_logger()


class RequestIdMiddleware:
    HEADER = "X-Request-Id"

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        rid = request.headers.get(self.HEADER) or str(uuid.uuid4())
        request.request_id = rid  # type: ignore[attr-defined]
        response = self.get_response(request)
        response[self.HEADER] = rid
        return response


class TenantMiddleware:
    """Resolve tenant from JWT claim `tenant_id`.

    Anonymous users get `request.tenant = None`; auth'd users with no membership
    are rejected by the ViewSet permission layer, not here.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        request.tenant = None  # type: ignore[attr-defined]
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                from rest_framework_simplejwt.tokens import UntypedToken

                token = UntypedToken(auth.removeprefix("Bearer "))
                tenant_id = token.get("tenant_id")
                if tenant_id:
                    from apps.tenants.models import Tenant

                    request.tenant = Tenant.objects.filter(id=tenant_id).first()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — log only, never block
                log.warning("tenant.resolve_failed", request_id=getattr(request, "request_id", None))
        return self.get_response(request)


class StructlogContextMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=getattr(request, "request_id", None),
            tenant_id=getattr(getattr(request, "tenant", None), "id", None),
            user_id=getattr(getattr(request, "user", None), "id", None),
        )
        return self.get_response(request)


class IntradayAsOfMiddleware:
    """Read ?date=YYYY-MM-DD from the query string and scope it as the
    intraday session override for the request.

    Any service that calls trading.utils.time_utils.intraday_session_date()
    transparently returns the picked date — so cockpit time-travel works
    without editing each view. Invalid or missing param = no override.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        from datetime import date
        from trading.utils.time_utils import use_session_date

        raw = request.GET.get("date") or request.GET.get("as_of")
        target = None
        if raw:
            try:
                target = date.fromisoformat(raw.strip())
            except (ValueError, TypeError):
                target = None

        with use_session_date(target):
            return self.get_response(request)
