"""BrokerLink CRUD + connect/refresh/positions actions."""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from django.db.models import OuterRef, Subquery
from django.utils import timezone
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.market_data.adapters.factory import (
    build_adapter, get_adapter_class, list_adapters,
)
from apps.market_data.models import BrokerLink, BrokerPositionSnapshot
from apps.market_data.services.crypto import encrypt_credentials


# Brokers whose access_token expires daily ~6 AM IST and needs a fresh
# OAuth-redirect login each trading morning. Kept here (not in adapter)
# because it drives UI behaviour, not the network call itself.
DAILY_TOKEN_BROKERS = {"zerodha", "fyers"}
IST = ZoneInfo("Asia/Kolkata")
# Conservative cutoff — both Kite and Fyers expire ~6 AM IST. We treat any
# `last_refreshed_at` before today's 6 AM IST as needing re-login.
DAILY_TOKEN_CUTOFF_IST = dtime(hour=6, minute=0)


def _next_token_expiry_ist(now=None) -> datetime:
    """Next 6 AM IST after ``now`` — when the daily token will expire."""
    n = (now or timezone.now()).astimezone(IST)
    cutoff_today = n.replace(hour=DAILY_TOKEN_CUTOFF_IST.hour,
                              minute=DAILY_TOKEN_CUTOFF_IST.minute,
                              second=0, microsecond=0)
    if n >= cutoff_today:
        cutoff_today = cutoff_today + timedelta(days=1)
    return cutoff_today


def _token_valid_today(link: BrokerLink, now=None) -> bool:
    """True iff link was last refreshed after the most recent 6 AM IST."""
    if link.broker_name not in DAILY_TOKEN_BROKERS:
        return True
    if not link.last_refreshed_at:
        return False
    n = (now or timezone.now()).astimezone(IST)
    cutoff = n.replace(hour=DAILY_TOKEN_CUTOFF_IST.hour,
                        minute=DAILY_TOKEN_CUTOFF_IST.minute,
                        second=0, microsecond=0)
    if n < cutoff:
        cutoff = cutoff - timedelta(days=1)
    return link.last_refreshed_at.astimezone(IST) >= cutoff


class BrokerLinkSerializer(serializers.ModelSerializer):
    """Read-side serializer — credential_blob is intentionally excluded.

    ``last_snapshot_at`` / ``last_snapshot_ok`` are populated via Subquery
    annotations on the viewset queryset to avoid N+1 against
    ``BrokerPositionSnapshot`` when listing many links.
    """
    last_snapshot_at = serializers.SerializerMethodField()
    last_snapshot_ok = serializers.SerializerMethodField()
    requires_daily_login = serializers.SerializerMethodField()
    token_valid_today = serializers.SerializerMethodField()
    next_token_expiry_at = serializers.SerializerMethodField()

    class Meta:
        model = BrokerLink
        fields = [
            "id", "broker_name", "display_name", "owner",
            "status", "last_refreshed_at", "last_error",
            "credential_meta", "is_default",
            "last_snapshot_at", "last_snapshot_ok",
            "requires_daily_login", "token_valid_today",
            "next_token_expiry_at",
        ]
        read_only_fields = [
            "id", "owner", "status", "last_refreshed_at", "last_error",
            "last_snapshot_at", "last_snapshot_ok",
            "requires_daily_login", "token_valid_today",
            "next_token_expiry_at",
        ]

    def get_last_snapshot_at(self, obj):
        at = getattr(obj, "_last_snapshot_at", None)
        if at is None:
            snap = obj.snapshots.only("fetched_at").first()
            at = snap.fetched_at if snap else None
        return at.isoformat() if at else None

    def get_last_snapshot_ok(self, obj):
        if hasattr(obj, "_last_snapshot_ok"):
            return obj._last_snapshot_ok
        snap = obj.snapshots.only("ok").first()
        return snap.ok if snap else None

    def get_requires_daily_login(self, obj):
        return obj.broker_name in DAILY_TOKEN_BROKERS

    def get_token_valid_today(self, obj):
        # Only meaningful for daily-login brokers; for others it's always
        # true so the UI can treat it uniformly.
        return _token_valid_today(obj)

    def get_next_token_expiry_at(self, obj):
        if obj.broker_name not in DAILY_TOKEN_BROKERS:
            return None
        return _next_token_expiry_ist().isoformat()


class BrokerLinkViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = BrokerLinkSerializer

    def get_queryset(self):
        latest = BrokerPositionSnapshot.objects.filter(link=OuterRef("pk")).order_by("-fetched_at")
        return (
            BrokerLink.objects
            .filter(tenant=self.request.tenant)
            .annotate(
                _last_snapshot_at=Subquery(latest.values("fetched_at")[:1]),
                _last_snapshot_ok=Subquery(latest.values("ok")[:1]),
            )
        )

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant, owner=self.request.user)

    # ── /api/v1/brokers/available/ ────────────────────────────────────
    @action(detail=False, methods=["get"])
    def available(self, request):
        """List broker plugins the backend knows how to talk to."""
        return Response({"brokers": list_adapters()})

    # ── /api/v1/brokers/{name}/connect/ ───────────────────────────────
    @action(detail=False, methods=["post"], url_path=r"(?P<broker_name>[\w-]+)/connect")
    def connect(self, request, broker_name=None):
        """Create or update a BrokerLink with encrypted credentials.

        Body: {"credentials": {...broker-specific...}, "meta": {...}, "display_name": "..."}
        Tests auth before persisting — bad creds never reach the DB.
        """
        cls = get_adapter_class(broker_name)
        if cls is None:
            raise ValidationError(f"Unknown broker: {broker_name}")

        creds = request.data.get("credentials") or {}
        meta = request.data.get("meta") or {}
        display_name = request.data.get("display_name") or ""
        if not isinstance(creds, dict) or not creds:
            raise ValidationError("`credentials` must be a non-empty object")

        try:
            probe = cls(credentials=creds, meta=meta)
            ok = probe.authenticate()
        except Exception as e:
            raise ValidationError(f"Auth probe raised: {e}")
        if not ok:
            raise ValidationError("Broker rejected the credentials")

        link, _ = BrokerLink.objects.update_or_create(
            tenant=request.tenant,
            broker_name=broker_name,
            owner=request.user,
            display_name=display_name,
            defaults={
                "credential_blob": encrypt_credentials(creds),
                "credential_meta": meta,
                "status": BrokerLink.Status.ACTIVE,
                "last_error": "",
                "last_refreshed_at": timezone.now(),
            },
        )
        try:
            _take_snapshot(link)
        except Exception:
            pass
        return Response(BrokerLinkSerializer(link).data, status=status.HTTP_201_CREATED)

    # ── /api/v1/brokers/{id}/refresh/ ─────────────────────────────────
    @action(detail=True, methods=["post"])
    def refresh(self, request, pk=None):
        """Re-fetch positions/holdings/margin from the broker."""
        link = self.get_object()
        snap = _take_snapshot(link)
        return Response({
            "link_id": str(link.id),
            "ok": snap.ok,
            "fetched_at": snap.fetched_at.isoformat(),
            "error": snap.error,
            "positions": snap.positions,
            "holdings": snap.holdings,
            "margin": snap.margin,
        })

    # ── /api/v1/brokers/{id}/positions/ ───────────────────────────────
    @action(detail=True, methods=["get"], url_path="positions")
    def positions(self, request, pk=None):
        """Return the latest cached snapshot + freshness."""
        link = self.get_object()
        snap = link.snapshots.first()
        if not snap:
            return Response({"detail": "no snapshot yet — call /refresh/"}, status=404)
        return Response({
            "link_id": str(link.id),
            "broker_name": link.broker_name,
            "display_name": link.display_name,
            "fetched_at": snap.fetched_at.isoformat(),
            "age_seconds": (timezone.now() - snap.fetched_at).total_seconds(),
            "positions": snap.positions,
            "holdings": snap.holdings,
            "margin": snap.margin,
        })

    # ── /api/v1/brokers/{id}/diagnose/ ────────────────────────────────
    @action(detail=True, methods=["get"], url_path="diagnose")
    def diagnose(self, request, pk=None):
        """Run each broker call independently and return raw outcomes.

        The /refresh/ endpoint is an atomic snapshot — one failure marks
        the whole snapshot failed and the operator has no visibility into
        whether auth, positions, holdings, or margin specifically broke.
        Diagnose runs each step in isolation so the UI can show exactly
        what the broker returned (and where the empty data came from).
        """
        import traceback

        link = self.get_object()
        result: dict = {
            "link_id": str(link.id),
            "broker_name": link.broker_name,
            "display_name": link.display_name,
            "checked_at": timezone.now().isoformat(),
            "checks": [],
        }

        adapter = build_adapter(link)
        if adapter is None:
            result["checks"].append({
                "step": "build_adapter", "ok": False,
                "detail": "adapter unavailable (missing plugin or credentials)",
            })
            return Response(result)

        # Auth probe
        try:
            auth_ok = bool(adapter.authenticate())
            result["checks"].append({
                "step": "authenticate", "ok": auth_ok,
                "detail": "" if auth_ok else "broker rejected stored credentials",
            })
        except Exception as e:
            result["checks"].append({
                "step": "authenticate", "ok": False,
                "detail": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
            })
            return Response(result)

        if not auth_ok:
            return Response(result)

        # Each fetch — independent so one failure doesn't mask the others.
        for step, fn in [
            ("fetch_positions", adapter.fetch_positions),
            ("fetch_holdings", adapter.fetch_holdings),
            ("fetch_margin", adapter.fetch_margin),
        ]:
            entry: dict = {"step": step}
            try:
                data = fn()
                entry["ok"] = True
                if isinstance(data, list):
                    entry["count"] = len(data)
                    entry["sample"] = [_diagnose_repr(x) for x in data[:3]]
                else:
                    entry["data"] = _diagnose_repr(data)
            except Exception as e:
                entry["ok"] = False
                entry["detail"] = f"{type(e).__name__}: {e}"
                entry["traceback"] = traceback.format_exc(limit=3)
            result["checks"].append(entry)

        return Response(result)

    # ── /api/v1/brokers/{id}/set-default/ ─────────────────────────────
    @action(detail=True, methods=["post"], url_path="set-default")
    def set_default(self, request, pk=None):
        link = self.get_object()
        BrokerLink.objects.filter(
            tenant=request.tenant, owner=request.user, is_default=True,
        ).update(is_default=False)
        link.is_default = True
        link.save(update_fields=["is_default"])
        return Response(BrokerLinkSerializer(link).data)


# ── Error classification ─────────────────────────────────────────────

# String fingerprints from the three broker SDKs (and a few of our own
# adapter messages) that indicate the daily access token / session is no
# longer valid. Used to translate a fetch exception into a proper
# EXPIRED status so the UI surfaces Re-login instead of a generic error.
# Auth hints chosen to avoid matching "access denied because of exceeding
# access rate" — that's a throttle, not an auth failure, and should stay
# ERRORED (transient) so the next beat retries instead of forcing re-login.
_AUTH_FAILURE_HINTS = (
    "authentication failed",
    "daily re-login required",
    "token expired",
    "invalid token",
    "invalid access_token",
    "tokenexception",
    "session expired",
    "unauthorized",
    "401",
    '"s":"error"',
    "missing handshake",
)


def _looks_like_rate_limit(msg: str) -> bool:
    if not msg:
        return False
    lower = msg.lower()
    return "rate" in lower and ("exceed" in lower or "limit" in lower or "throttl" in lower)


def _looks_like_auth_failure(msg: str) -> bool:
    if not msg:
        return False
    lower = msg.lower()
    return any(hint in lower for hint in _AUTH_FAILURE_HINTS)


# ── Diagnose helpers ─────────────────────────────────────────────────

def _diagnose_repr(obj):
    """Best-effort JSON-friendly representation of a Position / Holding / Margin
    instance — keeps the raw broker payload visible so the user can debug
    "why is my data empty" without us hiding fields."""
    if hasattr(obj, "__dict__"):
        return {
            k: (v if isinstance(v, (str, int, float, bool, type(None), dict, list)) else str(v))
            for k, v in obj.__dict__.items()
        }
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    return str(obj)


# ── Snapshot helper (shared by connect / refresh / Celery task) ───────

def _take_snapshot(link: BrokerLink) -> BrokerPositionSnapshot:
    adapter = build_adapter(link)
    if adapter is None:
        snap = BrokerPositionSnapshot.objects.create(
            tenant=link.tenant, link=link, fetched_at=timezone.now(),
            ok=False, error="adapter unavailable (missing plugin or credentials)",
        )
        link.status = BrokerLink.Status.ERRORED
        link.last_error = snap.error
        link.save(update_fields=["status", "last_error"])
        return snap

    # We deliberately do NOT pre-probe with adapter.authenticate() here —
    # for Angel One that means a fresh SmartAPI generateSession() on every
    # 30-second beat, which hits SmartAPI's rate limit ("Access denied
    # because of exceeding access rate"). The fetches themselves authenticate
    # lazily (Angel) or raise on token failure (Zerodha/Fyers), so any auth
    # problem still surfaces — just inside the except branch below.
    try:
        positions = adapter.fetch_positions()
        holdings = adapter.fetch_holdings()
        margin = adapter.fetch_margin()
        snap = BrokerPositionSnapshot.objects.create(
            tenant=link.tenant, link=link, fetched_at=timezone.now(),
            positions=type(adapter).serialise_positions(positions),
            holdings=type(adapter).serialise_holdings(holdings),
            margin=type(adapter).serialise_margin(margin),
            ok=True,
        )
        link.status = BrokerLink.Status.ACTIVE
        link.last_refreshed_at = snap.fetched_at
        link.last_error = ""
        link.save(update_fields=["status", "last_refreshed_at", "last_error"])
        return snap
    except Exception as e:
        err = str(e)[:500]
        # Classify auth/token failures as EXPIRED so the UI offers Re-login
        # instead of a generic "errored". Everything else stays ERRORED.
        if _looks_like_auth_failure(err):
            new_status = BrokerLink.Status.EXPIRED
            err = (
                f"Daily re-login required — {err}"
                if link.broker_name in DAILY_TOKEN_BROKERS
                else f"Broker rejected credentials — {err}"
            )
        else:
            new_status = BrokerLink.Status.ERRORED
        snap = BrokerPositionSnapshot.objects.create(
            tenant=link.tenant, link=link, fetched_at=timezone.now(),
            ok=False, error=err,
        )
        link.status = new_status
        link.last_error = err
        link.save(update_fields=["status", "last_error"])
        return snap
