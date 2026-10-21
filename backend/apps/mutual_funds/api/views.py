"""Mutual fund REST surface (Phase MF-UI) — mounted at /api/v1/mutual-funds/.

  GET  holdings/                 tenant's MFHolding rows valued at the latest
                                 NAV + a summary (totals, by-category mix)
  GET  schemes/?q=               search the global AMFI scheme catalog
  GET  schemes/<code>/?days=     one scheme + its NAV history
  GET  advisor/findings/         the tenant's latest MF advisor findings
  POST advisor/run/              run the advisor now (owner only; emits via
                                 the advisor's existing event_writer path)
  GET  status/                   NAV freshness + stale count for HELD schemes
  GET  screener/?category=...    within-category ranking with per-component
                                 scores + index-vs-active check (MF-S)
  GET  schemes/<code>/metrics/   latest metric snapshot + data-quality issues
  GET  schemes/<code>/nav-history/?days=&benchmark=1   NAV series (+ TRI)

MFScheme / MFNav are global reference data; MFHolding / MFTransaction and
the advisor Events are tenant-scoped and always filtered by request.tenant.
Read-only apart from advisor/run, which only writes Event rows.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from decimal import Decimal

from django.db.models import Count, Max, OuterRef, Q, Subquery
from django.shortcuts import get_object_or_404
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, inline_serializer
from rest_framework import generics, serializers
from rest_framework.exceptions import ValidationError
from rest_framework.pagination import LimitOffsetPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import OwnerOnly, TenantScoped
from apps.events.models import Event
from apps.mutual_funds.models import MFHolding, MFNav, MFScheme
from apps.mutual_funds.services.advisor import (
    DEFAULT_ASSUMED_DRAG_PCT,
    category_label,
    run_advisor,
)
from apps.mutual_funds.services.staleness import DEFAULT_MAX_TRADING_DAYS, trading_days_back
from apps.trading.models import Portfolio

ADVISOR_STEP_PREFIX = "mf_advisor."
# Findings of one advisor run are emitted back-to-back; anything within this
# window of the newest finding counts as the "latest run".
ADVISOR_RUN_WINDOW = timedelta(minutes=10)
MAX_NAV_HISTORY_DAYS = 3650
DEFAULT_NAV_HISTORY_DAYS = 365


def _truthy(v: str | None) -> bool:
    return (v or "").lower() in ("1", "true", "yes")


def _latest_nav_subqueries():
    base = MFNav.objects.filter(scheme_id=OuterRef("scheme_code")).order_by("-nav_date")
    return {
        "latest_nav": Subquery(base.values("nav")[:1]),
        "latest_nav_date": Subquery(base.values("nav_date")[:1]),
    }


def _stale_cutoff():
    return trading_days_back(timezone.localdate(), DEFAULT_MAX_TRADING_DAYS)


def _f(x) -> float | None:
    return None if x is None else float(x)


def _pct(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return round(num / den * 100, 2)


def _resolve_portfolio(request) -> Portfolio | None:
    """``?portfolio=<uuid>`` limited to the caller's tenant; 400 on a bad id."""
    raw = request.query_params.get("portfolio")
    if not raw:
        return None
    try:
        pid = uuid.UUID(str(raw))
    except ValueError as exc:
        raise ValidationError({"portfolio": "Must be a portfolio UUID."}) from exc
    pf = Portfolio.objects.filter(tenant=request.tenant, id=pid).first()
    if pf is None:
        raise ValidationError({"portfolio": "Portfolio not found for this tenant."})
    return pf


# ── holdings ─────────────────────────────────────────────────────────────

def build_holdings_payload(tenant, portfolio=None, include_closed=False) -> dict:
    qs = (
        MFHolding.objects.filter(tenant=tenant)
        .select_related("scheme", "portfolio")
        .order_by("scheme__scheme_name", "folio_number")
    )
    if portfolio is not None:
        qs = qs.filter(portfolio=portfolio)
    if not include_closed:
        qs = qs.filter(units__gt=0)

    holdings = list(qs)
    codes = {h.scheme_id for h in holdings}
    latest = {
        s.scheme_code: (s.latest_nav, s.latest_nav_date)
        for s in MFScheme.objects.filter(scheme_code__in=codes).annotate(**_latest_nav_subqueries())
    }
    cutoff = _stale_cutoff()

    rows = []
    tot_invested = tot_value = tot_gain = 0.0
    basis_invested = 0.0
    unvalued = missing_cost = 0
    by_cat: dict[str, float] = {}
    for h in holdings:
        nav, nav_date = latest.get(h.scheme_id, (None, None))
        units = h.units or Decimal("0")
        invested = float(units * h.avg_cost_per_unit) if h.avg_cost_per_unit is not None else None
        value = float(units * nav) if nav is not None else None
        gain = (value - invested) if (value is not None and invested is not None) else None
        cat = category_label(h.scheme)

        if invested is None:
            missing_cost += 1
        else:
            tot_invested += invested
        if value is None:
            unvalued += 1
        else:
            tot_value += value
            by_cat[cat] = by_cat.get(cat, 0.0) + value
        if gain is not None:
            tot_gain += gain
            basis_invested += invested

        rows.append({
            "id": h.id,
            "portfolio_id": str(h.portfolio_id),
            "portfolio_name": h.portfolio.name,
            "folio_number": h.folio_number,
            "source": h.source,
            "scheme_code": h.scheme.scheme_code,
            "scheme_name": h.scheme.scheme_name,
            "amc_name": h.scheme.amc_name,
            "plan": h.scheme.plan,
            "option": h.scheme.option,
            "category": cat,
            "is_direct": h.scheme.is_direct,
            "units": float(units),
            "avg_cost_per_unit": _f(h.avg_cost_per_unit),
            "invested": None if invested is None else round(invested, 2),
            "latest_nav": _f(nav),
            "nav_date": nav_date.isoformat() if nav_date else None,
            "nav_is_stale": nav_date is None or nav_date < cutoff,
            "current_value": None if value is None else round(value, 2),
            "gain": None if gain is None else round(gain, 2),
            "gain_pct": _pct(gain, invested),
        })

    allocation = [
        {"category": c, "value": round(v, 2), "pct": _pct(v, tot_value) or 0.0}
        for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])
    ]
    summary = {
        "holding_count": len(rows),
        "invested": round(tot_invested, 2),
        "current_value": round(tot_value, 2),
        # Gain only over holdings that have BOTH a cost basis and a NAV, so a
        # missing cost never shows up as a fake 100% gain.
        "gain": round(tot_gain, 2),
        "gain_pct": _pct(tot_gain, basis_invested),
        "gain_basis_invested": round(basis_invested, 2),
        "unvalued_count": unvalued,
        "missing_cost_count": missing_cost,
        "allocation": allocation,
    }
    return {"results": rows, "summary": summary, "include_closed": include_closed}


class HoldingsView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(
        parameters=[
            OpenApiParameter("portfolio", OpenApiTypes.UUID, OpenApiParameter.QUERY, required=False,
                             description="Limit to one of the tenant's portfolios."),
            OpenApiParameter("include_closed", OpenApiTypes.BOOL, OpenApiParameter.QUERY, required=False,
                             description="Include zero-unit (closed) folios. Default false."),
        ],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request):
        return Response(build_holdings_payload(
            request.tenant,
            portfolio=_resolve_portfolio(request),
            include_closed=_truthy(request.query_params.get("include_closed")),
        ))


# ── schemes ──────────────────────────────────────────────────────────────

class SchemeSearchPagination(LimitOffsetPagination):
    default_limit = 20
    max_limit = 50


class SchemeSerializer(serializers.ModelSerializer):
    category_label = serializers.SerializerMethodField()
    latest_nav = serializers.DecimalField(max_digits=14, decimal_places=4, read_only=True, allow_null=True)
    latest_nav_date = serializers.DateField(read_only=True, allow_null=True)

    class Meta:
        model = MFScheme
        fields = [
            "scheme_code", "scheme_name", "amc_name", "plan", "option", "category",
            "category_label", "is_direct", "isin_growth", "isin_div_reinvestment",
            "latest_nav", "latest_nav_date",
        ]

    def get_category_label(self, obj) -> str:
        return category_label(obj)


class SchemeSearchView(generics.ListAPIView):
    """Search the global AMFI catalog. Every whitespace-separated term must
    match the scheme name or AMC (or the whole query an exact scheme code).
    Queries shorter than 2 characters return nothing."""

    permission_classes = [IsAuthenticated, TenantScoped]
    serializer_class = SchemeSerializer
    pagination_class = SchemeSearchPagination

    @extend_schema(parameters=[
        OpenApiParameter("q", OpenApiTypes.STR, OpenApiParameter.QUERY, required=True,
                         description="Search text (scheme name / AMC / scheme code), min 2 chars."),
        OpenApiParameter("direct", OpenApiTypes.BOOL, OpenApiParameter.QUERY, required=False,
                         description="Only direct plans."),
    ])
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        q = (self.request.query_params.get("q") or "").strip()
        if len(q) < 2:
            return MFScheme.objects.none()
        cond = Q()
        for term in q.split():
            cond &= Q(scheme_name__icontains=term) | Q(amc_name__icontains=term)
        qs = MFScheme.objects.filter(cond | Q(scheme_code=q))
        if _truthy(self.request.query_params.get("direct")):
            qs = qs.filter(is_direct=True)
        return qs.annotate(**_latest_nav_subqueries()).order_by("scheme_name", "scheme_code")


class SchemeDetailView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(
        parameters=[OpenApiParameter("days", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False,
                                     description=f"NAV history window in days (default {DEFAULT_NAV_HISTORY_DAYS}, "
                                                 f"max {MAX_NAV_HISTORY_DAYS}).")],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request, scheme_code: str):
        scheme = get_object_or_404(
            MFScheme.objects.annotate(**_latest_nav_subqueries()), scheme_code=scheme_code,
        )
        try:
            days = int(request.query_params.get("days", DEFAULT_NAV_HISTORY_DAYS))
        except ValueError as exc:
            raise ValidationError({"days": "Must be an integer."}) from exc
        days = max(1, min(days, MAX_NAV_HISTORY_DAYS))

        navs = []
        if scheme.latest_nav_date:
            since = scheme.latest_nav_date - timedelta(days=days)
            navs = [
                {"date": d.isoformat(), "nav": float(n)}
                for d, n in MFNav.objects.filter(scheme=scheme, nav_date__gte=since)
                .order_by("nav_date").values_list("nav_date", "nav")
            ]
        held_units = sum(
            (h.units for h in MFHolding.objects.filter(tenant=request.tenant, scheme=scheme, units__gt=0)),
            Decimal("0"),
        )
        return Response({
            "scheme": SchemeSerializer(scheme).data,
            "days": days,
            "navs": navs,
            # Tenant-scoped: only the caller's own units are ever reported.
            "held_units": float(held_units),
        })


# ── advisor ──────────────────────────────────────────────────────────────

def _finding_row(e: Event) -> dict:
    payload = e.payload or {}
    return {
        "id": e.id,
        "ts": e.ts.isoformat(),
        "kind": payload.get("kind") or e.step_name.removeprefix(ADVISOR_STEP_PREFIX),
        "severity": e.severity,
        "text": e.text,
        "is_estimate": bool(payload.get("is_estimate")),
        "payload": payload,
    }


def latest_advisor_findings(tenant, limit: int = 200) -> dict:
    qs = Event.objects.filter(
        tenant=tenant,
        type=Event.Type.WORKFLOW_STEP_COMPLETED,
        step_name__startswith=ADVISOR_STEP_PREFIX,
    ).order_by("-ts", "-id")
    newest = qs.first()
    if newest is None:
        return {"latest_run_at": None, "count": 0, "results": []}
    batch = list(qs.filter(ts__gte=newest.ts - ADVISOR_RUN_WINDOW)[:limit])
    batch.reverse()  # emission order within the run
    return {
        "latest_run_at": newest.ts.isoformat(),
        "count": len(batch),
        "results": [_finding_row(e) for e in batch],
    }


class AdvisorFindingsView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(
        parameters=[OpenApiParameter("limit", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False,
                                     description="Max findings (default 200, cap 500).")],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request):
        try:
            limit = min(max(int(request.query_params.get("limit", 200)), 1), 500)
        except ValueError as exc:
            raise ValidationError({"limit": "Must be an integer."}) from exc
        return Response(latest_advisor_findings(request.tenant, limit=limit))


class AdvisorRunView(APIView):
    """Runs the deterministic advisor for the caller's tenant. Reads holdings
    only; findings are written through advisor.emit_findings ->
    event_writer.emit() (non-blocking). Owner-only because it writes events."""

    permission_classes = [OwnerOnly, TenantScoped]

    @extend_schema(
        request=inline_serializer("MFAdvisorRunRequest", fields={
            "portfolio": serializers.UUIDField(required=False),
            "assumed_drag_pct": serializers.FloatField(required=False),
        }),
        responses=OpenApiTypes.OBJECT,
    )
    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        portfolio = None
        if data.get("portfolio"):
            try:
                pid = uuid.UUID(str(data["portfolio"]))
            except ValueError as exc:
                raise ValidationError({"portfolio": "Must be a portfolio UUID."}) from exc
            portfolio = Portfolio.objects.filter(tenant=request.tenant, id=pid).first()
            if portfolio is None:
                raise ValidationError({"portfolio": "Portfolio not found for this tenant."})
        try:
            drag = float(data.get("assumed_drag_pct", DEFAULT_ASSUMED_DRAG_PCT))
        except (TypeError, ValueError) as exc:
            raise ValidationError({"assumed_drag_pct": "Must be a number."}) from exc
        if not 0 <= drag <= 5:
            raise ValidationError({"assumed_drag_pct": "Must be between 0 and 5."})

        report = run_advisor(request.tenant, portfolio, assumed_drag_pct=drag, emit_events=True)
        return Response({
            "total_value": report.total_value,
            "notes": report.notes,
            "unvalued": report.unvalued,
            "count": len(report.findings),
            "findings": [
                {"kind": f.kind, "severity": f.severity, "text": f.text,
                 "is_estimate": bool(f.payload.get("is_estimate")), "payload": f.payload}
                for f in report.findings
            ],
        })


# ── status / freshness ───────────────────────────────────────────────────

def catalog_nav_date(max_nav_date):
    """The NAV date most schemes are priced at. AMFI's file has liquid /
    overnight funds publish NAVs for upcoming non-business days (e.g. a
    Saturday/Sunday NAV on Friday night), so a plain MAX(nav_date) can sit
    in the future — the busiest date of the last 10 days is the honest
    "catalog as of" date."""
    if max_nav_date is None:
        return None
    row = (
        MFNav.objects.filter(nav_date__gte=max_nav_date - timedelta(days=10))
        .values("nav_date").annotate(n=Count("id")).order_by("-n", "-nav_date").first()
    )
    return row["nav_date"] if row else max_nav_date


class StatusView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        max_nav_date = MFNav.objects.aggregate(latest=Max("nav_date"))["latest"]
        latest_nav_date = catalog_nav_date(max_nav_date)
        # pull_amfi_nav update_or_create()s every scheme, so the newest
        # MFScheme.updated_at is the last successful catalog pull.
        last_pull_at = MFScheme.objects.aggregate(t=Max("updated_at"))["t"]
        cutoff = _stale_cutoff()

        held_codes = set(
            MFHolding.objects.filter(tenant=request.tenant, units__gt=0).values_list("scheme_id", flat=True)
        )
        stale = []
        if held_codes:
            for s in (
                MFScheme.objects.filter(scheme_code__in=held_codes)
                .annotate(latest=Max("navs__nav_date")).order_by("scheme_name")
            ):
                if s.latest is None or s.latest < cutoff:
                    stale.append({
                        "scheme_code": s.scheme_code,
                        "scheme_name": s.scheme_name,
                        "latest_nav_date": s.latest.isoformat() if s.latest else None,
                    })
        return Response({
            "latest_nav_date": latest_nav_date.isoformat() if latest_nav_date else None,
            "max_nav_date": max_nav_date.isoformat() if max_nav_date else None,
            "last_pull_at": last_pull_at.isoformat() if last_pull_at else None,
            "stale_cutoff_date": cutoff.isoformat(),
            "stale_after_trading_days": DEFAULT_MAX_TRADING_DAYS,
            "catalog_is_stale": latest_nav_date is None or latest_nav_date < cutoff,
            "scheme_count": MFScheme.objects.count(),
            "held_scheme_count": len(held_codes),
            "stale_holdings_count": len(stale),
            "stale_holdings": stale,
        })


# ── screener (MF-S S9) ───────────────────────────────────────────────────
# Shared catalog data (snapshots / NAV / benchmarks) is identical for every
# tenant; the only tenant-scoped side effect is the audit Event a screen run
# emits, which is written to request.tenant only.

def _float_param(request, name):
    raw = request.query_params.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValidationError({name: "Must be a number."}) from exc


class ScreenerView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(
        parameters=[
            OpenApiParameter("category", OpenApiTypes.STR, OpenApiParameter.QUERY, required=False,
                             description="Canonical category label. Omit to list available categories only."),
            OpenApiParameter("min_years", OpenApiTypes.NUMBER, OpenApiParameter.QUERY, required=False),
            OpenApiParameter("min_sharpe", OpenApiTypes.NUMBER, OpenApiParameter.QUERY, required=False),
            OpenApiParameter("max_drawdown", OpenApiTypes.NUMBER, OpenApiParameter.QUERY, required=False,
                             description="Floor, e.g. -0.35 excludes funds that fell more than 35%."),
            OpenApiParameter("exclude_flag", OpenApiTypes.STR, OpenApiParameter.QUERY, required=False, many=True),
            OpenApiParameter("top", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False),
            OpenApiParameter("index_check", OpenApiTypes.BOOL, OpenApiParameter.QUERY, required=False,
                             description="Include the index-vs-active check (default true)."),
        ],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request):
        from apps.mutual_funds.models import MFMetricSnapshot
        from apps.mutual_funds.services.screener import PERSONAL_USE_NOTE, ScreenCriteria, latest_as_of, screen

        cats = (
            MFMetricSnapshot.objects.filter(is_active=True).values("category")
            .annotate(n=Count("id"), as_of=Max("as_of")).order_by("category")
        )
        available = [{"category": c["category"], "count": c["n"], "as_of": c["as_of"].isoformat()} for c in cats]
        category = (request.query_params.get("category") or "").strip()
        newest = latest_as_of()
        base = {"available_categories": available, "personal_use_note": PERSONAL_USE_NOTE,
                "metrics_as_of": newest.isoformat() if newest else None}
        if not category:
            return Response({**base, "result": None})
        match = next((c["category"] for c in available if c["category"].lower() == category.lower()), None)
        if match is None:
            raise ValidationError({"category": "Unknown category (see available_categories)."})
        top = request.query_params.get("top")
        try:
            top = None if top in (None, "") else max(1, min(int(top), 500))
        except ValueError as exc:
            raise ValidationError({"top": "Must be an integer."}) from exc
        min_years = _float_param(request, "min_years")
        crit = ScreenCriteria(
            categories=[match],
            min_history_years=3.0 if min_years is None else min_years,
            min_sharpe=_float_param(request, "min_sharpe"),
            max_drawdown_floor=_float_param(request, "max_drawdown"),
            exclude_flags=request.query_params.getlist("exclude_flag"),
            top=top,
        )
        with_ic = request.query_params.get("index_check") in (None, "", "1", "true", "yes")
        res = screen(crit, with_index_check=with_ic, tenant=request.tenant)
        return Response({**base, "result": res["categories"][0]})


class SchemeMetricsView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request, scheme_code: str):
        from apps.mutual_funds.models import MFDataIssue, MFMetricSnapshot
        from apps.mutual_funds.services.metrics_runner import SNAPSHOT_FIELDS
        from apps.mutual_funds.services.screener_config import RISK_FREE_RATE
        from apps.mutual_funds.services.universe import canonical_category

        scheme = get_object_or_404(MFScheme, scheme_code=scheme_code)
        snap = MFMetricSnapshot.objects.filter(scheme=scheme).order_by("-as_of").first()
        issues = [
            {"kind": i.kind, "start_date": i.start_date.isoformat(), "end_date": i.end_date.isoformat(),
             "detail": i.detail}
            for i in MFDataIssue.objects.filter(scheme=scheme).order_by("start_date")[:200]
        ]
        metrics = None
        if snap is not None:
            metrics = {"as_of": snap.as_of.isoformat(), "category": snap.category}
            for f in SNAPSHOT_FIELDS:
                v = getattr(snap, f)
                metrics[f] = v.isoformat() if hasattr(v, "isoformat") else v
        return Response({
            "scheme_code": scheme.scheme_code,
            "scheme_name": scheme.scheme_name,
            "amc_name": scheme.amc_name,
            "category": canonical_category(scheme.category, scheme.scheme_name),
            "is_active": scheme.is_active,
            "metrics": metrics,
            "data_missing": metrics is None,
            # S2 reference data not sourced yet — explicit, never imputed.
            "reference_data": {"expense_ratio_pct": None, "aum_cr": None,
                               "note": "TER / AUM not sourced yet (see docs/MF_SCREENER_DATA_SOURCES.md)."},
            "risk_free_rate": RISK_FREE_RATE,
            "data_issues": issues,
        })


class SchemeNavHistoryView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(
        parameters=[
            OpenApiParameter("days", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False,
                             description="Window in days (default 1825, max 5500)."),
            OpenApiParameter("benchmark", OpenApiTypes.BOOL, OpenApiParameter.QUERY, required=False,
                             description="Include the category benchmark TRI, rebased to the first NAV."),
        ],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request, scheme_code: str):
        from apps.mutual_funds.models import MFBenchmarkNav
        from apps.mutual_funds.services.screener_config import CATEGORY_BENCHMARKS
        from apps.mutual_funds.services.universe import canonical_category

        scheme = get_object_or_404(MFScheme, scheme_code=scheme_code)
        try:
            days = int(request.query_params.get("days", 1825))
        except ValueError as exc:
            raise ValidationError({"days": "Must be an integer."}) from exc
        days = max(1, min(days, 5500))
        last = MFNav.objects.filter(scheme=scheme).aggregate(m=Max("nav_date"))["m"]
        navs = []
        if last:
            navs = [
                {"date": d.isoformat(), "nav": float(n)}
                for d, n in MFNav.objects.filter(scheme=scheme, nav_date__gte=last - timedelta(days=days))
                .order_by("nav_date").values_list("nav_date", "nav")
            ]
        bench = None
        cat = canonical_category(scheme.category, scheme.scheme_name)
        spec = CATEGORY_BENCHMARKS.get(cat)
        if _truthy(request.query_params.get("benchmark")) and spec and navs:
            first_d = date_from_iso(navs[0]["date"])
            rows = list(MFBenchmarkNav.objects.filter(index_name=spec[0], nav_date__gte=first_d,
                                                      nav_date__lte=last).order_by("nav_date")
                        .values_list("nav_date", "value"))
            if rows:
                base = float(rows[0][1])
                bench = {"index_name": spec[0], "is_tri": True,
                         "points": [{"date": d.isoformat(), "value": round(float(v) / base * navs[0]["nav"], 4)}
                                    for d, v in rows]}
        return Response({
            "scheme_code": scheme.scheme_code,
            "scheme_name": scheme.scheme_name,
            "category": cat,
            "is_active": scheme.is_active,
            "days": days,
            "first_date": navs[0]["date"] if navs else None,
            "last_date": navs[-1]["date"] if navs else None,
            "count": len(navs),
            "navs": navs,
            "benchmark": bench,
        })


def date_from_iso(s: str):
    from datetime import date as _date
    return _date.fromisoformat(s)
