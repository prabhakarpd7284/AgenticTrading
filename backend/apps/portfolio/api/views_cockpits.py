"""Trader-facing cockpit endpoints.

Each view is a thin wrapper around an `apps.portfolio.services.cockpits.*`
aggregation function. All return JSON-safe dicts. Reads only — no writes.

Routes are mounted **before** the PortfolioViewSet detail catchall in
``apps.portfolio.api.urls`` so paths like ``capital-cockpit/`` aren't
swallowed by ``^(?P<pk>[^/.]+)/$``.
"""
from __future__ import annotations

from datetime import date

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from django.core.cache import cache
from django.core.management import call_command

from apps.portfolio.services.cockpits import (
    build_broker_reconciliation,
    build_capital_cockpit,
    build_edge_decay,
    build_expiry_cockpit,
    build_greeks_heatmap,
    build_plan_vs_actual,
    build_regime_heatmap,
    build_risk_budget,
    build_signal_funnel,
    build_theta_forecast,
)
from apps.portfolio.services.correlation_matrix import build_correlation_report
from apps.portfolio.services.earnings_overlay import build_earnings_overlay
from apps.portfolio.services.edge_ledger import build_edge_ledger
from apps.portfolio.services.forced_flat import build_forced_flat, flatten_all
from apps.portfolio.services.intraday_rotation import build_intraday_rotation
from apps.portfolio.services.gap_risk import build_gap_risk_report
from apps.portfolio.services.post_mortem import build_post_mortem_report
from apps.portfolio.services.sizer_simulator import simulate as simulate_sizer
from apps.portfolio.services.slippage_edge import (
    compute as compute_slippage_edge, setup_catalog,
)
from apps.portfolio.services.structural_stops import build_structural_stops


class _BaseCockpitView(APIView):
    permission_classes = [IsAuthenticated]


class CapitalCockpitView(_BaseCockpitView):
    """GET /api/v1/portfolios/capital-cockpit/"""
    def get(self, request):
        return Response(build_capital_cockpit(getattr(request, "tenant", None)))


class SetCapitalView(_BaseCockpitView):
    """POST /api/v1/portfolios/capital/  body: {"capital": <inr>}

    Updates today's PortfolioSnapshot (or creates one) with the new capital.
    `available_cash` is recomputed as capital - invested so position sizing
    pulls the right denominator immediately.
    """
    def post(self, request):
        from trading.models import PortfolioSnapshot

        try:
            new_capital = float(request.data.get("capital"))
        except (TypeError, ValueError):
            return Response(
                {"error": "capital must be a number"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if new_capital <= 0:
            return Response(
                {"error": "capital must be > 0"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        today = date.today()
        existing = PortfolioSnapshot.objects.filter(snapshot_date=today).first()
        invested = float(existing.invested) if existing else 0.0
        available = max(new_capital - invested, 0.0)

        defaults = {
            "capital": new_capital,
            "available_cash": available,
            "invested": invested,
        }
        if existing:
            for k, v in defaults.items():
                setattr(existing, k, v)
            existing.save(update_fields=list(defaults.keys()) + ["last_updated"])
            snap = existing
        else:
            snap = PortfolioSnapshot.objects.create(snapshot_date=today, **defaults)

        return Response({
            "capital": float(snap.capital),
            "invested": float(snap.invested),
            "available_cash": float(snap.available_cash),
            "snapshot_date": snap.snapshot_date.isoformat(),
        })


class PlanVsActualView(_BaseCockpitView):
    """GET /api/v1/portfolios/plan-vs-actual/?limit=200&strategy=directional"""
    def get(self, request):
        limit = int(request.query_params.get("limit", 200) or 200)
        strategy = request.query_params.get("strategy") or None
        return Response(build_plan_vs_actual(
            getattr(request, "tenant", None), limit=limit, strategy=strategy,
        ))


class GreeksHeatmapView(_BaseCockpitView):
    """GET /api/v1/portfolios/greeks-heatmap/"""
    def get(self, request):
        return Response(build_greeks_heatmap(getattr(request, "tenant", None)))


class SignalFunnelView(_BaseCockpitView):
    """GET /api/v1/portfolios/signal-funnel/"""
    def get(self, request):
        return Response(build_signal_funnel(getattr(request, "tenant", None)))


class RiskBudgetView(_BaseCockpitView):
    """GET /api/v1/portfolios/risk-budget/?lookback_days=30"""
    def get(self, request):
        lookback = int(request.query_params.get("lookback_days", 30) or 30)
        return Response(build_risk_budget(
            getattr(request, "tenant", None), lookback_days=lookback,
        ))


class ExpiryCockpitView(_BaseCockpitView):
    """GET /api/v1/portfolios/expiry-cockpit/?underlying=NIFTY"""
    def get(self, request):
        underlying = (request.query_params.get("underlying") or "NIFTY").upper()
        return Response(build_expiry_cockpit(
            getattr(request, "tenant", None), underlying=underlying,
        ))


class BrokerReconView(_BaseCockpitView):
    """GET /api/v1/portfolios/broker-recon/?on_date=2026-05-14"""
    def get(self, request):
        raw = request.query_params.get("on_date")
        on_date = None
        if raw:
            try:
                on_date = date.fromisoformat(raw)
            except ValueError:
                on_date = None
        return Response(build_broker_reconciliation(
            getattr(request, "tenant", None), on_date=on_date,
        ))


class EdgeDecayView(_BaseCockpitView):
    """GET /api/v1/portfolios/edge-decay/?window=20"""
    def get(self, request):
        window = int(request.query_params.get("window", 20) or 20)
        return Response(build_edge_decay(
            getattr(request, "tenant", None), window=window,
        ))


class ThetaForecastView(_BaseCockpitView):
    """GET /api/v1/portfolios/theta-forecast/"""
    def get(self, request):
        return Response(build_theta_forecast(getattr(request, "tenant", None)))


class RegimeHeatmapView(_BaseCockpitView):
    """GET /api/v1/portfolios/regime-heatmap/"""
    def get(self, request):
        return Response(build_regime_heatmap(getattr(request, "tenant", None)))


class CorrelationMatrixView(_BaseCockpitView):
    """GET /api/v1/portfolios/correlation/"""
    def get(self, request):
        return Response(build_correlation_report(getattr(request, "tenant", None)))


class PostMortemView(_BaseCockpitView):
    """GET /api/v1/portfolios/post-mortem/?month=YYYY-MM"""
    def get(self, request):
        month = request.query_params.get("month") or None
        return Response(build_post_mortem_report(
            getattr(request, "tenant", None), month=month,
        ))


class GapRiskView(_BaseCockpitView):
    """GET /api/v1/portfolios/gap-risk/"""
    def get(self, request):
        return Response(build_gap_risk_report(getattr(request, "tenant", None)))


class SizerSimulatorView(_BaseCockpitView):
    """POST /api/v1/portfolios/sizer/simulate/
    body: {symbol, qty, side, stop, entry?, product?}
    """
    def post(self, request):
        return Response(simulate_sizer(request.data or {}))


class StructuralStopsView(_BaseCockpitView):
    """GET /api/v1/portfolios/structural-stops/"""
    def get(self, request):
        return Response(build_structural_stops(getattr(request, "tenant", None)))


class ForcedFlatView(_BaseCockpitView):
    """GET  /api/v1/portfolios/forced-flat/        — countdown + close list
    POST /api/v1/portfolios/forced-flat/flatten/  — square off all (paper)
    """
    def get(self, request):
        return Response(build_forced_flat(getattr(request, "tenant", None)))


class ForcedFlatFlattenView(_BaseCockpitView):
    def post(self, request):
        return Response(flatten_all(getattr(request, "tenant", None)))


class SlippageEdgeView(_BaseCockpitView):
    """POST /api/v1/portfolios/slippage-edge/  body: {symbol, qty, setup_avg_r_inr | setup_family}"""
    def post(self, request):
        return Response(compute_slippage_edge(request.data or {}))


class SlippageEdgeSetupsView(_BaseCockpitView):
    """GET /api/v1/portfolios/slippage-edge/setups/ — setup-family catalog"""
    def get(self, request):
        return Response(setup_catalog())


class ResetTradingDataView(_BaseCockpitView):
    """POST /api/v1/portfolios/reset/  body: {flags, capital?, no_reseed?}

    Mirrors the `python manage.py reset_trading_data --confirm` CLI.

      flags          list of: journal, straddles, snapshots, audit, signals,
                     watchlist, orders, runs, positions, cache, all, nuke
      keep_watchlist bool  — only honored when "all" is in flags
      capital        number — seed today's snapshot with this capital
      no_reseed      bool  — when "nuke" is in flags, skip the auto re-seed
    """
    def post(self, request):
        body = request.data or {}
        flags = [str(f).lower() for f in (body.get("flags") or [])]
        if not flags and body.get("capital") is None:
            return Response(
                {"error": "supply at least one flag in `flags` (or `capital`)"},
                status=400,
            )

        cli_kwargs: dict = {"confirm": True, "verbosity": 0}
        all_flags = (
            "journal", "straddles", "snapshots", "audit", "signals",
            "watchlist", "orders", "runs", "positions", "cache",
        )
        if "nuke" in flags:
            cli_kwargs["nuke"] = True
            if body.get("no_reseed"):
                cli_kwargs["no_reseed"] = True
        elif "all" in flags:
            cli_kwargs["all"] = True
            if body.get("keep_watchlist"):
                cli_kwargs["keep_watchlist"] = True
        else:
            for f in flags:
                if f in all_flags:
                    cli_kwargs[f] = True
        if body.get("capital") is not None:
            try:
                cli_kwargs["capital"] = float(body.get("capital"))
            except (TypeError, ValueError):
                return Response({"error": "capital must be a number"}, status=400)

        # Capture command output so the FE can show what was wiped.
        from io import StringIO
        buf = StringIO()
        try:
            call_command("reset_trading_data", stdout=buf, **cli_kwargs)
        except Exception as e:  # noqa: BLE001
            return Response({"error": str(e)}, status=500)

        # Always flush the local Django cache too — even if --cache wasn't
        # explicitly chosen — because every cockpit panel caches per-symbol
        # broker calls and we don't want stale numbers post-reset.
        cache.clear()

        return Response({
            "ok": True,
            "summary": buf.getvalue().strip(),
            "flags": flags,
            "capital": cli_kwargs.get("capital"),
            "reseeded": "nuke" in flags and not body.get("no_reseed"),
        })


class EdgeLedgerView(_BaseCockpitView):
    """GET /api/v1/portfolios/edge-ledger/?limit=200"""
    def get(self, request):
        try:
            limit = int(request.query_params.get("limit", 200) or 200)
        except ValueError:
            limit = 200
        return Response(build_edge_ledger(getattr(request, "tenant", None), limit=limit))


class IntradayRotationView(_BaseCockpitView):
    """GET /api/v1/portfolios/intraday-rotation/?date=YYYY-MM-DD"""
    def get(self, request):
        return Response(build_intraday_rotation(
            getattr(request, "tenant", None),
            on=request.query_params.get("date"),
        ))


class EarningsOverlayView(_BaseCockpitView):
    """GET /api/v1/portfolios/earnings-overlay/"""
    def get(self, request):
        return Response(build_earnings_overlay(getattr(request, "tenant", None)))
