from rest_framework import serializers, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.strategies.models import Backtest, StrategyInstance
from apps.strategies.services import backtester_lab as bt_lab
from apps.strategies.services.base_quality import build_base_quality
from apps.strategies.services.breakout_classifier import build_breakout_classifier
from apps.strategies.services.mtf_stage_scanner import build_mtf_stage_scanner


class StrategyInstanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StrategyInstance
        fields = "__all__"


class BacktestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Backtest
        fields = "__all__"


class StrategyInstanceViewSet(viewsets.ModelViewSet):
    serializer_class = StrategyInstanceSerializer

    def get_queryset(self):
        return StrategyInstance.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant)


class BacktestViewSet(viewsets.ModelViewSet):
    serializer_class = BacktestSerializer

    def get_queryset(self):
        return Backtest.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        from apps.strategies.tasks.backtest import run_backtest
        bt = serializer.save(tenant=self.request.tenant)
        run_backtest.delay(str(bt.id))


class BaseQualityView(APIView):
    """GET /api/v1/strategies/base-quality/?symbols=A,B,C
    (or no params to score the legacy watchlist).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        raw = request.query_params.get("symbols") or request.query_params.get("symbol") or ""
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()] if raw else None
        return Response(build_base_quality(symbols=syms))


class MTFStageScannerView(APIView):
    """GET /api/v1/strategies/mtf-stage/?symbols=A,B,C"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        raw = request.query_params.get("symbols") or ""
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()] if raw else None
        return Response(build_mtf_stage_scanner(symbols=syms))


class BreakoutClassifierView(APIView):
    """GET /api/v1/strategies/breakout-classifier/?symbols=A,B,C"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        raw = request.query_params.get("symbols") or ""
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()] if raw else None
        return Response(build_breakout_classifier(symbols=syms))


# ──────────────────────── Backtester Lab (10 endpoints) ──────────────────────
class _BTBase(APIView):
    permission_classes = [IsAuthenticated]

    def _qp(self, request, key, default, cast=int):
        try:
            return cast(request.query_params.get(key, default))
        except (TypeError, ValueError):
            return default


class BTWalkForwardView(_BTBase):
    """GET /api/v1/strategies/backtester/walk-forward/?symbol=&strategy=&splits=4&days=180"""
    def get(self, request):
        return Response(bt_lab.walk_forward(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 180),
            strategy=request.query_params.get("strategy") or "directional",
            splits=self._qp(request, "splits", 4),
        ))


class BTMonteCarloView(_BTBase):
    """GET /api/v1/strategies/backtester/monte-carlo/?symbol=&runs=500&ruin_pct=30"""
    def get(self, request):
        return Response(bt_lab.monte_carlo(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 120),
            strategy=request.query_params.get("strategy") or "directional",
            runs=self._qp(request, "runs", 500),
            ruin_pct=self._qp(request, "ruin_pct", 30.0, cast=float),
        ))


class BTRegimeStatsView(_BTBase):
    """GET /api/v1/strategies/backtester/regime-stats/?symbol=&strategy="""
    def get(self, request):
        return Response(bt_lab.regime_stats(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 180),
            strategy=request.query_params.get("strategy") or "directional",
        ))


class BTCostSensitivityView(_BTBase):
    """GET /api/v1/strategies/backtester/cost-sensitivity/?symbol=&base_cost=40"""
    def get(self, request):
        return Response(bt_lab.cost_sensitivity(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 120),
            strategy=request.query_params.get("strategy") or "directional",
            base_cost_inr=self._qp(request, "base_cost", 40.0, cast=float),
        ))


class BTEdgeDriftView(_BTBase):
    """GET /api/v1/strategies/backtester/edge-drift/?symbol=&window=20"""
    def get(self, request):
        return Response(bt_lab.edge_drift(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 120),
            strategy=request.query_params.get("strategy") or "directional",
            window=self._qp(request, "window", 20),
        ))


class BTCapacityView(_BTBase):
    """GET /api/v1/strategies/backtester/capacity/?symbol=&strategy="""
    def get(self, request):
        return Response(bt_lab.capacity_curve(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 120),
            strategy=request.query_params.get("strategy") or "directional",
        ))


class BTSweepView(_BTBase):
    """GET /api/v1/strategies/backtester/sweep/?symbol=&param_a=ema&values_a=10,20,30,40,50"""
    def get(self, request):
        def _parse_ints(raw):
            return [int(x) for x in (raw or "").split(",") if x.strip().lstrip("-").isdigit()]
        return Response(bt_lab.parameter_sweep(
            request.query_params.get("symbol") or "NIFTY",
            strategy=request.query_params.get("strategy") or "directional",
            param_a=request.query_params.get("param_a") or "ema",
            values_a=_parse_ints(request.query_params.get("values_a")) or None,
            param_b=request.query_params.get("param_b") or None,
            values_b=_parse_ints(request.query_params.get("values_b")) or None,
            days=self._qp(request, "days", 120),
        ))


class BTBootstrapView(_BTBase):
    """GET /api/v1/strategies/backtester/bootstrap/?symbol=&block_size=5&runs=200"""
    def get(self, request):
        return Response(bt_lab.block_bootstrap(
            request.query_params.get("symbol") or "NIFTY",
            days=self._qp(request, "days", 180),
            strategy=request.query_params.get("strategy") or "directional",
            block_size=self._qp(request, "block_size", 5),
            runs=self._qp(request, "runs", 200),
        ))


class BTRegistryView(_BTBase):
    """GET  /api/v1/strategies/backtester/registry/        — list saved runs
    POST /api/v1/strategies/backtester/registry/save/    — persist a new one
    POST /api/v1/strategies/backtester/registry/compare/ — compare N
    GET  /api/v1/strategies/backtester/registry/suggest/ — suggested next
    """
    def get(self, request):
        action = request.query_params.get("action") or "list"
        if action == "list":
            return Response(bt_lab.list_runs())
        if action == "suggest":
            return Response(bt_lab.suggest_next_test())
        return Response({"error": f"unknown action: {action}"}, status=400)

    def post(self, request):
        body = request.data or {}
        action = body.get("action") or "save"
        if action == "save":
            return Response(bt_lab.save_run(
                symbol=body.get("symbol") or "NIFTY",
                strategy=body.get("strategy") or "directional",
                params=body.get("params") or {},
                days=int(body.get("days") or 120),
                label=body.get("label") or "",
            ))
        if action == "compare":
            run_ids = body.get("run_ids") or []
            return Response(bt_lab.compare_runs(run_ids))
        return Response({"error": f"unknown action: {action}"}, status=400)
