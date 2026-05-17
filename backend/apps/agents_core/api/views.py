from __future__ import annotations

import asyncio
import uuid
from typing import Literal

from rest_framework import mixins, serializers, status, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.agents_core.domain.contracts import AgentContext
from apps.agents_core.models import AgentRun, AgentStep
from apps.agents_core.registry import strategy_registry
from apps.agents_core.tasks.run import execute_run


class AgentStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentStep
        fields = ["seq", "node", "event_type", "payload", "created_at"]


class AgentRunSerializer(serializers.ModelSerializer):
    """List form omits the heavy fields. Detail form includes `steps`
    so the React Stream tab can replay history on completed runs without
    waiting for the WebSocket (which doesn't replay)."""
    steps = serializers.SerializerMethodField()

    class Meta:
        model = AgentRun
        fields = [
            "id", "strategy_name", "strategy_version", "portfolio",
            "config", "status", "result", "error", "created_at",
            "started_at", "completed_at", "steps",
        ]
        read_only_fields = ["id", "status", "result", "error",
                            "created_at", "started_at", "completed_at",
                            "strategy_version", "steps"]

    def get_steps(self, obj):
        # Only return steps on the detail endpoint to keep the list view fast.
        view = self.context.get("view")
        if not view or getattr(view, "action", None) != "retrieve":
            return None
        rows = AgentStep.objects.filter(run=obj).order_by("seq")
        return AgentStepSerializer(rows, many=True).data


class AgentRunViewSet(mixins.CreateModelMixin,
                      mixins.ListModelMixin,
                      mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    serializer_class = AgentRunSerializer

    def get_queryset(self):
        return AgentRun.objects.filter(tenant=self.request.tenant)

    def create(self, request, *args, **kwargs):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)
        strat = strategy_registry.get(ser.validated_data["strategy_name"])
        run = AgentRun.objects.create(
            tenant=request.tenant,
            triggered_by=request.user,
            strategy_name=strat.name,
            strategy_version=strat.version,
            portfolio=ser.validated_data["portfolio"],
            config=ser.validated_data.get("config", {}),
        )
        execute_run.delay(str(run.id))
        return Response(self.get_serializer(run).data, status=status.HTTP_202_ACCEPTED)


class StrategyCatalogViewSet(viewsets.ViewSet):
    def list(self, request):
        data = []
        for name, strat in strategy_registry.items():
            s = strat.schema()
            data.append({
                "name": s.name, "version": s.version,
                "asset_class": s.asset_class,
                "params": s.params,
                "required_retrievers": s.required_retrievers,
            })
        return Response(data)


# ─────────────────────────────────────────────────────────────────────────
# Plan-stock orchestrator
# ─────────────────────────────────────────────────────────────────────────
Horizon = Literal["intraday", "swing", "monthly"]

# Each entry: (strategy_name, bucket_label, allocation_pct)
# Strategies that aren't in the registry are silently skipped.
HORIZON_POLICIES: dict[Horizon, list[tuple[str, str, float]]] = {
    "intraday": [
        ("directional",     "equity_intraday",   0.50),
        ("pyramid",         "options_intraday",  0.30),
        ("vertical_spread", "options_intraday",  0.10),
        ("reserve",         "reserve",           0.10),
    ],
    "swing": [
        ("directional",     "equity_delivery",   0.40),
        ("vertical_spread", "options_swing",     0.35),
        ("pyramid",         "options_intraday",  0.15),
        ("reserve",         "reserve",           0.10),
    ],
    "monthly": [
        ("directional",     "equity_delivery",   0.30),
        ("vertical_spread", "options_monthly",   0.40),
        ("pyramid",         "options_intraday",  0.20),
        ("reserve",         "reserve",           0.10),
    ],
}


class _NoopPublisher:
    """Discards events — orchestrator runs are not persisted as AgentSteps."""
    def emit(self, _event):
        pass


class _NoopJournal:
    def record(self, _entry):
        pass


def _make_ctx(request, portfolio_id, config: dict) -> AgentContext:
    """Build a minimal AgentContext for a planner-only run.

    Plugins reach into legacy modules directly for data/risk/broker, so the
    port stubs can be None. Publisher + journal are no-ops because the
    orchestrator returns the full state itself and doesn't persist AgentRuns.
    """
    return AgentContext(
        run_id=uuid.uuid4(),
        tenant_id=getattr(request.tenant, "id", uuid.uuid4()),
        user_id=request.user.id,
        portfolio_id=portfolio_id,
        market_data=None,  # type: ignore[arg-type]
        rag=None,          # type: ignore[arg-type]
        risk=None,         # type: ignore[arg-type]
        journal=_NoopJournal(),
        publisher=_NoopPublisher(),
        config=config,
    )


async def _run_strategy(request, portfolio_id, strategy_name: str, config: dict) -> dict:
    """Build + ainvoke one strategy graph; return its final state.

    Wraps every plugin call so a single broken strategy never fails the
    whole orchestrator response.
    """
    try:
        strat = strategy_registry.get(strategy_name)
    except LookupError:
        return {"error": f"strategy {strategy_name} not registered", "config": config}

    ctx = _make_ctx(request, portfolio_id, config)
    try:
        graph = strat.build_graph(ctx)
        state = await graph.ainvoke({"config": config})
        return state
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "config": config}


class PlanStockView(APIView):
    """POST /api/v1/agents/plan-stock/

    Body:
        {
          "symbol":       "CIPLA",
          "portfolio":    "<uuid>",
          "total_capital": 500000,
          "horizon":      "monthly" | "swing" | "intraday",
          "side_hint":    "BULL" | "BEAR" | null,
          "dry_run":      true
        }

    Response:
        {
          symbol, horizon, total_capital, dry_run,
          allocations: [{strategy, bucket, pct, capital}],
          results:     [{strategy, status, capital, summary, raw_state}],
          summary:     { strategies_succeeded, strategies_failed, total_runtime_ms }
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        import time
        from apps.portfolio.models import Portfolio

        symbol = (request.data.get("symbol") or "").strip().upper()
        portfolio_id = request.data.get("portfolio")
        total_capital = float(request.data.get("total_capital") or 0)
        horizon: Horizon = request.data.get("horizon") or "monthly"
        side_hint = request.data.get("side_hint")
        dry_run = bool(request.data.get("dry_run", True))

        if not symbol:
            return Response({"error": "symbol required"}, status=400)
        if horizon not in HORIZON_POLICIES:
            return Response({"error": f"unknown horizon {horizon!r}"}, status=400)
        if total_capital <= 0:
            return Response({"error": "total_capital must be > 0"}, status=400)
        if not portfolio_id:
            return Response({"error": "portfolio required"}, status=400)
        try:
            portfolio = Portfolio.objects.get(id=portfolio_id, tenant=request.tenant)
        except Portfolio.DoesNotExist:
            return Response({"error": "portfolio not found"}, status=404)

        policy = HORIZON_POLICIES[horizon]
        registered = set(strategy_registry.names())
        # Directional uses the equity DataService — only applies to stocks,
        # not index underlyings. Skip it on NIFTY/BANKNIFTY/SENSEX and
        # reallocate that slice into the reserve so the math still adds up.
        IS_INDEX = symbol in {"NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "BANKEX"}
        skip = {"directional"} if IS_INDEX else set()

        allocations = []
        reallocated = 0.0
        for strategy, bucket, pct in policy:
            if strategy in skip:
                reallocated += pct
                continue
            if strategy == "reserve" or strategy in registered:
                allocations.append({
                    "strategy": strategy,
                    "bucket": bucket,
                    "pct": pct,
                    "capital": round(total_capital * pct, 2),
                })
        # Push the skipped slice into reserve so the total still sums to 1.
        if reallocated > 0:
            for a in allocations:
                if a["strategy"] == "reserve":
                    a["pct"] = round(a["pct"] + reallocated, 4)
                    a["capital"] = round(total_capital * a["pct"], 2)
                    break

        # Per-strategy config slices.
        configs: dict[str, dict] = {
            "directional": {
                "symbol": symbol,
                "intent": f"Plan a {horizon} trade for {symbol}.",
                "capital": next((a["capital"] for a in allocations if a["strategy"] == "directional"), 0),
                "dry_run": dry_run,
            },
            "vertical_spread": {
                "underlying": symbol,
                "side": (side_hint or "BULL").upper(),
                "capital": next((a["capital"] for a in allocations if a["strategy"] == "vertical_spread"), 0),
                "max_lots": 10,
            },
            "pyramid": {
                "underlying": symbol,
                "option_type": "CE" if (side_hint or "BULL").upper() == "BULL" else "PE",
                "capital": next((a["capital"] for a in allocations if a["strategy"] == "pyramid"), 0),
                "risk_pct": 2.0,
                "max_pyramids": 5,
                "lookback_days": 3,  # cover weekend + last 1-2 trading sessions
            },
        }

        # Fan out — strategies run in parallel via asyncio.gather.
        fan_out = [a for a in allocations if a["strategy"] in configs]

        async def _all():
            tasks = [
                _run_strategy(request, portfolio.id, a["strategy"], configs[a["strategy"]])
                for a in fan_out
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)

        t0 = time.time()
        states = asyncio.run(_all())
        runtime_ms = int((time.time() - t0) * 1000)

        results = []
        succeeded = failed = 0
        for alloc, state in zip(fan_out, states):
            err = (state or {}).get("error")
            if err:
                failed += 1
            else:
                succeeded += 1
            results.append({
                "strategy": alloc["strategy"],
                "bucket": alloc["bucket"],
                "capital": alloc["capital"],
                "status": "failed" if err else "succeeded",
                "summary": _summarise(alloc["strategy"], state),
                "raw_state": _scrub(state),
            })

        return Response({
            "symbol": symbol,
            "horizon": horizon,
            "side_hint": side_hint,
            "total_capital": total_capital,
            "dry_run": dry_run,
            "allocations": allocations,
            "results": results,
            "summary": {
                "strategies_succeeded": succeeded,
                "strategies_failed": failed,
                "total_runtime_ms": runtime_ms,
            },
        })


def _summarise(strategy: str, state: dict) -> dict:
    """One-screen summary of a strategy's final state."""
    if state.get("error"):
        return {"error": state["error"]}

    if strategy == "directional":
        plan = state.get("plan") or {}
        risk = state.get("risk") or {}
        return {
            "side": plan.get("side"),
            "symbol": plan.get("symbol"),
            "quantity": plan.get("quantity"),
            "entry": plan.get("entry_price"),
            "stop_loss": plan.get("stop_loss"),
            "target": plan.get("target"),
            "confidence": plan.get("confidence"),
            "reasoning": (plan.get("reasoning") or "")[:300],
            "risk_approved": risk.get("approved"),
            "risk_reason": risk.get("reason"),
            "indicators": state.get("indicators"),
        }

    if strategy == "vertical_spread":
        plan = state.get("plan") or {}
        return {
            "error": plan.get("error"),  # surface "LTP zero" etc. so UI can render the cause
            "side": plan.get("side"),
            "option_type": plan.get("option_type"),
            "expiry": plan.get("expiry"),
            "long_strike": plan.get("long_strike"),
            "short_strike": plan.get("short_strike"),
            "long_ltp": state.get("long_ltp"),
            "short_ltp": state.get("short_ltp"),
            "net_debit": plan.get("net_debit"),
            "max_profit_inr": plan.get("max_profit_inr"),
            "max_loss_inr": plan.get("max_loss_inr"),
            "breakeven": plan.get("breakeven"),
            "lots": plan.get("lots"),
            "rr_ratio": plan.get("rr_ratio"),
            "capital_used": plan.get("capital_used"),
            "spot": plan.get("spot"),
        }

    if strategy == "pyramid":
        plan = state.get("plan") or {}
        return {
            "error": plan.get("error"),
            "underlying": state.get("underlying"),
            "strike": state.get("strike"),
            "expiry": state.get("expiry"),
            "spot": state.get("spot"),
            "symbol": plan.get("symbol") or state.get("symbol"),
            "entries": len(plan.get("entries", [])),
            "exit_price": plan.get("exit_price"),
            "exit_reason": plan.get("exit_reason"),
            "total_lots": plan.get("total_lots"),
            "peak_lots": plan.get("peak_lots"),
            "pnl_inr": plan.get("total_pnl_rupees"),
            "avg_entry": plan.get("avg_entry"),
            "candles": state.get("candles_raw") and len(state["candles_raw"]),
            "first_entry": (plan.get("entries") or [{}])[0] if plan.get("entries") else None,
        }
    return {}


def _scrub(state: dict | None) -> dict:
    """Drop the bulky candles arrays from the raw state — UI doesn't need them."""
    if not state:
        return {}
    keep = {k: v for k, v in state.items() if k not in ("candles_raw", "_seq")}
    # Pyramid stashes log_tail in plan; that's fine. Drop candles in market_data
    md = keep.get("market_data") or {}
    if isinstance(md, dict) and "candles" in md:
        md = {k: v for k, v in md.items() if k != "candles"}
        keep["market_data"] = md
    return keep
