"""Strategy Studio API (R4): strategies with versions, templates, spec
validation + signal preview, and real backtests. Every queryset is filtered
by ``request.tenant``; cross-tenant ids 404."""
from __future__ import annotations

from datetime import date, timedelta

from django.shortcuts import get_object_or_404
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response

from apps.strategies.models import Backtest, BacktestTrade, StrategyInstance, StrategyVersion
from apps.strategies.services import lifecycle
from apps.strategies.services import spec as spec_mod
from apps.strategies.services import templates as tpl

_UUID_PK = [OpenApiParameter("id", OpenApiTypes.UUID, OpenApiParameter.PATH)]
_uuid_detail_schema = extend_schema_view(
    retrieve=extend_schema(parameters=_UUID_PK, responses=OpenApiTypes.OBJECT),
    update=extend_schema(parameters=_UUID_PK, responses=OpenApiTypes.OBJECT),
    partial_update=extend_schema(parameters=_UUID_PK, responses=OpenApiTypes.OBJECT),
    destroy=extend_schema(parameters=_UUID_PK),
    list=extend_schema(responses=OpenApiTypes.OBJECT),
    create=extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT),
)

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX", "INDIAVIX"}
PREVIEW_MAX_INTRADAY_DAYS = 31
PREVIEW_MAX_DAILY_DAYS = 400
PREVIEW_MAX_POINTS = 2000


def _bad(errors, detail="invalid input", code=status.HTTP_400_BAD_REQUEST):
    return Response({"detail": detail, "errors": errors}, status=code)


def _parse_date(raw, field):
    try:
        return date.fromisoformat(str(raw)), None
    except (TypeError, ValueError):
        return None, {"path": field, "message": "date must be YYYY-MM-DD"}


# ── serialisation ───────────────────────────────────────────────────────────

def backtest_summary(bt: Backtest | None) -> dict | None:
    if bt is None:
        return None
    pv = bt.data_provenance or {}
    m = bt.metrics or {}
    return {
        "id": str(bt.id), "status": bt.status, "created_at": bt.created_at.isoformat(),
        "version": bt.version.version if bt.version_id and bt.version else None,
        "from_date": bt.from_date.isoformat(), "to_date": bt.to_date.isoformat(),
        "progress": bt.progress, "progress_note": bt.progress_note, "error": bt.error[:500],
        "metrics": {k: m.get(k) for k in ("trades", "win_rate", "avg_r", "expectancy_r", "profit_factor",
                                          "net_pnl", "max_drawdown", "costs", "too_few_trades")} if m else {},
        "provenance": {k: pv.get(k) for k in ("label", "synthetic", "timeframe", "sessions", "data_source",
                                              "sample", "research_oos_overlap")} if pv else None,
    }


def instance_dict(inst: StrategyInstance, *, full: bool = False) -> dict:
    b = tpl.backtestability(inst.kind, inst.strategy_name)
    last = lifecycle.last_backtest(inst)
    last_done = lifecycle.last_backtest(inst, done_only=True)
    blocker = lifecycle.delete_blocker(inst)
    out = {
        "id": str(inst.id), "name": inst.name, "description": inst.description, "kind": inst.kind,
        "strategy_name": inst.strategy_name, "strategy_version": inst.strategy_version,
        "template_key": inst.template_key, "stage": inst.stage, "current_version": inst.current_version,
        "archived": inst.archived_at is not None,
        "archived_at": inst.archived_at.isoformat() if inst.archived_at else None,
        "created_at": inst.created_at.isoformat(), "updated_at": inst.updated_at.isoformat(),
        "backtest": b["backtest"], "backtest_reason": b["reason"],
        "last_backtest": backtest_summary(last),
        "last_result": backtest_summary(last_done),
        "next_step": lifecycle.next_step(inst),
        "can_delete": blocker is None, "delete_blocker": blocker,
        "is_enabled": inst.is_enabled,
    }
    if inst.kind == "spec":
        out["live_gate"] = spec_mod.live_gate(inst.params) if inst.params else None
    else:
        out["live_gate"] = tpl.plugin_live_gate(inst.strategy_name)
    if full:
        cur = lifecycle.current_version(inst)
        definition = (cur.spec if inst.kind == "spec" else cur.params) if cur else inst.params
        out["definition"] = definition
        out["version_hash"] = cur.spec_hash if cur else None
        out["explain"] = spec_mod.explain(definition) if inst.kind == "spec" and definition else []
        if inst.kind == "plugin":
            sch = tpl.plugin_schema(inst.strategy_name)
            meta = tpl.PLUGIN_META.get(inst.strategy_name, {})
            out["schema"] = sch["params"] if sch else None
            out["template_notes"] = meta.get("notes", [])
            out["broken"] = meta.get("broken", {})
        out["versions"] = inst.versions.count()
    return out


def version_dict(v: StrategyVersion) -> dict:
    return {"id": str(v.id), "version": v.version, "spec_hash": v.spec_hash, "note": v.note,
            "created_at": v.created_at.isoformat(),
            "created_by": getattr(v.created_by, "email", None),
            "definition": v.spec if v.instance.kind == "spec" else v.params,
            "backtests": v.backtests.count()}


def backtest_dict(bt: Backtest) -> dict:
    out = backtest_summary(bt)
    out.update({
        "instance_id": str(bt.instance_id), "instance_name": bt.instance.name,
        "config": bt.config, "metrics": bt.metrics, "split_metrics": bt.split_metrics,
        "provenance": bt.data_provenance, "equity_curve": bt.equity_curve,
        "started_at": bt.started_at.isoformat() if bt.started_at else None,
        "finished_at": bt.finished_at.isoformat() if bt.finished_at else None,
        "cancel_requested": bt.cancel_requested,
        "trade_count": bt.trades.count(),
    })
    return out


# ── strategies ──────────────────────────────────────────────────────────────

@_uuid_detail_schema
class StrategyInstanceViewSet(viewsets.ViewSet):
    """/strategies/instances/ — CRUD with versioning."""

    def _get(self, pk) -> StrategyInstance:
        return get_object_or_404(StrategyInstance, pk=pk, tenant=self.request.tenant)

    def list(self, request):
        qs = StrategyInstance.objects.filter(tenant=request.tenant)
        arch = request.query_params.get("archived")
        if arch in ("1", "true"):
            qs = qs.filter(archived_at__isnull=False)
        elif arch != "all":
            qs = qs.filter(archived_at__isnull=True)
        return Response([instance_dict(i) for i in qs.order_by("-updated_at")])

    def create(self, request):
        d = request.data if isinstance(request.data, dict) else {}
        try:
            inst = lifecycle.create_strategy(
                tenant=request.tenant, user=request.user, name=d.get("name", ""),
                description=d.get("description", "") or "", template_key=d.get("template_key", "") or "",
                spec=d.get("spec"), strategy_name=d.get("strategy_name", "") or "",
                params=d.get("params"))
        except lifecycle.DefinitionError as e:
            return _bad(e.errors, str(e))
        return Response(instance_dict(inst, full=True), status=status.HTTP_201_CREATED)

    def retrieve(self, request, pk=None):
        return Response(instance_dict(self._get(pk), full=True))

    def partial_update(self, request, pk=None):
        inst = self._get(pk)
        d = request.data if isinstance(request.data, dict) else {}
        unknown = set(d) - {"name", "description"}
        if unknown:
            return _bad([{"path": k, "message": "not editable here — POST a new version to change the "
                                                 "definition"} for k in sorted(unknown)])
        if "name" in d:
            name = str(d["name"] or "").strip()
            if not name:
                return _bad([{"path": "name", "message": "name is required"}])
            inst.name = name[:120]
        if "description" in d:
            inst.description = str(d["description"] or "")
        inst.save(update_fields=["name", "description", "updated_at"])
        return Response(instance_dict(inst, full=True))

    update = partial_update

    def destroy(self, request, pk=None):
        inst = self._get(pk)
        try:
            lifecycle.delete(inst, request.user)
        except lifecycle.DeleteBlocked as e:
            return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
        return Response(status=status.HTTP_204_NO_CONTENT)

    @extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT, parameters=_UUID_PK)
    @action(detail=True, methods=["get", "post"])
    def versions(self, request, pk=None):
        inst = self._get(pk)
        if request.method == "GET":
            return Response([version_dict(v) for v in inst.versions.select_related("instance", "created_by")])
        d = request.data if isinstance(request.data, dict) else {}
        definition = d.get("spec") if inst.kind == "spec" else d.get("params")
        if not isinstance(definition, dict):
            key = "spec" if inst.kind == "spec" else "params"
            return _bad([{"path": key, "message": f"send the full {key} object"}])
        before = inst.current_version
        try:
            ver = lifecycle.new_version(inst, user=request.user, definition=definition, note=d.get("note", ""))
        except lifecycle.DefinitionError as e:
            return _bad(e.errors, str(e))
        inst.refresh_from_db()
        return Response({"version": version_dict(ver), "created": inst.current_version != before,
                         "instance": instance_dict(inst, full=True)},
                        status=status.HTTP_201_CREATED if inst.current_version != before else status.HTTP_200_OK)

    @extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT, parameters=_UUID_PK)
    @action(detail=True, methods=["post"])
    def duplicate(self, request, pk=None):
        inst = self._get(pk)
        name = (request.data or {}).get("name", "") if isinstance(request.data, dict) else ""
        try:
            new = lifecycle.duplicate(inst, user=request.user, name=name)
        except lifecycle.DefinitionError as e:
            return _bad(e.errors, str(e))
        return Response(instance_dict(new, full=True), status=status.HTTP_201_CREATED)

    @extend_schema(request=None, responses=OpenApiTypes.OBJECT, parameters=_UUID_PK)
    @action(detail=True, methods=["post"])
    def archive(self, request, pk=None):
        return Response(instance_dict(lifecycle.set_archived(self._get(pk), True, request.user), full=True))

    @extend_schema(request=None, responses=OpenApiTypes.OBJECT, parameters=_UUID_PK)
    @action(detail=True, methods=["post"])
    def unarchive(self, request, pk=None):
        return Response(instance_dict(lifecycle.set_archived(self._get(pk), False, request.user), full=True))


# ── templates / spec tools / universe ───────────────────────────────────────

@extend_schema(responses=OpenApiTypes.OBJECT)
@api_view(["GET"])
def templates_view(request):
    return Response(tpl.list_templates())


@extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT)
@api_view(["POST"])
def spec_validate(request):
    d = request.data if isinstance(request.data, dict) else {}
    return Response(spec_mod.validate_spec(d.get("spec")).as_dict())


def _loader(end):
    from apps.strategies.services.candles import DbCandleLoader
    return DbCandleLoader(end=end)


@extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT)
@api_view(["POST"])
def spec_preview(request):
    """Signals of one spec on one symbol over a short range (synchronous, no
    trades/costs) — the builder's instant feedback."""
    from apps.strategies.services.adapter import spec_signals
    from apps.strategies.services.rules import CompiledSpec

    d = request.data if isinstance(request.data, dict) else {}
    res = spec_mod.validate_spec(d.get("spec"))
    if not res.ok:
        return _bad(res.errors, "invalid spec")
    sym = str(d.get("symbol") or "").strip().upper()
    if not sym:
        return _bad([{"path": "symbol", "message": "pick a symbol"}])
    start, e1 = _parse_date(d.get("from_date"), "from_date")
    end, e2 = _parse_date(d.get("to_date"), "to_date")
    if e1 or e2:
        return _bad([e for e in (e1, e2) if e])
    daily = res.spec["timeframe"] == "1d"
    span = PREVIEW_MAX_DAILY_DAYS if daily else PREVIEW_MAX_INTRADAY_DAYS
    if end < start or (end - start).days > span:
        return _bad([{"path": "to_date", "message": f"preview range must be 0–{span} days"}])
    from apps.strategies.services.backtest_service import _max_period
    warm = max(_max_period(res.spec["indicators"]), _max_period(res.spec["entry"]), 14) * 2 + 5
    loaded = _loader(end).load(sym, res.spec["timeframe"], start, end, warm)
    if loaded is None or not loaded.bars:
        return Response({"symbol": sym, "bars": [], "signals": [], "stats": None,
                         "message": f"no {res.spec['timeframe']} candles for {sym} in this range"})
    sigs = spec_signals(CompiledSpec(res.spec), loaded.frame)
    lo, hi = start.isoformat(), end.isoformat()
    f = loaded.frame
    keep = [i for i in range(len(f)) if lo <= f.session[i] <= hi]
    step = max(1, -(-len(keep) // PREVIEW_MAX_POINTS))
    bars = [{"t": f.timestamps[i], "o": f.open[i], "h": f.high[i], "l": f.low[i], "c": f.close[i]}
            for i in keep[::step]]
    out = []
    for i in keep:
        s = sigs.get(i)
        if s is None:
            continue
        out.append({"t": s.timestamp, "fill_t": f.timestamps[i + 1] if i + 1 < len(f) else None,
                    "side": s.side.value, "price": round(s.entry_price, 2), "stop": round(s.stoploss, 2),
                    "target": round(s.target, 2) if s.target else None})
    return Response({"symbol": sym, "timeframe": res.spec["timeframe"], "bars": bars, "signals": out,
                     "stats": loaded.stats, "downsampled": step > 1,
                     "source": "market_data.Candle (" + ("NSE bhavcopy 1d" if daily else "Angel One 1m") + ")"})


@extend_schema(responses=OpenApiTypes.OBJECT)
@api_view(["GET"])
def universe_view(request):
    """Symbols the backtester has candles for (cached 1 h)."""
    from django.core.cache import cache
    from django.db.models import Max, Min
    from django.utils import timezone

    from apps.market_data.models import Candle

    hit = cache.get("strategies.universe.v1")
    if hit is not None:
        return Response(hit)
    since = timezone.now() - timedelta(days=30)
    intraday = sorted(set(Candle.objects.filter(interval="1m", t__gte=since)
                          .values_list("symbol__tradingsymbol", flat=True).distinct()))
    span = {}
    for iv in ("1m", "1d"):
        agg = Candle.objects.filter(interval=iv).aggregate(first=Min("t"), last=Max("t"))
        span[iv] = {k: (v.date().isoformat() if v else None) for k, v in agg.items()}
    daily_count = Candle.objects.filter(interval="1d", t__gte=since).values("symbol_id").distinct().count()
    out = {
        "intraday": {"symbols": [s for s in intraday if s not in INDEX_SYMBOLS],
                     "indices": [s for s in intraday if s in INDEX_SYMBOLS],
                     "first": span["1m"]["first"], "last": span["1m"]["last"],
                     "source": "Angel One 1m candles"},
        "daily": {"symbol_count": daily_count, "first": span["1d"]["first"], "last": span["1d"]["last"],
                  "source": "NSE bhavcopy (unadjusted)"},
        "research_oos": {"from": "2026-09-01", "to": "2026-10-19"},
    }
    cache.set("strategies.universe.v1", out, 3600)
    return Response(out)


# ── backtests ───────────────────────────────────────────────────────────────

@_uuid_detail_schema
class BacktestViewSet(viewsets.ViewSet):
    """/strategies/backtests/ — create (202, runs on the backtests queue),
    list, detail, trades, cancel."""

    def _get(self, pk) -> Backtest:
        return get_object_or_404(Backtest.objects.select_related("instance", "version"),
                                 pk=pk, tenant=self.request.tenant)

    def list(self, request):
        qs = Backtest.objects.filter(tenant=request.tenant).select_related("instance", "version")
        inst = request.query_params.get("instance")
        if inst:
            qs = qs.filter(instance_id=inst)
        return Response([{**backtest_summary(b), "instance_id": str(b.instance_id),
                          "instance_name": b.instance.name} for b in qs.order_by("-created_at")[:200]])

    def retrieve(self, request, pk=None):
        return Response(backtest_dict(self._get(pk)))

    def create(self, request):
        from apps.strategies.services.backtest_service import MAX_SYMBOLS
        d = request.data if isinstance(request.data, dict) else {}
        inst = StrategyInstance.objects.filter(tenant=request.tenant, pk=d.get("instance")).first() \
            if d.get("instance") else None
        if inst is None:
            return _bad([{"path": "instance", "message": "unknown strategy"}], code=status.HTTP_404_NOT_FOUND)
        b = tpl.backtestability(inst.kind, inst.strategy_name)
        if b["backtest"] != "backtest":
            return _bad([{"path": "instance", "message": b["reason"]}], "not backtestable")
        start, e1 = _parse_date(d.get("from_date"), "from_date")
        end, e2 = _parse_date(d.get("to_date"), "to_date")
        errs = [e for e in (e1, e2) if e]
        if not errs and end < start:
            errs.append({"path": "to_date", "message": "to_date is before from_date"})
        if not errs and end > date.today():
            errs.append({"path": "to_date", "message": "to_date is in the future"})
        vnum = d.get("version") or inst.current_version
        ver = inst.versions.filter(version=vnum).first()
        if ver is None:
            errs.append({"path": "version", "message": f"no version {vnum}"})
        syms = d.get("symbols")
        if syms is not None:
            if not isinstance(syms, list) or not all(isinstance(s, str) for s in syms):
                errs.append({"path": "symbols", "message": "symbols must be a list of strings"})
            else:
                syms = sorted({s.strip().upper() for s in syms if s.strip()})
                if len(syms) > MAX_SYMBOLS:
                    errs.append({"path": "symbols", "message": f"at most {MAX_SYMBOLS} symbols per run"})
        if inst.kind == "plugin" and not syms and not (inst.params or {}).get("symbols"):
            errs.append({"path": "symbols", "message": "pick the stocks to backtest"})
        capital = d.get("capital", 500_000)
        try:
            capital = float(capital)
            if not 10_000 <= capital <= 100_000_000:
                raise ValueError
        except (TypeError, ValueError):
            errs.append({"path": "capital", "message": "capital must be 10,000–10,00,00,000"})
        slip = d.get("slippage_pct")
        if slip is not None:
            try:
                slip = float(slip)
                if not 0 <= slip <= 2:
                    raise ValueError
            except (TypeError, ValueError):
                errs.append({"path": "slippage_pct", "message": "slippage_pct must be 0–2"})
        if errs:
            return _bad(errs)
        if Backtest.objects.filter(tenant=request.tenant,
                                   status__in=[Backtest.Status.QUEUED, Backtest.Status.RUNNING]).count() >= 2:
            return Response({"detail": "two backtests are already queued or running — wait for one to finish"},
                            status=status.HTTP_429_TOO_MANY_REQUESTS)
        bt = Backtest.objects.create(
            tenant=request.tenant, instance=inst, version=ver, from_date=start, to_date=end,
            config={"symbols": syms or None, "capital": capital, "slippage_pct": slip,
                    "oos_fraction": 0.3, "requested_by": getattr(request.user, "email", None)})
        from apps.strategies.tasks.backtest import run_backtest
        try:
            run_backtest.delay(str(bt.id))
        except Exception as e:  # noqa: BLE001 — broker down: say so instead of a silent queue
            bt.status = Backtest.Status.FAILED
            bt.error = f"could not queue the backtest task: {e}"
            bt.save()
        bt.refresh_from_db()
        return Response(backtest_dict(bt), status=status.HTTP_202_ACCEPTED)

    @extend_schema(responses=OpenApiTypes.OBJECT, parameters=_UUID_PK)
    @action(detail=True, methods=["get"])
    def trades(self, request, pk=None):
        bt = self._get(pk)
        qs = BacktestTrade.objects.filter(backtest=bt, tenant=request.tenant).order_by("seq")
        seg = request.query_params.get("segment")
        if seg in ("is", "oos"):
            qs = qs.filter(segment=seg)
        rows = list(qs.values("seq", "symbol", "side", "segment", "entry_time", "exit_time", "entry_price",
                              "exit_price", "stop", "target", "qty", "gross_pnl", "costs", "net_pnl",
                              "r_multiple", "exit_reason", "bars_held", "meta")[:5000])
        return Response(rows)

    @extend_schema(request=None, responses=OpenApiTypes.OBJECT, parameters=_UUID_PK)
    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        bt = self._get(pk)
        if bt.status not in (Backtest.Status.QUEUED, Backtest.Status.RUNNING):
            return Response({"detail": f"backtest is already {bt.status}"}, status=status.HTTP_409_CONFLICT)
        if bt.status == Backtest.Status.QUEUED and bt.started_at is None:
            # Not picked up by a worker yet: cancel now (the task, if it ever
            # runs, sees a terminal status and exits). Saved so the receipt follows.
            from django.utils import timezone
            bt.cancel_requested = True
            bt.status = Backtest.Status.CANCELLED
            bt.error = "cancelled before it started"
            bt.finished_at = timezone.now()
            bt.save()
        else:
            Backtest.objects.filter(pk=bt.pk).update(cancel_requested=True)
        bt.refresh_from_db()
        return Response(backtest_dict(bt), status=status.HTTP_202_ACCEPTED)
