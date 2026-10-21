"""Runs API — run receipts, today's schedule, and the top-bar status strip.

  GET  /api/v1/runs/receipts/                ?kind= &status=a,b &name= &date=YYYY-MM-DD
                                             &celery_task_id= &limit= (≤200)
  GET  /api/v1/runs/receipts/<uuid>/         one receipt + previous run of the same job
  GET  /api/v1/runs/schedule/                ?date=  — expected-schedule slots + states
  GET  /api/v1/runs/status/                  mode · data status · today's job health
  POST /api/v1/runs/rerun/<name>/            owner only — re-trigger a rerunnable job

Visibility: a member sees their own tenant's receipts plus system receipts
(``tenant IS NULL`` — process-wide jobs that hold only aggregate counts).
Another tenant's receipts are never listed and 404 on detail.
"""
from __future__ import annotations

import importlib
from datetime import date as date_cls
from datetime import datetime, time, timedelta

from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import OwnerOnly
from apps.system.models import RunReceipt
from apps.system.services import run_checks as rc
from apps.system.services.run_schedule import (
    IST,
    RERUNNABLE,
    registry,
    schedule_for_day,
    summarize,
)


def visible_receipts(request):
    tenant = getattr(request, "tenant", None)
    return RunReceipt.objects.filter(Q(tenant=tenant) | Q(tenant__isnull=True))


def receipt_dict(r: RunReceipt, *, full: bool = False) -> dict:
    checks = r.checks or []
    out = {
        "id": str(r.id),
        "kind": r.kind,
        "name": r.name,
        "scope": "tenant" if r.tenant_id else "system",
        "task_name": r.task_name,
        "schedule_key": r.schedule_key,
        "expected_at": r.expected_at.isoformat() if r.expected_at else None,
        "trigger": r.trigger,
        "status": r.status,
        "started_at": r.started_at.isoformat(),
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "duration_seconds": r.duration_seconds,
        "outputs": r.outputs or {},
        "warnings": r.warnings or [],
        "error": r.error or "",
        "check_counts": {s: sum(1 for c in checks if c.get("status") == s)
                         for s in ("pass", "warn", "fail", "info")},
        "detail_kind": r.detail_kind,
        "detail_id": r.detail_id,
        "rerunnable": r.name in RERUNNABLE,
        "celery_task_id": r.celery_task_id,
    }
    if full:
        out.update({"inputs": r.inputs or {}, "checks": checks,
                    "data_provenance": r.data_provenance or {},
                    "max_seconds": r.max_seconds,
                    "alerted_at": r.alerted_at.isoformat() if r.alerted_at else None})
    return out


def _parse_day(raw: str | None) -> date_cls | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


class ReceiptListView(APIView):
    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        qs = visible_receipts(request)
        p = request.query_params
        if p.get("kind"):
            qs = qs.filter(kind=p["kind"])
        if p.get("name"):
            qs = qs.filter(name=p["name"])
        if p.get("status"):
            qs = qs.filter(status__in=[s for s in p["status"].split(",") if s])
        if p.get("celery_task_id"):
            qs = qs.filter(celery_task_id=p["celery_task_id"])
        day = _parse_day(p.get("date"))
        if day:
            start = datetime.combine(day, time(0), tzinfo=IST)
            qs = qs.filter(started_at__gte=start, started_at__lt=start + timedelta(days=1))
        try:
            limit = max(1, min(int(p.get("limit", 50)), 200))
        except ValueError:
            limit = 50
        rows = list(qs.order_by("-started_at")[:limit])
        names = sorted(set(visible_receipts(request).values_list("name", flat=True)))
        return Response({"results": [receipt_dict(r) for r in rows], "count": len(rows),
                         "names": names})


class ReceiptDetailView(APIView):
    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request, receipt_id):
        r = visible_receipts(request).filter(id=receipt_id).first()
        if r is None:
            return Response({"detail": "Not found."}, status=404)
        prev = (visible_receipts(request)
                .filter(name=r.name, started_at__lt=r.started_at)
                .exclude(status__in=[RunReceipt.Status.NOT_RUN, RunReceipt.Status.RUNNING])
                .order_by("-started_at").first())
        return Response({**receipt_dict(r, full=True),
                         "previous": receipt_dict(prev) if prev else None,
                         "detail": _detail(r),
                         "rerun_note": RERUNNABLE.get(r.name, {}).get("note")})


def _detail(r: RunReceipt) -> dict | None:
    """Short summary of the linked detail row (PipelineRun / NightlyScan)."""
    try:
        if r.detail_kind == "pipeline_run" and r.detail_id:
            from apps.system.models import PipelineRun

            pr = PipelineRun.objects.filter(id=int(r.detail_id)).first()
            return pr and {"kind": "pipeline_run", "id": pr.id, "task": pr.task,
                           "status": pr.status, "summary": pr.summary, "href": "/pipeline"}
        if r.detail_kind == "nightly_scan" and r.detail_id:
            from apps.news.models import NightlyScan

            s = NightlyScan.objects.filter(id=int(r.detail_id)).first()
            return s and {"kind": "nightly_scan", "id": s.id, "for_date": s.for_date.isoformat(),
                          "status": s.status, "href": "/nightly-scan"}
        if r.detail_kind == "agent_run" and r.detail_id:
            from apps.agents_core.models import AgentRun

            a = AgentRun.objects.filter(id=r.detail_id, tenant_id=r.tenant_id).first()
            return a and {"kind": "agent_run", "id": str(a.id), "strategy": a.strategy_name,
                          "status": a.status, "error": (a.error or "")[:500], "href": f"/agents/{a.id}"}
        if r.detail_kind == "backtest" and r.detail_id:
            from apps.strategies.models import Backtest

            b = Backtest.objects.filter(id=r.detail_id, tenant_id=r.tenant_id).first()
            return b and {"kind": "backtest", "id": str(b.id), "status": b.status,
                          "from": b.from_date.isoformat(), "to": b.to_date.isoformat(),
                          "metrics": b.metrics, "href": "/strategies"}
    except Exception:  # noqa: BLE001
        return None
    return None


class ScheduleView(APIView):
    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        day = _parse_day(request.query_params.get("date")) or timezone.now().astimezone(IST).date()
        slots = schedule_for_day(day)
        return Response({
            "date": day.isoformat(),
            "slots": slots,
            "summary": summarize(slots),
            "jobs": [{"key": e.key, "label": e.label, "task": e.task, "schedule": e.describe(),
                      "tracked": e.tracked, "receipt_name": e.receipt_name,
                      "grace_minutes": int(e.grace.total_seconds() // 60),
                      "rerunnable": e.receipt_name in RERUNNABLE} for e in registry()],
        })


def trading_mode(tenant) -> dict:
    from apps.system.services.flags import is_kill_switch_on, is_live_trading_enabled

    tid = getattr(tenant, "id", None)
    env_mode = getattr(settings, "TRADING_MODE", "paper")
    kill = bool(is_kill_switch_on(tid))
    live_flag = bool(is_live_trading_enabled(tid))
    if env_mode == "halt" or kill:
        mode, reason = "HALT", "kill switch on" if kill else "TRADING_MODE=halt"
    elif env_mode == "live" and live_flag:
        mode, reason = "LIVE", "TRADING_MODE=live and live trading enabled for this tenant"
    else:
        mode = "PAPER"
        reason = ("TRADING_MODE=live but live trading is not enabled for this tenant"
                  if env_mode == "live" else f"TRADING_MODE={env_mode}")
    return {"mode": mode, "reason": reason, "env_mode": env_mode}


def data_status(tenant, now: datetime | None = None) -> dict:
    """live / stale / synthetic for the top bar."""
    from apps.market_data.models import BrokerLink
    from apps.market_data.services.market_calendar import is_trading_day

    now = now or timezone.now()
    links = list(BrokerLink.objects.filter(tenant=tenant)
                 .values("broker_name", "status", "last_refreshed_at", "last_error"))
    real_active = [lk for lk in links if lk["status"] == "active" and lk["broker_name"] not in rc.SYNTHETIC_BROKERS]
    brokers = [{"broker": lk["broker_name"], "status": lk["status"],
                "last_refreshed_at": lk["last_refreshed_at"].isoformat() if lk["last_refreshed_at"] else None,
                "last_error": (lk["last_error"] or "")[:160]} for lk in links]
    c1m = rc.candle_freshness("1m", now=now)
    c1d = rc.candle_freshness("1d", now=now)
    ist = now.astimezone(IST)
    market_open = is_trading_day(ist.date()) and time(9, 15) <= ist.time() < time(15, 30)

    if not real_active:
        state = "synthetic"
        reason = ("No active broker link — prices and option chains fall back to synthesised data."
                  if not links else "No broker link is active (" + ", ".join(
                      f"{b['broker']}: {b['status']}" for b in brokers) + ") — data may be synthetic.")
    else:
        state, reason = "live", f"{', '.join(sorted({lk['broker_name'] for lk in real_active}))} active"
        latest_1m = rc.latest_candle("1m")
        if market_open and (latest_1m is None or (now - latest_1m) > timedelta(minutes=15)):
            state = "stale"
            reason = "Market is open but the newest 1-minute candle is " + (
                f"{int((now - latest_1m).total_seconds() // 60)} min old" if latest_1m else "missing")
        elif not market_open and c1m.get("status") != rc.PASS:
            state, reason = "stale", c1m["message"]
        elif c1d.get("status") == rc.FAIL:
            state, reason = "stale", c1d["message"]
    return {"state": state, "reason": reason, "market_open": market_open, "brokers": brokers,
            "candles": {"1m": c1m, "1d": c1d}}


class StatusView(APIView):
    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        tenant = getattr(request, "tenant", None)
        slots = schedule_for_day()
        bad = [s for s in slots if s["state"] in ("failed", "missed")]
        return Response({
            "as_of": timezone.now().isoformat(),
            "mode": trading_mode(tenant),
            "data": data_status(tenant),
            "jobs": {**summarize(slots), "problems": [
                {"key": s["key"], "label": s["label"], "state": s["state"], "due_at": s["due_at"],
                 "receipt_id": (s.get("receipt") or {}).get("id")} for s in bad]},
        })


class RerunView(APIView):
    permission_classes = [OwnerOnly]

    @extend_schema(request=None, responses=OpenApiTypes.OBJECT)
    def post(self, request, name: str):
        spec = RERUNNABLE.get(name)
        if spec is None:
            return Response({"detail": f"'{name}' has no safe manual trigger."}, status=404)
        module_path, func = spec["task"].rsplit(".", 1)
        task = getattr(importlib.import_module(module_path), func)
        result = task.delay(**spec["kwargs"])
        return Response({"name": name, "dispatched": True, "celery_task_id": result.id,
                         "note": spec.get("note")}, status=202)
