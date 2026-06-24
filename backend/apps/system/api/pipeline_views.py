"""REST surface for the daily-pipeline debugger / flow-observability page.

  GET  /api/v1/system/pipeline/            — flow graph + live feed + runs
       ?date=YYYY-MM-DD                    — view a specific trading day
  POST /api/v1/system/pipeline/<task>/run/ — manually trigger a task

The daily pipeline (swing scan → live screener → EOD enrichment) runs
unattended via Celery beat. This endpoint exposes the *flow* of data
through it for any day: per-stage live metrics, per-strategy breakdown, a
chronological activity feed, and run history. Owner-gated, like Ops.
"""
from __future__ import annotations

from datetime import date as date_cls, datetime

from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.system.api.views import OwnerOnly
from apps.system.models import PipelineRun

# Static catalog — the runnable pipeline stages, in daily order.
PIPELINE_TASKS = [
    {
        "key": "swing_scan",
        "label": "Swing Scan",
        "schedule": "Mon–Fri · 08:20 IST",
        "description": "Oliver Kell daily/weekly scan across NIFTY 100.",
        "category": "data",
    },
    {
        "key": "premarket_basket",
        "label": "Premarket Basket",
        "schedule": "Mon–Fri · 08:45 IST",
        "description": "Market-mood basket — equity + index-option legs (scan only).",
        "category": "data",
    },
    {
        "key": "screener_session",
        "label": "Screener Session",
        "schedule": "Mon–Fri · 09:15 → 15:30 IST",
        "description": "Live intraday screener — fires signals while the market is open.",
        "category": "data",
    },
    {
        "key": "eod_enrichment",
        "label": "EOD Enrichment",
        "schedule": "Mon–Fri · 16:00 IST",
        "description": "Fills EOD price + outcome on the day's signals.",
        "category": "data",
    },
    {
        "key": "intraday_agent",
        "label": "Derive Trades",
        "schedule": "Mon–Fri · 16:30 IST",
        "description": "Replays the closed session into paper trades — feeds the "
                       "Monthly P&L / equity curve. Opt-in (auto-execute).",
        "category": "execution",
    },
]
_TASK_KEYS = {t["key"] for t in PIPELINE_TASKS}

# Signal.source → pipeline stage key used by the flow graph + feed tags.
_SOURCE_STAGE = {
    "SCREENER": "screener_session",
    "OK_SCANNER": "swing_scan",
    "PREMARKET": "premarket_basket",
    "TRADINGVIEW": "tradingview",
}

# Roster metadata for every registered strategy plugin. Keyed by the
# `alphadesk.strategies` entry-point name. Lets the /pipeline page show
# the full strategy estate — not just the ones wired into the daily flow.
STRATEGY_CATALOG = {
    "swing_scanner": {
        "label": "Oliver Kell Swing Scanner", "category": "Scanner",
        "automation": "daily", "pipeline_task": "swing_scan",
        "signal_source": "OK_SCANNER",
        "description": "Daily/weekly cycle scan across NIFTY 100.",
    },
    "intraday_screener": {
        "label": "Intraday Screener", "category": "Scanner",
        "automation": "daily", "pipeline_task": "screener_session",
        "signal_source": "SCREENER",
        "description": "Live tick-stream screener — 8 intraday strategies.",
    },
    "premarket_basket": {
        "label": "Premarket Basket", "category": "Scanner",
        "automation": "daily", "pipeline_task": "premarket_basket",
        "signal_source": "PREMARKET",
        "description": "Market-mood basket — equity + index-option legs.",
    },
    "directional": {
        "label": "Directional Trader", "category": "Execution",
        "automation": "on-demand", "pipeline_task": None, "signal_source": None,
        "description": "Per-symbol equity LangGraph planner — needs a symbol.",
    },
    "short_straddle": {
        "label": "Short Straddle Manager", "category": "Options",
        "automation": "on-demand", "pipeline_task": None, "signal_source": None,
        "description": "Lifecycle manager for an open straddle position.",
    },
    "pyramid": {
        "label": "Pyramid Options", "category": "Options",
        "automation": "on-demand", "pipeline_task": None, "signal_source": None,
        "description": "Momentum pyramiding on a chosen option strike.",
    },
    "vertical_spread": {
        "label": "Vertical Spread", "category": "Options",
        "automation": "on-demand", "pipeline_task": None, "signal_source": None,
        "description": "Directional option spreads — needs an underlying + bias.",
    },
    "backtest": {
        "label": "Backtester", "category": "Tooling",
        "automation": "on-demand", "pipeline_task": None, "signal_source": None,
        "description": "Generic historical replay across strategy engines.",
    },
}


def _strategy_roster(by_source: dict[str, int]) -> list[dict]:
    """Every registered strategy plugin + its automation / activity state."""
    try:
        from apps.agents_core.registry import strategy_registry
        registered = sorted(strategy_registry.names())
    except Exception:
        registered = sorted(STRATEGY_CATALOG)
    roster = []
    for name in registered:
        meta = STRATEGY_CATALOG.get(name, {
            "label": name.replace("_", " ").title(), "category": "Other",
            "automation": "on-demand", "pipeline_task": None,
            "signal_source": None, "description": "",
        })
        src = meta.get("signal_source")
        roster.append({
            "name": name,
            **meta,
            "wired": meta.get("pipeline_task") is not None,
            "signals": by_source.get(src, 0) if src else 0,
        })
    # Daily-automated scanners first, then by category + label.
    roster.sort(key=lambda s: (s["automation"] != "daily", s["category"], s["label"]))
    return roster


# ─── Serialisers ──────────────────────────────────────────────────────

def _run_dict(run: PipelineRun | None) -> dict | None:
    if run is None:
        return None
    duration = None
    if run.finished_at:
        duration = round((run.finished_at - run.started_at).total_seconds(), 1)
    return {
        "id": run.id,
        "task": run.task,
        "status": run.status,
        "trigger": run.trigger,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_seconds": duration,
        "summary": run.summary or {},
        "error": run.error or "",
    }


def _summarise(summary: dict) -> str:
    parts = [
        f"{k.replace('_', ' ')} {v}"
        for k, v in (summary or {}).items()
        if k not in ("ok", "skipped")
    ]
    return " · ".join(parts)


def _screener_is_live() -> bool:
    """True when the screener-session Redis lock is held — i.e. a session
    task is currently alive."""
    try:
        import redis
        from django.conf import settings

        from apps.strategies.tasks.daily_pipeline import _SCREENER_LOCK_KEY

        r = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=2)
        return bool(r.exists(_SCREENER_LOCK_KEY))
    except Exception:
        return False


# ─── Status / flow endpoint ───────────────────────────────────────────

class PipelineStatusView(APIView):
    """GET /api/v1/system/pipeline/ — flow graph + feed + per-strategy.

    Without ``?date=`` it shows the most recent day that actually has
    signal data (so the page is never blank just because today's
    screener hasn't run yet).
    """

    permission_classes = [OwnerOnly]

    @extend_schema(
        parameters=[
            OpenApiParameter(
                "date", OpenApiTypes.DATE, OpenApiParameter.QUERY,
                description="Trading day to view (YYYY-MM-DD). Defaults to the most "
                            "recent day that has signal data.",
                required=False,
            ),
        ],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request):
        from apps.market_data.services.market_calendar import (
            is_trading_day, next_trading_day,
        )
        from apps.strategies.models import Signal
        from trading.utils.time_utils import get_session_phase

        today = timezone.localdate()
        now = datetime.now()
        screener_live = _screener_is_live()

        # ── Resolve which day we're viewing ───────────────────────────
        date_options = list(
            Signal.objects.values_list("signal_date", flat=True)
            .distinct().order_by("-signal_date")[:30]
        )
        raw = request.query_params.get("date")
        sel_date: date_cls
        if raw:
            try:
                sel_date = datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                sel_date = today
        elif date_options:
            # Default to the latest day with data — or today if today has data.
            sel_date = today if today in date_options else date_options[0]
        else:
            sel_date = today
        viewing_today = sel_date == today

        # ── Signals for the selected day ──────────────────────────────
        sig_rows = list(
            Signal.objects.filter(signal_date=sel_date)
            .order_by("-signal_time")
            .values(
                "id", "symbol", "side", "source", "strategy",
                "entry_price", "signal_time", "outcome", "confidence",
                "eod_price", "max_favorable_move",
            )
        )
        total = len(sig_rows)
        enriched = sum(1 for s in sig_rows if s["eod_price"] is not None)
        traded = sum(1 for s in sig_rows if s["outcome"] == "TRADED")
        by_source: dict[str, int] = {}
        for s in sig_rows:
            by_source[s["source"]] = by_source.get(s["source"], 0) + 1

        # ── Per-strategy breakdown ────────────────────────────────────
        strat_acc: dict[tuple[str, str], dict] = {}
        for s in sig_rows:
            key = (s["strategy"] or "—", s["source"])
            acc = strat_acc.setdefault(key, {
                "strategy": s["strategy"] or "—",
                "source": s["source"],
                "count": 0, "traded": 0, "buy": 0, "sell": 0,
                "conf_sum": 0.0,
            })
            acc["count"] += 1
            acc["conf_sum"] += s["confidence"] or 0.0
            if s["outcome"] == "TRADED":
                acc["traded"] += 1
            if s["side"] == "BUY":
                acc["buy"] += 1
            else:
                acc["sell"] += 1
        by_strategy = []
        for acc in strat_acc.values():
            n = acc.pop("conf_sum")
            acc["avg_confidence"] = round(n / acc["count"], 2) if acc["count"] else 0.0
            by_strategy.append(acc)
        by_strategy.sort(key=lambda x: x["count"], reverse=True)

        # ── Run state for the selected day + last run overall ─────────
        day_run: dict[str, dict | None] = {}
        for spec in PIPELINE_TASKS:
            day_run[spec["key"]] = _run_dict(
                PipelineRun.objects
                .filter(task=spec["key"], started_at__date=sel_date)
                .first()
            )

        def node_status(task_key: str, metric: int, live: bool = False) -> str:
            if live:
                return "running"
            run = day_run.get(task_key)
            if run:
                return run["status"]
            return "success" if metric > 0 else "idle"

        swing_n = by_source.get("OK_SCANNER", 0)
        screener_n = by_source.get("SCREENER", 0)
        basket_n = by_source.get("PREMARKET", 0)

        flow = [
            {
                "key": "swing_scan", "label": "Swing Scanner", "role": "source",
                "status": node_status("swing_scan", swing_n),
                "metric": swing_n, "metric_label": "swing signals",
            },
            {
                "key": "premarket_basket", "label": "Premarket Basket", "role": "source",
                "status": node_status("premarket_basket", basket_n),
                "metric": basket_n, "metric_label": "basket signals",
            },
            {
                "key": "screener_session", "label": "Live Screener", "role": "source",
                "status": node_status(
                    "screener_session", screener_n, screener_live and viewing_today,
                ),
                "live": screener_live and viewing_today,
                "metric": screener_n, "metric_label": "intraday signals",
            },
            {
                "key": "ledger", "label": "Signal Ledger", "role": "hub",
                "status": "success" if total else "idle",
                "metric": total, "metric_label": "signals total",
            },
            {
                "key": "eod_enrichment", "label": "EOD Enrichment", "role": "stage",
                "status": node_status("eod_enrichment", enriched),
                "metric": enriched, "metric_label": f"of {total} enriched",
            },
            {
                "key": "feedback", "label": "Feedback / Monthly", "role": "sink",
                "status": "success" if traded else "idle",
                "metric": traded, "metric_label": "traded → capture loop",
            },
        ]

        # ── Activity feed — signals + run lifecycle, newest first ─────
        feed: list[dict] = []
        for s in sig_rows[:80]:
            feed.append({
                "ts": s["signal_time"].isoformat(),
                "kind": "signal",
                "stage": _SOURCE_STAGE.get(s["source"], "other"),
                "source": s["source"],
                "side": s["side"],
                "title": f'{s["side"]} {s["symbol"]}',
                "detail": (
                    f'{s["strategy"]} · entry {s["entry_price"]:.2f}'
                    + (f' · conf {s["confidence"]:.0%}' if s["confidence"] else "")
                ),
                "outcome": s["outcome"],
            })
        for run in PipelineRun.objects.filter(started_at__date=sel_date)[:25]:
            feed.append({
                "ts": (run.finished_at or run.started_at).isoformat(),
                "kind": "run",
                "stage": run.task,
                "status": run.status,
                "title": f"{run.get_task_display()} {run.status}",
                "detail": _summarise(run.summary) or (run.error[:140] if run.error else ""),
                "trigger": run.trigger,
            })
        feed.sort(key=lambda x: x["ts"], reverse=True)
        feed = feed[:100]

        # Enrichment backlog — un-enriched signals across *all* dates, so
        # the UI can offer a one-click backfill.
        unenriched_total = Signal.objects.filter(eod_price__isnull=True).count()

        from apps.system.services.flags import is_auto_execute_enabled
        _tid = getattr(getattr(request, "tenant", None), "id", None)

        return Response({
            "tasks": PIPELINE_TASKS,
            "auto_execute_enabled": is_auto_execute_enabled(_tid),
            "today": {
                "date": today.isoformat(),
                "is_trading_day": is_trading_day(today),
                "session_phase": get_session_phase(now),
                "next_trading_day": next_trading_day(today).isoformat(),
            },
            "selected_date": sel_date.isoformat(),
            "viewing_today": viewing_today,
            "date_options": [d.isoformat() for d in date_options],
            "flow": flow,
            "signals": {
                "total": total, "enriched": enriched, "traded": traded,
                "by_source": by_source,
            },
            "by_strategy": by_strategy,
            "strategies": _strategy_roster(by_source),
            "feed": feed,
            "unenriched_total": unenriched_total,
            "recent_runs": [_run_dict(r) for r in PipelineRun.objects.all()[:40]],
        })


# ─── Manual trigger endpoint ──────────────────────────────────────────

class PipelineTriggerView(APIView):
    """POST /api/v1/system/pipeline/<task>/run/ — manual run.

    Dispatches the Celery task with ``force=True`` so it bypasses the
    trading-day guard (the debugger needs to run things off-hours).
    """

    permission_classes = [OwnerOnly]

    @extend_schema(
        parameters=[
            OpenApiParameter(
                "task", OpenApiTypes.STR, OpenApiParameter.PATH,
                description="Pipeline task key (swing_scan, premarket_basket, "
                            "screener_session, eod_enrichment, intraday_agent).",
            ),
            OpenApiParameter(
                "backfill", OpenApiTypes.BOOL, OpenApiParameter.QUERY, required=False,
                description="eod_enrichment only — re-enrich all un-enriched signals "
                            "across every date (also accepted in the request body).",
            ),
            OpenApiParameter(
                "date", OpenApiTypes.DATE, OpenApiParameter.QUERY, required=False,
                description="eod_enrichment / intraday_agent — target a specific past "
                            "session (also accepted in the request body).",
            ),
        ],
        request=OpenApiTypes.OBJECT,
        responses=OpenApiTypes.OBJECT,
    )
    def post(self, request, task: str):
        if task not in _TASK_KEYS:
            return Response(
                {"detail": f"Unknown pipeline task: {task}"}, status=404,
            )

        from apps.strategies.tasks.daily_pipeline import (
            derive_intraday_trades, run_eod_enrichment, run_premarket_basket,
            run_screener_session, run_swing_scan,
        )

        callables = {
            "swing_scan": run_swing_scan,
            "premarket_basket": run_premarket_basket,
            "screener_session": run_screener_session,
            "eod_enrichment": run_eod_enrichment,
            "intraday_agent": derive_intraday_trades,
        }
        # Guard against a second screener session — the task itself holds a
        # Redis lock, but failing fast here gives the operator a clear 409.
        if task == "screener_session" and _screener_is_live():
            return Response(
                {"detail": "A screener session is already running."},
                status=409,
            )

        kwargs: dict = {"force": True}
        mode = "default"
        # EOD enrichment can target a specific day or backfill everything.
        # Both the request body and the query string are accepted so the
        # UI can POST either way.
        if task == "eod_enrichment":
            backfill = _truthy(
                request.data.get("backfill")
                or request.query_params.get("backfill")
            )
            date = (
                request.data.get("date")
                or request.query_params.get("date")
            )
            if backfill:
                kwargs["backfill"] = True
                mode = "backfill-all"
            elif date:
                kwargs["date"] = str(date)
                mode = str(date)

        # Derive-trades can target a specific past session (off-hours backfill);
        # default is today's just-closed session.
        if task == "intraday_agent":
            date = request.data.get("date") or request.query_params.get("date")
            if date:
                kwargs["date"] = str(date)
                mode = str(date)

        async_result = callables[task].delay(**kwargs)
        return Response({
            "task": task,
            "mode": mode,
            "dispatched": True,
            "celery_task_id": async_result.id,
        }, status=202)


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


class PipelineConfigView(APIView):
    """POST /api/v1/system/pipeline/config/ — toggle pipeline opt-ins.

    Currently just ``auto_execute_enabled`` (auto-derive trades daily). Owner
    only, since flipping it on lets the EOD beat task create paper trades.
    """

    permission_classes = [OwnerOnly]

    @extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT)
    def post(self, request):
        from apps.system.services.flags import AUTO_EXECUTE_KEY, set_flag

        _tid = getattr(getattr(request, "tenant", None), "id", None)
        if _tid is None:
            return Response({"detail": "No tenant in context."}, status=400)

        enabled = _truthy(
            request.data.get("auto_execute_enabled")
            if "auto_execute_enabled" in request.data
            else request.data.get("enabled")
        )
        set_flag(AUTO_EXECUTE_KEY, enabled, _tid)
        return Response({"auto_execute_enabled": enabled})
