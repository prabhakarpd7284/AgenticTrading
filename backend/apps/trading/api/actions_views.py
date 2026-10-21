"""Action Board API (UX redesign R3, plan §3–§4). Tenant-scoped throughout.

  GET  /api/v1/actions/today/?date=YYYY-MM-DD   board status + cards + track record
  POST /api/v1/actions/<id>/approve/            {note?}   → PlaceOrder (paper) → executed | rejected
  POST /api/v1/actions/<id>/skip/               {reason?}
  POST /api/v1/actions/<id>/snooze/             {until: ISO datetime} | {minutes: int}
  GET  /api/v1/actions/track-record/?strategy=directional

Reads need tenant membership; decisions need role owner / admin / trader.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from django.db.models import Q
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import TenantScoped
from apps.trading.models import TodayAction
from apps.trading.services import action_decisions as decisions
from apps.trading.services.action_board import (
    IST,
    STAGE_LABEL,
    STRATEGY,
    at,
    board_setting,
    expires_at_for,
    hhmm,
)
from apps.trading.services.action_outcomes import track_record


def _f(v):
    return float(v) if v is not None else None


def card_dict(c: TodayAction) -> dict:
    rc = c.risk_check or {}
    order = None
    if c.order_id:
        o = c.order
        order = {"id": str(o.id), "status": o.status, "qty": o.qty, "price": _f(o.price),
                 "error": o.error or ""}
    return {
        "id": str(c.id), "date": c.date.isoformat(), "kind": c.kind, "symbol": c.symbol,
        "strategy": c.strategy, "strategy_version": c.strategy_version, "stage": STAGE_LABEL,
        "horizon": c.horizon, "side": c.side,
        "entry": _f(c.entry_price), "entry_zone": [_f(c.entry_low), _f(c.entry_high)],
        "stoploss": _f(c.stoploss), "target": _f(c.target),
        "suggested_qty": c.suggested_qty, "capital_required": _f(c.capital_required),
        "risk_at_stake": _f(c.risk_at_stake), "risk_multiplier": _f(c.risk_multiplier),
        "risk_check": {"approved": bool(rc.get("approved")), "reason": rc.get("reason", ""),
                       "failed_criterion": rc.get("failed_criterion", ""), "criteria": rc.get("criteria", {}),
                       "gates": rc.get("gates", {}), "at_approve": rc.get("at_approve")},
        "risk_passed": c.risk_passed, "confidence": _f(c.confidence), "reasons": c.reasons or [],
        "source_refs": c.source_refs or {}, "shadow": c.shadow, "blocked_reason": c.blocked_reason,
        "planned_as_of": c.planned_as_of.isoformat(), "expires_at": c.expires_at.isoformat(),
        "status": c.status, "snoozed_until": c.snoozed_until.isoformat() if c.snoozed_until else None,
        "decided_at": c.decided_at.isoformat() if c.decided_at else None,
        "decision_note": c.decision_note, "order": order, "outcome": c.outcome or None,
    }


def _board_receipt(day: date):
    from apps.system.models import RunReceipt

    iso = day.isoformat()
    lo = at(day, hhmm("00:00"))
    return (RunReceipt.objects.filter(name="action_board")
            .filter(Q(outputs__date=iso) | Q(inputs__date=iso) |
                    Q(started_at__gte=lo, started_at__lt=lo + timedelta(days=1), inputs__date=None))
            .order_by("-started_at").first())


def board_status(day: date, now: datetime) -> dict:
    from apps.market_data.services.market_calendar import is_trading_day

    r = _board_receipt(day)
    sched = at(day, hhmm(board_setting("TIME")))
    receipt = None
    if r is not None:
        o = r.outputs or {}
        receipt = {"id": str(r.id), "status": r.status, "trigger": r.trigger,
                   "started_at": r.started_at.isoformat(),
                   "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                   "error": (r.error or "")[:400], "skipped": o.get("skipped"),
                   "as_of": o.get("as_of"), "replay": o.get("replay"),
                   "candidates": o.get("candidates"), "fresh": o.get("fresh"),
                   "planner_calls": o.get("planner_calls"), "planned": o.get("planned"),
                   "passed_risk": o.get("passed_risk"), "cards": o.get("cards"),
                   "dropped": o.get("dropped") or {}, "telegram_sent": o.get("telegram_sent"),
                   "warnings": (r.warnings or [])[:5]}
    return {"built": r is not None and r.status in ("success", "partial"),
            "is_trading_day": is_trading_day(day), "scheduled_for": sched.isoformat(),
            "schedule_label": f"{sched:%H:%M} IST", "expires_at": expires_at_for(day).isoformat(),
            "due": now >= sched, "receipt": receipt,
            "planner_cap": int(board_setting("MAX_PLANNER_CALLS"))}


def _parse_day(raw: str | None) -> date:
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return timezone.localdate()


class TodayActionsView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(parameters=[OpenApiParameter("date", OpenApiTypes.DATE, OpenApiParameter.QUERY,
                                                required=False)],
                   responses=OpenApiTypes.OBJECT)
    def get(self, request):
        now = timezone.now()
        day = _parse_day(request.query_params.get("date"))
        qs = TodayAction.objects.filter(tenant=request.tenant, date=day)
        decisions.refresh_statuses(qs, now)
        cards = list(qs.select_related("order").order_by("-risk_passed", "shadow", "symbol"))
        open_n = sum(1 for c in cards if c.status in decisions.OPEN)
        return Response({
            "date": day.isoformat(), "now": now.isoformat(),
            "strategy": {"name": STRATEGY, "stage": STAGE_LABEL, "mode": "paper"},
            "board": board_status(day, now),
            "counts": {"total": len(cards), "open": open_n,
                       "passed_risk": sum(1 for c in cards if c.risk_passed),
                       "by_status": {s: sum(1 for c in cards if c.status == s)
                                     for s in TodayAction.Status.values}},
            "can_decide": decisions.can_decide(request.user, request.tenant),
            "cards": [card_dict(c) for c in cards],
            "track_record": {STRATEGY: track_record(request.tenant, STRATEGY)},
        })


class TrackRecordView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        strategy = request.query_params.get("strategy") or STRATEGY
        return Response(track_record(request.tenant, strategy))


class ActionDecisionView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]
    ACTIONS = ("approve", "skip", "snooze")

    @extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT)
    def post(self, request, action_id, decision):
        if decision not in self.ACTIONS:
            return Response({"detail": f"unknown decision {decision!r}"}, status=status.HTTP_404_NOT_FOUND)
        card = TodayAction.objects.filter(tenant=request.tenant, id=action_id).select_related(
            "portfolio", "tenant").first()
        if card is None:
            return Response({"detail": "not found"}, status=status.HTTP_404_NOT_FOUND)
        if not decisions.can_decide(request.user, request.tenant):
            return Response({"detail": "your role cannot decide actions (owner/admin/trader only)"},
                            status=status.HTTP_403_FORBIDDEN)
        data = request.data or {}
        try:
            if decision == "approve":
                card = decisions.approve(card, request.user, note=str(data.get("note") or ""))
            elif decision == "skip":
                card = decisions.skip(card, request.user, reason=str(data.get("reason") or ""))
            else:
                until = self._until(data)
                if until is None:
                    return Response({"detail": "snooze needs 'until' (ISO datetime) or 'minutes'"},
                                    status=status.HTTP_400_BAD_REQUEST)
                card = decisions.snooze(card, request.user, until)
        except decisions.ActionConflict as e:
            return Response({"detail": str(e)}, status=status.HTTP_409_CONFLICT)
        card.refresh_from_db()
        return Response(card_dict(card))

    @staticmethod
    def _until(data) -> datetime | None:
        if data.get("minutes") not in (None, ""):
            try:
                return timezone.now() + timedelta(minutes=max(1, int(data["minutes"])))
            except (TypeError, ValueError):
                return None
        raw = data.get("until")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=IST)
