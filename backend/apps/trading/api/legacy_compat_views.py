"""Legacy bridge views — read from v2 Postgres tables.

These endpoints existed to expose legacy SQLite data to the React UI
without breaking the contract. As of redesign-v2, all queries route to
the new v2 models:

  Legacy table                 → v2 source
  -------------------------------------------------
  trading.TradeJournal         → apps.trading.Trade
  trading.StraddlePosition     → apps.trading.OptionsPosition + .OptionsLeg
  trading.AuditLog             → apps.events.Event
  trading.PortfolioSnapshot    → apps.trading.PortfolioSnapshot
  trading.SystemControl        → apps.system.SystemControl
  trading.SignalLog            → apps.strategies.Signal
  trading.WatchlistEntry       → apps.strategies.WatchlistEntry
  trading.StrategyDoc          → apps.rag.KnowledgeDoc

Response JSON shapes are unchanged — the frontend depends on them and
the legacy bridge URL routes (/api/v1/legacy/*) survive until the
frontend nav migrates to /api/v1/{trades,events,...}/ proper.

In Phase 6 these views move to their proper v2 locations and the
legacy bridge app is deleted entirely.
"""
from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta, timezone as dt_tz
from functools import wraps
from typing import Any, Callable

from django.conf import settings
from django.db.models import Count, Sum, F
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.events.models import Event
from apps.trading.models import Portfolio, PortfolioSnapshot
from apps.rag.models import KnowledgeDoc
from apps.strategies.models import Signal, WatchlistEntry
from apps.system.models import SystemControl
from apps.trading.models import OptionsLeg, OptionsPosition, Trade

logger = logging.getLogger(__name__)

DEFAULT_CAPITAL = 500_000.0
MAX_DAILY_LOSS_PCT = 3.0
MAX_POSITION_SIZE_PCT = 10.0
MAX_OPEN_POSITIONS = 5

# Statuses a Trade has at "in the market" — used for open-position counts.
LIVE_TRADE_STATUSES = (
    Trade.Status.SENT,
    Trade.Status.PARTIAL,
    Trade.Status.FILLED,
)
# Statuses with terminal P&L — used for daily-PNL aggregation.
TERMINAL_PNL_STATUSES = (
    Trade.Status.FILLED,
    Trade.Status.PARTIAL,
    Trade.Status.CANCELLED,
    Trade.Status.CLOSED,
)


# ── Error wrapper (now mostly a safety net) ───────────────────────────

def _with_legacy(view: Callable) -> Callable:
    """Catch unexpected errors and surface as 5xx JSON for the React UI.

    Used to translate legacy ImportError → 503; now the bridge no longer
    imports anything legacy, so this is purely a runtime-error boundary.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.exception("legacy bridge failed in %s", view.__name__)
            body = {
                "error": "legacy_runtime_error",
                "detail": str(e),
                "view": view.__name__,
                "exc_type": type(e).__name__,
            }
            if getattr(settings, "DEBUG", False):
                body["traceback"] = traceback.format_exc().splitlines()[-12:]
            return Response(body, status=500)
    return wrapper


# ── OptionsPosition computed properties (replace legacy methods) ──────

def _options_pos_premium_sold(pos: OptionsPosition) -> float:
    """sum(leg.open_price × leg.qty) for SHORT legs (matches legacy total_premium_sold)."""
    total = 0.0
    for leg in pos.legs.all():
        if leg.leg_role in (OptionsLeg.LegRole.SHORT_CE, OptionsLeg.LegRole.SHORT_PE):
            total += float(leg.open_price) * leg.qty
    return total


def _options_pos_total_pnl(pos: OptionsPosition) -> float:
    return float(pos.current_pnl_inr) + float(pos.realized_pnl)


def _options_pos_combined_pts(pos: OptionsPosition, which: str) -> float:
    """`which` ∈ {'sold','current'}. Sums per-lot prices across SHORT legs."""
    total = 0.0
    for leg in pos.legs.all():
        if leg.leg_role in (OptionsLeg.LegRole.SHORT_CE, OptionsLeg.LegRole.SHORT_PE):
            total += float(leg.open_price if which == "sold" else leg.current_price)
    return total


def _options_pos_display_strike(pos: OptionsPosition) -> int:
    """Display strike = average of leg strikes (single strike for straddle,
    midpoint for strangle/spread)."""
    strikes = [leg.strike for leg in pos.legs.all() if leg.strike is not None]
    if not strikes:
        return 0
    return int(sum(strikes) / len(strikes))


def _legs_by_role(pos: OptionsPosition) -> dict[str, OptionsLeg | None]:
    out: dict[str, OptionsLeg | None] = {"SHORT_CE": None, "SHORT_PE": None,
                                          "LONG_CE": None, "LONG_PE": None}
    for leg in pos.legs.all():
        out[leg.leg_role] = leg
    return out


# ── Portfolio + combined P&L ──────────────────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def portfolio(request):
    """Dashboard hero strip + KPI tiles."""
    tenant = request.tenant
    today = date.today()

    snap = (
        PortfolioSnapshot.objects.filter(tenant=tenant).order_by("-captured_at").first()
    )
    portfolio_row = Portfolio.objects.filter(tenant=tenant).first()
    capital = float(portfolio_row.capital) if portfolio_row else DEFAULT_CAPITAL

    if snap:
        daily_pnl = float(snap.day_pnl)
        invested = capital - float(snap.equity) + daily_pnl  # rough back-calc
        open_pos = snap.open_positions
        snap_date = snap.captured_at.date().isoformat()
    else:
        daily_pnl, invested, open_pos = 0.0, 0.0, 0
        snap_date = today.isoformat()

    available = capital - max(invested, 0)
    total_pnl = float(portfolio_row.realized_pnl) if portfolio_row else 0.0
    daily_loss = max(0.0, -daily_pnl)

    active_pos_qs = OptionsPosition.objects.filter(
        tenant=tenant, status__in=[
            OptionsPosition.Status.ACTIVE,
            OptionsPosition.Status.PARTIAL,
            OptionsPosition.Status.HEDGED,
        ],
    ).prefetch_related("legs")
    active_options = list(active_pos_qs)
    straddle_pnl = sum(_options_pos_total_pnl(p) for p in active_options)
    straddle_premium = sum(_options_pos_premium_sold(p) for p in active_options)

    today_trades = Trade.objects.filter(tenant=tenant, trade_date=today)
    today_count = today_trades.count()
    today_wins = today_trades.filter(realized_pnl__gt=0).count()
    today_losses = today_trades.filter(realized_pnl__lt=0).count()

    return Response({
        "capital": capital,
        "invested": invested,
        "available_cash": available,
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
        "daily_loss": daily_loss,
        "open_positions": open_pos,
        "snapshot_date": snap_date,
        "straddle_count": len(active_options),
        "straddle_pnl": straddle_pnl,
        "straddle_premium_sold": straddle_premium,
        "today_trades": today_count,
        "today_wins": today_wins,
        "today_losses": today_losses,
        "combined_pnl": daily_pnl + straddle_pnl,
        "combined": {
            "equity_pnl":  daily_pnl,
            "options_pnl": straddle_pnl,
            "total_pnl":   daily_pnl + straddle_pnl,
        },
    })


# ── Open positions (equity + options) ─────────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def positions(request):
    """Open intraday equity trades + active option positions."""
    tenant = request.tenant
    today = date.today()

    equity_positions = list(
        Trade.objects.filter(
            tenant=tenant,
            status__in=LIVE_TRADE_STATUSES,
            trade_date=today,
        ).values(
            "id", "symbol", "side",
            "entry_price", "stop_loss", "target",
            "quantity",
            # Legacy alias: TradeJournal had `pnl`, frontend keys on it
            pnl=F("realized_pnl"),
        ).annotate(
            status=F("status"),
            confidence=F("confidence"),
            fill_price=F("fill_price"),
        )
    )

    option_positions = []
    qs = (OptionsPosition.objects
          .filter(tenant=tenant,
                   status__in=[
                       OptionsPosition.Status.ACTIVE,
                       OptionsPosition.Status.PARTIAL,
                       OptionsPosition.Status.HEDGED,
                   ])
          .prefetch_related("legs"))
    for p in qs:
        legs = _legs_by_role(p)
        ce = legs.get("SHORT_CE") or legs.get("LONG_CE")
        pe = legs.get("SHORT_PE") or legs.get("LONG_PE")
        option_positions.append({
            "id": str(p.id),
            "underlying": p.underlying,
            "strike": _options_pos_display_strike(p),
            "ce_strike": ce.strike if ce else None,
            "pe_strike": pe.strike if pe else None,
            "expiry": p.expiry.isoformat(),
            "lots": p.lots,
            "lot_size": p.lot_size,
            "ce_sell": float(ce.open_price) if ce else 0,
            "pe_sell": float(pe.open_price) if pe else 0,
            "ce_current": float(ce.current_price) if ce else 0,
            "pe_current": float(pe.current_price) if pe else 0,
            "net_delta": float(p.net_delta),
            "pnl_inr": _options_pos_total_pnl(p),
            "realized_pnl": float(p.realized_pnl),
            "unrealized_pnl": float(p.current_pnl_inr),
            "status": p.status,
            "dte": max(0, (p.expiry - today).days),
        })

    return Response({"equity": equity_positions, "options": option_positions})


# ── Trade journal ─────────────────────────────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def trades(request):
    tenant = request.tenant
    limit = min(int(request.query_params.get("limit", 50)), 500)
    qs = Trade.objects.filter(tenant=tenant).order_by("-created_at")
    symbol = request.query_params.get("symbol")
    if symbol:
        qs = qs.filter(symbol=symbol.upper())

    rows = []
    for t in qs[:limit]:
        rows.append({
            "id": str(t.id),
            "trade_date": t.trade_date.isoformat(),
            "symbol": t.symbol,
            "side": t.side,
            "status": t.status,
            "entry_price": float(t.entry_price),
            "stop_loss": float(t.stop_loss),
            "target": float(t.target),
            "quantity": t.quantity,
            "fill_price": float(t.fill_price) if t.fill_price else None,
            "fill_quantity": t.fill_quantity,
            "pnl": float(t.realized_pnl) if t.realized_pnl is not None else None,
            "pnl_percent": float(t.pnl_percent) if t.pnl_percent is not None else None,
            "confidence": float(t.confidence),
            "reasoning": t.reasoning,
            "risk_approved": t.risk_approved,
            "risk_reason": t.risk_reason,
            "order_id": str(t.primary_order_id) if t.primary_order_id else "",
            "created_at": t.created_at.isoformat(),
        })
    return Response({"count": len(rows), "results": rows})


# ── Straddle / options position history ───────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def straddles(request):
    tenant = request.tenant
    qs = (OptionsPosition.objects
          .filter(tenant=tenant)
          .order_by("-trade_date")
          .prefetch_related("legs"))
    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter.upper())

    rows = []
    for p in qs[:200]:
        legs = _legs_by_role(p)
        ce = legs.get("SHORT_CE") or legs.get("LONG_CE")
        pe = legs.get("SHORT_PE") or legs.get("LONG_PE")
        rows.append({
            "id": str(p.id),
            "underlying": p.underlying,
            "strike": _options_pos_display_strike(p),
            "expiry": p.expiry.isoformat(),
            "trade_date": p.trade_date.isoformat(),
            "status": p.status,
            "lots": p.lots,
            "lot_size": p.lot_size,
            "ce_symbol": ce.symbol if ce else "",
            "pe_symbol": pe.symbol if pe else "",
            "ce_sell": float(ce.open_price) if ce else 0,
            "pe_sell": float(pe.open_price) if pe else 0,
            "ce_current": float(ce.current_price) if ce else 0,
            "pe_current": float(pe.current_price) if pe else 0,
            "premium_sold": _options_pos_premium_sold(p),
            "pnl_inr": _options_pos_total_pnl(p),
            "net_delta": float(p.net_delta),
            "action_taken": "",   # carried in events.Event now; not a column
        })

    cutoff = date.today() - timedelta(days=180)
    closed = (OptionsPosition.objects
              .filter(tenant=tenant, status=OptionsPosition.Status.CLOSED,
                       trade_date__gte=cutoff)
              .prefetch_related("legs"))
    closed_list = list(closed)
    history = {
        "count": len(closed_list),
        "total_pnl": sum(_options_pos_total_pnl(p) for p in closed_list),
        "wins": sum(1 for p in closed_list if _options_pos_total_pnl(p) > 0),
    }
    return Response({"count": len(rows), "results": rows, "history": history})


# ── Audit log / AI activity feed (now from events.Event) ──────────────

# Map v2 Event.type → legacy AuditLog.event_type for the activity feed.
_EVENT_TO_LEGACY = {
    Event.Type.LLM_REQUEST:  "PLANNER_REQ",
    Event.Type.LLM_RESPONSE: "PLANNER_RES",
    Event.Type.LLM_ERROR:    "PLANNER_ERR",
    Event.Type.RISK_APPROVED: "RISK_APPROVE",
    Event.Type.RISK_REJECTED: "RISK_REJECT",
    Event.Type.ORDER_SENT:   "EXECUTION",
}


def _format_event_detail(e: Event) -> str:
    sym = (e.payload or {}).get("symbol") or (e.text.split()[0] if e.text else "—")
    legacy_et = _EVENT_TO_LEGACY.get(e.type, e.type)
    if legacy_et == "PLANNER_REQ":   return f"Planning trade for {sym}"
    if legacy_et == "PLANNER_RES":   return f"AI planned trade for {sym}"
    if legacy_et == "PLANNER_ERR":   return f"AI planner error for {sym}"
    if legacy_et == "RISK_APPROVE":  return f"Risk APPROVED {sym}"
    if legacy_et == "RISK_REJECT":
        reason = (e.payload or {}).get("reason", "")
        return f"Risk REJECTED {sym}: {reason}" if reason else f"Risk REJECTED {sym}"
    if legacy_et == "EXECUTION":  return f"Order executed for {sym}"
    return e.text or e.type


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def audit(request):
    tenant = request.tenant
    limit = min(int(request.query_params.get("limit", 25)), 200)

    entries = (
        Event.objects
        .filter(tenant=tenant, type__in=list(_EVENT_TO_LEGACY.keys()))
        .order_by("-ts")[:limit]
    )
    feed = [
        {
            # Surface the underlying Event PK so the UI can click through to
            # /api/v1/events/<id>/ for the full row (payload, severity,
            # workflow_run linkage, etc.). Without this, the legacy compat
            # shape was a write-only projection and the audit list in the
            # Agents Console couldn't open a detail view.
            "id":     e.id,
            "time":   e.ts.strftime("%H:%M:%S"),
            "type":   _EVENT_TO_LEGACY.get(e.type, e.type),
            "symbol": (e.payload or {}).get("symbol", "") or (e.text.split()[0] if e.text else ""),
            "detail": _format_event_detail(e),
        }
        for e in entries
    ]
    return Response({"results": feed})


# ── Risk utilization + alerts ─────────────────────────────────────────

def _risk_utilization(tenant) -> dict:
    snap = (PortfolioSnapshot.objects.filter(tenant=tenant)
            .order_by("-captured_at").first())
    portfolio_row = Portfolio.objects.filter(tenant=tenant).first()
    capital = float(portfolio_row.capital) if portfolio_row else DEFAULT_CAPITAL

    if snap:
        daily_pnl = float(snap.day_pnl)
        open_pos = snap.open_positions
    else:
        daily_pnl, open_pos = 0.0, 0
    daily_loss = max(0.0, -daily_pnl)
    invested = float(portfolio_row.used_capital) if portfolio_row else 0.0

    max_daily_loss = capital * (MAX_DAILY_LOSS_PCT / 100)
    max_position_value = capital * (MAX_POSITION_SIZE_PCT / 100)
    daily_loss_pct = (daily_loss / max_daily_loss * 100) if max_daily_loss > 0 else 0
    capital_dep_pct = (invested / capital * 100) if capital > 0 else 0

    # Underwater = options where combined current premium > sold
    active = list(
        OptionsPosition.objects
        .filter(tenant=tenant, status=OptionsPosition.Status.ACTIVE)
        .prefetch_related("legs")
    )
    underwater = 0
    for p in active:
        if _options_pos_combined_pts(p, "current") > _options_pos_combined_pts(p, "sold"):
            underwater += 1

    options_margin = sum(_options_pos_premium_sold(p) for p in active)

    return {
        "capital": capital,
        "daily_loss": daily_loss,
        "daily_loss_pct": min(daily_loss_pct, 100),
        "max_daily_loss": max_daily_loss,
        "daily_loss_limit_pct": MAX_DAILY_LOSS_PCT,
        "capital_deployed": invested,
        "capital_deployed_pct": min(capital_dep_pct, 100),
        "max_position_value": max_position_value,
        "open_positions": open_pos,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "underwater_options": underwater,
        "active_straddles": len(active),
        "options_margin_exposure": options_margin,
        "total_exposure": invested + options_margin,
        "total_exposure_pct": min(
            ((invested + options_margin) / capital * 100) if capital > 0 else 0, 100,
        ),
    }


def _risk_status(r: dict) -> str:
    if r["daily_loss_pct"] >= 80 or r["underwater_options"] > 0:
        return "RED"
    if r["daily_loss_pct"] >= 50 or r["open_positions"] >= r["max_open_positions"] - 1:
        return "YELLOW"
    return "GREEN"


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def risk(request):
    r = _risk_utilization(request.tenant)
    r["status"] = _risk_status(r)
    return Response(r)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def alerts(request):
    tenant = request.tenant
    r = _risk_utilization(tenant)
    out: list[dict] = []

    if r["daily_loss_pct"] >= 80:
        out.append({
            "severity": "critical",
            "message": f"Daily loss at {r['daily_loss_pct']:.0f}% of limit "
                       f"({r['daily_loss']:.0f}/{r['max_daily_loss']:.0f} INR)",
            "action": "Consider stopping trading for the day",
        })
    elif r["daily_loss_pct"] >= 50:
        out.append({
            "severity": "warning",
            "message": f"Daily loss at {r['daily_loss_pct']:.0f}% of limit",
            "action": "Monitor closely",
        })

    today = date.today()
    qs = (OptionsPosition.objects
          .filter(tenant=tenant, status=OptionsPosition.Status.ACTIVE)
          .prefetch_related("legs"))
    for p in qs:
        if _options_pos_combined_pts(p, "current") > _options_pos_combined_pts(p, "sold"):
            out.append({
                "severity": "critical",
                "message": f"{p.underlying} {_options_pos_display_strike(p)} option "
                           f"position is UNDERWATER (P&L: {p.current_pnl_inr:+,.0f} INR)",
                "action": "Consider closing immediately",
            })
        dte = max(0, (p.expiry - today).days)
        if dte <= 1:
            out.append({
                "severity": "warning",
                "message": f"{p.underlying} {_options_pos_display_strike(p)} option "
                           f"expires {'TODAY' if dte == 0 else 'TOMORROW'}",
                "action": "Close before 3:15 PM",
            })

    if r["open_positions"] >= r["max_open_positions"]:
        out.append({
            "severity": "info",
            "message": f"Max open positions reached ({r['open_positions']}/{r['max_open_positions']})",
            "action": "No new trades until a position is closed",
        })

    return Response({"results": out})


# ── Journal analytics ─────────────────────────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def analytics(request):
    tenant = request.tenant
    days = min(int(request.query_params.get("days", 30)), 365)
    cutoff = date.today() - timedelta(days=days)

    trades_qs = list(
        Trade.objects.filter(tenant=tenant, trade_date__gte=cutoff,
                              realized_pnl__isnull=False)
        .order_by("created_at")
        .values("symbol", "side", "confidence", "trade_date",
                 pnl=F("realized_pnl"))
    )

    if not trades_qs:
        return Response({"trades": 0, "symbols": {}, "streaks": [], "calibration": []})

    symbols: dict[str, dict] = {}
    total_pnl = 0.0
    for t in trades_qs:
        sym = t["symbol"]
        pnl = float(t["pnl"] or 0)
        symbols.setdefault(sym, {"wins": 0, "losses": 0, "total_pnl": 0})
        symbols[sym]["total_pnl"] += pnl
        total_pnl += pnl
        if pnl > 0:
            symbols[sym]["wins"] += 1
        elif pnl < 0:
            symbols[sym]["losses"] += 1

    streak = 0
    streaks = []
    for t in trades_qs:
        pnl = float(t["pnl"] or 0)
        if pnl > 0:
            streak = max(0, streak) + 1
        elif pnl < 0:
            streak = min(0, streak) - 1
        streaks.append(streak)

    wins = sum(1 for t in trades_qs if float(t["pnl"] or 0) > 0)
    losses = sum(1 for t in trades_qs if float(t["pnl"] or 0) < 0)

    return Response({
        "trades": len(trades_qs),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades_qs) * 100) if trades_qs else 0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(trades_qs) if trades_qs else 0,
        "symbols": symbols,
        "max_win_streak": max(streaks) if streaks else 0,
        "max_loss_streak": abs(min(streaks)) if streaks else 0,
        "current_streak": streaks[-1] if streaks else 0,
    })


# ── Exposure breakdown ────────────────────────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def exposure(request):
    tenant = request.tenant
    today = date.today()
    portfolio_row = Portfolio.objects.filter(tenant=tenant).first()
    capital = float(portfolio_row.capital) if portfolio_row else DEFAULT_CAPITAL
    equity_invested = float(portfolio_row.used_capital) if portfolio_row else 0.0

    eq_pos = Trade.objects.filter(
        tenant=tenant,
        status__in=LIVE_TRADE_STATUSES,
        trade_date=today,
    )
    equity_risk = sum(t.risk_amount for t in eq_pos)
    equity_count = eq_pos.count()

    active = list(OptionsPosition.objects.filter(
        tenant=tenant,
        status__in=[OptionsPosition.Status.ACTIVE,
                     OptionsPosition.Status.PARTIAL,
                     OptionsPosition.Status.HEDGED],
    ).prefetch_related("legs"))
    options_premium = sum(_options_pos_premium_sold(p) for p in active)
    options_pnl = sum(_options_pos_total_pnl(p) for p in active)
    options_max_risk = sum(_options_pos_premium_sold(p) * 2 for p in active)
    total_at_risk = equity_risk + max(0, -options_pnl)

    return Response({
        "capital": capital,
        "equity": {
            "invested": equity_invested,
            "risk_amount": equity_risk,
            "position_count": equity_count,
            "pct_of_capital": (equity_invested / capital * 100) if capital > 0 else 0,
        },
        "options": {
            "premium_sold": options_premium,
            "current_pnl": options_pnl,
            "position_count": len(active),
            "max_risk_estimate": options_max_risk,
            "pct_of_capital": (options_premium / capital * 100) if capital > 0 else 0,
        },
        "total_at_risk": total_at_risk,
        "total_at_risk_pct": (total_at_risk / capital * 100) if capital > 0 else 0,
        "available_capital": capital - equity_invested,
    })


# ── System / kill switch ──────────────────────────────────────────────

def _get_system_control(tenant, key: str, default=None):
    row = SystemControl.objects.filter(tenant=tenant, key=key).first()
    return row.value if row else default


def _set_system_control(tenant, key: str, value):
    SystemControl.objects.update_or_create(
        tenant=tenant, key=key, defaults={"value": value},
    )


def _is_market_open_simple() -> bool:
    ist = dt_tz(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    from datetime import time as dt_time
    return dt_time(9, 15) <= now.time() <= dt_time(15, 30)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def system(request):
    import os
    tenant = request.tenant
    ist = dt_tz(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    is_open = _is_market_open_simple()
    is_weekday = now.weekday() < 5

    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if is_open:
        elapsed = (now - open_t).total_seconds()
        remaining = (close_t - now).total_seconds()
        total = (close_t - open_t).total_seconds()
        progress_pct = (elapsed / total * 100) if total > 0 else 0
    else:
        elapsed = remaining = 0
        progress_pct = 100 if (now > close_t and is_weekday) else 0

    if not is_open:
        if not is_weekday:
            phase = "WEEKEND"
        elif now.hour < 9 or (now.hour == 9 and now.minute < 15):
            phase = "PRE_MARKET"
        else:
            phase = "POST_MARKET"
    elif now.hour == 9 and now.minute < 30:
        phase = "OPENING"
    elif now.hour >= 14 and now.minute >= 45:
        phase = "CLOSING"
    else:
        phase = "REGULAR"

    return Response({
        "ai_paused": _get_system_control(tenant, "ai_trading_paused", False) is True,
        "is_market_open": is_open,
        "trading_mode": os.getenv("TRADING_MODE", "paper"),
        "session": {
            "is_open": is_open,
            "is_weekday": is_weekday,
            "current_time": now.strftime("%H:%M:%S"),
            "market_open": "09:15",
            "market_close": "15:30",
            "elapsed_minutes": int(elapsed / 60) if is_open else 0,
            "remaining_minutes": int(remaining / 60) if is_open else 0,
            "progress_pct": min(progress_pct, 100),
            "session_phase": phase,
        },
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@_with_legacy
def pause_ai(request):
    _set_system_control(request.tenant, "ai_trading_paused", True)
    logger.warning("AI TRADING PAUSED via legacy bridge")
    return Response({"ai_paused": True})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@_with_legacy
def resume_ai(request):
    _set_system_control(request.tenant, "ai_trading_paused", False)
    logger.info("AI trading RESUMED via legacy bridge")
    return Response({"ai_paused": False})


# ── Strategy library (now from KnowledgeDoc) ──────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def strategies(request):
    tenant = request.tenant
    rows = list(
        KnowledgeDoc.objects.filter(tenant=tenant, is_active=True)
        .order_by("category", "title")
        .values(
            "id", "title", "category", "content",
            "is_active", "created_at", "updated_at",
        )
    )
    return Response({"count": len(rows), "results": rows})


# ── Watchlist ─────────────────────────────────────────────────────────

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def watchlist(request):
    tenant = request.tenant
    cutoff = date.today() - timedelta(days=7)
    rows = list(
        WatchlistEntry.objects
        .filter(tenant=tenant, scan_date__gte=cutoff)
        .order_by("-scan_date", "-score")
        .values(
            "id", "symbol", "scan_date", "score", "bias",
            "setups", "prev_close", "prev_high", "prev_low", "prev_atr",
            "orb_high", "orb_low", "vwap",
            "outcome", "triggered_setup", "reason", "created_at",
        )
    )
    return Response({"count": len(rows), "results": rows})


# ── Pyramid (plugin-backed) ─────────────────────────────────────────
# Engine now lives at backend/plugins/strategy_pyramid/. This view still
# directly calls the plugin's pure functions (run_pyramid_with_chart_data)
# rather than going through the StrategyRun framework — Phase 3 follow-up
# will route this through the workflow runtime once the framework is wired.

@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def pyramid(request):
    from datetime import date as dt_date, timedelta as td

    from plugins.strategy_pyramid.strategy import (
        Candle, PyramidConfig, run_pyramid_with_chart_data,
        _generate_pyramid_sample,
    )
    from trading.options.data_service import find_option_token
    from trading.services.data_service import BrokerClient
    from trading.utils.expiry_utils import iso_to_angel, next_expiry_date
    from trading.utils.time_utils import cap_end_time

    strike = int(request.query_params.get("strike", 0))
    opt_type = request.query_params.get("type", "CE").upper()
    underlying = request.query_params.get("underlying", "NIFTY").upper()
    interval = request.query_params.get("interval", "FIVE_MINUTE")
    dry_run = request.query_params.get("dry_run", "false").lower() == "true"

    if strike <= 0:
        return Response({"error": "strike required"}, status=400)

    config = PyramidConfig(
        lot_size=int(request.query_params.get("lot_size", 65)),
        initial_capital=float(request.query_params.get("capital", 100000)),
        initial_risk_pct=float(request.query_params.get("risk_pct", 2.0)),
        profit_risk_pct=float(request.query_params.get("profit_risk", 0.80)),
        max_pyramids=int(request.query_params.get("max_pyramids", 5)),
    )

    expiry_str = request.query_params.get("expiry")
    if not expiry_str:
        exp_date = next_expiry_date(underlying)
        expiry_str = iso_to_angel(exp_date.isoformat()) if exp_date else None
        if not expiry_str:
            return Response({"error": "Cannot determine expiry"}, status=400)

    candle_date = request.query_params.get("date")
    if not candle_date:
        d = dt_date.today()
        while d.weekday() >= 5:
            d -= td(days=1)
        candle_date = d.isoformat()

    if dry_run:
        candles = _generate_pyramid_sample()
        symbol = f"{underlying} {strike} {opt_type} (sample)"
        data = run_pyramid_with_chart_data(candles, symbol=symbol, config=config)
        data["config"] = {
            "strike": strike, "type": opt_type, "underlying": underlying,
            "expiry": expiry_str, "date": candle_date, "interval": interval,
            "capital": config.initial_capital,
            "risk_pct": config.initial_risk_pct,
            "profit_risk": config.profit_risk_pct,
            "max_pyramids": config.max_pyramids,
            "lot_size": config.lot_size, "dry_run": True,
        }
        return Response(data)

    # Live: fetch candles from broker
    result = find_option_token(underlying, strike, expiry_str, opt_type)
    if not result:
        return Response({"error": f"Symbol not found for {underlying} {strike} {opt_type} (exp {expiry_str})"}, status=404)
    symbol, token = result
    broker = BrokerClient.get_instance()
    broker.ensure_login()
    exchange = "BFO" if underlying == "SENSEX" else "NFO"

    d = dt_date.fromisoformat(candle_date)
    candles = []
    for _ in range(6):
        if d.weekday() >= 5:
            d -= td(days=1)
            continue
        raw = broker.fetch_candles(
            symbol_token=token,
            start=f"{d.isoformat()} 09:15",
            end=cap_end_time(d.isoformat()),
            interval=interval,
            exchange=exchange,
        )
        if raw and len(raw) > 5:
            candles = [Candle.from_raw(r) for r in raw]
            break
        d -= td(days=1)

    if not candles:
        return Response({"error": f"No option candles found for {underlying} {strike} {opt_type} (exp {expiry_str})"}, status=404)

    symbol_label = f"{underlying} {strike} {opt_type} (exp {expiry_str})"
    data = run_pyramid_with_chart_data(candles, symbol=symbol_label, config=config)
    data["config"] = {
        "strike": strike, "type": opt_type, "underlying": underlying,
        "expiry": expiry_str, "date": d.isoformat(), "interval": interval,
        "capital": config.initial_capital,
        "risk_pct": config.initial_risk_pct,
        "profit_risk": config.profit_risk_pct,
        "max_pyramids": config.max_pyramids,
        "lot_size": config.lot_size, "dry_run": False,
    }
    return Response(data)
