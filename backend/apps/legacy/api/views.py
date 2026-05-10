"""DRF views that adapt the legacy `dashboard_utils.data_layer` helpers to
the v2 React UI.

Why this exists
---------------
The v2 schema (apps.portfolio, apps.orders, apps.agents_core …) is still
being migrated.  Meanwhile the legacy `trading` Django app already owns
~700 rows of real production-shape data inside `db.sqlite3`.  Exposing
those rows here lets the React UI render real data on day one.

Robustness
----------
The legacy code transitively imports `SmartApi`, `pyotp`, `dotenv`,
`mplfinance` and other broker / data-science deps that are NOT in the
v2 backend's `pyproject.toml`.  If we import `dashboard_utils.data_layer`
at module load time and any of those packages is missing, the URL conf
silently fails and every legacy route returns 404.

Therefore:

*   All legacy imports are **lazy**, performed inside each view.
*   Each view is wrapped in `_with_legacy(...)` which converts any
    `ImportError` into a structured 503 response so the React UI sees a
    sensible empty-state rather than a 404 / 500 / CORS surprise.
*   Pure-DB views (`trading.models`) only need Django itself and degrade
    gracefully when the broker stack is missing.
"""
from __future__ import annotations

import logging
import traceback
from functools import wraps
from typing import Any, Callable

from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _with_legacy(view: Callable) -> Callable:
    """Decorator: turn startup-style failures into a 503 JSON body the
    React layer can render as an empty-state.

    Three failure classes map to 503 "not provisioned yet":
      * ``ImportError``     — a broker/data dep (SmartApi, mplfinance …)
                              is missing from the v2 backend.
      * ``OperationalError``— the configured DB is reachable but the
                              ``trading_*`` tables haven't been migrated
                              into it yet (typical when switching from
                              the sqlite file to Postgres without
                              running ``manage.py migrate trading``).
      * ``ProgrammingError``— the table exists but a column doesn't
                              (schema drift between sqlite/postgres).

    When ``DEBUG=True`` every runtime 500 body also carries ``exc_type``
    plus the tail of the traceback — this avoids the "scroll through
    runserver stdout to find the real error" round-trip that has burned
    us repeatedly.  Production responses still only expose the short
    ``detail`` string.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except ImportError as e:
            logger.warning("legacy bridge missing dep for %s: %s", view.__name__, e)
            return Response(
                {
                    "error": "legacy_dependency_missing",
                    "detail": str(e),
                    "view": view.__name__,
                },
                status=503,
            )
        except (OperationalError, ProgrammingError) as e:
            logger.warning(
                "legacy bridge schema not provisioned for %s: %s",
                view.__name__, e,
            )
            return Response(
                {
                    "error": "legacy_schema_missing",
                    "detail": (
                        "Legacy trading tables are not provisioned in the "
                        "current database. Run `python manage.py migrate "
                        "trading` (or point DATABASE_URL back at the "
                        "sqlite file that holds the seed data)."
                    ),
                    "exc_type": type(e).__name__,
                    "view": view.__name__,
                    "raw": str(e)[:300],
                },
                status=503,
            )
        except Exception as e:  # noqa: BLE001 — boundary
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


def _safe(call, default: Any):
    """Run a broker-dependent helper without ever raising at the HTTP layer."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "fallback": default}


# ---------------------------------------------------------------------------
# Portfolio + combined P&L
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def portfolio(_request):
    """Snapshot used by the Dashboard hero strip + KPI tiles.

    Pure-DB read — no broker dependency.  We compute combined P&L
    ourselves from `trading.models` so we don't pull in the heavier
    `dashboard_utils.data_layer` import path here.
    """
    from datetime import date

    from trading.models import (
        TradeJournal, PortfolioSnapshot, StraddlePosition,
    )

    DEFAULT_CAPITAL = 500_000.0  # mirrors data_layer

    try:
        snap = PortfolioSnapshot.objects.latest()
        capital   = snap.capital
        invested  = snap.invested
        available = snap.available_cash
        daily_pnl = snap.daily_pnl
        total_pnl = snap.total_pnl
        daily_loss = snap.daily_loss
        open_pos   = snap.open_positions
        snap_date  = snap.snapshot_date.isoformat()
    except PortfolioSnapshot.DoesNotExist:
        capital, invested, available = DEFAULT_CAPITAL, 0.0, DEFAULT_CAPITAL
        daily_pnl = total_pnl = daily_loss = 0.0
        open_pos = 0
        snap_date = date.today().isoformat()

    active_straddles = list(
        StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"])
    )
    straddle_pnl     = sum(p.total_pnl for p in active_straddles)
    straddle_premium = sum(p.total_premium_sold for p in active_straddles)

    today = date.today()
    today_trades = TradeJournal.objects.filter(trade_date=today)
    today_count  = today_trades.count()
    today_wins   = today_trades.filter(pnl__gt=0).count()
    today_losses = today_trades.filter(pnl__lt=0).count()

    return Response({
        "capital": capital,
        "invested": invested,
        "available_cash": available,
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
        "daily_loss": daily_loss,
        "open_positions": open_pos,
        "snapshot_date": snap_date,
        "straddle_count": len(active_straddles),
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


# ---------------------------------------------------------------------------
# Active positions (equity + options)
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def positions(_request):
    """Open intraday equity trades + active straddle positions.

    Pure-DB read — re-implements the data_layer helper here so we don't
    have to import the broker-heavy module at request time."""
    from datetime import date
    from trading.models import TradeJournal, StraddlePosition

    today = date.today()

    equity_positions = list(
        TradeJournal.objects.filter(
            status__in=["EXECUTED", "FILLED", "PAPER"],
            trade_date=today,
        ).values(
            "id", "symbol", "side", "entry_price", "stop_loss", "target",
            "quantity", "pnl", "status", "confidence", "fill_price",
        )
    )

    option_positions = []
    for p in StraddlePosition.objects.filter(
        status__in=["ACTIVE", "PARTIAL", "HEDGED"]
    ):
        option_positions.append({
            "id": p.id,
            "underlying": p.underlying,
            "strike": p.display_strike,
            "ce_strike": p.ce_strike_actual,
            "pe_strike": p.pe_strike_actual,
            "expiry": p.expiry.isoformat(),
            "lots": p.lots,
            "lot_size": p.lot_size,
            "ce_sell": p.ce_sell_price,
            "pe_sell": p.pe_sell_price,
            "ce_current": p.ce_current_price,
            "pe_current": p.pe_current_price,
            "net_delta": p.net_delta,
            "pnl_inr": p.total_pnl,
            "realized_pnl": p.realized_pnl,
            "unrealized_pnl": p.current_pnl_inr,
            "status": p.status,
            "dte": max(0, (p.expiry - today).days),
        })

    return Response({"equity": equity_positions, "options": option_positions})


# ---------------------------------------------------------------------------
# Trade journal
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def trades(request):
    from trading.models import TradeJournal

    limit = min(int(request.query_params.get("limit", 50)), 500)
    qs = TradeJournal.objects.order_by("-created_at")
    symbol = request.query_params.get("symbol")
    if symbol:
        qs = qs.filter(symbol=symbol.upper())

    rows = list(qs[:limit].values(
        "id", "trade_date", "symbol", "side", "status",
        "entry_price", "stop_loss", "target",
        "quantity", "fill_price", "fill_quantity",
        "pnl", "pnl_percent", "confidence",
        "reasoning", "risk_approved", "risk_reason",
        "order_id", "created_at",
    ))
    return Response({"count": len(rows), "results": rows})


# ---------------------------------------------------------------------------
# Straddle history
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def straddles(request):
    from datetime import date, timedelta
    from trading.models import StraddlePosition

    qs = StraddlePosition.objects.order_by("-trade_date")
    status_filter = request.query_params.get("status")
    if status_filter:
        qs = qs.filter(status=status_filter.upper())

    rows = []
    for p in qs[:200]:
        rows.append({
            "id": p.id,
            "underlying": p.underlying,
            "strike": p.display_strike,
            "expiry": p.expiry.isoformat(),
            "trade_date": p.trade_date.isoformat(),
            "status": p.status,
            "lots": p.lots,
            "lot_size": p.lot_size,
            "ce_symbol": p.ce_symbol,
            "pe_symbol": p.pe_symbol,
            "ce_sell": p.ce_sell_price,
            "pe_sell": p.pe_sell_price,
            "ce_current": p.ce_current_price,
            "pe_current": p.pe_current_price,
            "premium_sold": p.total_premium_sold,
            "pnl_inr": p.total_pnl,
            "net_delta": p.net_delta,
            "action_taken": p.action_taken,
        })

    # Compact closed-history block (last 180d)
    cutoff = date.today() - timedelta(days=180)
    closed = list(StraddlePosition.objects.filter(status="CLOSED", trade_date__gte=cutoff))
    history = {
        "count": len(closed),
        "total_pnl": sum(p.total_pnl for p in closed),
        "wins": sum(1 for p in closed if p.total_pnl > 0),
    }
    return Response({"count": len(rows), "results": rows, "history": history})


# ---------------------------------------------------------------------------
# Audit log → AI activity feed for the Agent Console
# ---------------------------------------------------------------------------
def _format_audit_detail(entry) -> str:
    et = entry.event_type
    sym = entry.symbol or "—"
    if et == "PLANNER_REQ":   return f"Planning trade for {sym}"
    if et == "PLANNER_RES":   return f"AI planned trade for {sym}"
    if et == "PLANNER_ERR":   return f"AI planner error for {sym}"
    if et == "RISK_APPROVE":  return f"Risk APPROVED {sym}"
    if et == "RISK_REJECT":
        reason = ""
        if entry.risk_details and isinstance(entry.risk_details, dict):
            reason = entry.risk_details.get("reason", "")
        return f"Risk REJECTED {sym}: {reason}" if reason else f"Risk REJECTED {sym}"
    if et == "EXECUTION":  return f"Order executed for {sym}"
    if et == "RECONCILE":  return "Position reconciliation"
    return et


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def audit(request):
    from trading.models import AuditLog
    limit = min(int(request.query_params.get("limit", 25)), 200)
    entries = AuditLog.objects.order_by("-created_at")[:limit]
    feed = [
        {
            "time":   e.created_at.strftime("%H:%M:%S"),
            "type":   e.event_type,
            "symbol": e.symbol or "",
            "detail": _format_audit_detail(e),
        }
        for e in entries
    ]
    return Response({"results": feed})


# ---------------------------------------------------------------------------
# Risk utilization + alerts
# ---------------------------------------------------------------------------
def _risk_utilization() -> dict:
    """Mirror of data_layer.get_risk_utilization, no broker import."""
    from trading.models import PortfolioSnapshot, StraddlePosition

    DEFAULT_CAPITAL = 500_000.0
    MAX_DAILY_LOSS_PCT = 3.0
    MAX_POSITION_SIZE_PCT = 10.0
    MAX_OPEN_POSITIONS = 5

    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = snap.capital
        daily_loss = snap.daily_loss
        invested = snap.invested
        open_pos = snap.open_positions
    except PortfolioSnapshot.DoesNotExist:
        capital, daily_loss, invested, open_pos = DEFAULT_CAPITAL, 0.0, 0.0, 0

    max_daily_loss     = capital * (MAX_DAILY_LOSS_PCT / 100)
    max_position_value = capital * (MAX_POSITION_SIZE_PCT / 100)
    daily_loss_pct     = (daily_loss / max_daily_loss * 100) if max_daily_loss > 0 else 0
    capital_dep_pct    = (invested  / capital        * 100) if capital        > 0 else 0

    underwater = 0
    active = list(StraddlePosition.objects.filter(status="ACTIVE"))
    for p in active:
        if p.combined_current_pts > p.combined_sell_pts:
            underwater += 1

    options_margin = sum(p.total_premium_sold for p in active)

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
def risk(_request):
    r = _risk_utilization()
    r["status"] = _risk_status(r)
    return Response(r)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def alerts(_request):
    from datetime import date
    from trading.models import StraddlePosition

    r = _risk_utilization()
    out = []

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
    for p in StraddlePosition.objects.filter(status="ACTIVE"):
        if p.combined_current_pts > p.combined_sell_pts:
            out.append({
                "severity": "critical",
                "message": f"{p.underlying} {p.strike} straddle is UNDERWATER "
                           f"(P&L: {p.current_pnl_inr:+,.0f} INR)",
                "action": "Consider closing immediately",
            })
        dte = max(0, (p.expiry - today).days)
        if dte <= 1:
            out.append({
                "severity": "warning",
                "message": f"{p.underlying} {p.strike} straddle expires "
                           f"{'TODAY' if dte == 0 else 'TOMORROW'}",
                "action": "Close before 3:15 PM",
            })

    if r["open_positions"] >= r["max_open_positions"]:
        out.append({
            "severity": "info",
            "message": f"Max open positions reached ({r['open_positions']}/{r['max_open_positions']})",
            "action": "No new trades until a position is closed",
        })

    return Response({"results": out})


# ---------------------------------------------------------------------------
# Journal analytics (win rate, streaks, calibration)
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def analytics(request):
    from datetime import date, timedelta
    from trading.models import TradeJournal

    days = min(int(request.query_params.get("days", 30)), 365)
    cutoff = date.today() - timedelta(days=days)
    trades_qs = list(
        TradeJournal.objects.filter(trade_date__gte=cutoff, pnl__isnull=False)
        .order_by("created_at")
        .values("symbol", "side", "pnl", "confidence", "trade_date")
    )

    if not trades_qs:
        return Response({"trades": 0, "symbols": {}, "streaks": [], "calibration": []})

    symbols: dict[str, dict] = {}
    total_pnl = 0.0
    for t in trades_qs:
        sym = t["symbol"]
        symbols.setdefault(sym, {"wins": 0, "losses": 0, "total_pnl": 0})
        symbols[sym]["total_pnl"] += t["pnl"]
        total_pnl += t["pnl"]
        if t["pnl"] > 0:
            symbols[sym]["wins"] += 1
        elif t["pnl"] < 0:
            symbols[sym]["losses"] += 1

    streak = 0
    streaks = []
    for t in trades_qs:
        if t["pnl"] > 0:
            streak = max(0, streak) + 1
        elif t["pnl"] < 0:
            streak = min(0, streak) - 1
        streaks.append(streak)

    wins = sum(1 for t in trades_qs if t["pnl"] > 0)
    losses = sum(1 for t in trades_qs if t["pnl"] < 0)

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


# ---------------------------------------------------------------------------
# Exposure breakdown
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def exposure(_request):
    from datetime import date
    from trading.models import PortfolioSnapshot, TradeJournal, StraddlePosition

    DEFAULT_CAPITAL = 500_000.0
    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = snap.capital
        equity_invested = snap.invested
    except PortfolioSnapshot.DoesNotExist:
        capital, equity_invested = DEFAULT_CAPITAL, 0.0

    today = date.today()
    eq_pos = TradeJournal.objects.filter(
        status__in=["EXECUTED", "FILLED", "PAPER"], trade_date=today,
    )
    equity_risk = sum(t.risk_amount for t in eq_pos)
    equity_count = eq_pos.count()

    active = list(StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]))
    options_premium  = sum(p.total_premium_sold for p in active)
    options_pnl      = sum(p.total_pnl          for p in active)
    options_max_risk = sum(p.total_premium_sold * 2 for p in active)
    total_at_risk    = equity_risk + max(0, -options_pnl)

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


# ---------------------------------------------------------------------------
# System controls
# ---------------------------------------------------------------------------
def _get_system_control(key: str, default=None):
    from trading.models import SystemControl
    try:
        return SystemControl.objects.get(key=key).value
    except SystemControl.DoesNotExist:
        return default


def _set_system_control(key: str, value):
    from trading.models import SystemControl
    SystemControl.objects.update_or_create(key=key, defaults={"value": value})


def _is_market_open_simple() -> bool:
    """Simple IST market-hours check that doesn't import trading.utils."""
    from datetime import datetime, time, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def system(_request):
    import os
    from datetime import datetime, timezone, timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    is_open = _is_market_open_simple()
    is_weekday = now.weekday() < 5

    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
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
        "ai_paused": _get_system_control("ai_trading_paused", False) is True,
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
def pause_ai(_request):
    _set_system_control("ai_trading_paused", True)
    logger.warning("AI TRADING PAUSED via legacy bridge")
    return Response({"ai_paused": True})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@_with_legacy
def resume_ai(_request):
    _set_system_control("ai_trading_paused", False)
    logger.info("AI trading RESUMED via legacy bridge")
    return Response({"ai_paused": False})


# ---------------------------------------------------------------------------
# Strategy library — read-only list of legacy StrategyDoc rows
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def strategies(_request):
    from trading.models import StrategyDoc
    rows = list(
        StrategyDoc.objects.filter(is_active=True)
        .order_by("category", "title")
        .values(
            "id", "title", "category", "content",
            "is_active", "created_at", "updated_at",
        )
    )
    return Response({"count": len(rows), "results": rows})


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def watchlist(_request):
    from datetime import date, timedelta
    from trading.models import WatchlistEntry
    # Last 7 days only — keeps the response useful for the sidebar widget
    cutoff = date.today() - timedelta(days=7)
    rows = list(
        WatchlistEntry.objects
        .filter(scan_date__gte=cutoff)
        .order_by("-scan_date", "-score")
        .values(
            "id", "symbol", "scan_date", "score", "bias",
            "setups", "prev_close", "prev_high", "prev_low", "prev_atr",
            "orb_high", "orb_low", "vwap",
            "outcome", "triggered_setup", "reason", "created_at",
        )
    )
    return Response({"count": len(rows), "results": rows})


# ---------------------------------------------------------------------------
# Pyramid Strategy — run simulation on option candle data
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@_with_legacy
def pyramid(request):
    """Run pyramiding strategy simulation.

    Query params:
      strike (int, required), type (CE/PE), underlying (NIFTY/BANKNIFTY),
      expiry (DDMMMYY), date (YYYY-MM-DD), interval (FIVE_MINUTE),
      capital (float), risk_pct (float), profit_risk (float),
      max_pyramids (int), lot_size (int), dry_run (bool)
    """
    from trading.pyramid.strategy import (
        Candle, PyramidConfig, run_pyramid_with_chart_data,
    )

    strike = request.query_params.get("strike")
    if not strike:
        return Response({"error": "strike is required"}, status=400)
    strike = int(strike)

    opt_type = request.query_params.get("type", "CE").upper()
    underlying = request.query_params.get("underlying", "NIFTY").upper()
    interval = request.query_params.get("interval", "FIVE_MINUTE")
    dry_run = request.query_params.get("dry_run", "false").lower() == "true"

    config = PyramidConfig(
        lot_size=int(request.query_params.get("lot_size", 65)),
        initial_capital=float(request.query_params.get("capital", 100000)),
        initial_risk_pct=float(request.query_params.get("risk_pct", 2.0)),
        profit_risk_pct=float(request.query_params.get("profit_risk", 0.80)),
        max_pyramids=int(request.query_params.get("max_pyramids", 5)),
    )

    # Resolve expiry
    expiry_str = request.query_params.get("expiry")
    if not expiry_str:
        from trading.utils.expiry_utils import next_expiry_date, iso_to_angel
        exp_date = next_expiry_date(underlying)
        expiry_str = iso_to_angel(exp_date.isoformat()) if exp_date else None
        if not expiry_str:
            return Response({"error": "Cannot determine expiry"}, status=400)

    # Resolve date
    candle_date = request.query_params.get("date")
    if not candle_date:
        from trading.utils.time_utils import get_candle_date_range
        candle_date = get_candle_date_range()[0].isoformat()

    symbol_label = f"{underlying} {strike} {opt_type} (exp {expiry_str})"

    if dry_run:
        candles = _generate_pyramid_sample()
        actual_date = candle_date
    else:
        candles = _fetch_pyramid_candles(
            underlying, strike, expiry_str, opt_type, candle_date, interval,
        )
        # Detect actual date from candle data
        actual_date = candle_date
        if candles:
            ts = candles[0].timestamp
            actual_date = ts[:10] if len(ts) >= 10 else candle_date

    if not candles:
        return Response({
            "error": "No candle data",
            "detail": (
                f"No option candles found for {underlying} {strike} {opt_type} "
                f"(exp {expiry_str}) on {candle_date} or recent trading days. "
                f"The option may not have been listed yet, or the strike is too far OTM."
            ),
        }, status=404)

    data = run_pyramid_with_chart_data(candles, symbol=symbol_label, config=config)
    data["config"] = {
        "strike": strike,
        "type": opt_type,
        "underlying": underlying,
        "expiry": expiry_str,
        "date": actual_date,
        "interval": interval,
        "capital": config.initial_capital,
        "risk_pct": config.initial_risk_pct,
        "profit_risk": config.profit_risk_pct,
        "max_pyramids": config.max_pyramids,
        "lot_size": config.lot_size,
        "dry_run": dry_run,
    }
    return Response(data)


def _fetch_pyramid_candles(underlying, strike, expiry_str, opt_type, candle_date, interval):
    """Fetch option candles, walking back up to 5 days to handle holidays."""
    from datetime import date as dt_date, timedelta
    from trading.options.data_service import find_option_token
    from trading.services.data_service import BrokerClient
    from trading.pyramid.strategy import Candle
    from trading.utils.time_utils import cap_end_time

    result = find_option_token(underlying, strike, expiry_str, opt_type)
    if not result:
        return []
    symbol, token = result
    broker = BrokerClient.get_instance()
    broker.ensure_login()

    # SENSEX options trade on BSE (BFO segment), everything else on NSE (NFO)
    exchange = "BFO" if underlying == "SENSEX" else "NFO"

    # Try the requested date first, then walk back up to 5 days
    # to handle holidays (e.g. May Day) that get_candle_date_range misses
    d = dt_date.fromisoformat(candle_date)
    for attempt in range(6):
        if d.weekday() >= 5:  # skip weekends
            d -= timedelta(days=1)
            continue
        ds = d.isoformat()
        end_str = cap_end_time(ds)
        raw = broker.fetch_candles(
            symbol_token=token,
            start=f"{ds} 09:15",
            end=end_str,
            interval=interval,
            exchange=exchange,
        )
        if raw and len(raw) > 5:
            return [Candle.from_raw(r) for r in raw]
        d -= timedelta(days=1)

    return []


def _generate_pyramid_sample():
    import random
    from datetime import datetime
    from trading.pyramid.strategy import Candle

    candles = []
    price = 180.0
    base_time = datetime(2026, 5, 5, 9, 15)
    random.seed(77)
    phases = {
        (0, 15): (0.1, 0.8), (15, 25): (0.8, 1.0), (25, 45): (1.2, 0.7),
        (45, 55): (0.6, 0.5), (55, 65): (1.5, 0.9), (65, 75): (-0.3, 1.2),
    }
    for i in range(75):
        total_min = 15 + i * 5
        ts = base_time.replace(hour=9 + total_min // 60, minute=total_min % 60)
        if ts.hour >= 15 and ts.minute > 30:
            break
        drift, vol = 0.1, 0.8
        for (s, e), (d, v) in phases.items():
            if s <= i < e:
                drift, vol = d, v
                break
        open_p = price
        close_p = open_p + drift + random.gauss(0, vol)
        high_p = max(open_p, close_p) + abs(random.gauss(0, vol * 0.6))
        low_p = min(open_p, close_p) - abs(random.gauss(0, vol * 0.5))
        candles.append(Candle(
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            open=round(open_p, 2), high=round(high_p, 2),
            low=round(low_p, 2), close=round(close_p, 2),
            volume=random.randint(8000, 60000),
        ))
        price = close_p
    return candles
