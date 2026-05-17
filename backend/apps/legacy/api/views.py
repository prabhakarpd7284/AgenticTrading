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

    DATE-RANGE BACKTEST:
      date_from (YYYY-MM-DD) + date_to (YYYY-MM-DD) — when BOTH are
      present we iterate over every trading day in the range and
      return per-day runs + aggregated stats.
    """
    from trading.pyramid.strategy import (
        Candle, PyramidConfig, run_pyramid_with_chart_data,
    )

    # If a date range is supplied, dispatch to the range backtest helper.
    df = request.query_params.get("date_from")
    dt = request.query_params.get("date_to")
    if df and dt:
        return _pyramid_range(request, date_from=df, date_to=dt)

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


def _pyramid_range(request, *, date_from: str, date_to: str):
    """Date-range backtest — runs the pyramid strategy on each trading day
    between `date_from` and `date_to` (inclusive). Returns:

      runs: [{date, total_pnl_inr, total_pnl_pts, total_lots, trades, ...}]
      aggregate: { days_traded, profitable_days, total_pnl_inr,
                   win_rate_pct, avg_daily_pnl_inr, best_day, worst_day,
                   sharpe }

    Skips broker-failed days transparently — `runs` only contains days
    where the engine actually returned a value.
    """
    from datetime import date as dt_date
    from trading.pyramid.strategy import (
        Candle, PyramidConfig, run_pyramid_with_chart_data,
    )

    # Cap range to 60 days to keep the round-trip bearable
    try:
        d_from = dt_date.fromisoformat(date_from)
        d_to = dt_date.fromisoformat(date_to)
    except ValueError:
        return Response({"error": "date_from/date_to must be YYYY-MM-DD"}, status=400)
    if d_to < d_from:
        return Response({"error": "date_to is before date_from"}, status=400)
    if (d_to - d_from).days > 60:
        return Response({"error": "range > 60 days; cap to 60"}, status=400)

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

    # Resolve expiry once — we want the SAME contract across the range
    expiry_str = request.query_params.get("expiry")
    if not expiry_str:
        from trading.utils.expiry_utils import next_expiry_date, iso_to_angel
        exp_date = next_expiry_date(underlying)
        expiry_str = iso_to_angel(exp_date.isoformat()) if exp_date else None
        if not expiry_str:
            return Response({"error": "Cannot determine expiry"}, status=400)

    runs: list[dict] = []
    cursor = d_from
    while cursor <= d_to:
        # Skip weekends; NSE handles holidays via the candle walk-back inside
        # _fetch_pyramid_candles, so we just iterate Mon-Fri.
        if cursor.weekday() < 5:
            candle_date = cursor.isoformat()
            try:
                if dry_run:
                    candles = _generate_pyramid_sample()
                else:
                    candles = _fetch_pyramid_candles(
                        underlying, strike, expiry_str, opt_type, candle_date, interval,
                    )
                if candles:
                    actual_date = (candles[0].timestamp[:10]
                                    if hasattr(candles[0], "timestamp") and candles[0].timestamp
                                    else candle_date)
                    data = run_pyramid_with_chart_data(
                        candles,
                        symbol=f"{underlying} {strike} {opt_type} ({actual_date})",
                        config=config,
                    )
                    # The engine puts totals inside `kpis`; flatten the
                    # 4 fields we need at the row level so the range UI
                    # doesn't have to dig.
                    kpis = data.get("kpis") or {}
                    runs.append({
                        "date": actual_date,
                        "total_pnl_inr": kpis.get("total_pnl_inr", data.get("total_pnl_inr", 0)),
                        "total_pnl_pts": kpis.get("total_pnl_pts", data.get("total_pnl_pts", 0)),
                        "total_lots":    kpis.get("total_lots",    data.get("total_lots", 0)),
                        "trades":        len(data.get("trades", []) or data.get("entries", []) or []),
                        "kpis":          kpis,
                    })
            except Exception:  # noqa: BLE001
                pass
        cursor = cursor.fromordinal(cursor.toordinal() + 1)

    # Aggregate
    if not runs:
        return Response({
            "config": {
                "strike": strike, "type": opt_type, "underlying": underlying,
                "expiry": expiry_str, "date_from": date_from, "date_to": date_to,
                "dry_run": dry_run, "lot_size": config.lot_size,
                "capital": config.initial_capital,
            },
            "runs": [],
            "aggregate": None,
            "error": "No trading days produced candle data in the range.",
        })

    pnls = [r["total_pnl_inr"] for r in runs]
    profitable = sum(1 for p in pnls if p > 0)
    losing = sum(1 for p in pnls if p < 0)
    best = max(runs, key=lambda r: r["total_pnl_inr"])
    worst = min(runs, key=lambda r: r["total_pnl_inr"])
    avg = sum(pnls) / len(pnls)
    import statistics, math
    sd = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
    sharpe = round((avg / sd) * math.sqrt(252), 3) if sd > 0 else 0.0
    aggregate = {
        "days_traded": len(runs),
        "profitable_days": profitable,
        "losing_days": losing,
        "total_pnl_inr": round(sum(pnls), 2),
        "avg_daily_pnl_inr": round(avg, 2),
        "win_rate_pct": round(profitable / len(runs) * 100, 2),
        "best_day": {"date": best["date"], "pnl_inr": best["total_pnl_inr"]},
        "worst_day": {"date": worst["date"], "pnl_inr": worst["total_pnl_inr"]},
        "sharpe": sharpe,
    }
    return Response({
        "config": {
            "strike": strike, "type": opt_type, "underlying": underlying,
            "expiry": expiry_str, "date_from": date_from, "date_to": date_to,
            "interval": interval, "dry_run": dry_run,
            "lot_size": config.lot_size, "capital": config.initial_capital,
        },
        "runs": runs,
        "aggregate": aggregate,
    })


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


# ---------------------------------------------------------------------------
# Stock summary — one payload powering the React Stock View
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stock_summary(request):
    """Per-stock cross-strategy roll-up.

    Query params:
        symbol  required, case-insensitive (HDFCBANK, NIFTY, ...)
        period  one of weekly|monthly|half-yearly|yearly  (default monthly)
        from    ISO date — lower bound, inclusive (optional)
        to      ISO date — upper bound, inclusive   (optional)

    Response shape:
        {
          symbol, kind, period,
          kpis: { capital_deployed, money_in_play, open_count, period_pnl, live_leverage },
          buckets: [{key, count, notional, margin, pnl, premium_received}],
          open_positions: [...],
          rollups: [{period_label, trades_planned, trades_taken, capital_deployed,
                     pnl, positions_opened, positions_closed, win_rate}],
          strategies: [{name, runs, approved, rejected, pnl}],
          indicators: { source: "directional"|"straddle", values: {...} },
        }
    """
    from datetime import datetime, timedelta

    from trading.models import TradeJournal, StraddlePosition
    from apps.agents_core.models import AgentRun
    from apps.common.margin_calc import (
        MarginEstimate, equity_margin, short_straddle_margin, summarize_buckets,
    )

    symbol = (request.query_params.get("symbol") or "").upper().strip()
    if not symbol:
        return Response({"error": "symbol query param required"}, status=400)

    period = (request.query_params.get("period") or "monthly").lower()
    if period not in ("weekly", "monthly", "half-yearly", "yearly"):
        return Response({"error": f"unknown period {period!r}"}, status=400)

    # ── 1. Pull the raw rows for this symbol ──
    trades = list(TradeJournal.objects.filter(symbol=symbol).order_by("-created_at"))
    straddles = list(StraddlePosition.objects.filter(underlying=symbol).order_by("-opened_at"))

    runs = list(
        AgentRun.objects
        .filter(strategy_name__in=("directional", "short_straddle"))
        .order_by("-created_at")[:500]
    )
    # Filter v2 runs by symbol — directional via result.plan.symbol, straddle via position underlying
    straddle_ids = {p.id for p in straddles}
    def _run_matches(r):
        if r.strategy_name == "directional":
            res = r.result or {}
            plan_sym = (res.get("plan") or {}).get("symbol")
            cfg_sym = (r.config or {}).get("symbol")
            return (plan_sym or cfg_sym or "").upper() == symbol
        if r.strategy_name == "short_straddle":
            pid = (r.config or {}).get("position_id")
            return pid in straddle_ids
        return False
    runs = [r for r in runs if _run_matches(r)]

    # ── 2. Margin estimates for open positions ──
    estimates: list[MarginEstimate] = []
    open_positions: list[dict] = []

    # Open equity trades = status in (PENDING, APPROVED, EXECUTED, PAPER) and not closed
    OPEN_TRADE_STATUSES = ("PENDING", "APPROVED", "EXECUTED", "PAPER")
    for t in trades:
        if t.status not in OPEN_TRADE_STATUSES:
            continue
        m = equity_margin(t.side, int(t.quantity), float(t.entry_price), product="MIS")
        estimates.append(m)
        open_positions.append({
            "kind": "equity",
            "strategy": "directional",
            "id": t.id,
            "leg": f"{t.symbol} ({t.side})",
            "side": t.side,
            "quantity": t.quantity,
            "entry_price": float(t.entry_price),
            "current_price": float(t.fill_price or t.entry_price),
            "stop_loss": float(t.stop_loss),
            "expected_exit": float(t.target),
            "pnl_inr": float(t.pnl or 0.0),
            "status": t.status,
            "opened_at": t.created_at.isoformat() if t.created_at else None,
            "notional": m.notional,
            "margin": m.total_margin,
            "leverage": m.leverage,
        })

    for p in straddles:
        if p.status != "ACTIVE":
            continue
        m = short_straddle_margin(
            lots=int(p.lots), lot_size=int(p.lot_size),
            strike=float(p.strike),
            ce_premium=float(p.ce_sell_price), pe_premium=float(p.pe_sell_price),
            underlying_spot=float(p.strike),  # best estimate available without live spot
        )
        estimates.append(m)
        ce_now = float(p.ce_current_price or 0)
        pe_now = float(p.pe_current_price or 0)
        combined_now = ce_now + pe_now
        combined_sold = float(p.ce_sell_price + p.pe_sell_price)
        pnl_pts = combined_sold - combined_now
        pnl_inr = pnl_pts * float(p.lot_size) * float(p.lots)
        # Expected exit = target buy-back at ~50% premium decay (industry default)
        expected_buyback = combined_sold * 0.5
        open_positions.append({
            "kind": "options",
            "strategy": "short_straddle",
            "id": p.id,
            "leg": f"{p.underlying} {p.strike} STRADDLE ({p.expiry})",
            "side": "SHORT",
            "quantity": int(p.lots * p.lot_size * 2),  # both legs
            "entry_price": combined_sold,
            "current_price": combined_now,
            "stop_loss": combined_sold * 1.3,  # 1.3× hard stop
            "expected_exit": expected_buyback,
            "pnl_inr": pnl_inr,
            "status": p.status,
            "opened_at": p.opened_at.isoformat() if p.opened_at else None,
            "expiry": p.expiry.isoformat() if p.expiry else None,
            "notional": m.notional,
            "margin": m.total_margin,
            "leverage": m.leverage,
            "premium_received": m.premium_received,
        })

    bucket_summary = summarize_buckets(estimates)

    # ── 3. KPIs ──
    capital_deployed = sum(p.get("notional", 0) for p in open_positions)
    money_in_play = sum(p.get("margin", 0) for p in open_positions)
    period_window = _period_window(period, request)
    period_trades = [t for t in trades if t.created_at and period_window[0] <= t.created_at.date() <= period_window[1]]
    period_pnl = sum(float(t.pnl or 0) for t in period_trades)
    period_pnl += sum(float(p.realized_pnl or 0) for p in straddles
                       if p.opened_at and period_window[0] <= p.opened_at.date() <= period_window[1])
    live_leverage = bucket_summary["totals"]["leverage"]

    kpis = {
        "capital_deployed": round(capital_deployed, 2),
        "money_in_play": round(money_in_play, 2),
        "open_count": len(open_positions),
        "period_pnl": round(period_pnl, 2),
        "live_leverage": round(live_leverage, 2),
        "period_window": [period_window[0].isoformat(), period_window[1].isoformat()],
    }

    # ── 4. Period rollups ──
    rollups = _period_rollups(symbol, period, trades, straddles, runs)

    # ── 5. Per-strategy roll-up ──
    strat_map: dict[str, dict] = {}
    for r in runs:
        s = strat_map.setdefault(r.strategy_name, {"runs": 0, "approved": 0, "rejected": 0})
        s["runs"] += 1
        risk = (r.result or {}).get("risk") or {}
        if risk.get("approved") is True:
            s["approved"] += 1
        elif risk.get("approved") is False:
            s["rejected"] += 1
    strategies = [
        {"name": name, **stats, "pnl": 0.0}
        for name, stats in strat_map.items()
    ]
    # add equity p&l to directional, straddle p&l to short_straddle
    for s in strategies:
        if s["name"] == "directional":
            s["pnl"] = round(sum(float(t.pnl or 0) for t in trades), 2)
        if s["name"] == "short_straddle":
            s["pnl"] = round(sum(float(p.realized_pnl or 0) + float(p.current_pnl_inr or 0) for p in straddles), 2)

    # ── 6. Latest indicator snapshot (most recent run for this stock) ──
    indicators = {"source": None, "values": {}}
    if runs:
        latest = runs[0]
        res = latest.result or {}
        if latest.strategy_name == "directional" and res.get("indicators"):
            indicators = {"source": "directional", "run_id": str(latest.id), "values": res["indicators"]}
        elif latest.strategy_name == "short_straddle":
            an = res.get("analysis") or {}
            indicators = {
                "source": "short_straddle",
                "run_id": str(latest.id),
                "values": {
                    "ce_delta": an.get("ce_delta"),
                    "pe_delta": an.get("pe_delta"),
                    "net_delta": an.get("net_delta"),
                    "delta_bias": an.get("delta_bias"),
                    "vix_phase": an.get("vix_phase"),
                    "vix_current": an.get("vix_current"),
                    "market_phase": an.get("market_phase"),
                    "premium_decayed_pct": an.get("premium_decayed_pct"),
                    "days_to_expiry": an.get("days_to_expiry"),
                    "is_underwater": an.get("is_underwater"),
                    "nearest_itm_leg": an.get("nearest_itm_leg"),
                },
            }

    return Response({
        "symbol": symbol,
        "kind": _kind_of(symbol),
        "period": period,
        "kpis": kpis,
        "buckets": [
            {"key": k, **{kk: round(vv, 2) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}}
            for k, v in bucket_summary["by_bucket"].items()
        ],
        "bucket_totals": {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in bucket_summary["totals"].items()},
        "open_positions": open_positions,
        "rollups": rollups,
        "strategies": strategies,
        "indicators": indicators,
    })


def _period_window(period: str, request) -> tuple:
    """Return (start_date, end_date) inclusive for the chosen window."""
    from datetime import date, timedelta
    today = date.today()
    end_param = request.query_params.get("to")
    start_param = request.query_params.get("from")
    end = date.fromisoformat(end_param) if end_param else today
    if start_param:
        start = date.fromisoformat(start_param)
    elif period == "weekly":
        start = end - timedelta(days=6)
    elif period == "monthly":
        start = end.replace(day=1)
    elif period == "half-yearly":
        start = (end - timedelta(days=183)).replace(day=1)
    else:  # yearly
        start = end.replace(month=1, day=1)
    return (start, end)


def _period_rollups(symbol: str, period: str, trades, straddles, runs) -> list[dict]:
    """Bucket trades/runs into period slots and aggregate."""
    from datetime import date, timedelta
    today = date.today()

    if period == "weekly":
        slots = [(today - timedelta(days=7*i + 6), today - timedelta(days=7*i)) for i in range(8)][::-1]
        labeller = lambda lo, hi: f"{lo.strftime('%d %b')}–{hi.strftime('%d %b')}"
    elif period == "monthly":
        slots = []
        cursor = today.replace(day=1)
        for _ in range(6):
            month_start = cursor
            if cursor.month == 12:
                next_start = cursor.replace(year=cursor.year + 1, month=1)
            else:
                next_start = cursor.replace(month=cursor.month + 1)
            month_end = next_start - timedelta(days=1)
            slots.append((month_start, month_end))
            cursor = (cursor.replace(day=1) - timedelta(days=1)).replace(day=1)
        slots = slots[::-1]
        labeller = lambda lo, hi: lo.strftime("%b %Y")
    elif period == "half-yearly":
        slots = []
        # Use H1 (Jan-Jun) / H2 (Jul-Dec) blocks for the last 4 halves
        for i in range(4):
            y = today.year - (i // 2)
            half = (today.month <= 6) if (i % 2 == 0) else (today.month > 6)
            if i % 2 == 0:
                lo = date(y, 1 if today.month <= 6 else 7, 1)
                hi = date(y, 6 if today.month <= 6 else 12, 30 if today.month <= 6 else 31)
            else:
                lo = date(y, 7 if today.month <= 6 else 1, 1)
                hi = date(y, 12 if today.month <= 6 else 6, 31 if today.month <= 6 else 30)
            slots.append((lo, hi))
        slots = slots[::-1]
        labeller = lambda lo, hi: f"H{1 if lo.month <= 6 else 2} {lo.year}"
    else:  # yearly
        slots = [(date(today.year - i, 1, 1), date(today.year - i, 12, 31)) for i in range(4)][::-1]
        labeller = lambda lo, hi: str(lo.year)

    out = []
    for lo, hi in slots:
        slot_trades = [t for t in trades if t.created_at and lo <= t.created_at.date() <= hi]
        slot_straddles = [p for p in straddles if p.opened_at and lo <= p.opened_at.date() <= hi]
        slot_runs = [r for r in runs if r.created_at and lo <= r.created_at.date() <= hi]

        trades_taken = sum(1 for t in slot_trades if t.status in ("EXECUTED", "PAPER", "FILLED"))
        trades_planned = len(slot_runs)  # every agent run is a "plan attempt"
        capital_deployed = sum(float(t.entry_price) * t.quantity for t in slot_trades if t.status in ("EXECUTED", "PAPER", "FILLED"))
        capital_deployed += sum(float(p.ce_sell_price + p.pe_sell_price) * p.lot_size * p.lots for p in slot_straddles)
        pnl = sum(float(t.pnl or 0) for t in slot_trades) + sum(float(p.realized_pnl or 0) for p in slot_straddles)
        positions_opened = trades_taken + len(slot_straddles)
        positions_closed = sum(1 for t in slot_trades if t.pnl is not None) + sum(1 for p in slot_straddles if p.status == "CLOSED")
        wins = sum(1 for t in slot_trades if (t.pnl or 0) > 0)
        decided = sum(1 for t in slot_trades if t.pnl is not None)
        win_rate = (wins / decided * 100) if decided else 0
        out.append({
            "period_label": labeller(lo, hi),
            "from": lo.isoformat(),
            "to": hi.isoformat(),
            "trades_planned": trades_planned,
            "trades_taken": trades_taken,
            "capital_deployed": round(capital_deployed, 2),
            "pnl": round(pnl, 2),
            "positions_opened": positions_opened,
            "positions_closed": positions_closed,
            "win_rate": round(win_rate, 1),
        })
    return out


def _kind_of(symbol: str) -> str:
    if symbol in {"NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "BANKEX"}:
        return "index_underlying"
    return "equity"


# ---------------------------------------------------------------------------
# Expiries — feeds the agent-dialog dropdown
# ---------------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def expiries(request):
    """List option expiries available for an underlying.

    Query params:
        underlying  required (NIFTY, BANKNIFTY, HDFCBANK, …)
        kind        all|weekly|monthly (default all)
        limit       default 12

    Response:
        {
          underlying: "HDFCBANK", count: 3,
          results: [
            {expiry: "26MAY26", iso: "2026-05-26", dte: 9,  kind: "monthly", strikes: 63},
            ...
          ]
        }
    """
    from datetime import date, datetime
    from trading.services.ticker_service import ticker_service

    underlying = (request.query_params.get("underlying") or "").upper().strip()
    if not underlying:
        return Response({"error": "underlying required"}, status=400)
    kind_filter = (request.query_params.get("kind") or "all").lower()
    limit = min(int(request.query_params.get("limit", 12)), 50)

    try:
        ticker_service._ensure_loaded()
    except Exception as e:  # noqa: BLE001
        return Response({"error": f"scrip master not loaded: {e}"}, status=503)

    today = date.today()
    by_expiry: dict[date, int] = {}
    for inst in ticker_service._nfo_by_key.values():
        if inst.get("name") != underlying:
            continue
        if inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK"):
            continue
        try:
            d = datetime.strptime(inst.get("expiry", ""), "%d%b%Y").date()
        except Exception:  # noqa: BLE001
            continue
        if d < today:
            continue
        by_expiry[d] = by_expiry.get(d, 0) + 1

    # Classify monthly = last expiry per month (NSE convention).
    months_seen: dict[tuple[int, int], date] = {}
    for d in by_expiry:
        k = (d.year, d.month)
        if k not in months_seen or d > months_seen[k]:
            months_seen[k] = d
    monthly_set = set(months_seen.values())

    rows = []
    for d in sorted(by_expiry):
        is_monthly = d in monthly_set
        if kind_filter == "weekly" and is_monthly:
            continue
        if kind_filter == "monthly" and not is_monthly:
            continue
        rows.append({
            "expiry": d.strftime("%d%b%y").upper(),
            "iso": d.isoformat(),
            "dte": (d - today).days,
            "kind": "monthly" if is_monthly else "weekly",
            "strikes": by_expiry[d],
        })

    return Response({"underlying": underlying, "count": len(rows), "results": rows[:limit]})
