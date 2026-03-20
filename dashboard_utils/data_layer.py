"""
Dashboard Data Layer — centralized data fetching, caching, alerts, risk status.

Single authenticated broker session shared across all dashboard pages.
All heavy lifting happens here — pages only call these functions.
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

from logzero import logger

# Django bootstrap
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
django.setup()

from trading.models import (
    TradeJournal, PortfolioSnapshot, AuditLog, StraddlePosition,
)
from trading.services.risk_engine import (
    MAX_RISK_PER_TRADE_PCT, MAX_DAILY_LOSS_PCT,
    MAX_POSITION_SIZE_PCT, MAX_OPEN_POSITIONS,
)
from trading.options.data_service import OptionsDataService
from trading.options.straddle.analyzer import analyze_straddle

TRADING_MODE = os.getenv("TRADING_MODE", "paper")
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "500000"))

# ── Tokens for index spot data ──
BANKNIFTY_SPOT_TOKEN = "99926009"
NIFTY_SPOT_TOKEN = "99926000"
INDIA_VIX_TOKEN = "99926017"


# ──────────────────────────────────────────────
# Market Pulse (NIFTY, BANKNIFTY, VIX)
# ──────────────────────────────────────────────
def get_market_pulse(options_svc: OptionsDataService) -> dict:
    """
    Fetch NIFTY, BANKNIFTY, VIX serially. Each call has 0.3s rate-limit
    guard built in, and TTL cache means repeated calls within 5s are free.
    """
    results = {}

    for key, fn in [
        ("nifty", options_svc.fetch_nifty_spot),
        ("banknifty", options_svc.fetch_banknifty_spot),
        ("vix", options_svc.fetch_vix),
    ]:
        try:
            results[key] = fn()
        except Exception as e:
            logger.error(f"Market pulse '{key}' failed: {e}")
            results[key] = {}

    return results


# ──────────────────────────────────────────────
# Live Straddle P&L Sync
# ──────────────────────────────────────────────
def sync_straddle_prices(options_svc: OptionsDataService) -> list:
    """
    Refresh all active straddle positions with live CE/PE LTPs from broker.
    Updates StraddlePosition rows in DB and returns list of updated snapshots.

    Call this before compute_combined_pnl() or get_risk_utilization()
    to ensure P&L numbers reflect live market prices.
    """
    active = list(StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]))
    if not active:
        return []

    options_svc._ensure_broker()
    updated = []

    # Fetch NIFTY spot once (needed for delta calc)
    nifty_data = options_svc.fetch_nifty_spot()
    nifty_spot = nifty_data.get("ltp", 0)

    # Serial fetch — each ltpData call has 0.3s rate-limit guard built in
    ltp_results = {}  # (pos.id, "ce"|"pe") → ltp float
    for pos in active:
        for leg, symbol, token in [("ce", pos.ce_symbol, pos.ce_token),
                                    ("pe", pos.pe_symbol, pos.pe_token)]:
            try:
                data = options_svc.fetch_option_ltp(symbol, token)
                ltp_results[(pos.id, leg)] = data.get("ltp", 0.0)
            except Exception as e:
                logger.error(f"Straddle LTP fetch failed for {leg} pos {pos.id}: {e}")

    # Update each position with fresh prices
    for pos in active:
        ce_ltp = ltp_results.get((pos.id, "ce"), pos.ce_current_price)
        pe_ltp = ltp_results.get((pos.id, "pe"), pos.pe_current_price)

        pos.ce_current_price = ce_ltp
        pos.pe_current_price = pe_ltp

        # Recompute P&L
        net_pnl_pts = (pos.ce_sell_price + pos.pe_sell_price) - (ce_ltp + pe_ltp)
        pos.current_pnl_inr = net_pnl_pts * pos.lot_size * pos.lots

        # Recompute delta (approximate) — needs NIFTY spot, not option LTP
        if nifty_spot > 0:
            from trading.options.straddle.analyzer import _approx_delta
            dte = max(0, (pos.expiry - date.today()).days)
            ce_delta = -_approx_delta(nifty_spot, pos.strike, "CE", dte)  # short
            pe_delta = -_approx_delta(nifty_spot, pos.strike, "PE", dte)  # short
            pos.net_delta = ce_delta + pe_delta

        pos.save(update_fields=[
            "ce_current_price", "pe_current_price",
            "current_pnl_inr", "net_delta", "last_updated",
        ])

        updated.append({
            "id": pos.id,
            "underlying": pos.underlying,
            "strike": pos.display_strike,
            "ce_strike": pos.ce_strike_actual,
            "pe_strike": pos.pe_strike_actual,
            "ce_ltp": ce_ltp,
            "pe_ltp": pe_ltp,
            "pnl_inr": pos.current_pnl_inr,
            "net_delta": pos.net_delta,
            "status": pos.status,
        })

    logger.info(f"Synced {len(updated)} straddle positions with live prices")
    return updated


# ──────────────────────────────────────────────
# Straddle Analysis (pure Python — no LLM)
# ──────────────────────────────────────────────
def run_straddle_analysis(
    options_svc: OptionsDataService,
    position_id: int,
    include_candles: bool = True,
) -> dict:
    """
    Run the full straddle analyzer for a position using live market data.
    Returns the StraddleAnalysis as a dict (includes summary_text for display).

    This is the same analyzer used by the LangGraph straddle workflow,
    but callable directly from the dashboard for instant read-only analysis.
    """
    try:
        pos = StraddlePosition.objects.get(id=position_id)
    except StraddlePosition.DoesNotExist:
        return {"error": f"Position {position_id} not found"}

    # Fetch all market data in one parallel call
    snapshot = options_svc.fetch_straddle_snapshot(
        ce_symbol=pos.ce_symbol,
        ce_token=pos.ce_token,
        pe_symbol=pos.pe_symbol,
        pe_token=pos.pe_token,
        date_str=date.today().isoformat(),
        include_candles=include_candles,
    )

    nifty = snapshot.get("nifty", {})
    vix = snapshot.get("vix", {})
    ce = snapshot.get("ce", {})
    pe = snapshot.get("pe", {})

    if not nifty.get("ltp"):
        return {"error": "Failed to fetch NIFTY spot data"}

    analysis = analyze_straddle(
        underlying=pos.underlying,
        strike=pos.strike,
        expiry=pos.expiry.isoformat(),
        lot_size=pos.lot_size,
        lots=pos.lots,
        ce_sell_price=pos.ce_sell_price,
        pe_sell_price=pos.pe_sell_price,
        ce_ltp=ce.get("ltp", 0),
        pe_ltp=pe.get("ltp", 0),
        nifty_spot=nifty.get("ltp", 0),
        nifty_prev_close=nifty.get("prev_close", 0),
        vix_current=vix.get("ltp", 0),
        vix_prev_close=vix.get("prev_close", 0),
        candles=snapshot.get("candles", []),
    )

    return analysis.model_dump()


# ──────────────────────────────────────────────
# Combined P&L (equity + options) — LIVE
# ──────────────────────────────────────────────
def compute_combined_pnl(live: bool = False, options_svc: OptionsDataService = None) -> dict:
    """
    Sum equity daily P&L + all active straddle P&L.

    Args:
        live: If True and options_svc provided, sync straddle prices first.
        options_svc: Required when live=True.
    """
    equity_pnl = 0.0
    options_pnl = 0.0

    # Sync live prices if requested
    if live and options_svc:
        sync_straddle_prices(options_svc)

    try:
        snap = PortfolioSnapshot.objects.latest()
        equity_pnl = snap.daily_pnl
    except PortfolioSnapshot.DoesNotExist:
        pass

    active_straddles = StraddlePosition.objects.filter(status="ACTIVE")
    for pos in active_straddles:
        options_pnl += pos.total_pnl

    return {
        "equity_pnl": equity_pnl,
        "options_pnl": options_pnl,
        "total_pnl": equity_pnl + options_pnl,
    }


# ──────────────────────────────────────────────
# Broker Holdings & Positions
# ──────────────────────────────────────────────
def get_broker_holdings(options_svc: OptionsDataService = None) -> list:
    """Fetch current portfolio holdings from broker (delivery positions)."""
    from trading.services.data_service import BrokerClient
    b = BrokerClient.get_instance()
    b.ensure_login()
    try:
        holdings = b.fetch_holdings()
        if isinstance(holdings, dict) and "data" in holdings:
            return holdings["data"] or []
        if isinstance(holdings, list):
            return holdings
        return []
    except Exception as e:
        logger.error(f"Holdings fetch failed: {e}")
        return []


def get_broker_positions(options_svc: OptionsDataService = None) -> dict:
    """Fetch open positions from broker (intraday + carryforward)."""
    from trading.services.data_service import BrokerClient
    b = BrokerClient.get_instance()
    b.ensure_login()
    try:
        positions = b.fetch_positions()
        if isinstance(positions, dict) and "data" in positions:
            return positions["data"] or {}
        if isinstance(positions, dict):
            return positions
        return {}
    except Exception as e:
        logger.error(f"Positions fetch failed: {e}")
        return {}


def get_order_book(options_svc: OptionsDataService = None) -> list:
    """Fetch today's order book from broker."""
    from trading.services.data_service import BrokerClient
    b = BrokerClient.get_instance()
    b.ensure_login()
    try:
        return b.fetch_order_book()
    except Exception as e:
        logger.error(f"Order book fetch failed: {e}")
        return []


# ──────────────────────────────────────────────
# Portfolio Context (from RAG retriever)
# ──────────────────────────────────────────────
def get_portfolio_summary() -> dict:
    """
    Comprehensive portfolio summary combining DB snapshot + active positions.
    Used by dashboard Command Center for the portfolio overview panel.
    """
    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = snap.capital
        invested = snap.invested
        available = snap.available_cash
        daily_pnl = snap.daily_pnl
        total_pnl = snap.total_pnl
        daily_loss = snap.daily_loss
        open_pos = snap.open_positions
        snap_date = snap.snapshot_date.isoformat()
    except PortfolioSnapshot.DoesNotExist:
        capital = DEFAULT_CAPITAL
        invested = 0.0
        available = DEFAULT_CAPITAL
        daily_pnl = 0.0
        total_pnl = 0.0
        daily_loss = 0.0
        open_pos = 0
        snap_date = date.today().isoformat()

    # Active straddle stats
    active_straddles = StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"])
    straddle_count = active_straddles.count()
    straddle_pnl = sum(p.total_pnl for p in active_straddles)
    straddle_premium = sum(p.total_premium_sold for p in active_straddles)

    # Today's equity trades
    today_trades = TradeJournal.objects.filter(trade_date=date.today())
    today_count = today_trades.count()
    today_wins = today_trades.filter(pnl__gt=0).count()
    today_losses = today_trades.filter(pnl__lt=0).count()

    return {
        "capital": capital,
        "invested": invested,
        "available_cash": available,
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
        "daily_loss": daily_loss,
        "open_positions": open_pos,
        "snapshot_date": snap_date,
        "straddle_count": straddle_count,
        "straddle_pnl": straddle_pnl,
        "straddle_premium_sold": straddle_premium,
        "today_trades": today_count,
        "today_wins": today_wins,
        "today_losses": today_losses,
        "combined_pnl": daily_pnl + straddle_pnl,
    }


def get_portfolio_context_text() -> str:
    """
    Get formatted portfolio context string from RAG retriever.
    Useful for feeding into LLM prompts or displaying as text.
    """
    from trading.rag.retriever import retrieve_portfolio_context
    return retrieve_portfolio_context()


# ──────────────────────────────────────────────
# Risk Utilization
# ──────────────────────────────────────────────
def get_risk_utilization() -> dict:
    """Current risk utilization vs limits."""
    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = snap.capital
        daily_loss = snap.daily_loss
        invested = snap.invested
        open_pos = snap.open_positions
    except PortfolioSnapshot.DoesNotExist:
        capital = DEFAULT_CAPITAL
        daily_loss = 0.0
        invested = 0.0
        open_pos = 0

    max_daily_loss = capital * (MAX_DAILY_LOSS_PCT / 100)
    max_position_value = capital * (MAX_POSITION_SIZE_PCT / 100)

    daily_loss_pct = (daily_loss / max_daily_loss * 100) if max_daily_loss > 0 else 0
    capital_deployed_pct = (invested / capital * 100) if capital > 0 else 0

    # Count underwater options
    underwater_count = 0
    active_straddles = list(StraddlePosition.objects.filter(status="ACTIVE"))
    for pos in active_straddles:
        if pos.combined_current_pts > pos.combined_sell_pts:
            underwater_count += 1

    # Options margin exposure (approximate)
    options_margin = sum(p.total_premium_sold for p in active_straddles)

    return {
        "capital": capital,
        "daily_loss": daily_loss,
        "daily_loss_pct": min(daily_loss_pct, 100),
        "max_daily_loss": max_daily_loss,
        "daily_loss_limit_pct": MAX_DAILY_LOSS_PCT,
        "capital_deployed": invested,
        "capital_deployed_pct": min(capital_deployed_pct, 100),
        "max_position_value": max_position_value,
        "open_positions": open_pos,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "underwater_options": underwater_count,
        "active_straddles": len(active_straddles),
        "options_margin_exposure": options_margin,
        "total_exposure": invested + options_margin,
        "total_exposure_pct": min(((invested + options_margin) / capital * 100) if capital > 0 else 0, 100),
    }


def compute_risk_status(risk: dict) -> str:
    """Return GREEN / YELLOW / RED based on risk utilization."""
    if risk["daily_loss_pct"] >= 80 or risk["underwater_options"] > 0:
        return "RED"
    if risk["daily_loss_pct"] >= 50 or risk["open_positions"] >= risk["max_open_positions"] - 1:
        return "YELLOW"
    return "GREEN"


# ──────────────────────────────────────────────
# Holistic Exposure View
# ──────────────────────────────────────────────
def get_exposure_breakdown() -> dict:
    """
    Full exposure breakdown across equity + options.
    Shows exactly where capital is deployed and total risk.
    """
    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = snap.capital
        equity_invested = snap.invested
    except PortfolioSnapshot.DoesNotExist:
        capital = DEFAULT_CAPITAL
        equity_invested = 0.0

    # Equity positions
    today = date.today()
    equity_positions = TradeJournal.objects.filter(
        status__in=["EXECUTED", "FILLED", "PAPER"],
        trade_date=today,
    )
    equity_risk = sum(t.risk_amount for t in equity_positions)
    equity_count = equity_positions.count()

    # Options positions
    active_straddles = list(StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]))
    options_premium_sold = sum(p.total_premium_sold for p in active_straddles)
    options_current_pnl = sum(p.total_pnl for p in active_straddles)
    options_count = len(active_straddles)

    # Max loss scenario for options (if all straddles go fully ITM)
    options_max_risk = 0.0
    for pos in active_straddles:
        # Worst case: one leg goes deep ITM, other expires worthless
        # Approximate max loss = 2x premium sold (rule of thumb)
        options_max_risk += pos.total_premium_sold * 2

    total_at_risk = equity_risk + max(0, -options_current_pnl)

    return {
        "capital": capital,
        "equity": {
            "invested": equity_invested,
            "risk_amount": equity_risk,
            "position_count": equity_count,
            "pct_of_capital": (equity_invested / capital * 100) if capital > 0 else 0,
        },
        "options": {
            "premium_sold": options_premium_sold,
            "current_pnl": options_current_pnl,
            "position_count": options_count,
            "max_risk_estimate": options_max_risk,
            "pct_of_capital": (options_premium_sold / capital * 100) if capital > 0 else 0,
        },
        "total_at_risk": total_at_risk,
        "total_at_risk_pct": (total_at_risk / capital * 100) if capital > 0 else 0,
        "available_capital": capital - equity_invested,
    }


# ──────────────────────────────────────────────
# Active Alerts
# ──────────────────────────────────────────────
def get_active_alerts(market_pulse: dict = None, risk: dict = None) -> list:
    """Check conditions that need human attention. Returns list of alert dicts."""
    alerts = []

    # 1. Daily loss approaching limit
    if risk is None:
        risk = get_risk_utilization()
    if risk["daily_loss_pct"] >= 80:
        alerts.append({
            "severity": "critical",
            "message": f"Daily loss at {risk['daily_loss_pct']:.0f}% of limit ({risk['daily_loss']:.0f}/{risk['max_daily_loss']:.0f} INR)",
            "action": "Consider stopping trading for the day",
        })
    elif risk["daily_loss_pct"] >= 50:
        alerts.append({
            "severity": "warning",
            "message": f"Daily loss at {risk['daily_loss_pct']:.0f}% of limit",
            "action": "Monitor closely",
        })

    # 2. Underwater options
    active_straddles = StraddlePosition.objects.filter(status="ACTIVE")
    for pos in active_straddles:
        if pos.combined_current_pts > pos.combined_sell_pts:
            alerts.append({
                "severity": "critical",
                "message": f"{pos.underlying} {pos.strike} straddle is UNDERWATER (P&L: {pos.current_pnl_inr:+,.0f} INR)",
                "action": "Consider closing immediately",
            })

        # Expiry tomorrow
        dte = max(0, (pos.expiry - date.today()).days)
        if dte <= 1:
            alerts.append({
                "severity": "warning",
                "message": f"{pos.underlying} {pos.strike} straddle expires {'TODAY' if dte == 0 else 'TOMORROW'}",
                "action": "Close before 3:15 PM",
            })

    # 3. VIX spike
    if market_pulse and market_pulse.get("vix"):
        vix = market_pulse["vix"]
        vix_ltp = vix.get("ltp", 0)
        vix_prev = vix.get("prev_close", 0)
        if vix_prev > 0:
            vix_change = (vix_ltp - vix_prev) / vix_prev * 100
            if vix_change > 10:
                alerts.append({
                    "severity": "critical",
                    "message": f"VIX spiked {vix_change:+.1f}% ({vix_prev:.1f} → {vix_ltp:.1f})",
                    "action": "Review all short option positions",
                })
            elif vix_change > 5:
                alerts.append({
                    "severity": "warning",
                    "message": f"VIX up {vix_change:+.1f}%",
                    "action": "Monitor option positions",
                })

    # 4. Max positions reached
    if risk["open_positions"] >= risk["max_open_positions"]:
        alerts.append({
            "severity": "info",
            "message": f"Max open positions reached ({risk['open_positions']}/{risk['max_open_positions']})",
            "action": "No new trades until a position is closed",
        })

    # 5. Total exposure warning
    if risk.get("total_exposure_pct", 0) > 80:
        alerts.append({
            "severity": "warning",
            "message": f"Total capital exposure at {risk['total_exposure_pct']:.0f}% (equity + options)",
            "action": "Reduce position sizes or close a position",
        })

    return alerts


# ──────────────────────────────────────────────
# AI Activity Feed
# ──────────────────────────────────────────────
def get_ai_activity_feed(limit: int = 15) -> list:
    """Last N audit log entries formatted for display."""
    entries = AuditLog.objects.order_by("-created_at")[:limit]
    feed = []
    for entry in entries:
        feed.append({
            "time": entry.created_at.strftime("%H:%M:%S"),
            "type": entry.event_type,
            "symbol": entry.symbol or "",
            "detail": _format_audit_detail(entry),
        })
    return feed


def _format_audit_detail(entry) -> str:
    """Human-readable one-liner from AuditLog entry."""
    et = entry.event_type
    sym = entry.symbol or "—"

    if et == "PLANNER_REQ":
        return f"Planning trade for {sym}"
    elif et == "PLANNER_RES":
        return f"AI planned trade for {sym}"
    elif et == "PLANNER_ERR":
        return f"AI planner error for {sym}"
    elif et == "RISK_APPROVE":
        return f"Risk APPROVED {sym}"
    elif et == "RISK_REJECT":
        reason = ""
        if entry.risk_details and isinstance(entry.risk_details, dict):
            reason = entry.risk_details.get("reason", "")
        return f"Risk REJECTED {sym}: {reason}" if reason else f"Risk REJECTED {sym}"
    elif et == "EXECUTION":
        return f"Order executed for {sym}"
    elif et == "RECONCILE":
        return f"Position reconciliation"
    return et


# ──────────────────────────────────────────────
# Active Positions (unified equity + options)
# ──────────────────────────────────────────────
def get_active_positions() -> dict:
    """All open positions — equity from TradeJournal + options from StraddlePosition."""
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
    for pos in StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]):
        option_positions.append({
            "id": pos.id,
            "underlying": pos.underlying,
            "strike": pos.display_strike,
            "ce_strike": pos.ce_strike_actual,
            "pe_strike": pos.pe_strike_actual,
            "expiry": pos.expiry.isoformat(),
            "lots": pos.lots,
            "lot_size": pos.lot_size,
            "ce_sell": pos.ce_sell_price,
            "pe_sell": pos.pe_sell_price,
            "ce_current": pos.ce_current_price,
            "pe_current": pos.pe_current_price,
            "net_delta": pos.net_delta,
            "pnl_inr": pos.total_pnl,
            "realized_pnl": pos.realized_pnl,
            "unrealized_pnl": pos.current_pnl_inr,
            "status": pos.status,
            "dte": max(0, (pos.expiry - today).days),
        })

    return {"equity": equity_positions, "options": option_positions}


# ──────────────────────────────────────────────
# Journal Analytics
# ──────────────────────────────────────────────
def get_journal_analytics(days: int = 30) -> dict:
    """Compute win streaks, per-symbol stats, confidence calibration."""
    cutoff = date.today() - timedelta(days=days)
    trades = list(
        TradeJournal.objects.filter(
            trade_date__gte=cutoff,
            pnl__isnull=False,
        ).order_by("created_at").values("symbol", "side", "pnl", "confidence", "trade_date")
    )

    if not trades:
        return {"trades": 0, "symbols": {}, "streaks": [], "calibration": []}

    # Per-symbol P&L
    symbols = {}
    total_pnl = 0.0
    for t in trades:
        sym = t["symbol"]
        if sym not in symbols:
            symbols[sym] = {"wins": 0, "losses": 0, "total_pnl": 0}
        symbols[sym]["total_pnl"] += t["pnl"]
        total_pnl += t["pnl"]
        if t["pnl"] > 0:
            symbols[sym]["wins"] += 1
        elif t["pnl"] < 0:
            symbols[sym]["losses"] += 1

    # Streaks
    streaks = []
    current_streak = 0
    for t in trades:
        if t["pnl"] > 0:
            current_streak = max(0, current_streak) + 1
        elif t["pnl"] < 0:
            current_streak = min(0, current_streak) - 1
        streaks.append(current_streak)

    max_win_streak = max(streaks) if streaks else 0
    max_loss_streak = abs(min(streaks)) if streaks else 0
    current = streaks[-1] if streaks else 0

    # Confidence calibration (bucket by 0.1 intervals)
    calibration = []
    for low in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        high = low + 0.1
        bucket = [t for t in trades if low <= t["confidence"] < high]
        if bucket:
            wins = sum(1 for t in bucket if t["pnl"] > 0)
            calibration.append({
                "range": f"{low:.1f}-{high:.1f}",
                "count": len(bucket),
                "win_rate": wins / len(bucket) * 100,
            })

    # Win rate
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades) * 100) if trades else 0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(trades) if trades else 0,
        "symbols": symbols,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "current_streak": current,
        "calibration": calibration,
    }


# ──────────────────────────────────────────────
# Straddle History & Analytics
# ──────────────────────────────────────────────
def get_straddle_history(days: int = 90) -> dict:
    """Analytics for closed straddle positions."""
    cutoff = date.today() - timedelta(days=days)
    closed = list(StraddlePosition.objects.filter(
        status="CLOSED",
        trade_date__gte=cutoff,
    ))

    if not closed:
        return {"count": 0, "total_pnl": 0, "win_rate": 0, "positions": []}

    total_pnl = sum(p.total_pnl for p in closed)
    wins = sum(1 for p in closed if p.total_pnl > 0)

    positions = []
    for p in closed:
        positions.append({
            "id": p.id,
            "underlying": p.underlying,
            "strike": p.strike,
            "expiry": p.expiry.isoformat(),
            "trade_date": p.trade_date.isoformat(),
            "pnl_inr": p.total_pnl,
            "premium_sold": p.total_premium_sold,
            "action_taken": p.action_taken,
            "lots": p.lots,
        })

    return {
        "count": len(closed),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(closed),
        "win_rate": (wins / len(closed) * 100) if closed else 0,
        "wins": wins,
        "losses": len(closed) - wins,
        "positions": positions,
    }


# ──────────────────────────────────────────────
# System Controls (pause/resume AI)
# ──────────────────────────────────────────────
def get_system_control(key: str, default=None):
    """Read a system control flag from DB."""
    from trading.models import SystemControl
    try:
        ctrl = SystemControl.objects.get(key=key)
        return ctrl.value
    except SystemControl.DoesNotExist:
        return default


def set_system_control(key: str, value):
    """Set a system control flag in DB."""
    from trading.models import SystemControl
    SystemControl.objects.update_or_create(key=key, defaults={"value": value})


def is_ai_paused() -> bool:
    """Check if AI trading is paused."""
    return get_system_control("ai_trading_paused", False) is True


def pause_ai_trading():
    set_system_control("ai_trading_paused", True)
    logger.warning("AI TRADING PAUSED by user")


def resume_ai_trading():
    set_system_control("ai_trading_paused", False)
    logger.info("AI trading RESUMED by user")


# ──────────────────────────────────────────────
# Market Hours Check
# ──────────────────────────────────────────────
def is_market_open() -> bool:
    """Check if Indian market is currently open (9:15 AM - 3:30 PM IST, weekdays)."""
    now = datetime.now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    return market_open <= now <= market_close


def get_market_session_info() -> dict:
    """Detailed market session info for dashboard header."""
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_open = is_market_open()

    if is_open:
        elapsed = (now - market_open_time).total_seconds()
        remaining = (market_close_time - now).total_seconds()
        total = (market_close_time - market_open_time).total_seconds()
        progress_pct = (elapsed / total * 100) if total > 0 else 0
    else:
        elapsed = 0
        remaining = 0
        progress_pct = 100 if now > market_close_time and now.weekday() < 5 else 0

    return {
        "is_open": is_open,
        "is_weekday": now.weekday() < 5,
        "current_time": now.strftime("%H:%M:%S"),
        "market_open": "09:15",
        "market_close": "15:30",
        "elapsed_minutes": int(elapsed / 60) if is_open else 0,
        "remaining_minutes": int(remaining / 60) if is_open else 0,
        "progress_pct": min(progress_pct, 100),
        "session_phase": _get_session_phase(now, is_open),
    }


def _get_session_phase(now: datetime, is_open: bool) -> str:
    """Classify current trading session phase."""
    if not is_open:
        if now.weekday() >= 5:
            return "WEEKEND"
        if now.hour < 9 or (now.hour == 9 and now.minute < 15):
            return "PRE_MARKET"
        return "POST_MARKET"
    if now.hour == 9 and now.minute < 30:
        return "OPENING"
    if now.hour >= 14 and now.minute >= 45:
        return "CLOSING"
    return "REGULAR"
