"""What-If Position Sizer — stateless pre-trade simulator.

Given a hypothetical {symbol, qty, side, stop}, project the post-trade
portfolio state:
  - margin used by the new leg
  - free cash + leverage ratio after entry
  - worst-case loss at stop
  - distance to the 3% daily-loss cap

No DB writes. Pure calculation over current open positions.
"""
from __future__ import annotations

from typing import Any

from apps.common.margin_calc import equity_margin, short_straddle_margin


_DAILY_LOSS_CAP_PCT = 3.0   # mirrors MAX_DAILY_LOSS_PCT in CLAUDE.md


def _current_used_margin() -> float:
    """Sum of margins for everything currently open."""
    used = 0.0
    try:
        from trading.models import TradeJournal, StraddlePosition
        for t in TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER", "APPROVED")):
            m = equity_margin(t.side, int(t.quantity or 0), float(t.entry_price or 0), product="MIS")
            used += float(getattr(m, "total_margin", 0) or 0)
        for p in StraddlePosition.objects.filter(status="ACTIVE"):
            m = short_straddle_margin(
                lots=int(p.lots), lot_size=int(p.lot_size), strike=float(p.strike),
                ce_premium=float(p.ce_sell_price), pe_premium=float(p.pe_sell_price),
            )
            used += float(getattr(m, "total_margin", 0) or 0)
    except Exception:  # noqa: BLE001
        pass
    return used


def _current_capital() -> float:
    try:
        from trading.models import PortfolioSnapshot
        snap = PortfolioSnapshot.objects.latest()
        return float(snap.capital)
    except Exception:  # noqa: BLE001
        return 500_000.0


def _current_realised_pnl_today() -> float:
    from datetime import date
    try:
        from trading.models import TradeJournal
        today = date.today()
        rows = TradeJournal.objects.filter(timestamp__date=today)
        return float(sum((t.pnl or 0) for t in rows))
    except Exception:  # noqa: BLE001
        return 0.0


def _estimate_entry_price(symbol: str, fallback: float) -> float:
    try:
        from apps.market_data.services.data_port import DefaultMarketData
        port = DefaultMarketData(tenant_id=None)
        v = port.ltp(symbol)
        return float(v) if v else fallback
    except Exception:  # noqa: BLE001
        return fallback


def simulate(payload: dict) -> dict[str, Any]:
    symbol = (payload.get("symbol") or "").upper()
    qty = int(payload.get("qty") or 0)
    side = (payload.get("side") or "BUY").upper()
    stop = float(payload.get("stop") or 0)
    entry_hint = float(payload.get("entry") or 0)
    product = (payload.get("product") or "MIS").upper()

    if not symbol or qty <= 0:
        return {"error": "symbol and qty>0 are required"}

    entry = entry_hint if entry_hint > 0 else _estimate_entry_price(symbol, fallback=0.0)
    if entry <= 0:
        return {"error": f"no live price for {symbol} — supply ?entry= to simulate"}

    cap = _current_capital()
    used = _current_used_margin()
    realised_today = _current_realised_pnl_today()

    new_leg = equity_margin(side, qty, entry, product=product)
    new_margin = float(getattr(new_leg, "total_margin", 0) or 0)

    post_used = used + new_margin
    free_cash = max(cap - post_used, 0.0)
    leverage = (post_used / cap) if cap > 0 else 0.0

    worst_loss = 0.0
    if stop > 0:
        sign = 1 if side in ("BUY", "LONG") else -1
        worst_loss = abs((entry - stop) * sign) * qty

    daily_loss_cap_inr = cap * (_DAILY_LOSS_CAP_PCT / 100.0)
    available_loss_room = max(daily_loss_cap_inr + realised_today, 0.0)   # realised is signed
    distance_pct = (available_loss_room / cap * 100.0) if cap > 0 else 0.0

    return {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "entry": entry,
        "stop": stop,
        "post_trade_delta": {
            "margin_added": round(new_margin, 2),
            "notional_added": round(qty * entry, 2),
        },
        "capital": cap,
        "margin_used_before": round(used, 2),
        "margin_used": round(post_used, 2),
        "free_cash": round(free_cash, 2),
        "leverage_ratio": round(leverage, 3),
        "worst_case_loss_inr": round(worst_loss, 2),
        "distance_to_daily_loss_cap_pct": round(distance_pct, 2),
        "daily_loss_cap_inr": round(daily_loss_cap_inr, 2),
        "realised_pnl_today": round(realised_today, 2),
    }
